#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Dédoublonnage INTER-PHOTOS des détections (pipeline
"application").

Distinct du dédoublonnage INTRA-photo (recouvrement des tuiles 640px internes
à UNE photo, déjà résolu par `src/review/tiled_inference.py::nms_merge`,
réutilisé tel quel une fois par photo AVANT d'arriver ici). Ce module traite
l'AUTRE source de doublons : un même déchet réel photographié plusieurs fois
(jusqu'à ~25x, recouvrement de vol ~80%) par des PHOTOS DIFFÉRENTES, chacune
avec son propre repère pixel - impossible à comparer par IoU en pixels,
d'où la reprojection en coordonnées sol (mètres) faite par geolocation.py en
amont.

Méthode :
1. Grouper par classe (deux détections de classes différentes ne sont jamais
   candidates à un doublon - même convention que `nms_merge`/`matching.py`).
2. Regrouper en clusters ("même objet réel vu plusieurs fois") par IoU des
   polygones REPROJETÉS AU SOL >= `iou_threshold` - Union-Find plutôt qu'un
   appariement par paires, car un objet peut être vu par plus de deux photos
   à la fois. Index spatial (STRtree) pour éviter une comparaison O(n²) sur
   un batch de plusieurs milliers de détections.
   Pourquoi l'IoU au sol plutôt qu'une distance entre centroïdes GPS : la
   précision GPS du DJI Air 2S en GNSS seul est ±1.5m (spec DJI officielle) -
   largement suffisant pour confondre deux déchets DISTINCTS mais proches
   (ex : un tas de déchets rassemblés, où plusieurs objets peuvent être à
   moins d'1m les uns des autres) si on ne comparait que des points. L'IoU de
   forme reste discriminant même avec du bruit GPS, contrairement à une
   distance de centroïdes.
3. Dans chaque cluster, sélection du "gagnant", DANS CET ORDRE :
   a. Disqualification : un masque touchant le bord de sa photo D'ORIGINE
      (pas un bord de tuile 640px, déjà géré ailleurs) est écarté SI le
      cluster contient au moins une version qui ne touche aucun bord. Si
      TOUTES les versions touchent un bord (aucune vue complète disponible),
      aucune disqualification n'est possible - on garde tout le monde pour
      l'étape suivante.
   b. Parmi les survivants, on garde celui dont le masque est le plus proche
      du CENTRE de sa photo d'origine (pas le plus grand : un objet proche du
      bord d'une photo peut être coupé/déformé par la perspective, le
      centrage est un meilleur indicateur de fidélité - critère tranché).
   c. Égalité de centrage (rare) : on départage par confiance décroissante.

Exemple :
    from src.application.dedup import PhotoDetection, deduplicate_detections
    winners = deduplicate_detections(all_detections, iou_threshold=0.3)

Limite connue de l'IoU seul : un même objet réel vu par plusieurs photos très
rapprochées peut avoir un IoU quasi nul entre ses reprojections - le
géoréférencement direct (sans ajustement de faisceaux inter-photos, voir
geolocation.py) introduit assez de bruit de position/cap PAR PHOTO pour que
la FORME reprojetée d'un même objet ne se recoupe plus du tout d'une photo à
l'autre, même si le point reste proche. Baisser `iou_threshold` ne règle pas
ce cas sans risquer de fusionner à tort des objets distincts dans une zone
dense.

`deduplicate_detections_visual` (voir plus bas) est la réponse à cette
limite : décorréler le critère de correspondance de la géométrie reprojetée
(peu fiable dans ce cas) en comparant l'APPARENCE des chips au moyen d'un
modèle de similarité visuelle pré-entraîné (voir visual_similarity.py) -
candidats générés par PROXIMITÉ DE CENTROÏDE (pas par intersection de
polygone, qui rate justement les cas ci-dessus), fusion décidée par
similarité visuelle >= seuil, EN PLUS de l'IoU (une vraie forte intersection
reste un signal gratuit et fiable, pas besoin d'un embedding pour ça - voir
_cluster_by_iou_or_visual_similarity). `deduplicate_detections` (IoU seul,
ci-dessous) reste disponible telle quelle - utile en comparaison, ou pour un
batch où le bruit de géoréférencement serait un jour réduit
(calibration/correction de cap - question ouverte).
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from src.review.matching import polygon_iou

# Marge de tolérance (pixels) pour considérer qu'un polygone "touche" le bord
# de sa photo d'origine - un contour de segmentation peut avoir un vertex à
# 1-2px du bord réel par bruit de discrétisation sans que l'objet soit
# vraiment coupé ; au-delà, l'objet est considéré intact.
_BORDER_MARGIN_PX = 3.0


@dataclass
class PhotoDetection:
    """Une détection unique, telle que produite par l'inférence sur UNE
    photo (avant tout dédoublonnage inter-photos). `pixel_polygon` reste dans
    le repère de SA photo d'origine (nécessaire pour juger bord/centrage) ;
    `local_polygon` est la même géométrie reprojetée au sol en mètres
    (nécessaire pour comparer entre photos différentes - voir
    geolocation.pixel_polygon_to_local_polygon).

    `embedding` (optionnel) : vecteur de similarité visuelle pré-calculé
    (voir visual_similarity.py), nécessaire UNIQUEMENT pour
    `deduplicate_detections_visual` - None pour un usage IoU seul
    (`deduplicate_detections`) ou dans les tests synthétiques existants, qui
    restent valides sans le renseigner."""
    photo_id: str
    class_id: int
    confidence: float
    pixel_polygon: Polygon
    local_polygon: Polygon
    photo_width_px: int
    photo_height_px: int
    embedding: Optional[np.ndarray] = None


class DedupReport(NamedTuple):
    n_input: int
    n_output: int
    n_clusters: int
    cluster_sizes: List[int]
    n_border_disqualified: int


def touches_image_border(polygon: Polygon, width_px: int, height_px: int, margin_px: float = _BORDER_MARGIN_PX) -> bool:
    """Vrai si le polygone (coordonnées pixel de SA photo d'origine) touche
    ou dépasse le bord de l'image, à `margin_px` près."""
    minx, miny, maxx, maxy = polygon.bounds
    return (
        minx <= margin_px
        or miny <= margin_px
        or maxx >= width_px - margin_px
        or maxy >= height_px - margin_px
    )


def normalized_center_distance(polygon: Polygon, width_px: int, height_px: int) -> float:
    """Distance du centroïde du polygone au centre de l'image, normalisée par
    la demi-diagonale de l'image (0 = plein centre, 1 = coin de l'image) -
    normalisation nécessaire pour rester comparable même si toutes les photos
    du batch n'ont pas exactement la même résolution."""
    cx, cy = width_px / 2.0, height_px / 2.0
    dist = ((polygon.centroid.x - cx) ** 2 + (polygon.centroid.y - cy) ** 2) ** 0.5
    half_diag = ((width_px / 2.0) ** 2 + (height_px / 2.0) ** 2) ** 0.5
    return dist / half_diag if half_diag > 0 else 0.0


class _UnionFind:
    """Union-Find (disjoint set) minimal - un objet réel peut être vu par
    plus de deux photos à la fois, un simple appariement par paires
    (type matching.py) ne suffirait pas à regrouper un cluster de N vues."""

    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]  # path halving
            i = self._parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _cluster_by_iou(detections: List[PhotoDetection], iou_threshold: float) -> List[List[int]]:
    """Regroupe les indices de `detections` (déjà filtrés sur UNE classe) en
    clusters connectés par IoU au sol >= `iou_threshold`. Utilise un index
    spatial (STRtree) pour ne comparer que les paires dont les empreintes se
    recoupent réellement, plutôt qu'une comparaison O(n²) systématique -
    nécessaire dès que le batch produit plusieurs milliers de détections
    brutes (jusqu'à ~25 vues x nombreux déchets par photo)."""
    n = len(detections)
    if n <= 1:
        return [[i] for i in range(n)]

    geoms = [d.local_polygon for d in detections]
    tree = STRtree(geoms)
    uf = _UnionFind(n)

    for i, geom in enumerate(geoms):
        candidate_idxs = tree.query(geom, predicate="intersects")
        for j in candidate_idxs:
            j = int(j)
            if j <= i:
                continue  # évite de tester deux fois la même paire (i,j)/(j,i)
            if polygon_iou(geom, geoms[j]) >= iou_threshold:
                uf.union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())


def select_best_in_cluster(cluster: List[PhotoDetection]) -> Tuple[PhotoDetection, bool]:
    """Applique la règle de sélection décrite dans la docstring du module
    à un cluster de détections jugées "même objet réel".

    Retourne (détection gagnante, au_moins_une_disqualification_appliquée) -
    le 2e élément sert uniquement au reporting (DedupReport ci-dessous).
    """
    if len(cluster) == 1:
        return cluster[0], False

    touches = [touches_image_border(d.pixel_polygon, d.photo_width_px, d.photo_height_px) for d in cluster]
    disqualification_applied = any(touches) and not all(touches)

    if disqualification_applied:
        candidates = [d for d, t in zip(cluster, touches) if not t]
    else:
        # Soit personne ne touche un bord, soit tout le monde - dans les deux
        # cas, le critère de bord ne peut pas départager, on garde tout le monde.
        candidates = cluster

    best = min(
        candidates,
        key=lambda d: (
            normalized_center_distance(d.pixel_polygon, d.photo_width_px, d.photo_height_px),
            -d.confidence,
        ),
    )
    return best, disqualification_applied


def deduplicate_detections(
    detections: List[PhotoDetection], iou_threshold: float = 0.3, return_report: bool = False,
):
    """Point d'entrée principal : dédoublonne une liste de détections
    provenant de PLUSIEURS photos d'un même batch. Regroupe par classe,
    clusterise par IoU au sol, sélectionne un gagnant par cluster.

    `iou_threshold` (0.3 par défaut) : seuil de fusion géométrique - question
    ouverte, pas encore calibré empiriquement sur un batch réel ; exposé en
    paramètre explicite (plutôt que codé en dur) pour faciliter les essais de
    réglage.
    """
    by_class: Dict[int, List[PhotoDetection]] = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    winners: List[PhotoDetection] = []
    cluster_sizes: List[int] = []
    n_disqualified = 0

    for class_id, class_detections in by_class.items():
        clusters_idx = _cluster_by_iou(class_detections, iou_threshold)
        for idxs in clusters_idx:
            cluster = [class_detections[i] for i in idxs]
            best, disqualified = select_best_in_cluster(cluster)
            winners.append(best)
            cluster_sizes.append(len(cluster))
            if disqualified:
                n_disqualified += 1

    if not return_report:
        return winners

    report = DedupReport(
        n_input=len(detections),
        n_output=len(winners),
        n_clusters=len(cluster_sizes),
        cluster_sizes=cluster_sizes,
        n_border_disqualified=n_disqualified,
    )
    return winners, report


def _cluster_by_iou_or_visual_similarity(
    detections: List[PhotoDetection],
    iou_threshold: float,
    similarity_threshold: float,
    distance_threshold_m: float,
) -> List[List[int]]:
    """Comme `_cluster_by_iou`, mais deux détections sont aussi fusionnées si
    leurs CENTROÏDES sont à moins de `distance_threshold_m` ET que leur
    similarité visuelle (cosinus des `embedding`) atteint `similarity_threshold`
    - même sans le moindre recouvrement géométrique (voir la limite de l'IoU
    seul décrite en tête de module).

    Les deux critères de fusion utilisent chacun l'index spatial le moins
    coûteux qui leur suffit, pour éviter de bufferiser un polygone de
    segmentation (souvent des dizaines à quelques centaines de sommets) - une
    opération GEOS coûteuse si répétée pour chaque détection :
    - IoU : index spatial sur les polygones BRUTS (jamais bufferisés),
      `predicate=\"intersects\"` - aucune fusion géométrique n'est possible
      sans intersection réelle, pas besoin d'élargir la recherche.
    - Similarité visuelle : index spatial sur les seuls CENTROÏDES (des
      points, jamais des polygones complets) avec `predicate=\"dwithin\"` -
      une requête de distance native GEOS, sans construire le moindre
      polygone bufferisé côté détection ni côté requête ; son coût ne dépend
      donc jamais de la complexité des contours de segmentation.
    Cette séparation est importante sur une zone dense (justement le genre de
    zone où ce module est le plus utile) : bufferiser systématiquement les
    polygones ferait exploser à la fois le nombre de candidats et le coût par
    candidat, alors qu'une requête de distance sur des points reste bornée."""
    n = len(detections)
    if n <= 1:
        return [[i] for i in range(n)]

    geoms = [d.local_polygon for d in detections]
    centroids = [g.centroid for g in geoms]
    uf = _UnionFind(n)

    # Candidats géométriques (IoU) : polygones bruts, jamais bufferisés.
    poly_tree = STRtree(geoms)
    for i, geom in enumerate(geoms):
        for j in poly_tree.query(geom, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            if polygon_iou(geom, geoms[j]) >= iou_threshold:
                uf.union(i, j)

    # Candidats visuels : proximité de CENTROÏDE (points, pas polygones),
    # `dwithin` fait la recherche par distance directement côté GEOS.
    centroid_tree = STRtree(centroids)
    for i, c in enumerate(centroids):
        emb_i = detections[i].embedding
        if emb_i is None:
            continue  # pas d'embedding calculé (usage géométrique seul)
        for j in centroid_tree.query(c, predicate="dwithin", distance=distance_threshold_m):
            j = int(j)
            if j <= i:
                continue
            emb_j = detections[j].embedding
            if emb_j is None:
                continue
            similarity = float(np.dot(emb_i, emb_j))  # embeddings déjà normalisés L2 - produit scalaire = cosinus
            if similarity >= similarity_threshold:
                uf.union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())


def deduplicate_detections_visual(
    detections: List[PhotoDetection],
    iou_threshold: float = 0.3,
    similarity_threshold: float = 0.5,
    distance_threshold_m: float = 2.0,
    return_report: bool = False,
):
    """Comme `deduplicate_detections`, mais chaque `PhotoDetection` doit avoir
    son `embedding` renseigné (voir visual_similarity.py::compute_embeddings)
    - fusionne par IoU géométrique OU par similarité visuelle entre
    détections proches (voir `_cluster_by_iou_or_visual_similarity` et la
    limite de l'IoU seul en tête de module).

    `similarity_threshold` (0.5 par défaut) : question ouverte - calibré sur
    un seul exemple manuel (similarité 0.55-0.76 observée entre vues
    confirmées d'un même objet, contre 0.40-0.44 pour un objet visuellement
    différent dans le même voisinage), à affiner sur plus d'exemples réels,
    même statut que les autres seuils de ce pipeline (iou_threshold,
    conf_threshold...).

    `distance_threshold_m` (2m par défaut) : marge de recherche des
    candidats - question ouverte également, à resserrer/desserrer selon le
    bruit de position réellement observé sur d'autres batches."""
    by_class: Dict[int, List[PhotoDetection]] = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    winners: List[PhotoDetection] = []
    cluster_sizes: List[int] = []
    n_disqualified = 0

    for class_id, class_detections in by_class.items():
        clusters_idx = _cluster_by_iou_or_visual_similarity(
            class_detections, iou_threshold, similarity_threshold, distance_threshold_m,
        )
        for idxs in clusters_idx:
            cluster = [class_detections[i] for i in idxs]
            best, disqualified = select_best_in_cluster(cluster)
            winners.append(best)
            cluster_sizes.append(len(cluster))
            if disqualified:
                n_disqualified += 1

    if not return_report:
        return winners

    report = DedupReport(
        n_input=len(detections),
        n_output=len(winners),
        n_clusters=len(cluster_sizes),
        cluster_sizes=cluster_sizes,
        n_border_disqualified=n_disqualified,
    )
    return winners, report
