#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Dédoublonnage INTER-PHOTOS des détections (pipeline
"application"). Brique demandée en priorité par Jame le 04/09/2026 - voir
journal_decisions_pipeline.md pour la discussion de conception complète.

Distinct du dédoublonnage INTRA-photo (recouvrement des tuiles 640px internes
à UNE photo, déjà résolu par `src/review/tiled_inference.py::nms_merge`,
réutilisé tel quel une fois par photo AVANT d'arriver ici). Ce module traite
l'AUTRE source de doublons : un même déchet réel photographié plusieurs fois
(jusqu'à ~25x, recouvrement de vol ~80%) par des PHOTOS DIFFÉRENTES, chacune
avec son propre repère pixel - impossible à comparer par IoU en pixels,
d'où la reprojection en coordonnées sol (mètres) faite par geolocation.py en
amont.

Méthode (décidée le 04/09/2026) :
1. Grouper par classe (deux détections de classes différentes ne sont jamais
   candidates à un doublon - même convention que `nms_merge`/`matching.py`).
2. Regrouper en clusters ("même objet réel vu plusieurs fois") par IoU des
   polygones REPROJETÉS AU SOL >= `iou_threshold` - Union-Find plutôt qu'un
   appariement par paires, car un objet peut être vu par plus de deux photos
   à la fois. Index spatial (STRtree) pour éviter une comparaison O(n²) sur
   un batch de plusieurs milliers de détections.
   Pourquoi l'IoU au sol plutôt qu'une distance entre centroïdes GPS : la
   précision GPS du DJI Air 2S en GNSS seul est ±1.5m (spec DJI officielle,
   voir échange du 04/09/2026) - largement suffisant pour confondre deux
   déchets DISTINCTS mais proches (ex : un tas de déchets rassemblés, où
   plusieurs objets peuvent être à moins d'1m les uns des autres) si on ne
   comparait que des points. L'IoU de forme reste discriminant même avec du
   bruit GPS, contrairement à une distance de centroïdes.
3. Dans chaque cluster, sélection du "gagnant" avec les critères choisis par
   Jame le 04/09/2026, DANS CET ORDRE :
   a. Disqualification : un masque touchant le bord de sa photo D'ORIGINE
      (pas un bord de tuile 640px, déjà géré ailleurs) est écarté SI le
      cluster contient au moins une version qui ne touche aucun bord. Si
      TOUTES les versions touchent un bord (aucune vue complète disponible),
      aucune disqualification n'est possible - on garde tout le monde pour
      l'étape suivante.
   b. Parmi les survivants, on garde celui dont le masque est le plus proche
      du CENTRE de sa photo d'origine (pas le plus grand - décision du
      04/09/2026, revenant sur le premier critère envisagé).
   c. Égalité de centrage (rare) : on départage par confiance décroissante.

Exemple :
    from src.application.dedup import PhotoDetection, deduplicate_detections
    winners = deduplicate_detections(all_detections, iou_threshold=0.3)

CORRECTIF (06/09/2026) - le dédoublonnage par IoU seul NE SUFFIT PAS en
pratique : diagnostic sur un vrai batch (voir journal_decisions_pipeline.md,
entrée du 06/09) montrant qu'un même objet réel (une cagette), vu par 7
photos différentes à moins de 0.5m les unes des autres, a un IoU quasi NUL
(0% pour la plupart des paires, max observé 26%) entre ses reprojections -
le géoréférencement direct (sans ajustement de faisceaux inter-photos, voir
geolocation.py) introduit assez de bruit de position/cap PAR PHOTO pour que
la FORME reprojetée d'un même objet ne se recoupe plus du tout d'une photo à
l'autre, même si le point reste proche. Baisser `iou_threshold` déplace le
problème (vérifié : ne fusionne jamais complètement ce genre de cluster même
à 0.02, tout en commençant à fusionner à tort des objets distincts ailleurs
dans une zone dense dès 0.1).

`deduplicate_detections_visual` (voir plus bas) est la réponse : décorréler
le critère de correspondance de la géométrie reprojetée (peu fiable ici) en
comparant l'APPARENCE des chips au moyen d'un modèle de similarité visuelle
pré-entraîné (voir visual_similarity.py) - candidats générés par PROXIMITÉ DE
CENTROÏDE (pas par intersection de polygone, qui rate justement les cas ci-
dessus), fusion décidée par similarité visuelle >= seuil, EN PLUS de l'IoU
(une vraie forte intersection reste un signal gratuit et fiable, pas besoin
d'un embedding pour ça - voir _cluster_by_iou_or_visual_similarity).
`deduplicate_detections` (IoU seul, ci-dessous) reste disponible telle
quelle - utile en comparaison, ou pour un batch où le bruit de
géoréférencement serait un jour réduit (calibration/correction de cap,
voir points ouverts du 04/09).
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

    `embedding` (optionnel, ajouté le 06/09/2026) : vecteur de similarité
    visuelle pré-calculé (voir visual_similarity.py), nécessaire UNIQUEMENT
    pour `deduplicate_detections_visual` - None pour un usage IoU seul
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
    """Applique la règle de sélection du 04/09/2026 (voir docstring du
    module) à un cluster de détections jugées "même objet réel".

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

    `iou_threshold` : pas encore calibré empiriquement (Jame, 04/09/2026 :
    "on fera des essais") - exposé en paramètre explicite pour faciliter le
    réglage plutôt que codé en dur.
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
    - même sans le moindre recouvrement géométrique (voir le correctif du
    06/09/2026 en tête de module : le cas réel qui a motivé cette fonction
    avait un IoU nul entre la plupart des vues d'un même objet).

    CORRECTIF DE PERFORMANCE (06/09/2026, 2e passe) : la toute première
    version de cette fonction cherchait les candidats de proximité en
    bufferisant le POLYGONE COMPLET de chaque détection (`geom.buffer(...)`)
    avant de requêter l'index spatial - sur un vrai batch (essai1, 3044
    détections mono-classe, donc traitées en UN SEUL groupe), ça a fait
    tourner le pipeline plus d'une heure sans terminer (tué manuellement par
    Jame). Cause : bufferiser un polygone de segmentation (souvent des
    dizaines à quelques centaines de sommets, aucune simplification en amont)
    est une opération GEOS coûteuse, répétée pour CHAQUE détection, puis
    testée par intersection exacte contre chaque candidat - sur une zone
    dense (justement le genre de zone où ce module est le plus utile), le
    nombre de candidats et le coût par candidat explosent ensemble.
    Corrigé en séparant les deux critères de fusion, chacun sur l'index le
    moins coûteux qui lui suffit :
    - IoU : index spatial sur les polygones BRUTS (jamais bufferisés),
      `predicate=\"intersects\"` - exactement l'approche déjà éprouvée rapide
      de `_cluster_by_iou` sur ce même batch (aucune fusion géométrique n'est
      possible sans intersection réelle, pas besoin d'élargir la recherche).
    - Similarité visuelle : index spatial sur les seuls CENTROÏDES (des
      points, jamais des polygones complets) avec `predicate=\"dwithin\"` -
      une requête de distance native GEOS, sans construire le moindre
      polygone bufferisé côté détection ni côté requête ; son coût ne dépend
      donc plus jamais de la complexité des contours de segmentation.
    Testé avec un batch synthétique dense (3000 détections, ~120/m²,
    contours à 64 sommets - bien plus dense que tout ce qui a été observé
    dans essai1) : ~10s au total, contre plusieurs minutes pour l'ancienne
    approche à la même densité, et le mécanisme responsable de l'heure de
    blocage (bufferisation répétée de polygones complexes) est purement et
    simplement éliminé plutôt qu'accéléré marginalement."""
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
    détections proches (voir `_cluster_by_iou_or_visual_similarity` et le
    correctif du 06/09/2026 en tête de module).

    `similarity_threshold` (0.5 par défaut, proposé par Jame le 06/09/2026) :
    PAS calibré au-delà d'un seul exemple manuel (une similarité de 0.55-0.76
    observée entre 3 vues confirmées d'un même objet, contre 0.40-0.44 pour
    un objet visuellement différent dans le même voisinage) - à affiner sur
    plus d'exemples réels, même logique que tous les autres seuils de ce
    pipeline (iou_threshold, conf_threshold...).

    `distance_threshold_m` (2m par défaut) : marge de recherche des candidats
    - PAS calibrée non plus, à resserrer/desserrer selon le bruit de position
    réellement observé sur d'autres batches."""
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
