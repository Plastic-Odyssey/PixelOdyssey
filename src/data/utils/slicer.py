#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Slicer d'images géospatiales & labels YOLO-seg.

Découpe une image (orthomosaïque brute ou tuile déjà pré-découpée) et ses
annotations YOLO-seg en tuiles carrées par fenêtre glissante (sliding
window, avec overlap configurable).

Fonctionnalités :
- Slicing par fenêtre glissante en tuiles de taille configurable, avec overlap.
- Gestion des bords tronqués (discard_truncated) : False -> recadre le polygone
  sur la bordure de la tuile (clipping) et recalcule les coordonnées relatives ;
  True -> supprime l'annotation coupée par le bord.
- Protection contre les micro-débris (min_area_ratio) : écarte un objet tronqué
  dont l'aire restante passe sous ce ratio de son aire d'origine.
- Rétention des tuiles "sans déchet" (background) pour équilibrer
  l'entraînement : si l'image parente entière n'a aucun objet annoté, toutes
  ses tuiles sont gardées ; au sein d'une image par ailleurs annotée, une
  tuile vide sur 10 est gardée.
- Support multi-résolution : traite aussi bien des orthomosaïques brutes que
  des tuiles déjà découpées.
- Filtre anti-bordure noire (max_nodata_fraction, depuis le 24/09/2026) : écarte
  TOUTE tuile (annotée ou non) dont plus de max_nodata_fraction des pixels sont du
  "nodata" - zones noires qui touchent le bord de l'image parente (bordures de
  rotation d'orthomosaïque, bord noir des imagettes SB), voir compute_nodata_mask().
  Les pixels sombres À L'INTÉRIEUR de la scène (ombres, rochers, objets noirs) ne
  comptent PAS : mesuré le 24/09/2026 sur les 561 images, un filtre "tout pixel
  noir" aurait retiré 55 % des tuiles annotées et 25 % des objets (99 % à Saint
  Brandon), contre 1 % des objets avec cette définition au seuil de 25 %.

Ce module ne connaît pas les classes : il lit et écrit les IDs de classe
présents dans les fichiers .txt tels quels, sans les traduire. La traduction
(nom de classe du lot -> super-classe cible, ou exclusion) se fait en amont,
à l'étape 2 (split_dataset.py) - voir class_config.py pour le référentiel.
Les fichiers que ce slicer reçoit sont donc déjà dans l'espace de classes
final : il ne fait que de la géométrie (fenêtre glissante, clipping aux
bords de tuile, filtrage des micro-débris).

Entrée : une paire image + labels YOLO-seg, et les dossiers de sortie.
Sortie : fichiers image (.png) et labels (.txt) de chaque tuile, écrits sur
disque ; nombre de tuiles produites.

Exemple :
    from src.data.utils.slicer import PlasticImageSlicer
    slicer = PlasticImageSlicer(tile_size=640, overlap=256)
    n_tiles = slicer.slice_single_pair(
        "img.jpg", "img.txt", "out/images", "out/labels"
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from src.data.utils.image_io import load_image_bgr
from src.data.utils.tiling_geometry import iter_tile_windows

# Incrémenter quand la logique interne de tuilage change de façon à produire
# une sortie différente pour les MÊMES paramètres de constructeur. Les
# paramètres du constructeur (tile_size, overlap, ...) sont déjà inclus dans
# le fingerprint du cache incrémental (voir slice_dataset.py) ; ce numéro
# couvre les changements de comportement qui n'ont pas de paramètre dédié -
# sans lui, un tel changement passerait inaperçu par le garde-fou de cache et
# mélangerait silencieusement deux logiques différentes dans le même dossier
# de sortie.
LOGIC_VERSION = 5  # 24/09/2026 : filtre nodata (composantes noires de bord) sur toutes les tuiles


# Seuil "noir" par canal (BGR) : <= 10 plutôt que 0 strict pour absorber le bruit de
# compression JPEG autour du noir (bords des imagettes SB).
NODATA_MAX_VALUE = 10
# Une composante noire connectée au bord de l'image n'est du nodata que si elle pèse au
# moins cette fraction de l'image : écarte les coins sombres d'une photo drone (vignettage,
# ombre portée au bord - 0,01 à 0,02 % de l'image mesurés sur 4 photos DJI d'Aldabra, qui
# n'ont par construction aucun nodata).
NODATA_MIN_COMPONENT_FRACTION = 0.001


def compute_nodata_mask(img: np.ndarray) -> np.ndarray:
    """Masque booléen (H, W) des pixels "nodata" d'une image BGR : pixels quasi noirs
    (tous canaux <= NODATA_MAX_VALUE) appartenant à une composante connexe (8-voisinage)
    qui TOUCHE le bord de l'image et pèse au moins NODATA_MIN_COMPONENT_FRACTION de sa
    surface. Un objet sombre ou une ombre au milieu de la scène n'est jamais du nodata.

    Réutilisable côté application (ex: ignorer une prédiction posée sur le bord noir d'une
    orthomosaïque) - même définition qu'à l'entraînement."""
    h, w = img.shape[:2]
    near_black = np.all(img <= NODATA_MAX_VALUE, axis=2).astype(np.uint8)
    if not near_black.any():
        return np.zeros((h, w), dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(near_black, connectivity=8)
    edge_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    min_area = NODATA_MIN_COMPONENT_FRACTION * h * w
    keep = [int(i) for i in edge_labels if i != 0 and stats[int(i), cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return np.zeros((h, w), dtype=bool)
    return np.isin(labels, keep)


class PlasticImageSlicer:
    def __init__(
        self,
        tile_size: int = 640,
        overlap: int = 256,
        min_area_ratio: float = 0.05,
        discard_truncated: bool = False,
        max_nodata_fraction: float = 0.25,
    ) -> None:
        """
        Args:
            tile_size: Dimension de la tuile carrée cible en pixels (ex: 640).
            overlap: Recouvrement entre deux tuiles adjacentes en pixels (ex: 256).
            min_area_ratio: Ratio de surface minimale conservé pour un polygone tronqué (ex: 0.05 = 5%).
            discard_truncated: Si True, supprime les objets coupés par le bord.
                               Si False, recadre le polygone sur la bordure de la tuile.
            max_nodata_fraction: Fraction (0-1) de pixels nodata (voir compute_nodata_mask)
                               au-delà de laquelle une tuile est écartée, qu'elle contienne
                               un objet annoté ou non. 0,25 retenu le 24/09/2026 : perte
                               mesurée de 1 % des objets (6 % sur les orthomosaïques SL), contre
                               5,6 % (38 % sur les orthos SL) à tolérance 0.
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_area_ratio = min_area_ratio
        self.discard_truncated = discard_truncated
        self.max_nodata_fraction = max_nodata_fraction

    def _load_yolo_labels(self, label_path: Union[str, Path], img_w: int, img_h: int) -> List[Dict]:
        """Lit un fichier YOLO-seg déjà dans l'espace de classes final (voir docstring
        du module) - aucune traduction, juste le parsing géométrique."""
        polygons: List[Dict] = []
        label_file = Path(label_path)

        if not label_file.exists():
            return polygons

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                pixels = [
                    (coords[i] * img_w, coords[i + 1] * img_h)
                    for i in range(0, len(coords), 2)
                ]

                if len(pixels) >= 3:
                    poly_geom = Polygon(pixels)
                    if poly_geom.is_valid and poly_geom.area > 0:
                        polygons.append(
                            {
                                "class_id": class_id,
                                "geom": poly_geom,
                                "original_area": poly_geom.area,
                            }
                        )
        return polygons

    def slice_single_pair(
        self,
        img_path: Union[str, Path],
        label_path: Union[str, Path],
        output_img_dir: Union[str, Path],
        output_label_dir: Union[str, Path],
        prefix: str = "tile",
        override_discard_truncated: Optional[bool] = None,
    ) -> int:
        """Découpe une paire image/annotation YOLO en tuiles au format cible."""
        img_p = Path(img_path)
        label_p = Path(label_path)
        out_img_d = Path(output_img_dir)
        out_lab_d = Path(output_label_dir)

        out_img_d.mkdir(parents=True, exist_ok=True)
        out_lab_d.mkdir(parents=True, exist_ok=True)

        discard_truncated = (
            override_discard_truncated
            if override_discard_truncated is not None
            else self.discard_truncated
        )

        img = load_image_bgr(img_p)
        if img is None:
            print(f"[ERREUR] Impossible de charger l'image : {img_p}")
            return 0

        img_h, img_w, _ = img.shape

        # Cas d'une image déjà plus petite ou égale à la tuile cible dans les deux
        # dimensions (ex: une "imagette" déjà découpée par l'annotateur, pas
        # forcément exactement tile_size x tile_size) : rien à faire glisser,
        # l'image entière tient dans une seule tuile, écrite à sa taille native
        # (Ultralytics redimensionne/letterbox chaque image à l'entraînement de
        # toute façon).
        if img_h <= self.tile_size and img_w <= self.tile_size:
            polygons = self._load_yolo_labels(label_p, img_w, img_h)
            tile_labels = []
            for poly in polygons:
                coords = list(poly["geom"].exterior.coords)
                norm_coords = []
                for x, y in coords:
                    norm_coords.extend([max(0.0, min(1.0, x / img_w)), max(0.0, min(1.0, y / img_h))])
                coords_str = " ".join([f"{c:.6f}" for c in norm_coords])
                tile_labels.append(f"{poly['class_id']} {coords_str}\n")

            # Filtre nodata : cette image (déjà <= tile_size) est sa propre et unique
            # tuile - écartée entièrement si sa part de nodata dépasse le seuil, annotée
            # ou non (voir compute_nodata_mask).
            if compute_nodata_mask(img).mean() > self.max_nodata_fraction:
                return 0

            cv2.imwrite(str(out_img_d / f"{prefix}.png"), img)
            with open(out_lab_d / f"{prefix}.txt", "w", encoding="utf-8") as f:
                f.writelines(tile_labels)
            return 1

        polygons = self._load_yolo_labels(label_p, img_w, img_h)
        # Masque nodata calculé UNE fois sur l'image parente entière (la connexité au bord
        # n'a de sens qu'à cette échelle), puis image intégrale pour obtenir la part de
        # nodata de chaque fenêtre en temps constant.
        nodata_integral = np.pad(compute_nodata_mask(img).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        tile_count = 0
        # Vrai si l'image PARENTE entière n'a aucun objet annoté (label absent, ou vide -
        # ex: une photo de terrain confirmée sans déchet). À distinguer d'une tuile
        # individuellement vide au sein d'une image parente qui contient des objets
        # ailleurs (cf. le sous-échantillonnage 1/10 ci-dessous, qui ne s'applique qu'à
        # ce second cas).
        parent_is_all_background = not polygons

        for x_start, y_start, x_end, y_end in iter_tile_windows(img_w, img_h, self.tile_size, self.stride):
            tile_img = img[y_start:y_end, x_start:x_end]
            xe, ye = min(x_end, img_w), min(y_end, img_h)
            n_nodata = (nodata_integral[ye, xe] - nodata_integral[y_start, xe]
                        - nodata_integral[ye, x_start] + nodata_integral[y_start, x_start])
            if n_nodata > self.max_nodata_fraction * max(1, (xe - x_start) * (ye - y_start)):
                # Tuile majoritairement hors image (bordure noire) : écartée AVANT toute
                # autre règle, annotée ou non - elle ne compte pas non plus dans le
                # tirage des tuiles de fond ci-dessous.
                continue
            tile_box = box(x_start, y_start, x_end, y_end)
            tile_boundary = tile_box.boundary
            tile_labels: List[str] = []

            for poly in polygons:
                geom = poly["geom"]
                if not tile_box.intersects(geom):
                    continue

                if discard_truncated:
                    if geom.intersects(tile_boundary):
                        continue
                    intersection = geom
                else:
                    intersection = tile_box.intersection(geom)
                    if intersection.is_empty or intersection.area <= 0:
                        continue
                    if (intersection.area / poly["original_area"]) < self.min_area_ratio:
                        continue

                if isinstance(intersection, Polygon):
                    parts = [intersection]
                elif isinstance(intersection, MultiPolygon):
                    parts = list(intersection.geoms)
                else:
                    continue

                for part in parts:
                    if part.area <= 0:
                        continue

                    local_coords: List[float] = []
                    for x_glob, y_glob in part.exterior.coords:
                        x_loc = max(0.0, min(1.0, (x_glob - x_start) / self.tile_size))
                        y_loc = max(0.0, min(1.0, (y_glob - y_start) / self.tile_size))
                        local_coords.extend([x_loc, y_loc])

                    if len(local_coords) >= 6:
                        coords_str = " ".join([f"{c:.6f}" for c in local_coords])
                        tile_labels.append(f"{poly['class_id']} {coords_str}\n")

            # On garde une tuile si : elle contient un objet, OU l'image parente est
            # entièrement background (auquel cas on garde tout - ce sont précisément
            # les images "sans déchets" utiles pour équilibrer l'entraînement), OU 1
            # tuile vide sur 10 sinon (pour garder quelques exemples de fond au sein
            # d'une image qui contient par ailleurs des objets, sans exploser le
            # nombre de tuiles vides).
            #
            # Précision (24/09/2026) : "1 tuile vide sur 10" est en réalité "une tuile vide
            # gardée chaque fois que le nombre de tuiles DÉJÀ gardées est un multiple de 10" -
            # soit au plus ~1 tuile vide pour 9 tuiles annotées, positionnellement. D'où
            # seulement 1 à 6 % de tuiles vides à Santa Luzia et Aldabra. Comportement
            # inchangé ici (le rééquilibrage positifs/négatifs relève de l'échantillonnage
            # à l'entraînement, pas du tuilage).
            keep_as_background = False
            if not tile_labels:
                keep_as_background = parent_is_all_background or (tile_count % 10 == 0)

            if tile_labels or keep_as_background:
                base_name = f"{prefix}_{x_start}_{y_start}"
                cv2.imwrite(str(out_img_d / f"{base_name}.png"), tile_img)
                with open(out_lab_d / f"{base_name}.txt", "w", encoding="utf-8") as f:
                    f.writelines(tile_labels)

                tile_count += 1

        return tile_count
