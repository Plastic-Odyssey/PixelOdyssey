#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Slicer d'Images Géospatiales & Labels YOLO-seg.

Fonctionnalités principales :
-----------------------------
1. Slicing Dynamique (Sliding Window) : Découpage d'images géantes/tuiles en fenêtres 640x640 avec overlap.
2. Gestion des Bords Tronqués (discard_truncated) :
   - False -> Recadre les polygones sur les limites de la tuile (Clipping) et re-calcule les coordonnées relatives.
   - True  -> Supprime strictement les annotations coupées par la bordure.
3. Protection contre les Micro-Débris (min_area_ratio) : Écarte les objets amputés à plus de (1 - min_area_ratio).
4. Rétention des images "sans déchet" (background) : si l'image parente entière n'a
   aucun objet annoté, TOUTES ses tuiles sont gardées (voir `parent_is_all_background`
   ci-dessous) - utile pour équilibrer l'entraînement. Une tuile vide au sein d'une
   image par ailleurs non-vide reste sous-échantillonnée (1 sur 10).
5. Support multi-résolution : Traitement unifié des orthomosaïques brutes ou tuiles 1024x1024.

Ce module NE CONNAÎT PLUS RIEN aux classes (retiré le 19/08/2026) : il lit et
écrit les IDs de classe présents dans les fichiers .txt tels quels, sans les
traduire. Toute la traduction (nom de classe du lot -> super-classe cible, ou
exclusion) se fait désormais en amont, une seule fois, à l'étape 2
(split_dataset.py) - voir class_config.py pour le référentiel. Par construction,
les fichiers que ce slicer reçoit sont donc déjà dans l'espace de classes FINAL :
il ne fait plus que de la géométrie (sliding window, clipping aux bords de tuile,
filtrage des micro-débris).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

# Incrémenter quand la LOGIQUE interne de tuilage change de façon à produire
# une sortie différente pour les MÊMES paramètres de constructeur (ex: la
# règle de sous-échantillonnage des tuiles vides ci-dessous, v2). Les
# paramètres du constructeur (tile_size, overlap, ...) sont déjà inclus dans
# le fingerprint du cache incrémental (voir slice_dataset.py) ; ce numéro de
# version couvre les changements de comportement qui n'ont pas de paramètre
# dédié - sans lui, un tel changement passerait inaperçu par le garde-fou de
# cache et mélangerait silencieusement deux logiques différentes dans le même
# dossier de sortie.
LOGIC_VERSION = 2


class PlasticImageSlicer:
    def __init__(
        self,
        tile_size: int = 640,
        overlap: int = 256,
        min_area_ratio: float = 0.05,
        discard_truncated: bool = False,
    ) -> None:
        """
        Args:
            tile_size: Dimension de la tuile carrée cible en pixels (ex: 640).
            overlap: Recouvrement entre deux tuiles adjacentes en pixels (ex: 256).
            min_area_ratio: Ratio de surface minimale conservé pour un polygone tronqué (ex: 0.05 = 5%).
            discard_truncated: Si True, supprime les objets coupés par le bord.
                               Si False, recadre le polygone sur la bordure de la tuile.
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_area_ratio = min_area_ratio
        self.discard_truncated = discard_truncated

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

        img = cv2.imread(str(img_p))
        if img is None:
            print(f"[ERREUR] Impossible de charger l'image : {img_p}")
            return 0

        img_h, img_w, _ = img.shape

        # Cas d'une tuile déjà au format cible (ex: 640x640)
        if img_h == self.tile_size and img_w == self.tile_size:
            polygons = self._load_yolo_labels(label_p, img_w, img_h)
            tile_labels = []
            for poly in polygons:
                coords = list(poly["geom"].exterior.coords)
                norm_coords = []
                for x, y in coords:
                    norm_coords.extend([max(0.0, min(1.0, x / img_w)), max(0.0, min(1.0, y / img_h))])
                coords_str = " ".join([f"{c:.6f}" for c in norm_coords])
                tile_labels.append(f"{poly['class_id']} {coords_str}\n")

            cv2.imwrite(str(out_img_d / f"{prefix}.png"), img)
            with open(out_lab_d / f"{prefix}.txt", "w", encoding="utf-8") as f:
                f.writelines(tile_labels)
            return 1

        polygons = self._load_yolo_labels(label_p, img_w, img_h)
        tile_count = 0
        # Vrai si l'image PARENTE entière n'a aucun objet annoté (label absent, ou vide -
        # ex: une photo de terrain confirmée sans déchet). À distinguer d'une tuile
        # individuellement vide au sein d'une image parente qui contient des objets
        # ailleurs (cf. le sous-échantillonnage 1/10 ci-dessous, qui ne s'applique qu'à
        # ce second cas).
        parent_is_all_background = not polygons

        for y_offset in range(0, img_h, self.stride):
            for x_offset in range(0, img_w, self.stride):
                x_start = x_offset
                y_start = y_offset

                if x_start + self.tile_size > img_w:
                    x_start = max(0, img_w - self.tile_size)
                if y_start + self.tile_size > img_h:
                    y_start = max(0, img_h - self.tile_size)

                x_end = x_start + self.tile_size
                y_end = y_start + self.tile_size

                tile_img = img[y_start:y_end, x_start:x_end]
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
                # entièrement background (auquel cas ON GARDE TOUT - ce sont précisément
                # les images "sans déchets" utiles pour équilibrer l'entraînement, jeter
                # 90% de leurs tuiles serait contre-productif), OU 1 tuile vide sur 10 sinon
                # (juste pour garder quelques exemples de fond au sein d'une image qui
                # contient par ailleurs des objets, sans exploser le nombre de tuiles vides).
                if tile_labels or parent_is_all_background or (tile_count % 10 == 0):
                    base_name = f"{prefix}_{x_start}_{y_start}"
                    cv2.imwrite(str(out_img_d / f"{base_name}.png"), tile_img)
                    with open(out_lab_d / f"{base_name}.txt", "w", encoding="utf-8") as f:
                        f.writelines(tile_labels)

                    tile_count += 1

        return tile_count
