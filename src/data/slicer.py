#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Pipeline de Slicing avec Filtrage d'Inconnu et Mapping 4 Classes.
"""

import os
import cv2
import numpy as np
from shapely.geometry import Polygon, box

# Table de passage officielle : 20 classes CVAT -> 4 Super-Classes
# La classe 3 (Inconnu) est absente et sera explicitement ignorée lors du chargement.
CLASS_MAPPING = {
    # 0: Mousse_Fragments_Souple (Films, sacs, mousses, textiles, flip-flops)
    7: 0,   # Plastique souple (sac)
    8: 0,   # Mousse expansée
    10: 0,  # Polystyrène
    11: 0,  # fliflops (mousse EVA)
    18: 0,  # Tissu
    
    # 1: Plastique_Rigide (Bouteilles, bidons, bouchons, fragments rigides, casques, cagettes)
    0: 1,   # bouteille PET
    2: 1,   # Bouées
    4: 1,   # Fragments plastique rigide
    5: 1,   # Bouées pieuvre
    6: 1,   # Petit bidon
    9: 1,   # Sceau
    12: 1,  # casques
    13: 1,  # bouchons
    14: 1,  # cagette
    15: 1,  # Bouteille Plastique rigide
    19: 1,  # flotteurs
    
    # 2: Cordage_Filet (Fils, bout, filet de pêche)
    1: 2,   # Cordage
    
    # 3: NonPlastique (Bois, verre - éléments inertes)
    16: 3,  # Morceaux de bois
    17: 3   # Verre
}


class PlasticImageSlicer:
    def __init__(self, tile_size=640, overlap=256, min_area_ratio=0.10, discard_truncated=True):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_area_ratio = min_area_ratio
        self.discard_truncated = discard_truncated

    def _load_yolo_labels(self, label_path, img_w, img_h):
        """
        Lit un fichier d'annotation YOLO, remappe vers 4 classes et EXCLUT la classe 3 (Inconnu).
        """
        polygons = []
        if not os.path.exists(label_path):
            return polygons

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                
                raw_class_id = int(parts[0])

                # 🛑 FILTRAGE : Si l'objet est un 'Inconnu' (ID 3) ou absente du mapping, on l'ignore.
                if raw_class_id == 3 or raw_class_id not in CLASS_MAPPING:
                    continue
                
                # Conversion vers l'un des 4 nouveaux identifiants (0, 1, 2 ou 3)
                mapped_class_id = CLASS_MAPPING[raw_class_id]
                
                coords = [float(x) for x in parts[1:]]
                pixels = []
                for i in range(0, len(coords), 2):
                    x_abs = coords[i] * img_w
                    y_abs = coords[i+1] * img_h
                    pixels.append((x_abs, y_abs))
                
                if len(pixels) >= 3:
                    poly_geom = Polygon(pixels)
                    polygons.append({
                        "class_id": mapped_class_id,
                        "geom": poly_geom,
                        "original_area": poly_geom.area
                    })
        return polygons

    def slice_single_pair(self, img_path, label_path, output_img_dir, output_label_dir, prefix="tile"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERREUR] Impossible de charger l'image : {img_path}")
            return

        img_h, img_w, _ = img.shape
        polygons = self._load_yolo_labels(label_path, img_w, img_h)
        tile_count = 0

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
                tile_labels = []

                for poly in polygons:
                    if tile_box.intersects(poly["geom"]):
                        if self.discard_truncated:
                            if poly["geom"].intersects(tile_boundary):
                                continue
                            else:
                                intersection = poly["geom"]
                        else:
                            intersection = tile_box.intersection(poly["geom"])
                            if intersection.area / poly["original_area"] < self.min_area_ratio:
                                continue

                        if intersection.geom_type == "MultiPolygon":
                            parts = list(intersection.geoms)
                        elif intersection.geom_type == "Polygon":
                            parts = [intersection]
                        else:
                            continue

                        for part in parts:
                            local_coords = []
                            for x_glob, y_glob in part.exterior.coords:
                                x_loc = max(0.0, min(1.0, (x_glob - x_start) / self.tile_size))
                                y_loc = max(0.0, min(1.0, (y_glob - y_start) / self.tile_size))
                                local_coords.extend([x_loc, y_loc])

                            coords_str = " ".join([f"{c:.6f}" for c in local_coords])
                            tile_labels.append(f"{poly['class_id']} {coords_str}\n")

                if tile_labels or (tile_count % 10 == 0):
                    base_name = f"{prefix}_{x_start}_{y_start}"
                    out_img_path = os.path.join(output_img_dir, f"{base_name}.png")
                    cv2.imwrite(out_img_path, tile_img)
                    
                    out_lab_path = os.path.join(output_label_dir, f"{base_name}.txt")
                    with open(out_lab_path, "w", encoding="utf-8") as f:
                        f.writelines(tile_labels)
                    
                    tile_count += 1

        print(f"[INFO] Image {os.path.basename(img_path)} découpée -> {tile_count} tuiles.")