#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Pipeline de Préparation des Données
Module de Slicing (Fenêtre Glissante) pour YOLO Segmentation.
Gère le découpage synchrone des images et le filtrage des arêtes artificielles.
"""

import os
import cv2
import numpy as np
from shapely.geometry import Polygon, box


class PlasticImageSlicer:
    """
    Classe responsable du découpage par fenêtre glissante d'images de drone.
    Intègre une gestion avancée des objets coupés aux bordures pour éviter
    les arêtes droites artificielles lors de l'entraînement de l'IA.
    """

    def __init__(self, tile_size=640, overlap=256, min_area_ratio=0.10, discard_truncated=True):
        """
        Configuration des hyperparamètres du Slicer.
        
        Args:
            tile_size (int): Taille de la tuile carrée de sortie.
            overlap (int): Chevauchement entre deux tuiles (ex: 256 pour capturer les objets entiers).
            min_area_ratio (float): Ratio de surface minimale sous lequel un fragment est ignoré.
            discard_truncated (bool): Si True, rejette tous les objets coupés par les bords de la tuile.
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_area_ratio = min_area_ratio
        self.discard_truncated = discard_truncated

    def _load_yolo_labels(self, label_path, img_w, img_h):
        """
        Lit un fichier d'annotation YOLO et convertit les coordonnées normalisées
        en polygones absolus Shapely.
        """
        polygons = []
        if not os.path.exists(label_path):
            return polygons

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                pixels = []
                for i in range(0, len(coords), 2):
                    x_abs = coords[i] * img_w
                    y_abs = coords[i+1] * img_h
                    pixels.append((x_abs, y_abs))
                
                if len(pixels) >= 3:
                    poly_geom = Polygon(pixels)
                    polygons.append({
                        "class_id": class_id,
                        "geom": poly_geom,
                        "original_area": poly_geom.area
                    })
        return polygons

    def slice_single_pair(self, img_path, label_path, output_img_dir, output_label_dir, prefix="tile"):
        """
        Découpe un couple unique (Image + Fichier Texte) en morceaux de 640x640.
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERREUR] Impossible de charger l'image : {img_path}")
            return

        img_h, img_w, _ = img.shape
        polygons = self._load_yolo_labels(label_path, img_w, img_h)
        tile_count = 0

        # Algorithme de la fenêtre glissante
        for y_offset in range(0, img_h, self.stride):
            for x_offset in range(0, img_w, self.stride):
                
                # Ajustement de sécurité pour les bordures de la grande image
                x_start = x_offset
                y_start = y_offset
                if x_start + self.tile_size > img_w:
                    x_start = max(0, img_w - self.tile_size)
                if y_start + self.tile_size > img_h:
                    y_start = max(0, img_h - self.tile_size)

                x_end = x_start + self.tile_size
                y_end = y_start + self.tile_size

                # 1. Extraction de la sous-matrice de pixels (Slicing NumPy)
                tile_img = img[y_start:y_end, x_start:x_end]
                
                # Définition de la boîte de la tuile courante et de sa bordure extérieure
                tile_box = box(x_start, y_start, x_end, y_end)
                tile_boundary = tile_box.boundary  # La ligne extérieure du carré 640x640
                
                tile_labels = []

                # 2. Analyse géométrique des polygones
                for poly in polygons:
                    if tile_box.intersects(poly["geom"]):
                        
                        # STRATÉGIE A : Si on choisit de rejeter les objets coupés par les bords
                        if self.discard_truncated:
                            if poly["geom"].intersects(tile_boundary):
                                # L'objet touche la bordure de la tuile -> on l'ignore pour cette tuile
                                continue
                            else:
                                # L'objet est entièrement au milieu de la tuile, on le garde intact
                                intersection = poly["geom"]
                        
                        # STRATÉGIE B (Fallback) : Garder l'intersection et gérer les MultiPolygons
                        else:
                            intersection = tile_box.intersection(poly["geom"])
                            
                            # Si le filtre d'aire minimale rejette le fragment
                            if intersection.area / poly["original_area"] < self.min_area_ratio:
                                continue

                        # Extraction des sous-polygones (cas des MultiPolygons)
                        if intersection.geom_type == "MultiPolygon":
                            parts = list(intersection.geoms)
                        elif intersection.geom_type == "Polygon":
                            parts = [intersection]
                        else:
                            continue

                        # Calcul local et normalisation pour chaque partie exploitable
                        for part in parts:
                            local_coords = []
                            for x_glob, y_glob in part.exterior.coords:
                                x_loc = (x_glob - x_start) / self.tile_size
                                y_loc = (y_glob - y_start) / self.tile_size
                                # Éviter les micro-dépassements d'arrondis hors du repère [0, 1]
                                x_loc = max(0.0, min(1.0, x_loc))
                                y_loc = max(0.0, min(1.0, y_loc))
                                local_coords.extend([x_loc, y_loc])

                            coords_str = " ".join([f"{c:.6f}" for c in local_coords])
                            tile_labels.append(f"{poly['class_id']} {coords_str}\n")

                # 3. Sauvegarde physique (avec 10% d'images "fonds propres" pour l'équilibre)
                if tile_labels or (tile_count % 10 == 0):
                    base_name = f"{prefix}_{x_start}_{y_start}"
                    
                    # Sauvegarde Image
                    out_img_path = os.path.join(output_img_dir, f"{base_name}.png")
                    cv2.imwrite(out_img_path, tile_img)
                    
                    # Sauvegarde Label
                    out_lab_path = os.path.join(output_label_dir, f"{base_name}.txt")
                    with open(out_lab_path, "w", encoding="utf-8") as f:
                        f.writelines(tile_labels)
                    
                    tile_count += 1

        print(f"[INFO] Image {os.path.basename(img_path)} découpée en {tile_count} tuiles (Bordures filtrées: {self.discard_truncated}).")


if __name__ == "__main__":
    # Test unitaire rapide sur ton SSD externe
    RAW_DIR = r"D:\PixelOdyssey_Data\plastic_dataset\raw"
    OUT_IMG = r"D:\PixelOdyssey_Data\plastic_dataset\images\train"
    OUT_LAB = r"D:\PixelOdyssey_Data\plastic_dataset\labels\train"
    
    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_LAB, exist_ok=True)
    
    # Instance configurée selon tes nouveaux paramètres de précision
    slicer = PlasticImageSlicer(tile_size=640, overlap=256, discard_truncated=True)
    print("[RUN] Slicer configuré avec discard_truncated=True et overlap=256.")