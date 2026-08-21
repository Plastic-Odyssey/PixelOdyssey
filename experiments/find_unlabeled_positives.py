#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Data-Centric AI : Détecteur de labels manquants
Isole les faux positifs à haute confiance pour corriger le dataset CVAT.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from shapely.geometry import Polygon, box

# 1. Chemins
WEIGHTS_PATH = r"C:\Users\alexa\Documents\Jame\PixelOdyssey\output\runs\baseline_yolo11n-seg_20260819_161350\weights\best.pt"
DATASET_DIR = r"E:\PixelOdyssey\dataset\processed_dataset"
OUTPUT_AUDIT_DIR = r"C:\Users\alexa\Documents\Jame\PixelOdyssey\output\audit_missing_labels"

CONF_THRESHOLD = 0.60  # Seuil élevé : l'IA est très sûre d'elle
IOU_THRESHOLD = 0.10   # Si chevauchement < 0.10 avec un label existant, c'est un oubli potentiel


def load_gt_polygons(label_path, img_w=640, img_h=640):
    """Charge les polygones réels annotés dans le fichier .txt."""
    polys = []
    if not os.path.exists(label_path):
        return polys
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                coords = [float(x) for x in parts[1:]]
                pixels = [(coords[i] * img_w, coords[i+1] * img_h) for i in range(0, len(coords), 2)]
                if len(pixels) >= 3:
                    polys.append(Polygon(pixels))
    return polys


def audit_dataset():
    print("--- 🕵️ RECHERCHE DES LABELS OUBLIÉS PAR L'HUMAIN ---")
    os.makedirs(OUTPUT_AUDIT_DIR, exist_ok=True)
    model = YOLO(WEIGHTS_PATH)

    images_dir = os.path.join(DATASET_DIR, "images", "train")
    labels_dir = os.path.join(DATASET_DIR, "labels", "train")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg'))]
    print(f"📁 Analyse de {len(image_files)} tuiles d'entraînement...")

    flagged_count = 0

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, f"{Path(img_name).stem}.txt")

        gt_polys = load_gt_polygons(label_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Inférence du modèle
        results = model.predict(source=img, conf=CONF_THRESHOLD, verbose=False, device=0)[0]

        if results.masks is None or len(results.masks) == 0:
            continue

        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_clss = results.boxes.cls.cpu().numpy()
        pred_confs = results.boxes.conf.cpu().numpy()

        has_unlabeled_plastic = False

        for i, pred_box in enumerate(pred_boxes):
            p_poly = box(pred_box[0], pred_box[1], pred_box[2], pred_box[3])
            
            # Vérifier si cette prédiction correspond à un objet déjà annoté
            matched = False
            for gt_poly in gt_polys:
                if p_poly.intersects(gt_poly):
                    inter_area = p_poly.intersection(gt_poly).area
                    union_area = p_poly.union(gt_poly).area
                    if (inter_area / union_area) > IOU_THRESHOLD:
                        matched = True
                        break

            # Si l'IA voit un objet avec >60% de confiance mais qu'AUCUN label n'existe :
            if not matched:
                has_unlabeled_plastic = True
                cls_name = model.names[int(pred_clss[i])]
                conf = pred_confs[i]
                
                # Encadrer l'oubli potentiel en rouge vif
                x1, y1, x2, y2 = map(int, pred_box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"OUBLI: {cls_name} {conf:.2f}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Si des oublis sont repérés sur cette tuile, on sauvegarde l'image pour revue
        if has_unlabeled_plastic:
            flagged_count += 1
            out_file = os.path.join(OUTPUT_AUDIT_DIR, f"audit_{img_name}")
            cv2.imwrite(out_file, img)

    print("\n" + "="*60)
    print(f"🎯 AUDIT TERMINÉ : {flagged_count} tuile(s) contiennent des déchets non annotés !")
    print(f"📁 Ouvre le dossier suivant pour voir les oublis encadrés en rouge :\n{OUTPUT_AUDIT_DIR}")
    print("="*60)


if __name__ == "__main__":
    audit_dataset()