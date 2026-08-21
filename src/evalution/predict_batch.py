#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Inférence par Lot (Batch Prediction)
Traite l'ensemble d'un dossier de vol drone, allège les étiquettes 
et génère un rapport global des détections.
"""

import os
import time
from pathlib import Path
from ultralytics import YOLO

# 1. Chemins d'accès
WEIGHTS_PATH = r"C:\Users\alexa\Documents\Jame\PixelOdyssey\output\runs\baseline_yolo11n-seg_20260819_161350\weights\best.pt"
INPUT_FOLDER = r"E:\PixelOdyssey\2. Raw data\3. Santa Luzia\Raw pictures\Santa Luzia W5 - mar transect (6-10) avant ramassage (done)"
OUTPUT_DIR = r"C:\Users\alexa\Documents\Jame\PixelOdyssey\output\predictions"
RUN_NAME = "Aldabra_LEG1_15H46"

def run_batch_prediction():
    print("--- 🚀 DÉMARRAGE DE L'INFÉRENCE PAR LOT (ALREADY BATCHED) ---")
    
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Poids introuvables : {WEIGHTS_PATH}")
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"Dossier source introuvable : {INPUT_FOLDER}")

    # 2. Chargement du modèle
    print(f"📦 Modèle : {Path(WEIGHTS_PATH).name}")
    model = YOLO(WEIGHTS_PATH)

    start_time = time.time()

    # 3. Lancement de l'inférence sur le dossier complet
    print(f"📂 Traitement du dossier : {INPUT_FOLDER}")
    results = model.predict(
        source=INPUT_FOLDER,
        conf=0.25,                 # Seuil de confiance minimal
        imgsz=1280,                # Résolution adaptée aux photos aériennes
        device=0,                  # RTX 3070
        save=True,                 # Sauvegarde des images annotées
        project=OUTPUT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        line_width=1,              # 🪶 ÉTIQUETTES FINES : Réduit la taille du texte et des traits
        show_labels=True,          # Affiche le nom de la classe
        show_conf=False,           # Masque le score % pour aérer l'affichage (ex: "Plastique_Rigide" sans "0.85")
        # retina_masks=True,       # (Optionnel) Pour des contours de masques ultra-lisses
    )

    elapsed = time.time() - start_time

    # 4. Calcul du bilan statistique global
    total_objects = 0
    class_summary = {name: 0 for name in model.names.values()}
    images_with_debris = 0

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            images_with_debris += 1
            total_objects += len(boxes)
            for cls_id in boxes.cls:
                c_name = model.names[int(cls_id)]
                class_summary[c_name] += 1

    # 5. Affichage du compte-rendu
    print("\n" + "="*60)
    print(f"📊 INVENTAIRE GLOBAL — {RUN_NAME}")
    print("="*60)
    print(f"⏱️  Temps d'exécution total   : {elapsed:.2f} secondes (~{elapsed/len(results):.3f}s / image)")
    print(f"🖼️  Photos analysées          : {len(results)}")
    print(f"🏖️  Photos avec déchets       : {images_with_debris} / {len(results)}")
    print(f"🎯 Total des objets détectés  : {total_objects}")
    print("-" * 60)
    print("Répartition par Super-Classe :")
    for name, count in class_summary.items():
        pct = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  • {name:<25} : {count:>4} objet(s) ({pct:>5.1f}%)")
    print("="*60)
    print(f"📁 Images annotées sauvegardées dans :\n{os.path.join(OUTPUT_DIR, RUN_NAME)}")

if __name__ == "__main__":
    run_batch_prediction()