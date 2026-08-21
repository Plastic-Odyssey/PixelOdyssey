#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Test rapide d'inférence sur une image de drone.
"""

import os
from pathlib import Path
from ultralytics import YOLO

# 1. Définition des chemins
WEIGHTS_PATH = r"C:\Users\alexa\Documents\Jame\PixelOdyssey\output\runs\baseline_yolo11n-seg_20260819_161350\weights\best.pt"
IMAGE_PATH = r"E:\PixelOdyssey\2. Raw data\2. Saint Brandon\Raw pictures\Vers le nord 3\DJI_0513.JPG"

# Dossier où enregistrer le résultat détouré
OUTPUT_DIR = Path(r"C:\Users\alexa\Documents\Jame\PixelOdyssey\output\predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_quick_inference():
    print(f"--- 🔍 TEST D'INFÉRENCE PIXELODYSSEY ---")
    print(f"Modèle : {WEIGHTS_PATH}")
    print(f"Image  : {IMAGE_PATH}")

    # Vérification de l'existence des fichiers
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"❌ Poids introuvables à l'adresse : {WEIGHTS_PATH}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"❌ Image introuvable à l'adresse : {IMAGE_PATH}")

    # 2. Chargement du modèle
    model = YOLO(WEIGHTS_PATH)

    # 3. Inférence avec seuil de confiance
    # conf=0.25 : bon compromis pour une première visualisation
    results = model.predict(
        source=IMAGE_PATH,
        conf=0.25,        # Seuil de confiance (ajustable de 0.10 à 0.50)
        imgsz=640,        # Résolution de traitement
        save=True,        # Sauvegarde automatiquement l'image annotée
        project=str(OUTPUT_DIR),
        name="test_dji_0513",
        exist_ok=True
    )

    # 4. Affichage du compte-rendu dans le terminal
    res = results[0]
    total_detections = len(res.boxes)
    
    print("\n" + "="*45)
    print(f"📊 RÉSULTATS DE LA DÉTECTION ({total_detections} objets trouvés)")
    print("="*45)

    if total_detections == 0:
        print("Aucun déchet détecté avec un seuil de confiance >= 25%.")
    else:
        for i, box in enumerate(res.boxes):
            cls_id = int(box.cls[0])
            cls_name = res.names[cls_id]
            conf = float(box.conf[0])
            print(f"  • Déchet #{i+1:02d} : {cls_name:<25} (Confiance : {conf*100:.1f}%)")

    # Localisation de l'image de sortie
    saved_path = OUTPUT_DIR / "test_dji_0513" / Path(IMAGE_PATH).name
    print("="*45)
    print(f"🖼️ Image avec masques générée dans :\n👉 {saved_path}\n")

if __name__ == "__main__":
    run_quick_inference()