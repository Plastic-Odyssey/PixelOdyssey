#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Benchmark Externe (TACO Dataset)
Télécharge et évalue un modèle YOLO-seg pré-entraîné sur TACO / TrashNet.
"""

import os
import urllib.request
from pathlib import Path
from ultralytics import YOLO

# 1. Configuration des chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
TACO_WEIGHTS_PATH = PRETRAINED_DIR / "taco_yolov8m_seg.pt"

# Image test de Saint-Brandon
TEST_IMAGE = r"E:\PixelOdyssey\2. Raw data\2. Saint Brandon\Raw pictures\Vers le nord 3\DJI_0513.JPG"
OUTPUT_DIR = PROJECT_ROOT / "output" / "predictions"
RUN_NAME = "comparison_taco_model"

# URL Hugging Face du modèle YOLOv8-seg entraîné sur TACO
TACO_MODEL_URL = "https://huggingface.co/turhancan97/yolov8-segment-trash-detection/resolve/main/yolov8m-seg.pt"


def download_taco_weights():
    """Télécharge les poids TACO dans models/pretrained s'ils sont absents."""
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    
    if not TACO_WEIGHTS_PATH.exists():
        print(f"📥 Téléchargement du modèle TACO depuis Hugging Face (~54 Mo)...")
        urllib.request.urlretrieve(TACO_MODEL_URL, TACO_WEIGHTS_PATH)
        print("✅ Téléchargement terminé avec succès !")
    else:
        print(f"📦 Modèle TACO déjà présent : {TACO_WEIGHTS_PATH.name}")


def run_taco_inference():
    download_taco_weights()

    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError(f"Image introuvable : {TEST_IMAGE}")

    print(f"\n--- 🔍 TEST DU MODÈLE PUBLIC TACO SUR VUE DRONE ---")
    model = YOLO(str(TACO_WEIGHTS_PATH))
    
    print(f"Classes reconnues par le modèle TACO : {list(model.names.values())[:6]}...")

    # Lancement de l'inférence
    results = model.predict(
        source=TEST_IMAGE,
        conf=0.20,              # Seuil de confiance adapté
        imgsz=1280,             # Résolution haute pour vue drone
        device=0,               # RTX 3070
        save=True,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        line_width=1,           # Traits et texte fins
        show_labels=True,
        show_conf=True
    )

    result = results[0]
    boxes = result.boxes

    print("\n" + "="*55)
    print("🎯 RÉSULTATS DU MODÈLE TACO")
    print("="*55)
    if boxes is not None and len(boxes) > 0:
        print(f"✅ {len(boxes)} objet(s) détecté(s) par le modèle TACO :")
        for cls_id in boxes.cls:
            print(f"  • {model.names[int(cls_id)]}")
    else:
        print("⚠️ Aucun déchet détecté par le modèle TACO sur cette image.")

    out_file = OUTPUT_DIR / RUN_NAME / Path(TEST_IMAGE).name
    print("="*55)
    print(f"📁 Image annotée sauvegardée dans :\n{out_file}")


if __name__ == "__main__":
    run_taco_inference()