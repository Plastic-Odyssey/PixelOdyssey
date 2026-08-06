"""
PixelOdyssey - Core Training Module (Version Nano).
Fine-tuning de YOLOv11n-seg (Nano) sur le dataset à 4 super-classes.
"""

import os
import urllib.request
from pathlib import Path
from ultralytics import YOLO

def launch_training():
    print("--- 🏋️ INITIALISATION DE L'ENTRAÎNEMENT PIXELODYSSEY (YOLOv11 Nano) ---")
    
    # 1. Racine du projet et chemins absolus
    project_root = Path(__file__).resolve().parent.parent.parent
    pretrained_dir = project_root / "models" / "pretrained"
    model_path = pretrained_dir / "yolo11n-seg.pt"
    config_path = project_root / "config" / "data_config.yaml"
    output_dir = project_root / "output" / "runs"
    run_name = "pixel_odyssey_v1_nano"

    # Création automatique du dossier models/pretrained si besoin
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # 2. Téléchargement propre dans models/pretrained/ si absente
    if not model_path.exists():
        print(f"📥 Téléchargement de yolo11n-seg.pt dans {pretrained_dir}...")
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
        urllib.request.urlretrieve(url, model_path)
        print("✅ Téléchargement terminé.")

    print(f"Modèle chargé      : {model_path}")
    print(f"Config data        : {config_path}")
    print(f"Dossier de sortie  : {output_dir / run_name}")

    # Charger le modèle Nano
    model = YOLO(str(model_path))

    # 3. Lancement de l'entraînement
    results = model.train(
        data=str(config_path),
        epochs=100,
        imgsz=640,
        batch=16,                  # Si erreur 'out of memory', passe à 8
        device=0,
        workers=4,
        project=str(output_dir),
        name=run_name,
        plots=True,
        patience=20,
        exist_ok=True
    )
    
    print("\n--- ✅ ENTRAÎNEMENT NANO TERMINÉ ---")
    print(f"Les résultats sont enregistrés dans : {output_dir / run_name}")

if __name__ == "__main__":
    launch_training()