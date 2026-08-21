#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Entraînement YOLO (détection + segmentation d'instances).

Ce fichier est volontairement coupé en deux zones :

  1. PARAMÈTRES À AJUSTER : tout ce que tu es amené à changer d'un run à
     l'autre pour comparer des architectures/tailles de modèle (YOLOv8 vs
     YOLOv11, nano/small/medium/...) ou pour adapter batch/epochs à ta
     machine. C'est la SEULE section à modifier pour lancer une nouvelle
     expérience.
  2. Le reste : orchestration (nommage des runs, non-écrasement, génération
     du rapport de fin d'entraînement) - tu n'as normalement pas besoin d'y
     toucher.

Non-écrasement des runs
------------------------
Chaque run est enregistré dans un dossier UNIQUE, nommé à partir du modèle
choisi et d'un horodatage à la seconde (voir `_build_run_name`) :

    output/runs/<RUN_TAG>_<modèle>_<AAAAMMJJ_HHMMSS>/

`exist_ok=False` est passé à `model.train()` comme garde-fou supplémentaire :
si jamais deux runs partageaient malgré tout le même nom (cas extrême : deux
lancements dans la même seconde), Ultralytics lèvera une erreur au lieu
d'écraser silencieusement les résultats précédents.

Rapport de lecture
-------------------
À la fin de l'entraînement, `training_report.py` évalue le meilleur modèle
(`best.pt`) sur les splits val ET test, et génère un rapport HTML autonome
dans le dossier du run (`rapport_lecture.html`) : tableau par classe avec
taux de faux positifs/négatifs, graphique précision/rappel, matrice de
confusion. Il peut aussi être regénéré à tout moment pour un run passé, sans
relancer l'entraînement :

    python -m src.training.training_report --run output/runs/<nom_du_run>
"""

import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.training.training_report import generate_report

# ============================================================================
# 1. PARAMÈTRES À AJUSTER
# ============================================================================

# --- Choix de l'architecture -------------------------------------------------
# Ultralytics télécharge automatiquement le poids pré-entraîné correspondant
# au premier lancement (mis en cache localement) : pas besoin de gérer
# l'URL/le téléchargement toi-même, contrairement à l'ancienne version de ce
# script.
#
# Nom de fichier attendu, en combinant une famille et une taille :
#
#   Famille  | Fichier            | Repère mémoire / vitesse
#   ---------|--------------------|------------------------------------
#   YOLOv8   | yolov8{n,s,m,l,x}-seg.pt |
#   YOLOv11  | yolo11{n,s,m,l,x}-seg.pt |
#
#   n (nano) = le plus léger/rapide, le moins précis
#   s (small), m (medium), l (large), x (xlarge) = de plus en plus lourd,
#   lent à entraîner, mais généralement plus précis.
#
# Le suffixe "-seg" est obligatoire : ce sont les variantes segmentation
# (les seules pertinentes ici, PixelOdyssey délimite des masques de déchets,
# pas juste des boîtes).
MODEL_WEIGHTS = "yolo11n-seg.pt"   # <-- change UNIQUEMENT cette ligne pour tester un autre modèle
# Exemples à tester : "yolov8n-seg.pt", "yolov8s-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"

# --- Hyperparamètres d'entraînement -----------------------------------------
EPOCHS = 100
IMGSZ = 640
BATCH = 16          # -1 = laisse Ultralytics choisir automatiquement selon la VRAM dispo
PATIENCE = 20       # arrêt anticipé si aucune amélioration après N epochs
DEVICE = 0          # 0 = 1er GPU ; "cpu" = CPU ; "0,1" = multi-GPU
WORKERS = 4

# --- Étiquette libre pour retrouver ce run dans output/runs/ ---------------
# Sert uniquement à la lisibilité du nom de dossier - mets ce que tu veux,
# par ex. "baseline", "test_yolov8s", "sans_class_nonplastique", etc.
RUN_TAG = "baseline"

# ============================================================================
# 2. Orchestration (nommage, non-écrasement, rapport) - pas besoin d'y toucher
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output" / "runs"


def _build_run_name(model_weights: str, tag: str) -> str:
    """<tag>_<modèle-sans-extension>_<horodatage> - unique à la seconde
    près, donc jamais de collision avec un run précédent."""
    model_slug = Path(model_weights).stem  # "yolo11n-seg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{tag}_{model_slug}_{timestamp}"


def launch_training():
    run_name = _build_run_name(MODEL_WEIGHTS, RUN_TAG)
    run_dir = OUTPUT_DIR / run_name

    print("--- 🏋️ INITIALISATION DE L'ENTRAÎNEMENT PIXELODYSSEY ---")
    print(f"Modèle              : {MODEL_WEIGHTS}")
    print(f"Config data          : {CONFIG_PATH}")
    print(f"Dossier de sortie    : {run_dir}")

    # YOLO(...) télécharge automatiquement le poids pré-entraîné si besoin -
    # plus de gestion manuelle d'URL/urllib comme dans l'ancienne version.
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=str(CONFIG_PATH),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(OUTPUT_DIR),
        name=run_name,
        plots=True,
        exist_ok=False,  # garde-fou anti-écrasement, voir docstring en tête de fichier
    )

    print("\n--- ✅ ENTRAÎNEMENT TERMINÉ ---")
    print(f"Les résultats bruts Ultralytics sont dans : {run_dir}")

    print("\n--- 📊 GÉNÉRATION DU RAPPORT DE LECTURE (val + test, faux positifs/négatifs par classe) ---")
    try:
        report_path = generate_report(run_dir, data_config_path=CONFIG_PATH)
        print(f"Rapport HTML à ouvrir dans un navigateur : {report_path}")
    except Exception as e:  # noqa: BLE001 - un échec du rapport ne doit pas remettre en cause le run lui-même
        print(f"⚠️ Le rapport n'a pas pu être généré automatiquement ({e}).")
        print("   Tu peux le regénérer plus tard sans relancer l'entraînement, avec :")
        print(f'   python -m src.training.training_report --run "{run_dir}"')


if __name__ == "__main__":
    launch_training()
