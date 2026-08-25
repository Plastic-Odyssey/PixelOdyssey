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
#
# Note (24/08/2026) sur "nano vs medium" : plus de capacité n'aide QUE si la faiblesse observée
# est un souci de PRÉCISION DU MODÈLE (ex: contours de masque imprécis) - pas un souci de VOLUME/
# DIVERSITÉ DE DONNÉES (ex: classe rare sous-représentée), que plus de paramètres ne fait
# qu'aggraver côté surapprentissage sur un dataset encore petit (~540 images parentes avant
# slicing). Recommandation : regarder le rapport val/test par classe de CE run (nano, pipeline
# corrigé) avant de sauter à "m" - si le nano généralise déjà bien mais plafonne en précision,
# tester "yolo11s-seg.pt" d'abord (saut de capacité plus mesuré) plutôt que "m" directement.

# --- Hyperparamètres d'entraînement -----------------------------------------
EPOCHS = 100
IMGSZ = 640
BATCH = 16          # -1 = laisse Ultralytics choisir automatiquement selon la VRAM dispo
PATIENCE = 20       # arrêt anticipé si aucune amélioration après N epochs
DEVICE = 0          # 0 = 1er GPU ; "cpu" = CPU ; "0,1" = multi-GPU
WORKERS = 4

# --- Augmentation : rendus explicites le 24/08/2026 (jusqu'ici, valeurs par
# défaut d'Ultralytics appliquées SANS jamais avoir été choisies consciemment
# pour ce projet - voir le journal de décisions, section GSD/matériel drone) --
#
# Nos images sont des vues NADIR (drone à la verticale) : contrairement à une
# photo "normale" avec un horizon et un "haut" naturel (le cas pour lequel les
# défauts d'Ultralytics, pensés COCO, sont calibrés), un déchet photographié
# du dessus peut apparaître à N'IMPORTE QUELLE orientation - il n'y a pas de
# "haut" physique. Deux changements en découlent directement, pas des
# suppositions :
DEGREES = 180.0     # défaut Ultralytics = 0.0 (aucune rotation). Sans justification pour une vue
                    # nadir : le modèle voyait toujours les déchets dans l'orientation de prise de
                    # vue d'origine, jamais tournés - 180 = rotation aléatoire sur tout le cercle.
FLIPUD = 0.5        # défaut Ultralytics = 0.0. FLIPLR est à 0.5 par défaut (retournement horizontal
                    # aléatoire) mais pas FLIPUD (vertical) - illogique en nadir, où les deux sont
                    # équivalents. Aligné sur fliplr pour la même raison.

# copy_paste (défaut 0.0) : colle des instances segmentées d'une image sur une autre dans le batch -
# c'est LE mécanisme natif Ultralytics pour l'oversampling par augmentation des classes rares (Bidon,
# Bouee) qu'on avait explicitement différé le 22/08/2026 plutôt que de le construire à la main (voir
# journal, "Rééquilibrage des classes rares — DIFFÉRÉ").
#
# Remis à 0.0 le 25/08/2026 (essai à 0.3 sur le run diagnostic du 24/08) : c'est exactement la
# surveillance qu'on s'était promis de faire qui a payé - `visualize_predictions.py --scope test`
# a montré une explosion de fausses alertes ("Debris_Divers"/"Bouteille" etc. à confiance 0.26-0.72
# sur du sable vide) concentrée sur les grandes orthomosaïques SL, absente sur les tuiles SB. Suspect
# principal : copy_paste colle des instances sur des fonds dont l'éclairage/le grain ne correspond pas
# forcément, ce qui peut apprendre au modèle à repérer des "indices de collage" plutôt qu'un vrai
# objet - contrairement à degrees/flipud qui ne font que tourner la MÊME photo réelle, donc ne peuvent
# pas inventer un objet qui n'existe pas. Ce prochain run (copy_paste=0, degrees/flipud inchangés) sert
# à confirmer si c'est bien la cause avant de retenter une valeur de copy_paste plus prudente (voir
# journal, "Vérifications post-run : audit des tailles par classe + inspection visuelle").
COPY_PASTE = 0.0

# scale (défaut 0.5, soit un zoom aléatoire ~0.5x-1.5x) : PAS changé ici. C'est le paramètre qui
# répond à l'écart de GSD actuel (~0.5 cm/px) vs futur matériel (~1 cm/px, facteur ~2x) - voir le
# journal, section géolocalisation/GSD. Le défaut couvre déjà un facteur ~3x, donc probablement
# suffisant tel quel - mais ce run est justement l'occasion de le VÉRIFIER sur le rapport test
# (plutôt que de le deviner) avant de décider s'il faut l'élargir.
#
# hsv_h/hsv_s/hsv_v (variations teinte/saturation/luminosité), mosaic, erasing : laissés aux
# défauts Ultralytics - déjà actifs et raisonnables (hsv_v=0.4 couvre une variation de luminosité
# significative, pertinente pour du sable au soleil/à l'ombre). mixup : laissé désactivé (défaut
# 0.0) - mélange deux images entières, utile surtout sur des scènes complexes/multi-objets ; sur
# de petits déchets isolés sur fond de sable, le risque de brouiller des masques déjà petits
# dépasse probablement le bénéfice - pas de raison de l'activer sans preuve que ça aide.

# --- Étiquette libre pour retrouver ce run dans output/runs/ ---------------
# Sert uniquement à la lisibilité du nom de dossier - mets ce que tu veux,
# par ex. "baseline", "test_yolov8s", "sans_class_nonplastique", etc.
RUN_TAG = "ablation_no_copy_paste"

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
        # Augmentation choisie consciemment pour de la vue nadir + rééquilibrage
        # des classes rares - voir bloc de commentaires ci-dessus (24/08/2026).
        degrees=DEGREES,
        flipud=FLIPUD,
        copy_paste=COPY_PASTE,
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
