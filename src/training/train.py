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

Entrée : dataset configuré via config/data_config.yaml, paramètres de la section 1 ci-dessus.
Sortie : poids entraînés + rapport HTML dans output/runs/<nom_du_run>/.

Exemples (voir --help pour la liste complète des options) :

    # Cas normal : utilise CONFIG_PATH (config/data_config.yaml) et RUN_TAG tels que
    # définis en tête de fichier (section 1 ci-dessus).
    python -m src.training.train

    # Étiquette ponctuelle sans éditer RUN_TAG - utile pour un run isolé (ex: un
    # baseline de comparaison) sans laisser une étiquette de test active pour le
    # prochain run "normal".
    python -m src.training.train --tag test_manuel

    # Taxonomie/dataset alternatif : nécessite que le 4_sliced_dataset correspondant
    # existe déjà (voir data_pipeline.py --config/--suffix, --site/--suffix ou
    # --raw-dir/--suffix pour le générer). --tag n'est jamais déduit automatiquement
    # de --config : à fournir explicitement pour un nom de run lisible.
    python -m src.training.train --config config/data_config_mono_class.yaml --tag mono_class
    python -m src.training.train --config config/data_config_no_debris.yaml --tag no_debris
    python -m src.training.train --config config/data_config_SL.yaml --tag SL_dedie

    # Défauts Ultralytics purs pour l'augmentation/le rééquilibrage (degrees, flipud,
    # copy_paste, copy_paste_mode, cls_pw, mixup, overlap_mask NON transmis - voir la
    # docstring de launch_training() pour le point de vigilance sur degrees/flipud,
    # qui ne sont PAS de simples leviers de rééquilibrage).
    python -m src.training.train --default-augment --tag defaults_purs

    # Combinable : dataset alternatif ET défauts purs à la fois.
    python -m src.training.train --config config/data_config_mono_class.yaml --default-augment --tag mono_class_defaults
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
# Combine une famille et une taille de modèle. Suffixe "-seg" obligatoire :
# ce sont les variantes segmentation (les seules pertinentes ici, PixelOdyssey
# délimite des masques de déchets, pas juste des boîtes). Ultralytics
# télécharge et met en cache le poids pré-entraîné correspondant automatiquement.
#
#   Famille  | Fichier                   | Taille (léger -> lourd)
#   ---------|---------------------------|---------------------------------
#   YOLOv8   | yolov8{n,s,m,l,x}-seg.pt  | n < s < m < l < x
#   YOLOv11  | yolo11{n,s,m,l,x}-seg.pt  | n < s < m < l < x
#
# Choix pas définitivement tranché : dépend de l'objectif de chaque run.
# Repère pour choisir : monter en taille (nano -> medium) n'aide QUE si la
# faiblesse observée est un problème de PRÉCISION du modèle (contours de
# masque imprécis, généralisation qui plafonne) - jamais un problème de
# VOLUME/DIVERSITÉ de données (ex: classe rare sous-représentée), qu'une
# architecture plus grande ne fait qu'aggraver côté surapprentissage sur un
# dataset encore de taille modeste. Regarder le rapport par classe d'un run
# nano avant de sauter à une taille supérieure.
MODEL_WEIGHTS = "yolo11n-seg.pt"   # <-- change UNIQUEMENT cette ligne pour tester un autre modèle

# --- Hyperparamètres d'entraînement -----------------------------------------
EPOCHS = 100         # nombre maximal d'epochs - l'arrêt anticipé (PATIENCE) coupe généralement avant
IMGSZ = 640          # doit correspondre à la taille des tuiles du dataset (voir slice_dataset.py)
BATCH = 16           # -1 = laisse Ultralytics choisir automatiquement selon la VRAM disponible
PATIENCE = 20        # arrêt anticipé si aucune amélioration après N epochs (0 = désactivé, va au bout des EPOCHS)
DEVICE = 0           # 0 = 1er GPU ; "cpu" = CPU ; "0,1" = multi-GPU
WORKERS = 4          # parallélisme du chargement des données. En cas de crash de ressources système
                     # sous Windows (WinError 1450) pendant l'entraînement ou la validation, abaisser
                     # cette valeur (jusqu'à 1 si besoin) est la mitigation connue - au prix d'un
                     # chargement de données plus lent.

# --- Augmentation géométrique : correction d'un fait du dataset, pas un levier de rééquilibrage ----
#
# Nos images sont des vues NADIR (drone à la verticale) : contrairement à une photo "normale" avec
# un horizon et un "haut" naturel (le cas pour lequel les défauts d'Ultralytics sont calibrés), un
# déchet photographié du dessus peut apparaître à N'IMPORTE QUELLE orientation. Tranché : ces deux
# valeurs restent actives quel que soit le nombre de classes ou l'objectif du run - ce n'est jamais
# un levier à couper par réflexe.
DEGREES = 180.0      # défaut Ultralytics = 0.0 (aucune rotation) ; 180 = rotation aléatoire sur tout le cercle
FLIPUD = 0.5         # défaut Ultralytics = 0.0 ; aligné sur FLIPLR (déjà à 0.5 par défaut, retournement
                     # horizontal), pour la même raison d'absence d'orientation privilégiée en vue nadir

# --- Leviers de rééquilibrage des classes / d'augmentation avancée ----------------------------------
#
# Chacun des quatre paramètres ci-dessous répond à un problème identifié (classes rares, objets qui
# se touchent) mais aucun n'a démontré à ce jour un gain net et sans contrepartie sur le jeu de test -
# tous restent des leviers OUVERTS, à évaluer un par un, jamais plusieurs à la fois (une comparaison
# n'a de sens que si une seule variable change par rapport à un run de référence stable).

# copy_paste (0=désactivé) : colle des instances segmentées dans le même batch - mécanisme natif
# Ultralytics pour sur-échantillonner les classes rares et simuler des déchets rapprochés/qui se
# touchent. copy_paste_mode="flip" colle une copie MIROIR d'une instance de la MÊME image (même
# éclairage, même grain) ; "mixup" colle une instance d'une AUTRE image (risque de collage
# visuellement incohérent, à éviter sauf besoin explicite). Statut : désactivé, question ouverte.
COPY_PASTE = 0
COPY_PASTE_MODE = "flip"  # à garder explicite (jamais implicite) même quand COPY_PASTE=0, pour que
                          # la valeur soit correcte le jour où ce levier est réactivé

# cls_pw (0=désactivé, 1=inverse de fréquence complet) : pondère la loss de classification PAR CLASSE
# selon la fréquence des instances dans le train uniquement (poids = (1/n_instances)**cls_pw,
# normalisé à moyenne 1,0 sur les 7 classes). Vu le déséquilibre extrême du dataset (une catégorie
# fourre-tout très majoritaire), une valeur élevée risque de sacrifier la précision pour un gain
# incertain sur les classes rares. Statut : désactivé, question ouverte.
CLS_PW = 0.0

# mixup (0=désactivé) : mélange deux images ENTIÈRES par fondu (alpha blend) - mécanisme différent de
# copy_paste, qui ne colle que des instances découpées. Risque principal : brouiller des masques déjà
# petits. Statut : désactivé, question ouverte.
MIXUP = 0.0

# overlap_mask (True=défaut Ultralytics, False=ici) : quand deux instances se chevauchent dans une
# image, True fusionne tous les masques en un seul canal (seul le plus grand objet reste appris à
# chaque pixel de recouvrement) ; False donne un canal par instance (coût VRAM plus élevé - réduire
# BATCH si CUDA out of memory). Touche directement le rappel en zone de forte accumulation d'objets.
# Statut : désactivé (False), question ouverte.
OVERLAP_MASK = False

# scale (défaut Ultralytics ~0.5-1.5x, non modifié) : zoom aléatoire, répond à l'écart de résolution
# au sol (GSD) entre différents matériels de capture - le défaut couvre déjà un facteur ~3x, à
# vérifier sur le rapport test seulement si un nouveau matériel de capture change sensiblement le GSD.
#
# hsv_h/hsv_s/hsv_v (teinte/saturation/luminosité), erasing : laissés aux défauts Ultralytics, jamais
# évalués séparément à ce jour.
#
# mosaic (défaut Ultralytics 1.0, close_mosaic=10) : actif à son maximum sur tous les runs du projet,
# jamais désactivé ni testé comme variable.

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
# Emplacement dédié des poids pré-entraînés (nettoyage du 08/09/2026 : YOLO(MODEL_WEIGHTS)
# avec un simple nom de fichier téléchargeait auparavant dans le dossier courant, dispersant
# des .pt à la racine du repo à chaque changement de MODEL_WEIGHTS - Ultralytics télécharge
# à l'emplacement exact donné si le fichier n'y existe pas encore, donc ce chemin suffit à
# corriger ça pour de bon)
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"


def _build_run_name(model_weights: str, tag: str) -> str:
    """<tag>_<modèle-sans-extension>_<horodatage> - unique à la seconde
    près, donc jamais de collision avec un run précédent."""
    model_slug = Path(model_weights).stem  # "yolo11n-seg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{tag}_{model_slug}_{timestamp}"


def launch_training(config_path: Path = CONFIG_PATH, default_augment: bool = False, run_tag: str = RUN_TAG):
    """`config_path` : référentiel de classes/dataset à utiliser (défaut : config/data_config.yaml).
    Pour une variante expérimentale (ex: config/data_config_no_debris.yaml, qui pointe vers son
    propre `path:` de dataset tuilé - voir data_pipeline.py --config/--suffix pour la générer),
    passe --config en ligne de commande plutôt que d'éditer CONFIG_PATH ci-dessus : ça évite de
    laisser une expérience active par erreur pour le prochain run "normal".

    `default_augment` : True saute TOUS les kwargs d'augmentation/loss ci-dessus (degrees/flipud/
    copy_paste/copy_paste_mode/cls_pw/mixup/overlap_mask) - Ultralytics applique alors ses valeurs
    par défaut pures pour chacun. Point de vigilance : `degrees`/`flipud` ne sont PAS des réglages
    de rééquilibrage de classe comme les autres (copy_paste/cls_pw/mixup/overlap_mask le sont) -
    ils corrigent un fait géométrique du dataset (vue nadir, aucune orientation "haut" naturelle,
    voir le commentaire au-dessus de DEGREES/FLIPUD) qui reste vrai quel que soit le nombre de
    classes. Les repasser aux défauts Ultralytics (degrees=0, flipud=0) désactive donc aussi CETTE
    correction-là, pas seulement le rééquilibrage de classe - à faire consciemment, pas par
    défaut réflexe.

    `run_tag` : remplace RUN_TAG pour CE run (utile pour lancer un run ponctuel, ex: un baseline,
    sans éditer la constante en tête de fichier et risquer de l'oublier active au prochain run)."""
    run_name = _build_run_name(MODEL_WEIGHTS, run_tag)
    run_dir = OUTPUT_DIR / run_name

    print("--- 🏋️ INITIALISATION DE L'ENTRAÎNEMENT PIXELODYSSEY ---")
    print(f"Modèle              : {MODEL_WEIGHTS}")
    print(f"Config data          : {config_path}")
    print(f"Dossier de sortie    : {run_dir}")
    if default_augment:
        print("Augmentation        : DÉFAUTS ULTRALYTICS PURS (--default-augment) - degrees/flipud/"
              "copy_paste/copy_paste_mode/cls_pw/mixup/overlap_mask NON transmis, voir docstring.")

    # YOLO(...) télécharge automatiquement le poids pré-entraîné si besoin -
    # plus de gestion manuelle d'URL/urllib comme dans l'ancienne version.
    model = YOLO(str(PRETRAINED_DIR / MODEL_WEIGHTS))

    train_kwargs = dict(
        data=str(config_path),
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
    if not default_augment:
        # Augmentation choisie pour la vue nadir + rééquilibrage des classes rares -
        # voir bloc de commentaires ci-dessus. Omis entièrement si default_augment=True
        # (voir docstring de cette fonction) - Ultralytics applique alors ses propres
        # défauts pour CHACUN de ces paramètres, pas seulement une valeur neutre choisie ici.
        train_kwargs.update(
            degrees=DEGREES,
            flipud=FLIPUD,
            copy_paste=COPY_PASTE,
            copy_paste_mode=COPY_PASTE_MODE,
            cls_pw=CLS_PW,
            mixup=MIXUP,
            overlap_mask=OVERLAP_MASK,
        )

    model.train(**train_kwargs)

    print("\n--- ✅ ENTRAÎNEMENT TERMINÉ ---")
    print(f"Les résultats bruts Ultralytics sont dans : {run_dir}")

    print("\n--- 📊 GÉNÉRATION DU RAPPORT DE LECTURE (val + test, faux positifs/négatifs par classe) ---")
    try:
        report_path = generate_report(run_dir, data_config_path=config_path)
        print(f"Rapport HTML à ouvrir dans un navigateur : {report_path}")
    except Exception as e:  # noqa: BLE001 - un échec du rapport ne doit pas remettre en cause le run lui-même
        print(f"⚠️ Le rapport n'a pas pu être généré automatiquement ({e}).")
        print("   Tu peux le regénérer plus tard sans relancer l'entraînement, avec :")
        print(f'   python -m src.training.training_report --run "{run_dir}"')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement PixelOdyssey")
    parser.add_argument(
        "--config", default=str(CONFIG_PATH),
        help="Référentiel de classes/dataset à utiliser (défaut : config/data_config.yaml). "
             "Pour une variante expérimentale (ex: config/data_config_no_debris.yaml), le "
             "dataset tuilé correspondant doit déjà exister (voir data_pipeline.py --config/--suffix).",
    )
    parser.add_argument(
        "--default-augment", action="store_true",
        help="Ne transmet AUCUN kwarg d'augmentation/loss personnalisé à model.train() - "
             "Ultralytics applique alors ses propres défauts (degrees=0, flipud=0, copy_paste=0, "
             "cls_pw=0, mixup=0, overlap_mask=True). Voir la docstring de launch_training() pour "
             "un point de vigilance (degrees/flipud ne sont pas de simples réglages de "
             "rééquilibrage de classe).",
    )
    parser.add_argument(
        "--tag", default=RUN_TAG,
        help=f"Remplace RUN_TAG (défaut actuel : '{RUN_TAG}') pour CE run uniquement, sans "
             f"éditer le fichier.",
    )
    args = parser.parse_args()
    launch_training(config_path=Path(args.config), default_augment=args.default_augment, run_tag=args.tag)
