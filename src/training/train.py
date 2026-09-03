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

Exemple :
    python -m src.training.train
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
# Note sur "nano vs medium" : plus de capacité n'aide QUE si la faiblesse observée est un souci
# de PRÉCISION DU MODÈLE (ex: contours de masque imprécis) - pas un souci de VOLUME/DIVERSITÉ DE
# DONNÉES (ex: classe rare sous-représentée), que plus de paramètres ne fait qu'aggraver côté
# surapprentissage sur un dataset encore petit (~540 images parentes avant slicing).
# Recommandation : regarder le rapport val/test par classe avant de sauter à "m" - si le nano
# généralise déjà bien mais plafonne en précision, tester "yolo11s-seg.pt" d'abord (saut de
# capacité plus mesuré) plutôt que "m" directement.

# --- Hyperparamètres d'entraînement -----------------------------------------
EPOCHS = 100
IMGSZ = 640
BATCH = 16          # -1 = laisse Ultralytics choisir automatiquement selon la VRAM dispo
PATIENCE = 20       # arrêt anticipé si aucune amélioration après N epochs
DEVICE = 0          # 0 = 1er GPU ; "cpu" = CPU ; "0,1" = multi-GPU
WORKERS = 4

# --- Augmentation -------------------------------------------------------------
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

# copy_paste (défaut Ultralytics 0.0, désactivé) : colle des instances segmentées dans le même batch -
# c'est le mécanisme natif Ultralytics pour l'oversampling par augmentation des classes rares
# (Bidon, Bouee) - et, occasion identifiée le 28/08, un moyen de simuler des déchets rapprochés/qui se
# touchent (utile pour le problème de rappel en zone de forte accumulation).
#
# CORRECTIF (28/08/2026) : précédemment laissé à 0.0 par crainte de coller des instances sur des fonds
# dont l'éclairage/le grain ne correspond pas - MAIS cette crainte suppose `copy_paste_mode="mixup"`
# (colle une instance d'une AUTRE image). Vérifié dans le code source d'Ultralytics
# (`ultralytics/data/augment.py`, classe CopyPaste) : le mode PAR DÉFAUT est `copy_paste_mode="flip"`
# (voir explicitement ci-dessous), qui colle une copie MIROIR d'une instance de la MÊME image sur
# elle-même - même éclairage, même grain, même capteur/exposition. Le risque de "collage" que
# degrees/flipud n'ont pas est donc largement écarté par ce mode, contrairement à ce que le
# commentaire précédent laissait entendre. Réintroduit en isolant cette SEULE variable par rapport à
# `ref_split_corrige` (même split, mêmes autres hyperparamètres) pour mesurer proprement son effet.
# Valeur 0.5 : reprend l'exemple donné par Ultralytics lui-même dans la docstring de CopyPaste, assez
# fort pour produire un effet mesurable sur un seul run.
# À vérifier malgré tout après ce run via `visualize_predictions.py --scope test` : même en mode flip,
# un collage laisse une jointure (pas de fondu) - à surveiller pour des artefacts de bord, et à
# comparer aux résultats de `ref_split_corrige` via `compare_runs.py`.
#
# PAUSE (28/08/2026) : run `copy_paste_flip05` (0.5) comparé à `ref_split_corrige` - pas d'amélioration
# sur TEST (rappel macro -3,8pp, précision globale pondérée -8,4pp), malgré une nette amélioration sur
# VAL - décalage val/test à comprendre avant de remonter cette valeur. Remis à 0.0 le temps d'isoler
# proprement `cls_pw` (voir ci-dessous) contre `ref_split_corrige` - PAS empilé sur copy_paste tant que
# ce dernier n'est pas confirmé comme un progrès réel. Voir journal pour le détail complet.
COPY_PASTE = 0.0
COPY_PASTE_MODE = "flip"  # explicite : c'est le défaut Ultralytics, mais la distinction flip/mixup
                           # est le coeur du correctif ci-dessus - jamais la laisser implicite ici.
                           # Sans effet tant que COPY_PASTE=0.0 (voir PAUSE ci-dessus).

# cls_pw (défaut Ultralytics 0.0, désactivé) : PAS l'ancien "cls_pw" de YOLOv5 (poids BCE fixe) - dans
# cette version, pondère la loss de classification PAR CLASSE, selon la fréquence des instances dans
# le train du split (uniquement train, jamais val/test - vérifié dans
# `ultralytics/models/yolo/detect/train.py`, `DetectionTrainer.get_class_counts/set_class_weights`,
# dont hérite le trainer de segmentation) :
#   poids_classe = (1 / nb_instances_de_cette_classe_dans_train) ** cls_pw
# puis normalisé pour que la moyenne des poids sur les 7 classes vaille 1,0 (l'échelle globale de la
# loss ne change pas, seule sa répartition entre classes change). cls_pw=0 : tous les poids = 1
# (désactivé). cls_pw=1 : inverse de fréquence complet - vu le déséquilibre extrême de ce dataset
# (Debris_Divers ~75% des instances), risque de sur-corriger et de sacrifier sa précision pour un
# gain incertain sur les classes à très faible effectif (Bouee, Cagette, Bidon).
#
# Valeur 0.5 testée le 28/08 (run `cls_pw05`), comparaison chiffrée par classe pas encore faite
# (compare_runs.py pas encore relancé) - remis à 0.0 le temps de lancer l'expérience suivante
# ("sans Debris_Divers", voir --config/CONFIG_PATH ci-dessus et le journal) sans empiler deux
# changements non encore prouvés. Remettre à 0.5 (ou la valeur qui sera retenue) une fois cls_pw
# évalué, pour un run qui isole CETTE SEULE variable contre `ref_split_corrige`.
CLS_PW = 0.0

# scale (défaut 0.5, soit un zoom aléatoire ~0.5x-1.5x) : pas changé ici. C'est le paramètre qui
# répond à l'écart entre le GSD actuel (~0.5 cm/px) et un futur matériel (~1 cm/px, facteur ~2x) -
# le défaut couvre déjà un facteur ~3x, donc probablement suffisant tel quel. À vérifier sur le
# rapport test avant de décider s'il faut l'élargir.
#
# hsv_h/hsv_s/hsv_v (variations teinte/saturation/luminosité), erasing : laissés aux défauts
# Ultralytics - déjà actifs et raisonnables (hsv_v=0.4 couvre une variation de luminosité
# significative, pertinente pour du sable au soleil/à l'ombre).
#
# mosaic (défaut Ultralytics 1.0, `close_mosaic=10` - désactivé les 10 derniers epochs) : DÉJÀ
# actif à son maximum sur TOUS les runs de ce projet depuis le début, jamais une variable qui a
# changé - rien à "activer", contrairement à ce que suggérait la demande initiale. Aucun run
# supplémentaire ne teste ce paramètre, il n'y a rien de nouveau à isoler ici.
#
# mixup (défaut Ultralytics 0.0) : PAS le même mécanisme que copy_paste - mélange deux images
# ENTIÈRES par fondu (alpha blend), pas un collage d'instances découpées. Laissé désactivé
# jusqu'ici par crainte de brouiller des masques déjà petits (voir historique du fichier) - crainte
# jamais testée empiriquement. Réintroduit le 28/08 sur la base d'une étude 2024 trouvant
# spécifiquement mosaic+mixup efficaces sur des détecteurs mono-étage (famille YOLO) là où le
# rééquilibrage de loss/échantillonnage ne l'était pas (voir journal). Valeur 0.1 : départ prudent
# (mosaic était déjà à son max, donc c'est mixup qui porte tout le risque de ce run).
#
# PAUSE (28/08/2026) : run `mixup01` (0.1) comparé à `ref_split_corrige` sur TEST - précision macro
# +3,7pp et précision globale pondérée +6,9pp, MAIS rappel macro -3,2pp et surtout rappel global
# pondéré -6,9pp (31,8% -> 24,9%). mAP50-95 légèrement meilleur (+1,6pp) mais uniquement parce que
# mAP intègre sur tout le seuil de confiance - au seuil opérationnel réel (0,25 par défaut), c'est
# strictement plus de faux négatifs. Remis à 0.0 : troisième réglage d'affilée (après copy_paste_flip05
# et cls_pw05) qui échange du rappel contre de la précision sans gain net - or la priorité du projet
# est justement de RATTRAPER des détections manquées en zone dense, pas d'en perdre plus. Voir journal
# du 28/08 ("Pourquoi arrêter d'empiler augmentation/loss-reweighting") pour le raisonnement complet.
MIXUP = 0.0

# overlap_mask (défaut Ultralytics True, segmentation uniquement) : quand deux instances se
# chevauchent dans une image, Ultralytics fusionne TOUS les masques de l'image en un seul masque à
# 1 canal, en triant par aire décroissante (`polygons2masks_overlap`,
# `ultralytics/data/utils.py`) - au pixel où deux masques se chevauchent, seul le plus GRAND objet
# reste dans la cible d'entraînement, le plus petit y est effacé. Vérifié dans le code source
# (`ultralytics/data/augment.py` + `ultralytics/data/dataset.py`, `mask_overlap=hyp.overlap_mask`) :
# ce n'est pas une supposition, ce mécanisme tourne, actif, sur TOUS les runs de ce projet depuis le
# début (jamais examiné jusqu'ici) - contrairement à copy_paste/cls_pw/mixup qui rééquilibrent des
# CLASSES, celui-ci touche directement le problème n°2 signalé en tout début de cette série
# d'expériences (rappel qui chute en zone de forte accumulation, objets qui se touchent) : si un
# petit déchet est partiellement recouvert par un plus grand dans l'image, le modèle n'a
# actuellement JAMAIS vu son masque complet pendant l'entraînement, quelle que soit
# l'augmentation/le rééquilibrage testé par ailleurs - ça peut expliquer une partie du plafond de
# rappel observé sur les 3 runs précédents (aucun ne touchait à ce mécanisme).
# OVERLAP_MASK=False : chaque instance garde son propre canal de masque (pas de fusion/écrasement),
# au prix d'un coût mémoire/VRAM plus élevé (N canaux au lieu de 1, N = nb d'instances dans l'image -
# à surveiller sur ce dataset où Debris_Divers peut regrouper >15 instances dans une même tuile) ;
# réduire BATCH si CUDA out of memory. Isolé contre `ref_split_corrige` (copy_paste/cls_pw/mixup
# remis à 0.0 ci-dessus) - une seule variable nouvelle, et une famille de levier différente de
# celles déjà testées.
OVERLAP_MASK = False

# --- Étiquette libre pour retrouver ce run dans output/runs/ ---------------
# Sert uniquement à la lisibilité du nom de dossier - mets ce que tu veux,
# par ex. "baseline", "test_yolov8s", "sans_class_nonplastique", etc.
RUN_TAG = "overlap_mask_off"

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


def launch_training(config_path: Path = CONFIG_PATH, default_augment: bool = False, run_tag: str = RUN_TAG):
    """`config_path` : référentiel de classes/dataset à utiliser (défaut : config/data_config.yaml).
    Pour une variante expérimentale (ex: config/data_config_no_debris.yaml, qui pointe vers son
    propre `path:` de dataset tuilé - voir data_pipeline.py --config/--suffix pour la générer),
    passe --config en ligne de commande plutôt que d'éditer CONFIG_PATH ci-dessus : ça évite de
    laisser une expérience active par erreur pour le prochain run "normal".

    `default_augment` (ajouté le 02/09/2026, pour le baseline mono-classe) : True saute TOUS
    les kwargs d'augmentation/loss ci-dessus (degrees/flipud/copy_paste/copy_paste_mode/cls_pw/
    mixup/overlap_mask) - Ultralytics applique alors ses valeurs par défaut pures pour chacun.
    Point de vigilance à connaître avant de l'utiliser sur un dataset autre que le baseline
    mono-classe pour lequel ce flag a été ajouté : `degrees`/`flipud` ne sont PAS des réglages
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
    model = YOLO(MODEL_WEIGHTS)

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
