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

    E:\PixelOdyssey\6. Model outputs\runs\<RUN_TAG>_<modèle>_<AAAAMMJJ_HHMMSS>\

Garde-fou anti-écrasement (depuis le 23/09/2026, voir `launch_training`) : `run_dir`
est vérifié juste avant `model.train()` - s'il contient autre chose que le
`resolved_data_config.yaml` qu'on vient d'y écrire (donc un VRAI run antérieur au
même nom, cas extrême de deux lancements dans la même seconde), une RuntimeError
est levée plutôt que d'écraser silencieusement les résultats précédents.
(`exist_ok=True` est passé à `model.train()` : `run_dir` existe toujours déjà à ce
stade, pré-créé pour y écrire `resolved_data_config.yaml` - `exist_ok=False`
faisait dévier tous les vrais résultats vers "<run_name>2" à chaque run, bug
constaté le 23/09/2026.)

Rapport de lecture
-------------------
À la fin de l'entraînement, `training_report.py` évalue le meilleur modèle
(`best.pt`) sur les splits val ET test, et génère un rapport HTML autonome
dans le dossier du run (`rapport_lecture.html`) : tableau par classe avec
taux de faux positifs/négatifs, graphique précision/rappel, matrice de
confusion. Il peut aussi être regénéré à tout moment pour un run passé, sans
relancer l'entraînement :

    python -m src.training.training_report --run "E:\PixelOdyssey\6. Model outputs\runs\<nom_du_run>"

Entrée : dataset configuré via config/data_config.yaml, paramètres de la section 1 ci-dessus.
Sortie : poids entraînés + rapport HTML dans TRAINING_RUNS_DIR/<nom_du_run>/ (voir
         src/paths_config.py - déménagé hors du dépôt git le 10/09/2026).

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
    python -m src.training.train --config config/variants/data_config_no_debris.yaml --tag no_debris
    python -m src.training.train --config config/variants/data_config_SL.yaml --tag SL_dedie

    # Défauts Ultralytics purs pour l'augmentation/le rééquilibrage (degrees, flipud,
    # copy_paste, copy_paste_mode, cls_pw, mixup, overlap_mask NON transmis - voir la
    # docstring de launch_training() pour le point de vigilance sur degrees/flipud,
    # qui ne sont PAS de simples leviers de rééquilibrage).
    python -m src.training.train --default-augment --tag defaults_purs

    # Combinable : dataset alternatif ET défauts purs à la fois.
    python -m src.training.train --config config/data_config_mono_class.yaml --default-augment --tag mono_class_defaults
"""

import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

# Doit être fait AVANT tout import de torch/ultralytics (l'initialisation du
# runtime OpenMP a lieu à l'import) : évite "OMP: Error #15 - libiomp5md.dll
# already initialized" (torch et numpy embarquent chacun leur propre copie du
# runtime OpenMP sous Windows). Fixé ici plutôt que par variable d'environnement
# système/conda - fonctionne quel que soit comment le script est lancé (terminal
# activé ou non, noyau Jupyter, bouton "Run" de VS Code...), sans étape manuelle
# à reproduire sur chaque machine. setdefault : un choix explicite de
# l'utilisateur (variable déjà positionnée avant de lancer le script) n'est
# jamais écrasé.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import yaml
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.utils.class_config import DEFAULT_CLASS_CONFIG_PATH
from src.training.training_report import generate_report, resolve_run_class_config_path
from src.paths_config import DATA_ROOT, TRAINING_RUNS_DIR, resolve_data_config_path_field
from src.review.visualize_predictions import run_visualize


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
EPOCHS = 100         # nombre d'epochs (arrêt anticipé désactivé depuis le 09/10/2026, voir PATIENCE)
IMGSZ = 640          # doit correspondre à la taille des tuiles du dataset (voir slice_dataset.py)
BATCH = 16           # -1 = laisse Ultralytics choisir automatiquement selon la VRAM disponible
PATIENCE = 0         # 0 = arrêt anticipé désactivé : va au bout des EPOCHS (best.pt reste la meilleure epoch en val).
#                       Passé de 20 à 0 le 09/10/2026 : avec 20, le multi de la campagne du 08/10 s'est arrêté à
#                       l'epoch 64 (meilleure : 44), taux d'apprentissage encore à 40 % - sans jamais atteindre la
#                       fin du programme (taux bas + 10 dernières epochs sans mosaïque), où les modèles gagnent le
#                       plus (meilleure epoch du mono : 95/100). Résultat : multi 0,466 -> 0,368 sur les mêmes images.
DEVICE = 0           # 0 = 1er GPU ; "cpu" = CPU ; "0,1" = multi-GPU - repli automatique sur CPU si
                     # aucun GPU CUDA n'est détecté (voir _resolve_device), donc laisser 0 ici ne
                     # plante plus sur une machine/installation sans CUDA (torch CPU-only par ex.)
WORKERS = 2          # parallélisme du chargement des données. À 0, tout se fait dans le process
                     # principal : le GPU attend alors le CPU entre chaque batch (chargement +
                     # augmentation), d'où une utilisation GPU proche de 0% malgré de la VRAM occupée -
                     # observé le 11/09/2026 après être passé à 0 pour éviter un crash mémoire (voir
                     # ci-dessous). 2 restaure un minimum de recouvrement CPU/GPU sans redemander
                     # autant de mémoire que 4 : avec CUDA, CHAQUE worker réimporte torch (donc recharge
                     # les DLL CUDA, plusieurs centaines de Mo) dans son propre sous-processus - WORKERS
                     # élevé démultiplie d'autant la mémoire/pagination requise, cause du crash Windows
                     # WinError 1450/1455 ("ressources système"/"fichier de pagination insuffisant" au
                     # chargement de cublas64_12.dll). Ce réglage suppose le fichier d'échange Windows
                     # agrandi (voir guide de setup, 1.4) - sinon repasser à 0 en attendant.

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

# --- Recettes nommées (12/10/2026) --------------------------------------------
# "v3" = les constantes ci-dessus (recette de toutes les campagnes jusqu'au 11/10/2026), toujours
# utilisée par défaut. "v4" = recette du protocole v4 (--recette v4), à utiliser avec les datasets
# config/data_config*_v4.yaml. Ce qui change par rapport à v3, et pourquoi :
#   - epochs 200 (100 avant) : les tuiles v4 sont ~45 % moins nombreuses (recouvrement 128 px au lieu
#     de 256), 200 epochs v4 font donc à peu près autant de pas d'apprentissage que 100 epochs v3 ; le
#     mono v3 progressait encore à l'epoch 95/100. Pas d'arrêt anticipé (patience 0).
#   - optimiseur AdamW et taux fixés EXPLICITEMENT, à la valeur que choisissait `optimizer=auto`
#     (0,002 x 5 / (4 + nombre de classes) : 0,002 en mono, 0,000909 en multi). En automatique,
#     Ultralytics bascule sur MuSGD lr 0,01 dès 10 000 itérations : un simple changement d'epochs ou de
#     taille de dataset changeait d'optimiseur sans prévenir.
#   - overlap_mask revient au défaut Ultralytics (True) : seulement 3 % des contours se chevauchent (SL,
#     0,5 % à A, 0,3 % à SB - mesuré le 12/10/2026), le réglage n'a presque pas d'effet ; le test du
#     28/08 (False) n'avait pas montré de gain net, et True consomme moins de mémoire (utile en small).
#   - augmentations inchangées : celles qui ont fait leurs preuves (rotation 180°, retournements,
#     mosaïque, échelle, couleurs par défaut) ; copy_paste, mixup et cls_pw restent à 0 (aucun gain
#     mesuré sur le test). Pas d'augmentation de contraste : les orthos d'entraînement sont déjà aussi
#     ternes que le vol W5 (écart-type de luminance 13 à 25, contre 19 pour W5).
RECETTES = {
    "v3": dict(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE, degrees=DEGREES, flipud=FLIPUD,
               copy_paste=COPY_PASTE, copy_paste_mode=COPY_PASTE_MODE, cls_pw=CLS_PW, mixup=MIXUP,
               overlap_mask=OVERLAP_MASK),
    "v4": dict(epochs=200, imgsz=640, batch=16, patience=0, degrees=180.0, flipud=0.5, fliplr=0.5,
               mosaic=1.0, close_mosaic=10, scale=0.5, translate=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
               copy_paste=0.0, copy_paste_mode="flip", cls_pw=0.0, mixup=0.0, overlap_mask=True,
               optimizer="AdamW", lr0="auto_fit", cos_lr=False, seed=0, deterministic=True),
}


def _lr_fit(config_path: Path) -> float:
    """Taux que choisirait `optimizer=auto` en AdamW : 0,002 x 5 / (4 + nombre de classes)."""
    with open(config_path, "r", encoding="utf-8") as f:
        nc = len((yaml.safe_load(f) or {}).get("names", {})) or 1
    return round(0.002 * 5 / (4 + nc), 6)


# --- Étiquette libre pour retrouver ce run dans TRAINING_RUNS_DIR ----------
# Sert uniquement à la lisibilité du nom de dossier - mets ce que tu veux,
# par ex. "baseline", "test_yolov8s", "sans_class_nonplastique", etc.
RUN_TAG = "baseline"

# ============================================================================
# 2. Orchestration (nommage, non-écrasement, rapport) - pas besoin d'y toucher
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
# Sorties d'entraînement (poids, args.yaml, rapports) - déménagées le 10/09/2026 hors du
# dépôt git (impossible à pousser avec des poids de plusieurs dizaines/centaines de Mo par
# run) vers TRAINING_RUNS_DIR, voir src/paths_config.py. Comportement de nommage/non-
# écrasement des runs inchangé par ailleurs, seule la racine change.
OUTPUT_DIR = TRAINING_RUNS_DIR
# Emplacement dédié des poids pré-entraînés (nettoyage du 08/09/2026 : YOLO(MODEL_WEIGHTS)
# avec un simple nom de fichier téléchargeait auparavant dans le dossier courant, dispersant
# des .pt à la racine du repo à chaque changement de MODEL_WEIGHTS - Ultralytics télécharge
# à l'emplacement exact donné si le fichier n'y existe pas encore, donc ce chemin suffit à
# corriger ça pour de bon). Reste sous le dépôt (contrairement à OUTPUT_DIR ci-dessus) :
# ce sont des dépendances re-téléchargeables, pas des sorties du travail de Jame - déjà
# exclu de git via la règle `models/` du .gitignore, pas besoin de le déplacer physiquement.
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"

def _resolve_split_dir(config_path: Path) -> Path:
    """Détecte automatiquement le dossier de split (2_split_dataset*)

    associé à la configuration (ex: 4_sliced_dataset_mono_class ->
    2_split_dataset_mono_class).
    """
    # Corrigé le 23/09/2026 - deux erreurs cumulées faisaient TOUJOURS retomber sur le
    # repli (2_split_dataset, le dataset multi-classe complet), y compris pour un run
    # mono-classe ou dédié à un site : (1) le manifeste s'appelle ".parent_manifest.json"
    # (avec un point, voir split_dataset.PARENT_MANIFEST_FILENAME), pas "parent_manifest.json" ;
    # (2) `path` (relatif, versionné) n'était pas résolu contre DATA_ROOT, donc cherché
    # relativement au dossier courant. Conséquence : les visualisations de fin
    # d'entraînement (visualisations_test/) des runs mono/site montraient le test multi-classe
    # complet, pas celui du dataset réellement entraîné.
    from src.data.utils.split_dataset import PARENT_MANIFEST_FILENAME, SPLIT_DIR
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = resolve_data_config_path_field(yaml.safe_load(f) or {})
        dataset_path = cfg.get("path")
        if dataset_path:
            dataset_dir = Path(dataset_path)
            candidate_name = dataset_dir.name.replace("4_sliced_dataset", "2_split_dataset", 1)
            candidate_dir = dataset_dir.parent / candidate_name
            if (candidate_dir / PARENT_MANIFEST_FILENAME).exists():
                return candidate_dir
            print(f"⚠️  Dossier de split introuvable pour {config_path} ({candidate_dir}) - repli sur {SPLIT_DIR}.")
    except Exception as e:  # noqa: BLE001 - la visualisation ne doit jamais faire échouer un run
        print(f"⚠️  Résolution du dossier de split impossible ({e}) - repli sur {SPLIT_DIR}.")

    return Path(SPLIT_DIR)

def _prepare_resolved_data_config(config_path: Path, target_dir: Path) -> Path:
    """Adapte dynamiquement le champ 'path' du YAML pour correspondre au DATA_ROOT local.
    
    Permet à chaque machine (disque C:, D:, E:, etc.) d'exécuter l'entraînement
    sans modifier les fichiers YAML versionnés sous Git.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Résolution factorisée dans src/paths_config.py (resolve_data_config_path_field) -
    # réutilisée telle quelle par training_report.generate_report() et
    # src.review.scoped_report, qui ont besoin exactement du même traitement avant
    # de transmettre un data_config à Ultralytics (voir la docstring de cette fonction
    # pour le pourquoi : évite qu'Ultralytics résolve 'path' contre son propre
    # datasets_dir global plutôt que DATA_ROOT).
    config_data = resolve_data_config_path_field(config_data)

    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml_path = target_dir / "resolved_data_config.yaml"
    with open(resolved_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, sort_keys=False)

    return resolved_yaml_path

def _resolve_device(device):
    """Retombe sur "cpu" si `device` demande un GPU CUDA (int, "0", "0,1"...) et
    qu'aucun n'est disponible - CUDA absent (pas de carte NVIDIA) ou torch installé
    en version CPU-only (cas fréquent : `pip install ultralytics` seul tire par
    défaut le wheel torch CPU sous Windows, sans le build CUDA de pytorch.org).
    Sans ce garde-fou, Ultralytics lève une ValueError bloquante ("Invalid CUDA
    'device=0' requested") au lieu de simplement tourner sur CPU (plus lent, mais
    fonctionnel). Explicitement "cpu" n'est jamais réécrit : un choix explicite
    de l'utilisateur est toujours respecté tel quel."""
    if isinstance(device, str) and device.strip().lower() == "cpu":
        return device
    if not torch.cuda.is_available():
        print(f"⚠️  Aucun GPU CUDA détecté (device={device!r} demandé) - repli sur CPU. "
              "Si une carte NVIDIA est pourtant présente, vérifie l'installation de torch "
              "(https://pytorch.org/get-started/locally/ - il faut le build CUDA, pas la "
              "version CPU tirée par défaut avec `pip install ultralytics`).")
        return "cpu"
    return device


def _build_run_name(model_weights: str, tag: str) -> str:
    """<tag>_<modèle-sans-extension>_<horodatage> - unique à la seconde
    près, donc jamais de collision avec un run précédent."""
    model_slug = Path(model_weights).stem  # "yolo11n-seg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{tag}_{model_slug}_{timestamp}"


def launch_training(
    config_path: Path = CONFIG_PATH,
    default_augment: bool = False,
    run_tag: str = RUN_TAG,
    skip_viz: bool = False,
    viz_limit: Optional[int] = None,
    model_weights: Optional[str] = None,
    epochs: Optional[int] = None,
    recette: str = "v3",
):
    """`config_path` : référentiel de classes/dataset à utiliser (défaut : config/data_config.yaml).
    Pour une variante expérimentale (ex: config/variants/data_config_no_debris.yaml, qui pointe vers son
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
    sans éditer la constante en tête de fichier et risquer de l'oublier active au prochain run).

    `skip_viz` : saute la génération des images annotées du split test après l'entraînement (voir
    bloc EXPORT DES VISUALISATIONS ci-dessous) - utile pour un run rapide/exploratoire où ces
    images ne servent à rien (elles restent régénérables après coup, sans réentraîner, via
    src/review/visualize_predictions.py).

    `viz_limit` : nombre max d'images du split test à annoter (défaut : toutes) - utile pour un
    coup d'oeil rapide sur un gros split test sans attendre l'inférence sur chaque image."""
    if recette not in RECETTES:
        raise ValueError(f"Recette inconnue '{recette}' (disponibles : {', '.join(RECETTES)}).")
    if recette != "v3" and default_augment:
        raise ValueError("--default-augment ne se combine qu'avec la recette v3.")
    model_weights = model_weights or MODEL_WEIGHTS
    run_name = _build_run_name(model_weights, run_tag)
    run_dir = OUTPUT_DIR / run_name

    resolved_device = _resolve_device(DEVICE)

    print("--- 🏋️ INITIALISATION DE L'ENTRAÎNEMENT PIXELODYSSEY ---")
    print(f"Modèle              : {model_weights}")
    print(f"Recette             : {recette}")
    print(f"Config data          : {config_path}")
    print(f"Device               : {resolved_device}")
    print(f"Dossier de sortie    : {run_dir}")
    if default_augment:
        print("Augmentation        : DÉFAUTS ULTRALYTICS PURS (--default-augment) - degrees/flipud/"
              "copy_paste/copy_paste_mode/cls_pw/mixup/overlap_mask NON transmis, voir docstring.")

    # YOLO(...) télécharge automatiquement le poids pré-entraîné si besoin -
    # plus de gestion manuelle d'URL/urllib comme dans l'ancienne version.
    model = YOLO(str(PRETRAINED_DIR / model_weights))

    resolved_config = _prepare_resolved_data_config(config_path, run_dir)

    # Garde-fou anti-collision (ajouté le 23/09/2026 - bug constaté sur les 6 runs
    # relancés le jour même, ex: A_mono_v2_yolo11n-seg_20260923_155140) :
    # `_prepare_resolved_data_config` ci-dessus vient de CRÉER `run_dir` (pour y écrire
    # `resolved_data_config.yaml`, qui doit persister DANS le dossier du run - voir
    # `resolve_run_class_config_path` dans training_report.py). Avec `exist_ok=False`
    # ci-dessous, Ultralytics voit alors ce dossier déjà présent et dévie SILENCIEUSEMENT
    # tous les vrais résultats (poids, args.yaml, courbes...) vers "<run_name>2" au lieu
    # de `run_dir` - dossier que le reste de ce script (rapport, visualisations) ne
    # regarde jamais. `exist_ok=True` est donc nécessaire, MAIS seulement parce qu'on
    # sait exactement ce qu'on vient d'y mettre (juste ce yaml) : si `run_dir` contient
    # autre chose que ça, c'est le signe d'un VRAI run antérieur au même nom (le cas
    # extrême que le garde-fou d'origine visait) - on doit bloquer plutôt qu'écraser.
    unexpected_entries = sorted(p.name for p in run_dir.iterdir() if p.name not in ("resolved_data_config.yaml", "recette.json"))
    if unexpected_entries:
        raise RuntimeError(
            f"🛑 {run_dir} contient déjà {unexpected_entries} - probablement un run "
            f"antérieur au même nom. Abandon plutôt que d'écraser silencieusement "
            f"(voir 'Non-écrasement des runs' en tête de fichier)."
        )

    # Instantané de la répartition train/val/test RÉELLEMENT utilisée (23/09/2026) : le
    # manifeste du dataset peut changer après coup (split reconstruit, nouvelles images) ;
    # evaluation_matrix.py s'appuie sur CET instantané pour vérifier qu'aucune image vue à
    # l'entraînement ne se retrouve dans le test évalué.
    try:
        import shutil
        from src.data.utils.split_dataset import PARENT_MANIFEST_FILENAME
        train_split_dir = _resolve_split_dir(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            _sliced_name = Path(resolve_data_config_path_field(yaml.safe_load(f) or {})["path"]).name
        # N'enregistre l'instantané que si le split trouvé est bien CELUI de ce dataset (et pas
        # le repli par défaut de _resolve_split_dir) - sinon le contrôle de fuite lui ferait
        # confiance à tort.
        if train_split_dir.name == _sliced_name.replace("4_sliced_dataset", "2_split_dataset", 1):
            shutil.copy2(train_split_dir / PARENT_MANIFEST_FILENAME, run_dir / "training_parent_manifest.json")
        else:
            print(f"⚠️  Split d'entraînement non identifié avec certitude ({train_split_dir}) - "
                  f"instantané non enregistré (le contrôle de fuite utilisera le manifeste actuel).")
    except Exception as e:  # noqa: BLE001 - ne doit jamais empêcher l'entraînement
        print(f"⚠️  Instantané du split d'entraînement non enregistré ({e}).")

    train_kwargs = dict(
        data=str(resolved_config),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=resolved_device,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(OUTPUT_DIR),
        name=run_name,
        plots=True,
        # True (pas False) depuis le 23/09/2026 : `run_dir` est intentionnellement
        # pré-créé juste au-dessus par `_prepare_resolved_data_config` - voir le
        # garde-fou ci-dessus, qui couvre maintenant le cas qu'`exist_ok=False`
        # protégeait à l'origine (deux runs au même nom).
        exist_ok=True,
    )
    if recette != "v3":
        hp = dict(RECETTES[recette])
        if hp.get("lr0") == "auto_fit":
            hp["lr0"] = _lr_fit(config_path)
        if epochs:
            hp["epochs"] = epochs
        train_kwargs.update(hp)
        print("Hyperparamètres      : " + ", ".join(f"{k}={v}" for k, v in sorted(hp.items())))
        with open(run_dir / "recette.json", "w", encoding="utf-8") as f:
            import json as _json
            _json.dump({"recette": recette, "modele": model_weights, "config": str(config_path),
                        "hyperparametres": {**hp, "epochs": train_kwargs["epochs"]}}, f, indent=1, ensure_ascii=False)
    elif epochs:
        train_kwargs["epochs"] = epochs
    if not default_augment and recette == "v3":
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

    # --- 👁️ EXPORT DES VISUALISATIONS DU SPLIT TEST -------------------------
    if not skip_viz:
        print(
            "\n--- 👁️ GÉNÉRATION DES IMAGES DU SPLIT TEST (Vert=Réussi, Orange=Masque à ajuster, "
            "Rouge=Raté, Bleu=Faux Positif) ---"
        )
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            try:
                split_dir = _resolve_split_dir(config_path)
                viz_dir = run_dir / "visualisations_test"

                # Taxonomie dynamique : dérivée du run lui-même (args.yaml, écrit par
                # Ultralytics au lancement de l'entraînement) plutôt que du `config_path`
                # reçu en paramètre par CETTE fonction - qui pourrait en théorie diverger
                # (ex: `launch_training` appelée programmatiquement avec un config_path qui
                # ne correspond pas exactement au dataset réellement entraîné). Voir la
                # docstring de resolve_run_class_config_path pour le détail. Repli sur
                # `config_path` seulement si args.yaml est illisible pour une raison ou
                # une autre - ne doit jamais faire échouer tout le run.
                try:
                    viz_class_config_path = resolve_run_class_config_path(run_dir)
                except (FileNotFoundError, ValueError) as e:
                    print(f"⚠️ Taxonomie du run non retrouvée automatiquement ({e}) - repli sur --config.")
                    viz_class_config_path = config_path

                run_visualize(
                    model_path=str(best_pt),
                    scope="test",
                    class_config_path=viz_class_config_path,
                    split_dir=str(split_dir),
                    output_dir=str(run_dir),
                    run_id="visualisations_test",
                    limit=viz_limit,
                    tile_conf_threshold=0.25,
                    conf_threshold=0.25,
                )
                print(f"✅ Images annotées (.jpg) enregistrées dans : {viz_dir / 'images'}")
                print(f"📄 Visionneuse interactive générée : {viz_dir / 'index.html'}")
            except Exception as e:
                print(f"⚠️ Échec lors de la génération des visualisations ({e}).")
        else:
            print(f"⚠️ Fichier de poids introuvable : {best_pt}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement PixelOdyssey")
    parser.add_argument(
        "--config", default=str(CONFIG_PATH),
        help="Référentiel de classes/dataset à utiliser (défaut : config/data_config.yaml). "
             "Pour une variante expérimentale (ex: config/variants/data_config_no_debris.yaml), le "
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
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Ne génère pas les images annotées du split test après l'entraînement (régénérable "
             "après coup sans réentraîner, via src/review/visualize_predictions.py).",
    )
    parser.add_argument(
        "--viz-limit", type=int, default=None,
        help="Nombre max d'images du split test à annoter (défaut : toutes).",
    )
    parser.add_argument("--model", default=None,
                        help=f"Poids pré-entraînés (dans models/pretrained/), ex: yolo11s-seg.pt. Défaut : {MODEL_WEIGHTS}.")
    parser.add_argument("--epochs", type=int, default=None, help="Remplace le nombre d'epochs de la recette.")
    parser.add_argument("--recette", default="v3", choices=sorted(RECETTES),
                        help="Jeu d'hyperparamètres : v3 (défaut, constantes en tête de fichier) ou v4 (protocole "
                             "du 12/10/2026, avec les datasets config/data_config*_v4.yaml).")
    args = parser.parse_args()
    launch_training(
        config_path=Path(args.config),
        default_augment=args.default_augment,
        run_tag=args.tag,
        skip_viz=args.skip_viz,
        viz_limit=args.viz_limit,
        model_weights=args.model,
        epochs=args.epochs,
        recette=args.recette,
    )
