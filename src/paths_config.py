# -*- coding: utf-8 -*-
"""
PixelOdyssey - Résolution dynamique et portable des chemins du projet.

Localise automatiquement le dossier de données PixelOdyssey quel que soit
l'ordinateur ou la lettre de lecteur attribuée par Windows (D:, E:, F:...),
sans nécessiter de configuration manuelle ni de variable d'environnement.

Ordre de résolution de DATA_ROOT :
1. Variable d'environnement PIXELODYSSEY_DATA_DIR (si définie explicitement).
2. Chemins relatifs au dépôt (ex: structure portable sur clé/disque USB).
3. Scan automatique des disques Windows connectés à la recherche de la
   signature du dataset ('3. Processed dataset' et '5. Tests').
4. Valeur de repli historique (E:\\PixelOdyssey).
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

_ENV_VAR = "PIXELODYSSEY_DATA_DIR"
_DEFAULT_FALLBACK = r"E:\PixelOdyssey"

# Racine du dépôt de code (C:\...\PixelOdyssey ou <lecteur>:\...\repo)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_valid_data_root(candidate: Path) -> bool:
    """Vérifie si le dossier candidat contient bien la signature des données PixelOdyssey."""
    if not candidate.exists() or not candidate.is_dir():
        return False
    # Signature obligatoire : dossier des données traitées + tests ou brut
    has_processed = (candidate / "3. Processed dataset").is_dir()
    has_tests = (candidate / "5. Tests").is_dir()
    has_raw = (candidate / "1_annotated_dataset").is_dir() or (
        candidate / "3. Processed dataset" / "1_annotated_dataset"
    ).is_dir()
    return has_processed and (has_tests or has_raw)


def _get_available_windows_drives() -> List[str]:
    """Liste les lettres de lecteurs accessibles sous Windows (D:, E:, F:, etc.)."""
    drives = []
    if os.name == "nt":
        # Priorité aux disques externes/secondaires usuels, puis C: en dernier
        letters = "DEFGHIJKLMNOPQRSTUVWXYZABC"
        for letter in letters:
            drive_path = f"{letter}:\\"
            if os.path.exists(drive_path):
                drives.append(drive_path)
    return drives


def _resolve_data_root() -> Path:
    """Localise le dossier de données de façon totalement autonome."""
    # 1. Variable d'environnement explicite
    env_dir = os.environ.get(_ENV_VAR)
    if env_dir:
        env_path = Path(env_dir)
        if _is_valid_data_root(env_path):
            return env_path.resolve()

    # 2. Chemins relatifs (architecture bundle autonome sur disque portable)
    relative_candidates = [
        REPO_ROOT.parent / "PixelOdyssey",        # dossier frère
        REPO_ROOT.parent / "data",                # dossier bundle data/
        REPO_ROOT / "data",                       # data sous le repo
        REPO_ROOT.parent.parent / "PixelOdyssey", # arborescence décalée
    ]
    for cand in relative_candidates:
        if _is_valid_data_root(cand):
            return cand.resolve()

    # 3. Scan automatique des disques Windows connectés
    for drive in _get_available_windows_drives():
        cand = Path(drive) / "PixelOdyssey"
        if _is_valid_data_root(cand):
            return cand.resolve()

    # 4. Repli sur le chemin par défaut historique
    fallback_path = Path(_DEFAULT_FALLBACK)
    if _is_valid_data_root(fallback_path):
        return fallback_path.resolve()

    # Si rien n'a été trouvé, erreur guidée
    drives_scanned = ", ".join(_get_available_windows_drives()) or "aucun"
    raise FileNotFoundError(
        f"🛑 Dossier de données PixelOdyssey introuvable.\n"
        f"- Signature recherchée : un dossier contenant '3. Processed dataset' et '5. Tests'.\n"
        f"- Lecteurs Windows analysés : {drives_scanned}\n"
        f"- Solutions :\n"
        f"  1. Connecte le disque externe contenant les données.\n"
        f"  2. Ou définis la variable {_ENV_VAR} (ex: $env:{_ENV_VAR} = \"F:\\PixelOdyssey\")."
    )


# ===========================================================================
# RACINES GLOBALES EXPORTÉES
# ===========================================================================

DATA_ROOT = _resolve_data_root()


def resolve_data_config_path_field(config_data: dict) -> dict:
    """Retourne une COPIE de `config_data` (dict déjà chargé depuis un
    data_config*.yaml) avec son champ 'path' réécrit en ABSOLU, ancré sur
    DATA_ROOT (le disque détecté SUR CETTE MACHINE) - quel que soit le
    chemin enregistré dans le YAML versionné sous Git (relatif par
    construction, voir l'en-tête de ce module : "Permet à chaque machine
    d'exécuter l'entraînement sans modifier les fichiers YAML versionnés").

    Pourquoi indispensable avant de transmettre un data_config à Ultralytics
    (`YOLO(...).train()` ou `.val()`) : si 'path' est relatif, Ultralytics le
    résout contre SON PROPRE réglage global `datasets_dir`
    (AppData/Roaming/Ultralytics/settings.json sous Windows) - qui n'a AUCUNE
    raison de coïncider avec DATA_ROOT - plutôt que d'échouer bruyamment.
    Constaté le 11/09/2026 : `src.review.scoped_report` transmettait le
    data_config tel quel (sans ce traitement), et Ultralytics allait chercher
    les images sous "<datasets_dir Ultralytics>/3. Processed dataset/..." au
    lieu de "<DATA_ROOT>/3. Processed dataset/..." - échec silencieux
    ("images not found") sans rapport avec le modèle ou la donnée eux-mêmes.
    `train.py` s'en tirait déjà correctement (voir _prepare_resolved_data_config,
    qui appelle CETTE MÊME fonction) ; `training_report.generate_report` et
    `scoped_report._write_scoped_data_yaml` font maintenant pareil.

    Ne modifie jamais `config_data` en place (copie superficielle) - l'appelant
    reste libre de réutiliser le dict original pour autre chose (ex: lire
    `names`/`class_taxonomy` avant résolution du chemin)."""
    config_data = dict(config_data)
    raw_path_str = str(config_data.get("path", ""))
    sub_path = "3. Processed dataset"
    if sub_path in raw_path_str:
        # Chemin déjà enraciné sous "3. Processed dataset" (portable ou absolu
        # d'un AUTRE disque/machine) : ne garde que ce qui suit ce repère, pour
        # ne jamais dupliquer/mélanger un préfixe d'un autre DATA_ROOT.
        suffix = raw_path_str.split(sub_path)[-1].lstrip(r"\/")
        resolved_path = DATA_ROOT / sub_path / suffix
    else:
        resolved_path = DATA_ROOT / raw_path_str
    config_data["path"] = str(resolved_path)
    return config_data


# Données traitées et résultats (rétrocompatibilité stricte avec le code existant)
PROCESSED_DATASET_DIR = DATA_ROOT / "3. Processed dataset"
RESULTS_DIR = DATA_ROOT / "4. Results"

# Dossiers spécifiques nécessaires au banc d'évaluation autonome
ANNOTATED_DATASET_DIR = PROCESSED_DATASET_DIR / "1_annotated_dataset"
TESTS_ROOT_DIR = DATA_ROOT / "5. Tests"
TEST_GEN_DIR = TESTS_ROOT_DIR / "1_gen"       # Split de test général figé
TEST_SL_DIR = TESTS_ROOT_DIR / "2_SL"         # Split de test dédié Santa Luzia

# Sorties d'entraînement (poids, args.yaml, rapport_lecture.html, rapport_metrics.json,
# plots Ultralytics) - AJOUTÉ le 10/09/2026 : ces fichiers vivaient auparavant sous
# <racine du dépôt>/output/runs/, ce qui rendait le dépôt git impossible à pousser
# (poids de plusieurs dizaines/centaines de Mo par run). Nouveau slot numéroté au
# même niveau que "4. Results" et "5. Tests" (jamais "4." ou "5." - déjà pris) :
# chaque dossier de run individuel (ex: reference_multiclasse_yolo11n-seg_20260908_135356/)
# vit directement sous TRAINING_RUNS_DIR/, inchangé par ailleurs (train.py continue de
# nommer/organiser les runs exactement pareil, seule la racine change).
TRAINING_RUNS_DIR = DATA_ROOT / "6. Model outputs" / "runs"

# Dossiers de résultats d'inférence et d'assistance à l'annotation
PREDICTION_DIR = RESULTS_DIR / "2_prediction"
ASSISTED_ANNOTATION_DIR = RESULTS_DIR / "1_assisted_annotation"