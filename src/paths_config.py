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