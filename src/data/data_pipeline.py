#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Orchestrateur du pipeline de préparation de données.

Point d'entrée UNIQUE qui enchaîne les 4 étapes, dans l'ordre, avec un
garde-fou bloquant avant de commencer quoi que ce soit :

    0. VÉRIFICATION D'INTÉGRITÉ (raw_dataset_checker.py) sur 1_annotated_dataset.
       Si le dataset brut n'est pas exploitable (data.yaml d'un lot incohérent
       avec le référentiel, images/labels orphelins, classes hors référentiel,
       lignes de label mal formées...), le pipeline s'arrête ICI, avant de
       toucher à quoi que ce soit d'autre, en listant TOUS les problèmes
       bloquants trouvés (pas seulement le premier) - inutile de découvrir le
       3e problème seulement après avoir corrigé le 1er et relancé.
    1. SPLIT train/val/test (split_dataset.py, étanche au niveau image parente).
    2. AUGMENTATION (augment_dataset.py, pass-through pour l'instant - voir sa
       docstring pour les techniques envisagées).
    3. SLICING en tuiles 640x640 (slice_dataset.py, anciennement data_pipeline.py).

Chaque étape reste par ailleurs utilisable seule (ex: retoucher uniquement le
slicing sans repasser par le split) - voir le `if __name__` de chaque module.
Cet orchestrateur ne fait qu'appeler leurs fonctions `run_*` dans l'ordre ; il
ne duplique aucune logique.

Après le split et après le slicing, une vérification légère et NON bloquante
(dataset_sanity_check.py - orphelins images/labels) tourne automatiquement :
une anomalie à ce stade trahirait un bug du pipeline lui-même plutôt qu'un
problème de données sources (déjà filtré par la vérification de l'étape 0),
donc elle est signalée sans interrompre le run.

Usage :
    python src/data/data_pipeline.py             # pipeline complet
    python src/data/data_pipeline.py --force      # + reconstruit les étapes dont la config a changé
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.augment_dataset import AUGMENTED_DIR, run_augment
from src.data.dataset_sanity_check import PlasticDatasetChecker
from src.data.raw_dataset_checker import RawDatasetValidator
from src.data.slice_dataset import SLICED_DIR, run_slice
from src.data.split_dataset import RAW_DIR, SPLIT_DIR, run_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"


def _post_stage_check(label: str, base_path: str) -> None:
    """Vérification légère et non bloquante après le split, l'augmentation et le slicing.

    Contrairement à la vérification de l'étape 0 (qui porte sur les données
    SOURCES et bloque le pipeline), un vrai désalignement ici porterait sur une
    sortie que le pipeline vient de produire lui-même - un signal de bug
    plutôt qu'un problème de données à corriger en amont. On la signale donc
    sans interrompre le run. Un split simplement vide/absent à ce stade
    (petit échantillon, split pas encore peuplé) N'EST PAS traité comme une
    anomalie - voir la distinction orphan_issues/missing_splits dans
    dataset_sanity_check.py.
    """
    print(f"\n--- 🔎 Vérification post-étape : {label} ---")
    result = PlasticDatasetChecker(base_path=base_path).verify_all_splits()
    if result["orphan_issues"]:
        print(f"⚠️  Anomalie détectée sur {label} - probablement un bug du pipeline "
              f"(la donnée source a déjà été validée à l'étape 0). À investiguer, "
              f"mais le pipeline continue.")
    elif result["missing_splits"]:
        print(f"ℹ️  {label} : split(s) vide(s)/pas encore peuplé(s) à ce stade "
              f"({', '.join(result['missing_splits'])}) - normal avec un petit échantillon "
              f"ou un split encore non alimenté, pas un problème d'intégrité.")
    else:
        print(f"✅ {label} : rien à signaler.")


def run_full_pipeline(force: bool = False) -> None:
    print("=" * 70)
    print("ÉTAPE 0/4 — VÉRIFICATION D'INTÉGRITÉ DU DATASET BRUT (1_annotated_dataset)")
    print("=" * 70)
    validator = RawDatasetValidator(RAW_DIR, str(CONFIG_PATH))
    if not validator.run_all():
        print(
            "\n🛑 PIPELINE ARRÊTÉ AVANT LE SPLIT : le dataset brut n'est pas exploitable "
            f"en l'état ({len(validator.problems)} problème(s) listé(s) ci-dessus).\n"
            "   Corrige-les dans 1_annotated_dataset (ou complète class_taxonomy / "
            "class_aliases dans config/data_config.yaml selon le cas), puis relance ce script."
        )
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ÉTAPE 1/4 — SPLIT TRAIN / VAL / TEST")
    print("=" * 70)
    run_split(force=force)
    _post_stage_check("2_split_dataset", SPLIT_DIR)

    print("\n" + "=" * 70)
    print("ÉTAPE 2/4 — AUGMENTATION (train uniquement, val/test inchangés)")
    print("=" * 70)
    run_augment(force=force)
    _post_stage_check("3_augmented_dataset", AUGMENTED_DIR)

    print("\n" + "=" * 70)
    print("ÉTAPE 3/4 — DÉCOUPAGE EN TUILES (SLICING)")
    print("=" * 70)
    run_slice(force=force)
    _post_stage_check("4_sliced_dataset", SLICED_DIR)

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLET TERMINÉ")
    print(f"   Dataset prêt pour l'entraînement dans : {SLICED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline complet de préparation de données PixelOdyssey "
                     "(vérification -> split -> augmentation -> slicing)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Appliqué aux 3 étapes de traitement (split/augmentation/slicing) : si la "
             "config d'une étape a changé depuis le dernier run, reconstruit cette étape "
             "from scratch au lieu de s'arrêter avec une erreur. N'affecte pas la "
             "vérification d'intégrité (étape 0), qui n'a pas de notion de cache.",
    )
    args = parser.parse_args()
    try:
        run_full_pipeline(force=args.force)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
