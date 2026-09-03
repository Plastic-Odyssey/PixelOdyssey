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
    3. SLICING en tuiles 640x640 (slice_dataset.py).

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

Variante de taxonomie (ex: dataset expérimental "sans Debris_Divers") :
    python src/data/data_pipeline.py --config config/data_config_no_debris.yaml --suffix _no_debris
    -> écrit dans 2_split_dataset_no_debris / 3_augmented_dataset_no_debris / 4_sliced_dataset_no_debris,
       AUCUN fichier partagé avec le pipeline par défaut (seul 1_annotated_dataset, la donnée brute
       source, reste commun). --suffix est obligatoire dès que --config diffère du défaut - garde-fou
       explicite plutôt que de risquer d'écraser silencieusement le dataset principal avec une autre
       taxonomie (voir _validate_config_suffix_pairing ci-dessous).

Variante de donnée brute (ex: dataset corrigé par src/review/review_false_positives.py) :
    python src/data/data_pipeline.py --raw-dir "E:\\PixelOdyssey\\3. Processed dataset\\1bis_corrected_annotation" --suffix _corrected
    -> même mécanisme que --config/--suffix ci-dessus, mais pour changer la SOURCE brute (1_annotated_dataset
       par défaut) plutôt que la taxonomie - --suffix également obligatoire dès que --raw-dir diffère du
       défaut, pour la même raison (ne jamais écraser silencieusement 2_split_dataset/3_augmented_dataset/
       4_sliced_dataset avec une donnée source différente). --config et --raw-dir sont indépendants et
       combinables (ex: dataset corrigé ET sans Debris_Divers à la fois).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.augment_dataset import AUGMENTED_DIR, run_augment
from src.data.dataset_sanity_check import PlasticDatasetChecker
from src.data.pipeline_utils import RunConfirmation
from src.data.raw_dataset_checker import RawDatasetValidator
from src.data.slice_dataset import SLICED_DIR, run_slice
from src.data.split_dataset import RAW_DIR, SPLIT_DIR, run_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"


def _validate_config_suffix_pairing(config_path: str, suffix: str, raw_dir: str) -> None:
    """Garde-fou : --config et/ou --raw-dir non défaut sans --suffix écraserait le dataset
    principal (2_split_dataset/3_augmented_dataset/4_sliced_dataset) avec une taxonomie et/ou
    une donnée source différente - jamais silencieux, on arrête tout de suite plutôt que de
    laisser data_pipeline.py trancher à la place de l'utilisateur."""
    non_default = []
    if str(Path(config_path).resolve()) != str(CONFIG_PATH.resolve()):
        non_default.append(f"--config pointe vers une taxonomie non-défaut ({config_path})")
    if str(Path(raw_dir).resolve()) != str(Path(RAW_DIR).resolve()):
        non_default.append(f"--raw-dir pointe vers une donnée source non-défaut ({raw_dir})")

    if non_default and not suffix:
        raise RuntimeError(
            "🛑 " + " ET ".join(non_default) + " mais --suffix est vide : ça écrirait dans les "
            "MÊMES dossiers que le dataset principal (2_split_dataset/3_augmented_dataset/"
            "4_sliced_dataset), avec une donnée différente - dataset principal écrasé. Fournis "
            "--suffix (ex: --suffix _corrected) pour écrire dans des dossiers parallèles dédiés."
        )


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


def run_full_pipeline(
    force: bool = False, config_path: str = str(CONFIG_PATH), suffix: str = "", raw_dir: str = RAW_DIR,
    balance_by: str = "images",
) -> None:
    """`config_path`/`suffix` : voir la docstring du module ("Variante de taxonomie"). `raw_dir` :
    voir "Variante de donnée brute" (ex: 1bis_corrected_annotation produit par
    review_false_positives.py). Défauts inchangés (config/donnée principales, aucun suffixe) ->
    comportement strictement identique à avant l'ajout de ces paramètres."""
    _validate_config_suffix_pairing(config_path, suffix, raw_dir)
    split_dir = SPLIT_DIR + suffix
    augmented_dir = AUGMENTED_DIR + suffix
    sliced_dir = SLICED_DIR + suffix

    # Une SEULE RunConfirmation partagée par les 3 étapes de ce run : évite une invite
    # d'écrasement par étape quand un changement amont (ex: taxonomie) cascade sur les 3
    # (voir pipeline_utils.RunConfirmation). Sans effet si `force=True`, ou si aucune étape
    # ne rencontre de config changée (rien à confirmer dans ce cas).
    run_confirmation = RunConfirmation()

    print("=" * 70)
    print(f"ÉTAPE 0/4 — VÉRIFICATION D'INTÉGRITÉ DU DATASET BRUT ({raw_dir})")
    print("=" * 70)
    validator = RawDatasetValidator(raw_dir, str(config_path))
    if not validator.run_all():
        print(
            "\n🛑 PIPELINE ARRÊTÉ AVANT LE SPLIT : le dataset brut n'est pas exploitable "
            f"en l'état ({len(validator.problems)} problème(s) listé(s) ci-dessus).\n"
            "   Corrige-les dans 1_annotated_dataset (ou complète class_taxonomy / "
            f"class_aliases dans {config_path} selon le cas), puis relance ce script."
        )
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"ÉTAPE 1/4 — SPLIT TRAIN / VAL / TEST -> {split_dir}")
    print("=" * 70)
    run_split(
        force=force, raw_dir=raw_dir, split_dir=split_dir, class_config_path=config_path,
        run_confirmation=run_confirmation, balance_by=balance_by,
    )
    _post_stage_check(Path(split_dir).name, split_dir)

    print("\n" + "=" * 70)
    print(f"ÉTAPE 2/4 — AUGMENTATION (train uniquement, val/test inchangés) -> {augmented_dir}")
    print("=" * 70)
    run_augment(force=force, split_dir=split_dir, augmented_dir=augmented_dir, run_confirmation=run_confirmation)
    _post_stage_check(Path(augmented_dir).name, augmented_dir)

    print("\n" + "=" * 70)
    print(f"ÉTAPE 3/4 — DÉCOUPAGE EN TUILES (SLICING) -> {sliced_dir}")
    print("=" * 70)
    run_slice(force=force, augmented_dir=augmented_dir, sliced_dir=sliced_dir, run_confirmation=run_confirmation)
    _post_stage_check(Path(sliced_dir).name, sliced_dir)

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLET TERMINÉ")
    print(f"   Dataset prêt pour l'entraînement dans : {sliced_dir}")
    if suffix:
        print(f"   Rappel : lance l'entraînement avec --config {config_path} (voir train.py --config).")
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
    parser.add_argument(
        "--config", default=str(CONFIG_PATH),
        help="Référentiel de classes à utiliser (défaut : config/data_config.yaml). Pour une "
             "taxonomie alternative, fournis AUSSI --suffix (voir la docstring du module).",
    )
    parser.add_argument(
        "--raw-dir", default=RAW_DIR,
        help=f"Dossier de donnée brute à utiliser comme source (défaut : {RAW_DIR}). Pour une "
             "variante corrigée (ex: 1bis_corrected_annotation, voir "
             "src/review/review_false_positives.py), fournis AUSSI --suffix (voir la docstring "
             "du module, 'Variante de donnée brute').",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Suffixe ajouté aux 3 dossiers de sortie (2_split_dataset<suffix>, "
             "3_augmented_dataset<suffix>, 4_sliced_dataset<suffix>) - obligatoire dès que "
             "--config et/ou --raw-dir diffère du défaut, pour ne jamais écraser le dataset "
             "principal.",
    )
    parser.add_argument(
        "--balance-by", choices=["images", "items"], default="images",
        help="Transmis tel quel à split_dataset.py (étape 1) - voir son --balance-by. "
             "'items' équilibre train/val/test par nombre d'instances annotées plutôt que "
             "par nombre d'images (utile pour un dataset mono-classe ou très hétérogène en "
             "densité de déchets par image).",
    )
    args = parser.parse_args()
    try:
        run_full_pipeline(
            force=args.force, config_path=args.config, suffix=args.suffix, raw_dir=args.raw_dir,
            balance_by=args.balance_by,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
