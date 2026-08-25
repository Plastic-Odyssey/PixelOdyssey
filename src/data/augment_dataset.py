#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Étape 3 : Augmentation (PLACEHOLDER - pas encore implémentée).

Lit `2_split_dataset/{images,labels}/{train,val,test}` et écrit
`3_augmented_dataset/{images,labels}/{train,val,test}`.

RÈGLE STRUCTURELLE, PAS UNE SIMPLE CONVENTION :
------------------------------------------------
val/ et test/ sont TOUJOURS copiés tels quels (`_passthrough_split`), sans jamais
passer par la fonction d'augmentation. Seul train/ passe par `_augment_train_split`.
C'est délibérément deux fonctions distinctes plutôt qu'un paramètre optionnel sur
une fonction unique : ça rend impossible d'appeler par erreur une transformation
sur val/test en modifiant juste un flag - il faudrait changer quel appel de
fonction est fait, ce qui se voit à la revue de code.

Pourquoi c'est important : les techniques envisagées pour train/ (copy-paste de
masques de déchets, dédoublement d'images pour rééquilibrer des classes rares)
créent des images DÉRIVÉES d'images existantes. Si l'une de ces variantes finissait
en val ou en test, le modèle aurait "vu" une quasi-copie de ce qu'il est censé être
évalué sur - fuite de données. D'où le split (étape 2) avant l'augmentation (ici).

TODO (à discuter avant implémentation - cf. conversation projet) :
- Augmentation de contraste / couleur - vérifier d'abord si ça n'est pas déjà
  couvert par les augmentations à la volée d'Ultralytics pendant l'entraînement
  (hsv_h/hsv_s/hsv_v sont déjà actifs dans les args de train.py) avant de dupliquer
  cet effort en pré-calculant des variantes sur disque.
- Copy-paste de masques de déchets extraits - Ultralytics a un paramètre
  `copy_paste` natif (actuellement à 0.0 dans les args d'entraînement) : à évaluer
  avant de construire un pipeline de copy-paste maison.
- Dédoublement / oversampling des classes rares - probablement plus efficace après
  le découpage en tuiles (étape 4) qu'ici : dupliquer une tuile 640x640 contenant
  l'objet rare coûte beaucoup moins qu'dupliquer toute l'image parente. À trancher
  une fois qu'on aura les statistiques de classes par tuile.

Tant que ces décisions ne sont pas prises, ce script est un pass-through complet
(train/val/test copiés à l'identique) : 4_sliced_dataset peut déjà être généré à
partir de 3_augmented_dataset dès aujourd'hui, sans attendre que l'augmentation
soit implémentée.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.pipeline_utils import ensure_cache_is_safe, read_upstream_fingerprint
from src.data.split_dataset import MANIFEST_FILENAME as SPLIT_MANIFEST_FILENAME

BASE_DIR = r"E:\PixelOdyssey\3. Processed dataset"
SPLIT_DIR = os.path.join(BASE_DIR, "2_split_dataset")
AUGMENTED_DIR = os.path.join(BASE_DIR, "3_augmented_dataset")

MANIFEST_FILENAME = ".augment_manifest.json"

# Version des règles d'augmentation appliquées à train/. Tant qu'aucune technique
# n'est implémentée, ce n'est qu'un pass-through - mais on la fait quand même
# passer par le garde-fou de cache : le jour où une vraie technique est ajoutée,
# ce numéro DOIT être incrémenté pour forcer un ré-traitement de train/.
AUGMENTATION_RULES_VERSION = 0


def _augment_params(split_dir: Path) -> Dict:
    # "upstream_split_fingerprint" (24/08/2026) : propage le fingerprint de
    # l'étape amont (split) dans le nôtre - voir la docstring de
    # read_upstream_fingerprint() dans pipeline_utils.py pour le bug concret que
    # ça évite (fuite train/val silencieuse si le split change de logique SANS
    # que cette étape ait elle-même changé de paramètre).
    return {
        "augmentation_rules_version": AUGMENTATION_RULES_VERSION,
        "upstream_split_fingerprint": read_upstream_fingerprint(split_dir / SPLIT_MANIFEST_FILENAME),
    }


def _copy_split_dir(src_split_dir: Path, dst_split_dir: Path) -> int:
    """Copie un sous-dossier images/{split} ou labels/{split} tel quel. Retourne le nb de fichiers copiés."""
    if not src_split_dir.exists():
        return 0
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_split_dir.iterdir():
        if f.is_file():
            dst = dst_split_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                count += 1
    return count


def _passthrough_split(split: str, split_dir: Path, augmented_dir: Path) -> None:
    """val/ et test/ : copie strictement à l'identique, jamais de transformation."""
    n_img = _copy_split_dir(split_dir / "images" / split, augmented_dir / "images" / split)
    n_lab = _copy_split_dir(split_dir / "labels" / split, augmented_dir / "labels" / split)
    print(f"  [{split}] pass-through : {n_img} image(s), {n_lab} label(s) copiés tels quels.")


def _augment_train_split(split_dir: Path, augmented_dir: Path) -> None:
    """train/ : c'est ICI que les techniques d'augmentation s'appliqueront.

    Pour l'instant : pass-through identique à _passthrough_split, en attendant
    que les techniques (contraste, copy-paste, rééquilibrage) soient décidées et
    implémentées. Ne PAS appeler cette fonction pour val/ ou test/.
    """
    n_img = _copy_split_dir(split_dir / "images" / "train", augmented_dir / "images" / "train")
    n_lab = _copy_split_dir(split_dir / "labels" / "train", augmented_dir / "labels" / "train")
    print(f"  [train] pass-through (aucune augmentation implémentée pour l'instant) : "
          f"{n_img} image(s), {n_lab} label(s) copiés tels quels.")


def run_augment(force: bool = False, split_dir: str = SPLIT_DIR, augmented_dir: str = AUGMENTED_DIR):
    split_dir_p, augmented_dir_p = Path(split_dir), Path(augmented_dir)

    ensure_cache_is_safe(
        augmented_dir,
        _augment_params(split_dir_p),
        force=force,
        wipe_subdirs=["images", "labels"],
        manifest_filename=MANIFEST_FILENAME,
    )

    print("--- 🌱 AUGMENTATION (étape 3) ---")
    _augment_train_split(split_dir_p, augmented_dir_p)
    _passthrough_split("val", split_dir_p, augmented_dir_p)
    _passthrough_split("test", split_dir_p, augmented_dir_p)
    print("[SUCCÈS] Étape 3 terminée (pass-through - aucune technique d'augmentation "
          "implémentée pour l'instant).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Étape 3 : augmentation PixelOdyssey (placeholder)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Si les règles d'augmentation ont changé depuis le dernier run, "
             "supprime 3_augmented_dataset/images et /labels et régénère from "
             "scratch au lieu de s'arrêter avec une erreur.",
    )
    args = parser.parse_args()
    try:
        run_augment(force=args.force)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
