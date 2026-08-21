#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Étape 4 : Découpage en tuiles (slicing) pour YOLO.

Lit les images parentes complètes et déjà réparties par split depuis
`3_augmented_dataset/{images,labels}/{train,val,test}`, et les découpe en
imagettes 640x640 dans `4_sliced_dataset/{images,labels}/{train,val,test}`.

Cette étape ne décide plus du split (fait à l'étape 2, split_dataset.py) ni de
la traduction des classes par lot (faite elle aussi à l'étape 2, une seule fois
pour toutes - voir class_config.py) : les fichiers qu'elle lit sont déjà dans
le référentiel canonique de classes (super-classes cibles) et déjà répartis.
Elle ne fait plus que de la géométrie : sliding window, clipping des polygones
aux bords de tuile, filtrage des micro-débris. Elle n'a plus aucune notion de
classe (voir slicer.py, révisé le 19/08/2026).

(Anciennement `data_pipeline.py`. Renommé lors du passage à un orchestrateur
unique : `data_pipeline.py` est désormais le point d'entrée qui enchaîne les
4 étapes - vérification -> split -> augmentation -> slicing - et ce fichier
n'en est plus que la 4e brique. Reste utilisable seul, pour ne relancer QUE le
slicing après un ajustement du slicer sans repasser par tout le pipeline :
`python src/data/slice_dataset.py --force`.)

Garde-fou de cohérence de config
---------------------------------
Le cache incrémental (on ne re-tuile pas un parent déjà présent dans
4_sliced_dataset) n'est valable QUE si la configuration du slicer (tile_size,
overlap, discard_truncated, min_area_ratio) et la logique interne du slicer
(LOGIC_VERSION) sont identiques à celles utilisées lors du run précédent - voir
src/data/pipeline_utils.py. Le référentiel de classes (config/data_config.yaml)
n'a plus d'influence ICI (le slicer ne connaît plus les classes) : un
changement de class_taxonomy invalide le cache de l'étape 2 (split_dataset.py),
pas celui-ci. Sinon, --force permet de repartir de zéro.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.pipeline_utils import already_present, ensure_cache_is_safe
from src.data.slicer import LOGIC_VERSION, PlasticImageSlicer

BASE_DIR = r"E:\PixelOdyssey\3. Processed dataset"
AUGMENTED_DIR = os.path.join(BASE_DIR, "3_augmented_dataset")
SLICED_DIR = os.path.join(BASE_DIR, "4_sliced_dataset")

SPLITS = ["train", "val", "test"]
VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

MANIFEST_FILENAME = ".slicing_manifest.json"


def _slicing_params(slicer: PlasticImageSlicer) -> Dict:
    """Sérialise tous les paramètres qui influencent la sortie du slicer.

    Le slicer n'a plus aucune notion de classes (traduction faite une seule
    fois en amont, à l'étape 2 - voir split_dataset.py et class_config.py) :
    seuls les paramètres géométriques du constructeur comptent ici. Inclut
    aussi `LOGIC_VERSION` (slicer.py) : un changement de comportement interne
    du slicer sans changement de paramètre de constructeur (ex: la règle de
    rétention des tuiles de fond, v2) doit lui aussi invalider le cache -
    sinon deux logiques différentes se mélangeraient silencieusement dans le
    même dossier de sortie.
    """
    return {
        "slicer_logic_version": LOGIC_VERSION,
        "tile_size": slicer.tile_size,
        "overlap": slicer.overlap,
        "min_area_ratio": slicer.min_area_ratio,
        "discard_truncated": slicer.discard_truncated,
    }


def run_slice(force: bool = False, augmented_dir: str = AUGMENTED_DIR, sliced_dir: str = SLICED_DIR):
    # discard_truncated=False : conserve et découpe la géométrie au bord de la tuile.
    # La traduction des classes brutes -> super-classes cibles a désormais lieu UNE
    # SEULE FOIS, en amont, à l'étape 2 (split_dataset.py, voir class_config.py) : les
    # fichiers lus ici sont déjà dans le référentiel final. Le slicer n'a donc plus de
    # paramètre lié aux classes (voir sa docstring, révisée le 19/08/2026).
    slicer = PlasticImageSlicer(tile_size=640, overlap=256, discard_truncated=False)

    ensure_cache_is_safe(
        sliced_dir,
        _slicing_params(slicer),
        force=force,
        wipe_subdirs=["images", "labels"],
        manifest_filename=MANIFEST_FILENAME,
    )

    augmented_dir_p = Path(augmented_dir)
    processed_count = 0
    skipped_count = 0

    for split in SPLITS:
        img_src_dir = augmented_dir_p / "images" / split
        lab_src_dir = augmented_dir_p / "labels" / split
        if not img_src_dir.exists():
            print(f"⚠️  Split [{split}] : {img_src_dir} n'existe pas encore (rien à découper).")
            continue

        img_dst_dir = Path(sliced_dir) / "images" / split
        lab_dst_dir = Path(sliced_dir) / "labels" / split
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lab_dst_dir.mkdir(parents=True, exist_ok=True)

        existing_tiles = set(os.listdir(img_dst_dir))

        split_files = sorted(
            p for p in img_src_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS
        )
        print(f"--- ✂️  Split [{split}] : {len(split_files)} image(s) parente(s) ---")

        for img_path in split_files:
            parent_id = img_path.stem
            label_path = lab_src_dir / f"{parent_id}.txt"

            # Le slicer écrit toujours en .png (voir slicer.py), quel que soit le
            # format de l'image source (.jpg/.tif/...) - donc c'est bien ".png"
            # qu'il faut chercher dans existing_tiles, pas le suffixe de img_path.
            if already_present(parent_id, existing_tiles, suffix=".png"):
                skipped_count += 1
                continue

            slicer.slice_single_pair(
                img_path=img_path,
                label_path=label_path,
                output_img_dir=img_dst_dir,
                output_label_dir=lab_dst_dir,
                prefix=parent_id,
            )
            processed_count += 1

    print(f"\n[SUCCÈS] Découpage terminé !")
    print(f"  • Nouvelles images découpées : {processed_count}")
    print(f"  • Images ignorées (déjà là)  : {skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Étape 4 : découpage en tuiles PixelOdyssey")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Si la config de slicing a changé depuis le dernier run, supprime "
             "tout 4_sliced_dataset/images et /labels et retuile from scratch "
             "au lieu de s'arrêter avec une erreur.",
    )
    args = parser.parse_args()
    try:
        run_slice(force=args.force)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
