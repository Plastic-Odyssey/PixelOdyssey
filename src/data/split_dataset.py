#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Étape 2 : Split Train / Val / Test.

Lit les images parentes complètes (non découpées) et leurs labels depuis
`1_annotated_dataset`, décide de façon étanche (au niveau de l'image parente,
pas de la tuile) dans quel split chacune tombe, et écrit le résultat dans
`2_split_dataset/images/{train,val,test}` + `labels/{train,val,test}`.

Pourquoi cette étape doit avoir lieu AVANT l'augmentation (étape 3) :
--------------------------------------------------------------------
Certaines techniques d'augmentation prévues (copy-paste de masques de déchets,
dédoublement d'images pour rééquilibrer des classes rares) créent des images
DÉRIVÉES d'une image existante. Si l'augmentation avait lieu avant le split,
une image originale pourrait finir en train et sa quasi-copie en val/test :
le modèle aurait alors "vu" une variante de ce qu'il est censé évaluer sans
jamais l'avoir vu à l'identique - fuite de données, métriques trompeuses.
En décidant le split ici, sur des images encore non-augmentées, l'étape 3
peut ensuite augmenter librement la portion train sans jamais risquer de
faire fuiter quoi que ce soit vers val/test.

Ce que fait cette étape, en plus de router chaque image parente vers le bon
split - LA TRADUCTION DE CLASSES COMPLÈTE, en une seule passe (révisé le
19/08/2026, modèle par nom - voir class_config.py) :
  1. Lit le data.yaml LOCAL du lot d'origine de l'image (local_id -> nom).
  2. Normalise ce nom et le cherche dans `class_taxonomy` (config/data_config.yaml).
  3. Écrit directement l'ID de SUPER-CLASSE CIBLE final (celui de `names`), ou
     rien si la classe est exclue.
À partir de 2_split_dataset, les labels sont donc déjà dans leur espace de
classes FINAL - les étapes suivantes (augmentation, slicing) n'ont plus
aucune notion de classe à gérer, seulement de la géométrie/des fichiers.

Ce que cette étape NE fait PAS : elle ne découpe pas les images en tuiles
(étape 4) et n'altère pas la géométrie des polygones - seules les lignes de
label sont traduites/filtrées, les coordonnées ne sont pas touchées.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    EXCLUDE,
    load_batch_local_names,
    load_class_config,
    resolve_class_name,
)
from src.data.pipeline_utils import ensure_cache_is_safe
from src.data.raw_dataset import collect_parent_images

BASE_DIR = r"E:\PixelOdyssey\3. Processed dataset"
RAW_DIR = os.path.join(BASE_DIR, "1_annotated_dataset")
SPLIT_DIR = os.path.join(BASE_DIR, "2_split_dataset")

# NB : dataset encore petit et en phase d'expérimentation de méthode -> on garde un test
# set (pour préserver une évaluation finale non biaisée), mais un ratio réduit reste à
# discuter si l'entraînement manque cruellement de données.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
SEED = 42

MANIFEST_FILENAME = ".split_manifest.json"


def _translate_and_filter_label(
    src_label_path: Path,
    local_names: Dict[int, str],
    class_taxonomy,
    batch_name: str,
) -> List[str]:
    """Lit un label brut et traduit chaque ligne vers l'ID de super-classe final.

    `local_names` : data.yaml local DU LOT d'origine (local_id -> nom) - obligatoire,
    c'est la seule façon de savoir ce que "0" ou "8" veut dire pour CE lot précis.
    Les coordonnées ne sont jamais modifiées - seul l'ID de classe en tête de ligne
    change (ou la ligne disparaît si la classe est exclue). Retourne la liste des
    lignes de sortie (peut être vide - une image sans objet a un label vide, valide
    en YOLO, et c'est aussi le cas d'une image qui n'a carrément pas de .txt).
    """
    if not src_label_path.exists():
        return []  # Pas d'objets sur cette image (ou "sans déchet" confirmé) - label vide, valide en YOLO.

    out_lines = []
    with open(src_label_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            local_id = int(parts[0])
            name = local_names.get(local_id)
            if name is None:
                # L'ID utilisé dans le label n'est même pas déclaré dans le data.yaml
                # DE CE LOT - incohérence interne au lot, pas un problème de
                # référentiel. raw_dataset_checker.py (étape 0) est censé l'attraper
                # avant d'arriver ici ; ce garde-fou est une sécurité de secours.
                raise RuntimeError(
                    f"{src_label_path} (L{lineno}) : ID de classe {local_id} absent du "
                    f"data.yaml du lot '{batch_name}'. Corrige l'annotation ou le data.yaml "
                    f"de ce lot avant de relancer."
                )

            resolved = resolve_class_name(name, class_taxonomy)
            if resolved is None:
                # Nom réellement utilisé mais ni dans class_taxonomy ni dans class_aliases
                # (config/data_config.yaml) - c'est exactement le cas qui doit bloquer
                # plutôt que d'être ignoré en silence (cf. class_config.py). Encore une
                # sécurité de secours : raw_dataset_checker.py doit déjà avoir bloqué avant.
                raise RuntimeError(
                    f"{src_label_path} (L{lineno}) : classe '{name}' (lot '{batch_name}') "
                    f"n'est dans aucune entrée de class_taxonomy/class_aliases "
                    f"(config/data_config.yaml). Décide de son sort avant de relancer."
                )
            if resolved == EXCLUDE:
                continue

            out_lines.append(f"{resolved} {' '.join(parts[1:])}\n")

    return out_lines


def _split_params(class_taxonomy, target_names: Dict[int, str]) -> Dict:
    """Paramètres qui influencent la sortie de cette étape (pour le garde-fou de cache)."""
    return {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "class_taxonomy": {k: v for k, v in sorted(class_taxonomy.items())},
        "target_names": {str(k): v for k, v in sorted(target_names.items())},
    }


def run_split(force: bool = False, raw_dir: str = RAW_DIR, split_dir: str = SPLIT_DIR):
    class_taxonomy, target_names = load_class_config(DEFAULT_CLASS_CONFIG_PATH)

    ensure_cache_is_safe(
        split_dir,
        _split_params(class_taxonomy, target_names),
        force=force,
        wipe_subdirs=["images", "labels"],
        manifest_filename=MANIFEST_FILENAME,
    )

    all_parents = collect_parent_images(Path(raw_dir))
    if not all_parents:
        print(f"❌ Aucune image brute trouvée dans {raw_dir}.")
        return

    unique_parents = list({p["parent_id"]: p for p in all_parents}.values())

    random.seed(SEED)
    random.shuffle(unique_parents)

    n_total = len(unique_parents)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    split_map = {}
    for idx, item in enumerate(unique_parents):
        if idx < n_train:
            split_map[item["parent_id"]] = "train"
        elif idx < n_train + n_val:
            split_map[item["parent_id"]] = "val"
        else:
            split_map[item["parent_id"]] = "test"

    print(f"--- 📦 SPLIT ÉTANCHE ({n_total} images parentes) ---")
    print(f"  • Train ({int(TRAIN_RATIO*100)}%): {n_train}")
    print(f"  • Val   ({int(VAL_RATIO*100)}%): {n_val}")
    print(f"  • Test  ({int(TEST_RATIO*100)}%): {n_total - n_train - n_val}\n")

    processed_count = 0
    skipped_count = 0
    # Cache par lot : plusieurs images parentes partagent le même data.yaml local,
    # pas la peine de le relire à chaque fois.
    local_names_by_batch: Dict[str, Dict[int, str]] = {}

    for item in unique_parents:
        parent_id = item["parent_id"]
        split = split_map[parent_id]
        img_src = Path(item["img_path"])
        batch_name = item["batch"]

        img_dst_dir = Path(split_dir) / "images" / split
        lab_dst_dir = Path(split_dir) / "labels" / split
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lab_dst_dir.mkdir(parents=True, exist_ok=True)

        img_dst = img_dst_dir / f"{parent_id}{img_src.suffix.lower()}"
        lab_dst = lab_dst_dir / f"{parent_id}.txt"

        if img_dst.exists():
            skipped_count += 1
            continue

        shutil.copy2(img_src, img_dst)

        if batch_name not in local_names_by_batch:
            # Le data.yaml d'un lot vit à sa racine (ex: "SL 11-16/data.yaml"), donc
            # au premier niveau sous raw_dir - reconstruit à partir de raw_dir + batch_name.
            local_yaml = Path(raw_dir) / batch_name / "data.yaml"
            if not local_yaml.exists():
                raise RuntimeError(
                    f"data.yaml introuvable pour le lot '{batch_name}' ({local_yaml}). "
                    f"Chaque lot doit déclarer son propre data.yaml local (local_id -> nom)."
                )
            local_names_by_batch[batch_name] = load_batch_local_names(local_yaml)

        out_lines = _translate_and_filter_label(
            Path(item["label_path"]), local_names_by_batch[batch_name], class_taxonomy, batch_name
        )
        with open(lab_dst, "w", encoding="utf-8") as f:
            f.writelines(out_lines)

        processed_count += 1

    print(f"[SUCCÈS] Split terminé.")
    print(f"  • Nouvelles images copiées : {processed_count}")
    print(f"  • Images ignorées (déjà là) : {skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Étape 2 : split train/val/test PixelOdyssey")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Si la config de split a changé depuis le dernier run (ratio, seed, "
             "taxonomie de classes...), supprime 2_split_dataset/images et /labels "
             "et refait le split from scratch au lieu de s'arrêter avec une erreur.",
    )
    args = parser.parse_args()
    try:
        run_split(force=args.force)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
