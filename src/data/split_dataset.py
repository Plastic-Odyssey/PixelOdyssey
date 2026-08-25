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
split - LA TRADUCTION DE CLASSES COMPLÈTE, en une seule passe (modèle par nom - voir class_config.py) :
  1. Lit le data.yaml LOCAL du lot d'origine de l'image (local_id -> nom).
  2. Normalise ce nom et le cherche dans `class_taxonomy` (config/data_config.yaml).
  3. Écrit directement l'ID de SUPER-CLASSE CIBLE final (celui de `names`), ou
     rien si la classe est exclue.
À partir de 2_split_dataset, les labels sont donc déjà dans leur espace de
classes FINAL - les étapes suivantes (augmentation, slicing) n'ont plus
aucune notion de classe à gérer, seulement de la géométrie/des fichiers.
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
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

# Manifeste parent_id -> {batch, split, chemins bruts} écrit à chaque run de
# run_split() (voir la fin de la fonction). Sert de source de vérité fiable
# pour tout outil en aval qui a besoin de retrouver l'image/le label BRUTS
# d'origine d'un parent_id présent dans 2_split_dataset (ex: l'outil de
# relecture assistée par modèle, src/review/) - reconstruire ce chemin en
# "dérivant" le parent_id (remplacer les espaces/séparateurs par des
# underscores) est ambigu dès qu'un nom de fichier contient déjà un
# underscore, donc pas fiable pour écrire quoi que ce soit en retour dans
# 1_annotated_dataset.
PARENT_MANIFEST_FILENAME = ".parent_manifest.json"


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



# Incrémenter quand la LOGIQUE de split elle-même change (ex: passage d'un
# shuffle global à une stratification par lot, le 21/08/2026 - v2 ; passage de
# int() [toujours vers le bas, test hérite du reliquat] à round() pour val/test
# avec reliquat absorbé par train, le 22/08/2026 - v3, voir commentaire dans la
# boucle plus bas) de façon à produire une répartition différente pour les
# MÊMES seed/ratios/taxonomie. Sans ce marqueur, _split_params() aurait
# fingerprinté IDENTIQUEMENT avant et après un tel changement (aucun des
# paramètres ci-dessous n'a changé de valeur), et ensure_cache_is_safe() aurait
# silencieusement réutilisé un 2_split_dataset généré avec l'ANCIEN algorithme
# sans jamais le signaler.
SPLIT_LOGIC_VERSION = 3


def _split_params(class_taxonomy, target_names: Dict[int, str]) -> Dict:
    """Paramètres qui influencent la sortie de cette étape (pour le garde-fou de cache)."""
    return {
        "split_logic_version": SPLIT_LOGIC_VERSION,
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

    # Stratification PAR LOT (21/08/2026) - remplace l'ancien shuffle global.
    # Pourquoi : un shuffle global sur les 540 images parentes traite chaque
    # image comme interchangeable, alors que certains lots (ex: les A LEG,
    # ~8 images chacun) sont la SEULE source de certaines classes réelles
    # (ex: Sceau). Avec un shuffle global, le pur hasard peut renvoyer un
    # petit lot entier vers train et laisser val/test sans aucun exemple de
    # ses classes exclusives - ou l'inverse. Découper 70/20/10 SÉPARÉMENT à
    # l'intérieur de chaque lot, puis fusionner les résultats, garantit que
    # CHAQUE lot (et donc ses classes propres) est représenté dans les trois
    # splits dans les proportions voulues, peu importe la chance du tirage.
    parents_by_batch: Dict[str, List[Dict]] = defaultdict(list)
    for item in unique_parents:
        parents_by_batch[item["batch"]].append(item)

    random.seed(SEED)
    split_map: Dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    small_batch_warnings: List[str] = []
    for batch_name in sorted(parents_by_batch):
        batch_items = parents_by_batch[batch_name]
        random.shuffle(batch_items)
        n_b = len(batch_items)
        # Arrondi (22/08/2026, v3) : val et test sont arrondis chacun au plus proche
        # de leur part théorique (round(), pas int()) ; train absorbe le reliquat.
        # AVANT : train et val étaient tous les deux tronqués vers le BAS (int()),
        # et test récupérait TOUT le reliquat des deux troncatures - un biais
        # SYSTÉMATIQUE qui gonflait test au-delà de ses 10% nominaux, surtout sur
        # les petits lots (ex: un lot de 4 images visant train=2.8/val=0.8/test=0.4
        # donnait train=2/val=0/test=2, soit 5x la part théorique de test). Ce
        # biais a été découvert en aval : sur le run du 22/08/2026, le split test
        # avait MOINS d'images parentes que val (66 vs 102) mais PLUS de tuiles et
        # d'instances après slicing (836/1375 vs 752/909), parce que les petits
        # lots à résolution variable (A LEG, SL - qui produisent beaucoup de
        # tuiles par image via la fenêtre glissante, contrairement aux imagettes
        # SB) étaient surreprésentés dans test. Faire absorber le reliquat par
        # TRAIN plutôt que TEST est le bon choix : train ne sert jamais à mesurer
        # une performance, une distorsion d'arrondi y est donc sans conséquence,
        # alors qu'elle biaisait directement l'évaluation quand elle tombait sur
        # test.
        n_val_b = round(n_b * VAL_RATIO)
        n_test_b = round(n_b * TEST_RATIO)
        n_train_b = n_b - n_val_b - n_test_b

        for idx, item in enumerate(batch_items):
            if idx < n_train_b:
                s = "train"
            elif idx < n_train_b + n_val_b:
                s = "val"
            else:
                s = "test"
            split_map[item["parent_id"]] = s
            counts[s] += 1

        # Un lot trop petit pour peupler ses trois portions proportionnellement
        # (ex: un lot de 8 images -> 0 en val et/ou test avec ces ratios) n'est
        # pas une erreur - juste un fait à ne pas laisser passer en silence :
        # ses classes exclusives risquent alors de ne jamais apparaître dans
        # un des splits malgré la stratification.
        if n_b > 0 and (n_train_b == 0 or n_val_b == 0 or n_test_b == 0):
            small_batch_warnings.append(
                f"  ⚠️  [{batch_name}] lot de {n_b} image(s) trop petit pour peupler les 3 "
                f"splits proportionnellement (train={n_train_b}, val={n_val_b}, test={n_test_b})."
            )

    n_total = len(unique_parents)
    print(f"--- 📦 SPLIT ÉTANCHE, STRATIFIÉ PAR LOT ({n_total} images parentes, {len(parents_by_batch)} lots) ---")
    print(f"  • Train: {counts['train']}")
    print(f"  • Val  : {counts['val']}")
    print(f"  • Test : {counts['test']}")
    if small_batch_warnings:
        print()
        for w in small_batch_warnings:
            print(w)
    print()

    processed_count = 0
    skipped_count = 0
    # Cache par lot : plusieurs images parentes partagent le même data.yaml local,
    # pas la peine de le relire à chaque fois.
    local_names_by_batch: Dict[str, Dict[int, str]] = {}
    # Reconstruit à CHAQUE run (pas seulement pour les parents nouvellement copiés) -
    # voir PARENT_MANIFEST_FILENAME ci-dessus.
    parent_manifest: Dict[str, Dict] = {}

    for item in unique_parents:
        parent_id = item["parent_id"]
        split = split_map[parent_id]
        img_src = Path(item["img_path"])
        batch_name = item["batch"]

        parent_manifest[parent_id] = {
            "batch": batch_name,
            "split": split,
            "raw_img_path": str(img_src),
            "raw_label_path": item["label_path"],
        }

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

    with open(Path(split_dir) / PARENT_MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(parent_manifest, f, indent=2, ensure_ascii=False)

    print(f"[SUCCÈS] Split terminé.")
    print(f"  • Nouvelles images copiées : {processed_count}")
    print(f"  • Images ignorées (déjà là) : {skipped_count}")
    print(f"  • Manifeste parent -> brut : {PARENT_MANIFEST_FILENAME} ({len(parent_manifest)} entrées)")


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
