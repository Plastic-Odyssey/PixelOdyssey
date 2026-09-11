#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Étape 2 : Split Train / Val / Test.

Décide, pour chaque image PARENTE brute (jamais au niveau de la tuile), dans
quel split elle tombe, traduit ses labels vers la taxonomie de classes cible,
et écrit le résultat dans 2_split_dataset. Fait le pont entre les données
brutes par lot (1_annotated_dataset) et la suite du pipeline (augment_dataset.py
puis slice_dataset.py), qui n'ont ensuite plus aucune notion de classe à gérer.

Paramètres (CLI) :
    --force         Si la config de split a changé depuis le dernier run (ratio,
                     seed, taxonomie...), régénère 2_split_dataset from scratch
                     au lieu de s'arrêter en erreur.
    --config PATH   Référentiel de classes à utiliser (défaut : data_config.yaml).
                     Pour une taxonomie alternative, fournir aussi --split-dir.
    --split-dir DIR Dossier de sortie (défaut : 2_split_dataset). À changer
                     systématiquement avec --config ou --site.
    --balance-by    "images" (défaut) ou "items" - voir plus bas.
    --site CODES    Filtre par code(s) de site (ex: SL,A). Défaut : tous les sites.

Exemples :
    python src/data/split_dataset.py
    python src/data/split_dataset.py --site SL --split-dir 2_split_dataset_SL --config config/data_config_SL.yaml

Arbitrages notables :

Le split est décidé AVANT l'augmentation : trancher sur des images encore
non-augmentées évite qu'une image et sa quasi-copie augmentée se retrouvent
dans des splits différents (fuite de données).

La stratification se fait PAR LOT plutôt que sur un shuffle global : certains
lots sont la seule source de certaines classes, un tirage global pourrait donc
priver val/test de leurs exemples. Chaque lot est découpé 70/20/10 séparément
puis les résultats sont fusionnés.

Le split de chaque image parente est dérivé d'un hash stable de son propre
parent_id (voir `_stable_unit_interval`), jamais de sa position dans un shuffle
de la liste du lot. Conséquence recherchée : ajouter de nouvelles images à un
lot déjà existant (site enrichi progressivement) ne change jamais le split
d'une image déjà présente - seules les nouvelles images sont réparties. Le
mode `--balance-by items` (équilibrage par nombre d'instances annotées plutôt
que par nombre d'images, utile quand la densité de déchets varie beaucoup
d'une image à l'autre) n'a pas cette même garantie : son algorithme glouton
par déficit reste sensible à l'ordre et au volume total du lot.

Si `config/frozen_test_parents.json` existe (voir `freeze_benchmark_test.py`),
les parent_id qu'il liste sont épinglés en test définitivement, jamais
recalculés. Le reste du lot n'alimente alors plus que train/val - le test set
reste rigoureusement identique d'un run à l'autre, condition nécessaire pour
comparer deux entraînements sur un banc d'évaluation stable.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    EXCLUDE,
    load_batch_local_names,
    load_class_config,
    resolve_class_name,
)
from src.data.pipeline_utils import ensure_cache_is_safe
from src.data.raw_dataset import collect_parent_images, site_of_batch

from src.paths_config import PROCESSED_DATASET_DIR as BASE_DIR  # racine centralisee (08/09/2026), voir src/paths_config.py
RAW_DIR = os.path.join(BASE_DIR, "1_annotated_dataset")
SPLIT_DIR = os.path.join(BASE_DIR, "2_split_dataset")

# Dataset encore petit et en phase d'expérimentation de méthode : on garde un
# vrai test set pour une évaluation finale non biaisée. À revoir si
# l'entraînement venait à manquer cruellement de données.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
SEED = 42

MANIFEST_FILENAME = ".split_manifest.json"

# Banc de test figé (optionnel) : parent_id gelés en test une bonne fois pour
# toutes, écrits par freeze_benchmark_test.py. Voir _load_frozen_test_ids().
FROZEN_TEST_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "frozen_test_parents.json"

# Manifeste parent_id -> {batch, split, chemins bruts}, source de vérité pour
# tout outil en aval (ex: src/review/) qui a besoin de retrouver l'image/le
# label BRUTS d'un parent_id présent dans 2_split_dataset. Reconstruire ce
# chemin en "dérivant" le parent_id serait ambigu dès qu'un nom de fichier
# contient déjà un underscore - pas fiable.
PARENT_MANIFEST_FILENAME = ".parent_manifest.json"


def _count_resolved_instances(label_path: Path, local_names: Dict[int, str], class_taxonomy) -> int:
    """Comme `_translate_and_filter_label` mais ne retourne qu'un compte de lignes
    résolues (non exclues) - utilisé par `--balance-by items` pour peser chaque
    image parente par son nombre réel d'instances entraînables plutôt que par sa
    simple présence. Volontairement tolérant (jamais de RuntimeError ici) : une
    incohérence réelle est de toute façon bloquée par raw_dataset_checker.py en
    amont ; ce compte ne sert qu'à équilibrer les splits."""
    if not label_path.exists():
        return 0
    count = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            local_id = int(parts[0])
            name = local_names.get(local_id)
            if name is None:
                continue
            resolved = resolve_class_name(name, class_taxonomy)
            if resolved is None or resolved == EXCLUDE:
                continue
            count += 1
    return count


def _greedy_item_balanced_assignment(
    items: List[Dict], counts: Dict[str, int], val_share: float, test_share: float, seed_key: str,
) -> Dict[str, str]:
    """Répartit `items` (reliquat non figé d'UN lot) entre train/val/test en
    visant des proportions en NOMBRE D'ITEMS plutôt qu'en nombre d'images -
    utile quand deux images ont des densités de déchets très différentes.

    Pas un problème de partition résolu de façon optimale (NP-difficile en
    général) : un algorithme glouton par déficit suffit et reste simple à
    déboguer. Les items sont triés par poids décroissant avant la boucle
    (heuristique "Longest Processing Time first") plutôt que dans un ordre
    purement aléatoire : sur un lot très hétérogène (quelques images "denses"
    parmi beaucoup d'images "éparses"), un ordre aléatoire peut placer une
    image dense au pire moment et priver un petit split cible (ex: test) de
    presque tout son contenu. Trier par poids décroissant place les grosses
    images en premier, quand tous les déficits sont encore larges, et laisse
    les petites affiner le reste ensuite - convergence bien plus fidèle aux
    proportions cibles. Le mélange aléatoire (déterministe via `seed_key`) ne
    sert plus qu'à départager les ex-æquo.

    Entrée : items du lot, leurs comptes d'items, part de val/test parmi ce
    reliquat (train absorbe le reste), clé de seed.
    Sortie : dict parent_id -> split.
    """
    rng = random.Random(seed_key)
    shuffled = list(items)
    rng.shuffle(shuffled)
    # Tri stable par nb d'items décroissant (LPT) : préserve le mélange aléatoire
    # ci-dessus comme tie-break entre items de même poids.
    shuffled.sort(key=lambda it: counts[it["parent_id"]], reverse=True)

    total_items = sum(counts[it["parent_id"]] for it in shuffled)
    targets = {
        "train": total_items * (1.0 - val_share - test_share),
        "val": total_items * val_share,
        "test": total_items * test_share,
    }
    assigned: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    result: Dict[str, str] = {}
    # Un split à cible nulle (ex: test_share=0.0 quand le banc de test est figé)
    # ne doit jamais recevoir d'item, même par un effet de bord d'arrondi en fin
    # de lot - sans quoi une image pourrait fuiter dans le banc de test figé.
    eligible = [s for s in targets if targets[s] > 0] or list(targets)
    for it in shuffled:
        deficits = {s: targets[s] - assigned[s] for s in eligible}
        chosen = max(deficits, key=deficits.get)
        result[it["parent_id"]] = chosen
        assigned[chosen] += counts[it["parent_id"]]
    return result


def _translate_and_filter_label(
    src_label_path: Path,
    local_names: Dict[int, str],
    class_taxonomy,
    batch_name: str,
) -> List[str]:
    """Lit un label brut et traduit chaque ligne vers l'ID de super-classe cible.

    Entrée : `src_label_path` (label brut), `local_names` (data.yaml local du lot
    d'origine, local_id -> nom), `class_taxonomy`, `batch_name` (messages d'erreur).
    Sortie : liste des lignes traduites (peut être vide - image sans objet).
    Les coordonnées ne sont jamais modifiées, seul l'ID de classe change (ou la
    ligne disparaît si la classe est exclue). Lève RuntimeError si un ID ou un
    nom de classe n'est pas résolvable.
    """
    if not src_label_path.exists():
        return []  # Pas d'objets sur cette image (ou "sans déchet" confirmé) - valide en YOLO.

    out_lines = []
    with open(src_label_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            local_id = int(parts[0])
            name = local_names.get(local_id)
            if name is None:
                # ID non déclaré dans le data.yaml de CE lot - raw_dataset_checker.py
                # (étape 0) est censé l'attraper avant ; garde-fou de secours ici.
                raise RuntimeError(
                    f"{src_label_path} (L{lineno}) : ID de classe {local_id} absent du "
                    f"data.yaml du lot '{batch_name}'. Corrige l'annotation ou le data.yaml "
                    f"de ce lot avant de relancer."
                )

            resolved = resolve_class_name(name, class_taxonomy)
            if resolved is None:
                # Nom utilisé mais absent de class_taxonomy/class_aliases - doit
                # bloquer plutôt qu'être ignoré en silence (garde-fou de secours,
                # raw_dataset_checker.py doit déjà avoir bloqué avant).
                raise RuntimeError(
                    f"{src_label_path} (L{lineno}) : classe '{name}' (lot '{batch_name}') "
                    f"n'est dans aucune entrée de class_taxonomy/class_aliases "
                    f"(config/data_config.yaml). Décide de son sort avant de relancer."
                )
            if resolved == EXCLUDE:
                continue

            out_lines.append(f"{resolved} {' '.join(parts[1:])}\n")

    return out_lines


def _stable_unit_interval(key: str) -> float:
    """Valeur déterministe dans [0, 1), dérivée uniquement de `key` (SHA-256 -
    jamais `hash()` natif de Python, randomisé par process via PYTHONHASHSEED,
    donc non reproductible d'un run à l'autre).

    Assigne un split à chaque parent_id indépendamment du reste de son lot :
    `key` inclut (seed, nom de lot, parent_id), donc la valeur retournée pour un
    parent_id donné ne change jamais, que d'autres images soient ajoutées ou non
    à son lot - propriété nécessaire pour qu'enrichir un lot existant ne fasse
    jamais changer le split d'une image déjà répartie.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:15], 16) / float(16 ** 15)


# SPLIT_LOGIC_VERSION : à incrémenter quand la LOGIQUE de split elle-même change
# (algorithme de répartition, stratification, arrondi...) de façon à produire
# une répartition différente pour les mêmes seed/ratios/taxonomie. Sans ce
# marqueur, _split_params() fingerprinterait identiquement avant et après un
# tel changement, et ensure_cache_is_safe() réutiliserait silencieusement un
# 2_split_dataset généré avec l'ancien algorithme.
#
# v6 : mode "images" (défaut) reconstruit sur une assignation par hash stable
# de chaque parent_id plutôt qu'un shuffle déterministe + découpage par
# position - corrige une instabilité du split sous croissance d'un lot
# existant. Le mode "items" n'est pas concerné (toujours basé sur l'algorithme
# glouton par déficit, sensible à l'ordre/au volume total du lot).
SPLIT_LOGIC_VERSION = 6


def _load_frozen_test_ids() -> List[str]:
    """Charge la liste de parent_id épinglés en test (voir FROZEN_TEST_PATH).

    Sortie : liste triée (fingerprint reproductible), vide si le fichier
    n'existe pas (cas par défaut - aucun banc figé encore créé)."""
    if not FROZEN_TEST_PATH.exists():
        return []
    with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data.get("parent_ids", []))


def _split_params(
    class_taxonomy, target_names: Dict[int, str], frozen_test_ids: List[str], balance_by: str,
    site_filter: Optional[List[str]] = None,
) -> Dict:
    """Paramètres qui influencent la sortie de cette étape (garde-fou de cache)."""
    return {
        "split_logic_version": SPLIT_LOGIC_VERSION,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "balance_by": balance_by,
        "class_taxonomy": {k: v for k, v in sorted(class_taxonomy.items())},
        "target_names": {str(k): v for k, v in sorted(target_names.items())},
        # Inclus tel quel (pas un hash) : un banc figé qui change de contenu doit
        # invalider le cache comme tout autre paramètre de cette étape.
        "frozen_test_ids": frozen_test_ids,
        # None (tous les sites) sérialisé distinctement d'une liste vide, pour
        # permettre un entraînement dédié à un seul site sans toucher au dataset complet.
        "site_filter": sorted(site_filter) if site_filter else None,
    }


def run_split(
    force: bool = False,
    raw_dir: str = RAW_DIR,
    split_dir: str = SPLIT_DIR,
    run_confirmation=None,
    class_config_path: Union[str, Path] = DEFAULT_CLASS_CONFIG_PATH,
    balance_by: str = "images",
    site_filter: Optional[List[str]] = None,
):
    """Voir la docstring du module.

    `class_config_path` : référentiel de classes à utiliser - permet de faire
    tourner cette étape sur une taxonomie alternative sans toucher au fichier
    principal, à condition de fournir aussi un `split_dir` distinct (aucun
    garde-fou de ce script n'empêche d'écraser le dataset principal sinon).

    `balance_by` ("images" ou "items") : à l'intérieur de chaque lot
    (stratification par lot inchangée dans les deux modes), "images" répartit
    au prorata du nombre d'images parentes ; "items" au prorata du nombre
    d'instances annotées résolues, via `_greedy_item_balanced_assignment`.

    `site_filter` : liste de codes de site à retenir (voir
    raw_dataset.site_of_batch) ; None (défaut) = tous les sites. Ne filtre que
    les images parentes traitées ici - la stratification, les ratios et le
    banc de test figé restent la même logique, appliquée à ce sous-ensemble.
    Comme pour `class_config_path`, fournir un `split_dir` distinct est
    indispensable pour ne pas écraser le dataset complet (voir data_pipeline.py
    --site/--suffix pour le garde-fou explicite au niveau de l'orchestrateur).
    """
    if balance_by not in ("images", "items"):
        raise RuntimeError(f"--balance-by doit valoir 'images' ou 'items' (reçu '{balance_by}').")

    class_taxonomy, target_names = load_class_config(class_config_path)
    frozen_test_ids = _load_frozen_test_ids()

    ensure_cache_is_safe(
        split_dir,
        _split_params(class_taxonomy, target_names, frozen_test_ids, balance_by, site_filter),
        force=force,
        wipe_subdirs=["images", "labels"],
        manifest_filename=MANIFEST_FILENAME,
        run_confirmation=run_confirmation,
    )

    all_parents = collect_parent_images(Path(raw_dir))
    if not all_parents:
        print(f"❌ Aucune image brute trouvée dans {raw_dir}.")
        return

    unique_parents_all = list({p["parent_id"]: p for p in all_parents}.values())

    if frozen_test_ids:
        # Vérifié contre le dataset COMPLET (pas le sous-ensemble filtré par
        # site) : un parent_id gelé appartenant à un site exclu par --site
        # n'est pas "manquant", juste hors périmètre de ce run.
        found_ids = {p["parent_id"] for p in unique_parents_all}
        missing = [pid for pid in frozen_test_ids if pid not in found_ids]
        if missing:
            print(f"⚠️  {len(missing)} parent_id du banc de test figé sont introuvables dans "
                  f"{raw_dir} (déplacés/supprimés depuis le gel ?) : {', '.join(missing[:5])}"
                  f"{'...' if len(missing) > 5 else ''}")

    if site_filter:
        known_sites = sorted({site_of_batch(p["batch"]) for p in unique_parents_all})
        unknown = [s for s in site_filter if s not in known_sites]
        if unknown:
            raise RuntimeError(
                f"--site {unknown} : aucun lot ne correspond à ce (ces) code(s) de site parmi "
                f"{raw_dir}. Sites détectés dans la donnée brute : {', '.join(known_sites)}."
            )
        unique_parents = [p for p in unique_parents_all if site_of_batch(p["batch"]) in site_filter]
        n_frozen_in_scope = sum(
            1 for p in unique_parents_all
            if p["parent_id"] in frozen_test_ids and site_of_batch(p["batch"]) in site_filter
        )
        print(f"🔎 Filtre --site {','.join(site_filter)} : {len(unique_parents)}/{len(unique_parents_all)} "
              f"image(s) parente(s) retenue(s) (sites détectés au total : {', '.join(known_sites)})"
              + (f" - dont {n_frozen_in_scope} du banc de test figé." if frozen_test_ids else "."))
    else:
        unique_parents = unique_parents_all

    # Stratification par lot : un shuffle global traiterait chaque image comme
    # interchangeable, alors que certains lots sont la seule source de
    # certaines classes réelles. Découper 70/20/10 séparément dans chaque lot
    # garantit que chaque lot (et ses classes propres) est représenté dans les
    # trois splits, peu importe la chance du tirage.
    parents_by_batch: Dict[str, List[Dict]] = defaultdict(list)
    for item in unique_parents:
        parents_by_batch[item["batch"]].append(item)

    # Chargé tôt (avant l'allocation) : nécessaire pour peser chaque image par
    # son nombre d'items résolus, que ce soit pour décider l'allocation
    # (--balance-by items) ou simplement pour l'afficher à titre informatif en
    # mode "images" par défaut.
    local_names_by_batch: Dict[str, Dict[int, str]] = {}
    item_count_by_parent: Dict[str, int] = {}
    for item in unique_parents:
        batch_name = item["batch"]
        if batch_name not in local_names_by_batch:
            local_yaml = Path(raw_dir) / batch_name / "data.yaml"
            if not local_yaml.exists():
                raise RuntimeError(
                    f"data.yaml introuvable pour le lot '{batch_name}' ({local_yaml}). "
                    f"Chaque lot doit déclarer son propre data.yaml local (local_id -> nom)."
                )
            local_names_by_batch[batch_name] = load_batch_local_names(local_yaml)
        item_count_by_parent[item["parent_id"]] = _count_resolved_instances(
            Path(item["label_path"]), local_names_by_batch[batch_name], class_taxonomy
        )

    frozen_test_id_set = set(frozen_test_ids)
    split_map: Dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0, "test_frozen": 0}
    item_counts = {"train": 0, "val": 0, "test": 0, "test_frozen": 0}
    small_batch_warnings: List[str] = []
    for batch_name in sorted(parents_by_batch):
        batch_items = parents_by_batch[batch_name]

        # Le banc de test figé (s'il existe) est retiré du tirage AVANT toute
        # stratification : ces images sont en test de façon permanente, quel
        # que soit le lot, et ne participent plus au calcul train/val/test.
        frozen_items = [it for it in batch_items if it["parent_id"] in frozen_test_id_set]
        remaining_items = [it for it in batch_items if it["parent_id"] not in frozen_test_id_set]
        for it in frozen_items:
            split_map[it["parent_id"]] = "test"
            counts["test"] += 1
            counts["test_frozen"] += 1
            item_counts["test"] += item_count_by_parent[it["parent_id"]]
            item_counts["test_frozen"] += item_count_by_parent[it["parent_id"]]

        # RNG indépendant par lot (dérivé de SEED + nom du lot), plutôt qu'un
        # random.seed(SEED) global consommé au fil de la boucle : sinon le
        # tirage d'un lot dépendrait de l'ordre et de la taille de tous les
        # lots qui le précèdent - ajouter une nouvelle source de données
        # décalerait alors silencieusement la répartition de lots inchangés.
        random.Random(f"{SEED}:{batch_name}").shuffle(remaining_items)
        n_b = len(remaining_items)

        if frozen_test_ids:
            # Test est déjà couvert par le banc figé (globalement) : le
            # reliquat de ce lot ne se répartit plus qu'entre train et val,
            # aux ratios renormalisés pour sommer à 1.
            val_share = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
            test_share = 0.0
        else:
            val_share = VAL_RATIO
            test_share = TEST_RATIO

        if balance_by == "items":
            total_items_b = sum(item_count_by_parent[it["parent_id"]] for it in remaining_items)
            if total_items_b == 0:
                # Rien à équilibrer (lot restant 100% "sans déchet") - retombe
                # sur le comportement "images" pour ce lot plutôt que de diviser par 0.
                n_val_b = round(n_b * val_share)
                n_test_b = round(n_b * test_share)
                n_train_b = n_b - n_val_b - n_test_b
                for idx, item in enumerate(remaining_items):
                    s = "train" if idx < n_train_b else ("val" if idx < n_train_b + n_val_b else "test")
                    split_map[item["parent_id"]] = s
                    counts[s] += 1
            else:
                batch_split_map = _greedy_item_balanced_assignment(
                    remaining_items, item_count_by_parent, val_share, test_share, f"{SEED}:{batch_name}"
                )
                for item in remaining_items:
                    s = batch_split_map[item["parent_id"]]
                    split_map[item["parent_id"]] = s
                    counts[s] += 1
            n_train_b = sum(1 for it in remaining_items if split_map[it["parent_id"]] == "train")
            n_val_b = sum(1 for it in remaining_items if split_map[it["parent_id"]] == "val")
            n_test_b = sum(1 for it in remaining_items if split_map[it["parent_id"]] == "test")
        else:
            # Assignation par hash stable de chaque parent_id : chaque image
            # reçoit un split déterminé uniquement par (SEED, nom de lot, son
            # propre parent_id), jamais par le nombre ou l'ordre des autres
            # images du lot. `train_share` absorbe le reliquat - train ne sert
            # jamais à mesurer une performance, une distorsion d'arrondi y est
            # donc sans conséquence, contrairement à val/test.
            train_share = 1.0 - val_share - test_share
            for item in remaining_items:
                u = _stable_unit_interval(f"{SEED}:{batch_name}:{item['parent_id']}")
                if u < train_share:
                    s = "train"
                elif u < train_share + val_share:
                    s = "val"
                else:
                    s = "test"
                split_map[item["parent_id"]] = s
                counts[s] += 1
            n_train_b = sum(1 for it in remaining_items if split_map[it["parent_id"]] == "train")
            n_val_b = sum(1 for it in remaining_items if split_map[it["parent_id"]] == "val")
            n_test_b = sum(1 for it in remaining_items if split_map[it["parent_id"]] == "test")

        for it in remaining_items:
            item_counts[split_map[it["parent_id"]]] += item_count_by_parent[it["parent_id"]]

        # Un lot trop petit pour peupler ses portions proportionnellement
        # n'est pas une erreur, juste un fait à ne pas laisser passer en
        # silence : ses classes exclusives risquent de ne jamais apparaître
        # dans un des splits. Avec banc figé, test == 0 par lot est attendu et
        # ne doit pas déclencher ce warning.
        empty_portions = [n_train_b == 0, n_val_b == 0] if frozen_test_ids else [n_train_b == 0, n_val_b == 0, n_test_b == 0]
        if n_b > 0 and any(empty_portions):
            small_batch_warnings.append(
                f"  ⚠️  [{batch_name}] lot de {n_b} image(s) restante(s) trop petit pour peupler ses "
                f"portions proportionnellement (train={n_train_b}, val={n_val_b}, test={n_test_b})."
            )

    n_total = len(unique_parents)
    print(f"--- 📦 SPLIT ÉTANCHE, STRATIFIÉ PAR LOT ({n_total} images parentes, {len(parents_by_batch)} lots, "
          f"équilibrage par {'ITEMS' if balance_by == 'items' else 'images'}) ---")
    if frozen_test_ids:
        print(f"  • Banc de test figé : {counts['test_frozen']}/{len(frozen_test_ids)} parent_id retrouvés "
              f"(voir config/frozen_test_parents.json)")
    total_items = sum(item_counts[s] for s in ("train", "val", "test"))
    for s in ("train", "val", "test"):
        pct_img = round(100.0 * counts[s] / n_total, 1) if n_total else 0.0
        pct_item = round(100.0 * item_counts[s] / total_items, 1) if total_items else 0.0
        print(f"  • {s.capitalize():<6}: {counts[s]:>4} image(s) ({pct_img:>5.1f}%)   "
              f"{item_counts[s]:>6} item(s) ({pct_item:>5.1f}%)")
    if small_batch_warnings:
        print()
        for w in small_batch_warnings:
            print(w)
    print()

    processed_count = 0
    skipped_count = 0
    # `local_names_by_batch` déjà chargé plus haut - réutilisé tel quel, pas la
    # peine de relire chaque data.yaml de lot une 2e fois.
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
            # Le data.yaml d'un lot vit à sa racine (ex: "SL 11-16/data.yaml"),
            # reconstruit à partir de raw_dir + batch_name.
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
    parser.add_argument(
        "--config", default=str(DEFAULT_CLASS_CONFIG_PATH),
        help="Référentiel de classes à utiliser (défaut : config/data_config.yaml). "
             "Pour une taxonomie alternative, fournis AUSSI --split-dir pour ne "
             "jamais écraser le dataset principal.",
    )
    parser.add_argument(
        "--split-dir", default=SPLIT_DIR,
        help="Dossier de sortie de cette étape (défaut : 2_split_dataset). À changer "
             "systématiquement quand --config pointe vers une taxonomie alternative.",
    )
    parser.add_argument(
        "--balance-by", choices=["images", "items"], default="images",
        help="Critère d'équilibrage train/val/test à l'intérieur de chaque lot. "
             "'images' (défaut) : assignation stable par parent_id, insensible à "
             "l'ajout d'images dans un lot existant. 'items' : vise la proportion "
             "en nombre d'instances annotées résolues, utile quand la densité de "
             "déchets varie beaucoup d'une image à l'autre, mais sans la même "
             "garantie de stabilité sous croissance d'un lot - à éviter si tu "
             "prévois d'ajouter des images progressivement. Le nombre d'items par "
             "split est toujours affiché, même en mode 'images', à titre informatif.",
    )
    parser.add_argument(
        "--site", default=None,
        help="Filtre par code de site (ex: SL, SB, A), un ou plusieurs séparés par "
             "virgule (ex: SL,A). Défaut : aucun filtre, tous les sites. Fournis "
             "AUSSI --split-dir pour ne jamais écraser le dataset complet.",
    )
    args = parser.parse_args()
    site_filter = [s.strip() for s in args.site.split(",")] if args.site else None
    try:
        run_split(
            force=args.force, split_dir=args.split_dir, class_config_path=args.config,
            balance_by=args.balance_by, site_filter=site_filter,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
