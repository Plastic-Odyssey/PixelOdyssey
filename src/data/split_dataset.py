#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Étape 2 : Split Train / Val / Test.

Lit les images parentes complètes (non découpées) et leurs labels depuis
`1_annotated_dataset`, décide de façon étanche (au niveau de l'image parente,
pas de la tuile) dans quel split chacune tombe, traduit chaque ligne de label
vers l'ID de super-classe cible (via `class_taxonomy`/`class_aliases` de
config/data_config.yaml, ou l'exclut), et écrit le résultat dans
`2_split_dataset/images/{train,val,test}` + `labels/{train,val,test}`.

Cette étape doit avoir lieu AVANT l'augmentation (étape 3) : décider le split
sur des images encore non-augmentées évite qu'une image originale et sa
quasi-copie augmentée (copy-paste, dédoublement) finissent dans des splits
différents, ce qui serait une fuite de données.

À partir de 2_split_dataset, les labels sont déjà dans leur espace de classes
final : les étapes suivantes (augmentation, slicing) n'ont plus aucune notion
de classe à gérer, seulement de la géométrie/des fichiers. Cette étape ne
découpe pas les images en tuiles (étape 4) et n'altère pas la géométrie des
polygones - seules les lignes de label sont traduites/filtrées.

Si `config/frozen_test_parents.json` existe (voir `freeze_benchmark_test.py`),
les parent_id qu'il liste sont ÉPINGLÉS en test, définitivement - jamais
recalculés, quel que soit le lot ou la stratification. Le reste de leur lot
(et tout nouveau lot) n'alimente alors plus que train/val - test ne grossit
plus au fil des ajouts de données, exactement les mêmes images d'un run à
l'autre. Sans ce fichier (défaut), comportement inchangé : 70/20/10
stratifié par lot, comme avant.

Entrée : `1_annotated_dataset` (images + labels bruts par lot, chaque lot avec
son propre data.yaml local), config/data_config.yaml (taxonomie cible), et
optionnellement config/frozen_test_parents.json (banc de test figé).
Sortie : `2_split_dataset/images/{train,val,test}` + `labels/{train,val,test}`,
plus un manifeste `.parent_manifest.json` (parent_id -> infos brutes d'origine,
utile aux outils en aval comme src/review/).

Exemple :
    python src/data/split_dataset.py [--force]
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

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

# Banc de test figé (optionnel) : liste de parent_id gelés en test une bonne
# fois pour toutes, écrite par `freeze_benchmark_test.py`. Voir _load_frozen_test_ids().
FROZEN_TEST_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "frozen_test_parents.json"

# Manifeste parent_id -> {batch, split, chemins bruts} écrit à chaque run de
# run_split() (voir la fin de la fonction). Sert de source de vérité fiable
# pour tout outil en aval qui a besoin de retrouver l'image/le label BRUTS
# d'origine d'un parent_id présent dans 2_split_dataset (ex: l'outil de
# relecture assistée par modèle, src/review/) - reconstruire ce chemin en
# "dérivant" le parent_id (remplacer les espaces/séparateurs par des
# underscores) est ambigu dès qu'un nom de fichier contient déjà un
# underscore, donc pas fiable.
PARENT_MANIFEST_FILENAME = ".parent_manifest.json"


def _count_resolved_instances(label_path: Path, local_names: Dict[int, str], class_taxonomy) -> int:
    """Comme `_translate_and_filter_label` mais ne retourne qu'un COMPTE de lignes
    résolues (non exclues, non EXCLUDE) - utilisé par `--balance-by items` (voir
    `run_split`) pour peser chaque image parente par son nombre RÉEL d'instances
    entraînables plutôt que par la simple présence de l'image. Volontairement
    tolérant (jamais de RuntimeError ici) : une incohérence réelle est de toute
    façon détectée et bloquée par `raw_dataset_checker.py` (étape 0, avant que
    cette fonction ne tourne) et re-signalée par `_translate_and_filter_label`
    au moment de la traduction réelle - ce compte ne sert qu'à équilibrer les
    splits, pas de source de vérité sur la validité des labels."""
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
    """Répartit `items` (déjà réduits au reliquat non-figé d'UN lot) entre
    train/val/test en visant des proportions en NOMBRE D'ITEMS (`counts`,
    parent_id -> nb d'instances résolues) plutôt qu'en nombre d'images - voir
    la demande du 02/09/2026 (dataset mono-classe : équilibrer train/val/test
    en items, pas en images, deux images pouvant avoir des densités de déchets
    très différentes).

    Ce n'est PAS un problème de partition résolu de façon optimale (NP-difficile
    en général) : un algorithme glouton par DÉFICIT suffit très largement ici et
    reste simple à expliquer/déboguer - à chaque image (dans un ordre mélangé de
    façon déterministe via `seed_key`, même logique de RNG indépendant par lot
    que le mode --balance-by images), on l'assigne au split actuellement le
    PLUS EN DESSOUS de sa cible en items, ce qui converge naturellement vers les
    proportions demandées sans jamais viser un optimum exact.

    IMPORTANT - ordre de traitement (correctif du 02/09/2026) : un ordre
    purement aléatoire pose problème quand les poids (items/image) sont très
    hétérogènes. Exemple observé en test synthétique (3 images "denses" à 30
    items + 17 images "éparses" à 1 item, cible test = 10% des items) : selon
    l'instant où une image dense est tirée dans l'ordre aléatoire, elle peut
    soit dépasser largement une petite cible (ex. test), soit être écartée par
    le déficit d'un split plus gros, laissant test avec presque aucun item
    (0.9% observé, PIRE que le mode --balance-by images sur le même jeu, 1.9%).
    Correctif standard pour ce type de partition équilibrée : trier les items
    par poids DÉCROISSANT avant la boucle gloutonne (heuristique "Longest
    Processing Time first" / LPT), en ne gardant le mélange aléatoire que
    pour départager les ex-æquo (tri stable après un `shuffle` déterministe).
    Ainsi les grosses images sont placées en premier, quand tous les déficits
    sont encore larges (donc leur placement a peu d'impact relatif), et les
    petites images affinent le reste ensuite - convergence bien plus fidèle
    aux proportions cibles, tout en restant déterministe via `seed_key`.

    Entrée : items du lot (reliquat non figé), leurs comptes d'items, part de
    val/test parmi CE reliquat (train = 1 - val_share - test_share), clé de
    seed (même convention que le reste du module : "SEED:nom_du_lot").
    Sortie : dict parent_id -> split ("train"/"val"/"test")."""
    rng = random.Random(seed_key)
    shuffled = list(items)
    rng.shuffle(shuffled)
    # LPT : tri stable par nb d'items décroissant. Le tri stable préserve
    # l'ordre aléatoire ci-dessus entre items de même poids (tie-break).
    shuffled.sort(key=lambda it: counts[it["parent_id"]], reverse=True)

    total_items = sum(counts[it["parent_id"]] for it in shuffled)
    targets = {
        "train": total_items * (1.0 - val_share - test_share),
        "val": total_items * val_share,
        "test": total_items * test_share,
    }
    assigned: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    result: Dict[str, str] = {}
    # Splits à cible nulle (ex: test_share=0.0 quand le banc de test est figé -
    # voir _load_frozen_test_ids/freeze_benchmark_test.py) ne doivent JAMAIS
    # recevoir d'item, même par un effet de bord de l'arrondi flottant en toute
    # fin de lot (déficits de train/val légèrement négatifs simultanément,
    # ce qui rendrait le déficit nul de test artificiellement maximal). On les
    # exclut du candidat, sans quoi une image pourrait fuiter dans le banc de
    # test figé - inacceptable puisque ce banc doit rester strictement stable
    # pour comparer les runs (cf. demande du 02/09/2026).
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
    d'origine, local_id -> nom - seule façon de savoir ce que "0" ou "8" veut dire
    pour ce lot), `class_taxonomy`, `batch_name` (pour les messages d'erreur).
    Sortie : liste des lignes traduites (peut être vide - image sans objet).
    Les coordonnées ne sont jamais modifiées, seul l'ID de classe en tête de ligne
    change (ou la ligne disparaît si la classe est exclue). Lève RuntimeError si un
    ID ou un nom de classe n'est pas résolvable.
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
                # (config/data_config.yaml) - doit bloquer plutôt qu'être ignoré en
                # silence. Sécurité de secours : raw_dataset_checker.py doit déjà
                # avoir bloqué avant.
                raise RuntimeError(
                    f"{src_label_path} (L{lineno}) : classe '{name}' (lot '{batch_name}') "
                    f"n'est dans aucune entrée de class_taxonomy/class_aliases "
                    f"(config/data_config.yaml). Décide de son sort avant de relancer."
                )
            if resolved == EXCLUDE:
                continue

            out_lines.append(f"{resolved} {' '.join(parts[1:])}\n")

    return out_lines



# SPLIT_LOGIC_VERSION : à incrémenter quand la LOGIQUE de split elle-même
# change (algorithme de répartition, stratification, arrondi...) de façon à
# produire une répartition différente pour les MÊMES seed/ratios/taxonomie.
# Sans ce marqueur, _split_params() fingerprinterait IDENTIQUEMENT avant et
# après un tel changement, et ensure_cache_is_safe() réutiliserait
# silencieusement un 2_split_dataset généré avec l'ancien algorithme.
SPLIT_LOGIC_VERSION = 5


def _load_frozen_test_ids() -> List[str]:
    """Charge la liste de parent_id épinglés en test (voir FROZEN_TEST_PATH).

    Sortie : liste triée (ordre stable, utile pour un fingerprint reproductible),
    vide si le fichier n'existe pas (cas par défaut - aucun banc figé encore créé)."""
    if not FROZEN_TEST_PATH.exists():
        return []
    with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data.get("parent_ids", []))


def _split_params(
    class_taxonomy, target_names: Dict[int, str], frozen_test_ids: List[str], balance_by: str,
) -> Dict:
    """Paramètres qui influencent la sortie de cette étape (pour le garde-fou de cache)."""
    return {
        "split_logic_version": SPLIT_LOGIC_VERSION,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "balance_by": balance_by,  # "images" (défaut) ou "items" - change la répartition, voir run_split
        "class_taxonomy": {k: v for k, v in sorted(class_taxonomy.items())},
        "target_names": {str(k): v for k, v in sorted(target_names.items())},
        # Inclus tel quel (pas juste un hash) : un banc figé qui change de contenu
        # (créé, ou modifié à la main) doit invalider le cache comme tout autre
        # paramètre de cette étape.
        "frozen_test_ids": frozen_test_ids,
    }


def run_split(
    force: bool = False,
    raw_dir: str = RAW_DIR,
    split_dir: str = SPLIT_DIR,
    run_confirmation=None,
    class_config_path: Union[str, Path] = DEFAULT_CLASS_CONFIG_PATH,
    balance_by: str = "images",
):
    """Voir la docstring du module. `class_config_path` : référentiel de classes à utiliser -
    permet de faire tourner cette étape sur une taxonomie alternative (ex: variante
    expérimentale "sans Debris_Divers", voir config/data_config_no_debris.yaml) SANS toucher
    au fichier principal, à condition de fournir aussi un `split_dir` distinct (sinon le
    dataset principal serait écrasé par une taxonomie différente - aucun garde-fou de ce
    script n'empêche ça, c'est à l'appelant de garder les deux séparés).

    `balance_by` ("images", défaut, ou "items", ajouté le 02/09/2026 pour le dataset
    mono-classe) : à l'intérieur de chaque lot (stratification par lot inchangée dans les
    deux modes - voir le commentaire plus bas sur pourquoi ça reste utile même en
    mono-classe), "images" répartit train/val/test au prorata du nombre d'IMAGES PARENTES
    (comportement historique, inchangé) ; "items" répartit au prorata du nombre
    D'INSTANCES ANNOTÉES RÉSOLUES (résolues sous `class_config_path` - un compte 0 avec le
    référentiel 7-classes peut différer d'un compte avec le référentiel mono-classe, ex:
    "Inconnu" inclus dans l'un, exclu dans l'autre), via un algorithme glouton par déficit
    (voir `_greedy_item_balanced_assignment`) - utile quand la densité de déchets varie
    beaucoup d'une image à l'autre (le cas ici), pour que train/val/test soient comparables
    en VOLUME D'EXEMPLES, pas juste en nombre d'images."""
    if balance_by not in ("images", "items"):
        raise RuntimeError(f"--balance-by doit valoir 'images' ou 'items' (reçu '{balance_by}').")

    class_taxonomy, target_names = load_class_config(class_config_path)
    frozen_test_ids = _load_frozen_test_ids()

    ensure_cache_is_safe(
        split_dir,
        _split_params(class_taxonomy, target_names, frozen_test_ids, balance_by),
        force=force,
        wipe_subdirs=["images", "labels"],
        manifest_filename=MANIFEST_FILENAME,
        run_confirmation=run_confirmation,
    )

    all_parents = collect_parent_images(Path(raw_dir))
    if not all_parents:
        print(f"❌ Aucune image brute trouvée dans {raw_dir}.")
        return

    unique_parents = list({p["parent_id"]: p for p in all_parents}.values())

    if frozen_test_ids:
        found_ids = {p["parent_id"] for p in unique_parents}
        missing = [pid for pid in frozen_test_ids if pid not in found_ids]
        if missing:
            print(f"⚠️  {len(missing)} parent_id du banc de test figé sont introuvables dans "
                  f"{raw_dir} (déplacés/supprimés depuis le gel ?) : {', '.join(missing[:5])}"
                  f"{'...' if len(missing) > 5 else ''}")

    # Stratification par lot : un shuffle global traiterait chaque image parente
    # comme interchangeable, alors que certains lots sont la SEULE source de
    # certaines classes réelles - le pur hasard pourrait alors renvoyer un
    # petit lot entier vers train et laisser val/test sans aucun exemple de ses
    # classes exclusives. Découper 70/20/10 SÉPARÉMENT à l'intérieur de chaque
    # lot, puis fusionner les résultats, garantit que CHAQUE lot (et donc ses
    # classes propres) est représenté dans les trois splits dans les
    # proportions voulues, peu importe la chance du tirage.
    parents_by_batch: Dict[str, List[Dict]] = defaultdict(list)
    for item in unique_parents:
        parents_by_batch[item["batch"]].append(item)

    # Chargé TÔT (avant l'allocation, pas seulement au moment de la copie/traduction
    # plus bas) : nécessaire pour peser chaque image par son nombre d'items résolus,
    # que ce soit pour décider l'allocation (--balance-by items) ou simplement pour
    # l'AFFICHER à titre informatif même en mode "images" par défaut (voir plus bas -
    # utile de voir le déséquilibre en items même quand on ne corrige que sur les
    # images).
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
        # stratification : ces images sont en test de façon permanente, quel que
        # soit le lot, et ne participent plus jamais au calcul train/val/test de
        # leur lot - voir freeze_benchmark_test.py et le journal du 26/08/2026.
        frozen_items = [it for it in batch_items if it["parent_id"] in frozen_test_id_set]
        remaining_items = [it for it in batch_items if it["parent_id"] not in frozen_test_id_set]
        for it in frozen_items:
            split_map[it["parent_id"]] = "test"
            counts["test"] += 1
            counts["test_frozen"] += 1
            item_counts["test"] += item_count_by_parent[it["parent_id"]]
            item_counts["test_frozen"] += item_count_by_parent[it["parent_id"]]

        # RNG indépendant par lot (dérivé de SEED + du nom du lot), plutôt qu'un
        # random.seed(SEED) global consommé au fil de la boucle : sinon le tirage
        # d'un lot dépend de l'ORDRE et de la TAILLE de tous les lots qui le
        # précèdent dans parents_by_batch - ajouter ou retirer un lot (ex:
        # nouvelle source de données) décalerait alors silencieusement la
        # répartition train/val/test de lots qui n'ont pourtant pas changé. Avec un
        # flux propre à chaque lot, seul le nouveau lot est affecté par son ajout ;
        # condition nécessaire pour comparer deux runs sur un test set stable.
        random.Random(f"{SEED}:{batch_name}").shuffle(remaining_items)
        n_b = len(remaining_items)

        if frozen_test_ids:
            # Test est déjà entièrement couvert par le banc figé (globalement, pas
            # forcément dans CE lot précis) : le reliquat de ce lot ne se répartit
            # plus qu'entre train et val, aux ratios train/val renormalisés pour
            # sommer à 1 (ex: 70/20 -> 77.8%/22.2%).
            val_share = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
            test_share = 0.0
        else:
            val_share = VAL_RATIO
            test_share = TEST_RATIO

        if balance_by == "items":
            total_items_b = sum(item_count_by_parent[it["parent_id"]] for it in remaining_items)
            if total_items_b == 0:
                # Rien à équilibrer (lot restant 100% "sans déchet") - retombe sur le
                # comportement "images" pour ce lot précis plutôt que de diviser par 0.
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
            # val et test sont arrondis chacun au plus proche de leur part théorique
            # (round()) ; train absorbe le reliquat. Train ne sert jamais à mesurer
            # une performance, une distorsion d'arrondi y est donc sans conséquence,
            # alors qu'elle biaiserait directement l'évaluation si elle tombait sur
            # val ou test.
            n_val_b = round(n_b * val_share)
            n_test_b = round(n_b * test_share)
            n_train_b = n_b - n_val_b - n_test_b

            for idx, item in enumerate(remaining_items):
                if idx < n_train_b:
                    s = "train"
                elif idx < n_train_b + n_val_b:
                    s = "val"
                else:
                    s = "test"
                split_map[item["parent_id"]] = s
                counts[s] += 1

        for it in remaining_items:
            item_counts[split_map[it["parent_id"]]] += item_count_by_parent[it["parent_id"]]

        # Un lot trop petit pour peupler ses portions proportionnellement (ex: un
        # lot de 8 images -> 0 en val avec ces ratios) n'est pas une erreur - juste
        # un fait à ne pas laisser passer en silence : ses classes exclusives
        # risquent alors de ne jamais apparaître dans un des splits malgré la
        # stratification. Sans banc figé, n_test_b == 0 sur un lot non-vide compte
        # aussi comme un cas à signaler ; avec banc figé, test == 0 par lot est
        # attendu (voir ci-dessus) et ne doit pas déclencher ce warning.
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
    # `local_names_by_batch` déjà chargé plus haut (pour compter les items par image,
    # voir balance_by) - réutilisé tel quel ici, pas la peine de relire chaque
    # data.yaml de lot une 2e fois.
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
    parser.add_argument(
        "--config", default=str(DEFAULT_CLASS_CONFIG_PATH),
        help="Référentiel de classes à utiliser (défaut : config/data_config.yaml). "
             "Pour une taxonomie alternative (ex: config/data_config_no_debris.yaml), "
             "fournis AUSSI --split-dir pour ne jamais écraser le dataset principal.",
    )
    parser.add_argument(
        "--split-dir", default=SPLIT_DIR,
        help="Dossier de sortie de cette étape (défaut : 2_split_dataset). À changer "
             "systématiquement quand --config pointe vers une taxonomie alternative.",
    )
    parser.add_argument(
        "--balance-by", choices=["images", "items"], default="images",
        help="Critère d'équilibrage train/val/test À L'INTÉRIEUR DE CHAQUE LOT (défaut : "
             "'images', comportement historique inchangé - proportion au nombre d'images "
             "parentes). 'items' (ajouté le 02/09/2026 pour le dataset mono-classe) vise "
             "plutôt la proportion en NOMBRE D'INSTANCES ANNOTÉES RÉSOLUES - utile quand la "
             "densité de déchets varie beaucoup d'une image à l'autre. Le nombre d'items par "
             "split est toujours affiché, même en mode 'images', à titre informatif.",
    )
    args = parser.parse_args()
    try:
        run_split(
            force=args.force, split_dir=args.split_dir, class_config_path=args.config,
            balance_by=args.balance_by,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
