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
    --balance-by    "items" (défaut depuis le 24/09/2026) ou "images" - voir plus bas.
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
priver val/test de leurs exemples. Chaque lot est découpé 70/15/15 séparément
(70/20/10 jusqu'au 23/09/2026) puis les résultats sont fusionnés.

Équilibrage par OBJETS (défaut depuis le 24/09/2026, `--balance-by items`) : au sein
d'un lot, les images sont réparties pour que chaque split reçoive sa part du nombre
d'annotations, pas du nombre d'images. Le mode "images" donnait 54/33/13 % des objets
au lieu de 70/20/10 (69 % des objets d'Aldabra en val), faute d'images assez
nombreuses et de densités comparables. Le poids d'une image est son nombre TOTAL
d'annotations brutes, toutes classes confondues (exclues comprises) : il ne dépend
donc pas de la taxonomie, et toutes les variantes (multi, mono, mono avec Inconnu,
filtre de site) obtiennent EXACTEMENT la même répartition des mêmes images.

Groupes spatiaux (24/09/2026, `config/split_groups.json`, généré par
src/data/diagnostics/spatial_groups.py) : des images parentes qui montrent en partie
le même terrain (imagettes SB voisines, qui se recouvrent de 100 px ; orthomosaïques
SL adjacentes ; photos drone dont les empreintes se chevauchent) sont regroupées et
toujours placées dans le MÊME split - sinon un même objet peut se retrouver en train
et en test. Mesuré le 24/09/2026 : 14 % des annotations SB avaient leur double dans
un autre split. Depuis le 08/10/2026, un groupe peut couvrir deux lots d'un même site
(rattaché au lot de son membre le plus annoté), et les images qui recouvrent une image
du banc figé sont mises en QUARANTAINE (exclues de train/val/test : le test reste
exactement le banc figé).

Le split de chaque image parente est dérivé d'un hash stable de son propre
parent_id (voir `_stable_unit_interval`), jamais de sa position dans un shuffle
de la liste du lot. Conséquence recherchée : ajouter de nouvelles images à un
lot déjà existant (site enrichi progressivement) ne change jamais le split
d'une image déjà présente - seules les nouvelles images sont réparties. Le
mode `--balance-by items` (équilibrage par nombre d'instances annotées plutôt
que par nombre d'images, utile quand la densité de déchets varie beaucoup
d'une image à l'autre) n'a pas cette même garantie : son algorithme glouton
par déficit reste sensible à l'ordre et au volume total du lot.

Si FROZEN_TEST_BENCH_ENABLED vaut True ET que `config/frozen_test_parents.json`
existe (voir `freeze_benchmark_test.py`), les parent_id qu'il liste sont épinglés
en test définitivement, jamais recalculés. Le reste du lot n'alimente alors plus
que train/val. Réactivé le 24/09/2026 avec le banc v2 (stratifié par lot et par
objets, respectant les groupes spatiaux - voir freeze_benchmark_test.py --stratified).

Mise à jour incrémentale (23/09/2026) : un parent déjà présent n'est plus sauté
aveuglément - son label est retraduit et comparé à la copie existante (et l'image
source à sa copie, taille + date). Toute différence (annotation corrigée ou
complétée dans 1_annotated_dataset) met la copie à jour, et `run_split` renvoie la
liste de ces parents pour que data_pipeline.py invalide leurs copies/tuiles en aval.
Les fichiers de parents disparus du brut (ou changés de split) sont retirés.
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
from src.data.utils.class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    EXCLUDE,
    load_batch_local_names,
    load_class_config,
    normalize_class_name,
    resolve_class_name,
)
from src.data.utils.pipeline_utils import ensure_cache_is_safe
from src.data.utils.raw_dataset import VALID_IMG_EXTS, collect_parent_images, site_of_batch

from src.paths_config import PROCESSED_DATASET_DIR as BASE_DIR  # racine centralisee (08/09/2026), voir src/paths_config.py
RAW_DIR = os.path.join(BASE_DIR, "1_annotated_dataset")
SPLIT_DIR = os.path.join(BASE_DIR, "2_split_dataset")

# Dataset encore petit et en phase d'expérimentation de méthode : on garde un
# vrai test set pour une évaluation finale non biaisée. À revoir si
# l'entraînement venait à manquer cruellement de données.
# 70/15/15 depuis le 24/09/2026 (70/20/10 avant) : la val ne sert qu'à choisir l'epoch
# (best.pt) et n'a pas besoin de plus ; 15 % d'objets en test (banc figé v2) donnent un
# intervalle de confiance exploitable par site.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

MANIFEST_FILENAME = ".split_manifest.json"

# Banc de test figé (optionnel) : parent_id gelés en test une bonne fois pour
# toutes, écrits par freeze_benchmark_test.py. Voir _load_frozen_test_ids().
#
# Correctif du 23/09/2026 : ce chemin était calculé avec parent.parent.parent, correct
# quand ce fichier vivait dans src/data/, mais FAUX depuis son déplacement dans
# src/data/utils/ (réorganisation du code des 10-11/09/2026, commitée le 11/09 sous
# "refactor: deplacement des modules data vers src/data/utils") - il pointait vers
# src/config/frozen_test_parents.json, qui n'existe pas, et _load_frozen_test_ids()
# renvoyait alors [] SANS RIEN DIRE. Conséquence vérifiée le 23/09/2026 : les 8 manifestes
# .split_manifest.json du disque sont à `frozen_test_ids: []`, le plus ancien daté du
# 10/09/2026 au soir (les runs du 08/09 évaluaient encore sur le banc figé : 911 instances
# en test). Le test set de TOUS les runs depuis le 10/09 est le tirage par hash 70/20/10.
# parents[3] = racine du dépôt (utils -> data -> src -> racine), même convention que
# class_config.DEFAULT_CLASS_CONFIG_PATH, qui avait, lui, été corrigé lors du déplacement.
FROZEN_TEST_PATH = Path(__file__).resolve().parents[3] / "config" / "frozen_test_parents.json"

# Interrupteur EXPLICITE du banc figé (ajouté le 23/09/2026, en même temps que le
# correctif de chemin ci-dessus). False = comportement IDENTIQUE à celui de tous les
# datasets et de tous les modèles actuels (tirage par hash, banc figé ignoré) : corriger
# le chemin sans cet interrupteur aurait RÉACTIVÉ en silence, au prochain lancement du
# pipeline, un banc v1 (27/08/2026) devenu inadapté - simulé le 23/09/2026 sur les
# données actuelles : 334 objets seulement en test (4,8 % du total, contre 905 avec le
# tirage par hash), dont 6 Cagette et 12 Bouee, et une seule photo pour Aldabra - tout en
# obligeant à réentraîner les 8 modèles (39 des 54 parents figés sont aujourd'hui dans
# leur train). Passer à True seulement après avoir (re)construit un banc adapté
# (voir la proposition "banc de test v2" du 23/09/2026 dans le journal de décisions).
#
# Réactivé le 24/09/2026 avec le banc v2 (config/frozen_test_parents.json régénéré par
# `freeze_benchmark_test.py --stratified` ; l'ancien banc v1 est archivé dans
# config/frozen_test_parents_v1.json). À laisser à True : c'est ce qui garantit que tous
# les runs à venir sont évalués sur exactement les mêmes images.
FROZEN_TEST_BENCH_ENABLED = True

# Groupes spatiaux (24/09/2026) : parent_id -> identifiant de groupe, pour les images
# qui se recouvrent sur le terrain (voir la docstring du module). Généré par
# src/data/diagnostics/spatial_groups.py ; à régénérer quand un nouveau lot est ajouté.
# Absent = chaque image est son propre groupe (comportement d'avant le 24/09/2026),
# signalé en clair à chaque split.
SPLIT_GROUPS_PATH = Path(__file__).resolve().parents[3] / "config" / "split_groups.json"

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


def _count_raw_instances(label_path: Path) -> int:
    """Nombre TOTAL d'annotations (polygones) d'un label BRUT, toutes classes confondues,
    exclues comprises - poids d'équilibrage de `--balance-by items` depuis le 24/09/2026
    (ITEMS_LOGIC_VERSION 3). Indépendant de la taxonomie : le split est une propriété du
    dataset, pas de l'expérience - multi, mono et mono avec Inconnu partagent ainsi
    exactement les mêmes images en train/val/test."""
    if not label_path.exists():
        return 0
    with open(label_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if len(line.split()) >= 7)


def _load_split_groups() -> Dict[str, str]:
    """parent_id -> identifiant de groupe spatial (voir SPLIT_GROUPS_PATH). Dict vide si le
    fichier n'existe pas (chaque image est alors son propre groupe) - toujours annoncé."""
    if not SPLIT_GROUPS_PATH.exists():
        print(f"⚠️  {SPLIT_GROUPS_PATH} introuvable : aucune contrainte de groupe spatial (images "
              f"voisines potentiellement réparties dans des splits différents). Générer avec "
              f"`python -m src.data.diagnostics.spatial_groups`.")
        return {}
    with open(SPLIT_GROUPS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    groups: Dict[str, str] = {}
    for gid, members in sorted(data.get("groups", {}).items()):
        for pid in members:
            groups[pid] = gid
    print(f"ℹ️  Groupes spatiaux : {len(data.get('groups', {}))} groupe(s), {len(groups)} image(s) "
          f"regroupée(s), lus dans {SPLIT_GROUPS_PATH.name}.")
    return groups


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
    # Correctif du 23/09/2026 (ITEMS_LOGIC_VERSION 2) : les images SANS objet (poids 0)
    # ne passent plus par la boucle gloutonne ci-dessous. Avant, une fois les images
    # "pesantes" réparties, tous les déficits étaient figés et CHAQUE image à 0 objet
    # partait dans le même split (celui au plus grand déficit restant) - vérifié sur
    # 2_split_dataset_SB_mono : 0 image de fond en train, 145 en val, 219 en test
    # (78 imagettes d'entraînement au total sur 510). Elles sont désormais réparties
    # par le même hash stable que le mode "images" (proportions val/test respectées,
    # jamais de test si test_share == 0).
    zero_weight = [it for it in items if counts[it["parent_id"]] <= 0]
    items = [it for it in items if counts[it["parent_id"]] > 0]
    result_zero: Dict[str, str] = {}
    train_share_z = 1.0 - val_share - test_share
    for it in zero_weight:
        u = _stable_unit_interval(f"{seed_key}:{it['parent_id']}")
        result_zero[it["parent_id"]] = "train" if u < train_share_z else ("val" if u < train_share_z + val_share else "test")

    rng = random.Random(seed_key)
    # Tri par parent_id AVANT le mélange (23/09/2026) : l'ordre d'entrée venait de
    # Path.rglob(), qui dépend du système de fichiers (NTFS vs ext4...) - même seed,
    # répartition différente d'une machine à l'autre (68 écarts constatés entre le
    # disque de Jame et une reproduction sous Linux du même split SB_mono).
    shuffled = sorted(items, key=lambda it: it["parent_id"])
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
    result.update(result_zero)
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

# Version propre au mode "--balance-by items" (23/09/2026) : n'entre dans le fingerprint
# QUE pour ce mode (voir _split_params) - corriger ce mode ne doit pas invalider les
# datasets construits en mode "images" (le défaut, utilisé par les 8 modèles actuels),
# dont la sortie est inchangée.
#   v2 : images sans objet réparties par hash stable (plus toutes en val/test) +
#        ordre d'entrée trié (répartition identique quel que soit le système de fichiers).
#   v3 (24/09/2026) : poids = nombre total d'annotations BRUTES (toutes classes), plus le
#        nombre d'objets retenus par la taxonomie - même split pour toutes les variantes ;
#        répartition par groupe spatial (voir SPLIT_GROUPS_PATH) plutôt que par image.
#   v4 (08/10/2026) : groupes spatiaux pouvant couvrir plusieurs lots d'un même site (rattachés
#        au lot du membre le plus annoté) ; voisins d'une image figée mis en quarantaine
#        (exclus) au lieu d'être ajoutés au test.
ITEMS_LOGIC_VERSION = 4

# Variante "objets exclus = zones ignorées" (option --inpaint-excluded, 23/09/2026).
#   v1 : polygones des classes listées remplis dans un masque, dilaté de 3 px, puis
#        effacés de l'image par inpainting (cv2.INPAINT_TELEA, rayon 5).
INPAINT_LOGIC_VERSION = 1


def _inpaint_excluded_copy(
    img_src: Path, label_path: Path, local_names: Dict[int, str], class_taxonomy,
    names_to_inpaint: set, img_dst: Path,
) -> int:
    """Copie `img_src` vers `img_dst` en EFFAÇANT (inpainting) les objets dont la classe
    brute est exclue ET listée dans `names_to_inpaint` (noms normalisés).

    Pourquoi (constat du 23/09/2026) : une classe "exclue" (Inconnu, à déterminer...)
    disparaît du label mais l'objet reste VISIBLE dans l'image - le modèle apprend donc
    explicitement que ces objets sont du FOND. 3 645 objets annotés sur 10 536 sont dans ce
    cas (dont 1 098 "inconnu" à Saint Brandon, pour ~370 objets retenus sur ce site) ; en
    évaluation au niveau objet, 79 % des faux positifs du modèle mono à Saint Brandon
    tombent sur un objet "inconnu". Effacer ces objets de l'image en fait des zones
    ignorées (ni positif ni négatif) - équivalent, pour YOLO qui n'a pas de notion de
    région ignorée, du `iscrowd`/"ignore region" des jeux de référence.

    Appliqué aux 3 splits (même définition du dataset partout). Retourne le nombre
    d'objets effacés (0 = simple copie, identique au comportement par défaut)."""
    import cv2
    import numpy as np
    from src.data.utils.image_io import load_image_bgr

    polys = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                name = local_names.get(int(parts[0]))
                if name is None or normalize_class_name(name) not in names_to_inpaint:
                    continue
                if resolve_class_name(name, class_taxonomy) != EXCLUDE:
                    continue  # garde-fou : on n'efface JAMAIS une classe retenue
                polys.append([float(v) for v in parts[1:]])
    if not polys:
        shutil.copy2(img_src, img_dst)
        return 0

    img = load_image_bgr(img_src)
    if img is None:
        raise RuntimeError(f"Image illisible pour l'inpainting : {img_src}")
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for xy in polys:
        pts = np.array([[xy[i] * w, xy[i + 1] * h] for i in range(0, len(xy) - 1, 2)], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if img_dst.suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(img_dst), img, params):
        raise RuntimeError(f"Écriture impossible : {img_dst}")
    return len(polys)


def _load_frozen_test_ids() -> List[str]:
    """Charge la liste de parent_id épinglés en test (voir FROZEN_TEST_PATH).

    Sortie : liste triée (fingerprint reproductible), vide si le banc est désactivé
    (FROZEN_TEST_BENCH_ENABLED) ou si le fichier n'existe pas.

    Depuis le 23/09/2026, le mode réellement appliqué est TOUJOURS annoncé en clair
    dans la console : c'est précisément l'absence de ce message qui a laissé passer
    inaperçue, pendant deux semaines, la désactivation accidentelle du banc (voir
    FROZEN_TEST_PATH)."""
    if not FROZEN_TEST_BENCH_ENABLED:
        print("ℹ️  Banc de test figé DÉSACTIVÉ (FROZEN_TEST_BENCH_ENABLED=False dans split_dataset.py) : "
              "test = tirage stable par hash de chaque image parente (70/20/10 par lot).")
        return []
    if not FROZEN_TEST_PATH.exists():
        raise RuntimeError(
            f"🛑 FROZEN_TEST_BENCH_ENABLED=True mais {FROZEN_TEST_PATH} est introuvable. Génère le banc "
            f"(freeze_benchmark_test.py) ou repasse l'interrupteur à False - jamais d'abandon silencieux "
            f"du banc figé."
        )
    with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = sorted(data.get("parent_ids", []))
    print(f"ℹ️  Banc de test figé ACTIF : {len(ids)} parent_id lus dans {FROZEN_TEST_PATH}.")
    return ids


def _split_params(
    class_taxonomy, target_names: Dict[int, str], frozen_test_ids: List[str], balance_by: str,
    site_filter: Optional[List[str]] = None, inpaint_excluded: Optional[List[str]] = None,
    split_groups: Optional[Dict[str, str]] = None,
) -> Dict:
    """Paramètres qui influencent la sortie de cette étape (garde-fou de cache)."""
    extra = {"items_logic_version": ITEMS_LOGIC_VERSION} if balance_by == "items" else {}
    # Groupes spatiaux : empreinte du CONTENU (pas du fichier), n'entre dans le fingerprint
    # que s'il y en a - un changement de groupes change la répartition, donc le dataset.
    if split_groups:
        digest = hashlib.sha1(json.dumps(sorted(split_groups.items())).encode("utf-8")).hexdigest()[:16]
        extra["split_groups_digest"] = digest
    # N'entre dans le fingerprint QUE si l'option est utilisée : les datasets existants
    # (option absente) gardent exactement le même fingerprint qu'avant le 23/09/2026.
    if inpaint_excluded:
        extra["inpaint_excluded"] = sorted(inpaint_excluded)
        extra["inpaint_logic_version"] = INPAINT_LOGIC_VERSION
    return {
        **extra,
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


def _same_file_stat(src: Path, dst: Path) -> bool:
    """`shutil.copy2` préserve la date de modification : une copie intacte a donc la même
    taille ET la même mtime que sa source. Toute différence = image source remplacée."""
    try:
        s, d = src.stat(), dst.stat()
    except OSError:
        return False
    # Tolérance de 2 s : granularité des systèmes de fichiers FAT/exFAT (disques externes).
    return s.st_size == d.st_size and abs(s.st_mtime - d.st_mtime) <= 2.0


def _same_label_content(existing_label: Path, new_lines: List[str]) -> bool:
    """Compare ligne à ligne, insensible aux fins de ligne (CRLF sous Windows)."""
    try:
        old = [l.strip() for l in existing_label.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return False
    new = [l.strip() for l in new_lines if l.strip()]
    return old == new


PENDING_PURGE_FILENAME = ".pending_downstream_purge.json"


def _plan_stale_split_files(split_dir: Path, parent_manifest: Dict[str, Dict]):
    """Liste (SANS rien supprimer) les fichiers de split_dir/{images,labels}/<split>/ dont le
    parent_id n'est pas attendu dans CE split d'après le manifeste courant (parent disparu
    du brut, renommé, ou changé de split). Seuls les vrais fichiers de parent sont
    considérés - images d'extension connue et labels .txt, jamais un fichier caché
    (`._*` laissés par macOS sur exFAT, `.split_manifest.json`...) ni Thumbs.db/desktop.ini.
    Retourne (fichiers périmés, nombre de parents présents)."""
    stale, present = [], set()
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            d = split_dir / kind / split
            if not d.exists():
                continue
            for f in d.iterdir():
                if not f.is_file() or f.name.startswith("."):
                    continue
                ext = f.suffix.lower()
                if (kind == "images" and ext not in VALID_IMG_EXTS) or (kind == "labels" and ext != ".txt"):
                    continue
                present.add(f.stem)
                info = parent_manifest.get(f.stem)
                if info is None or info["split"] != split:
                    stale.append(f)
    return stale, len(present)


def _read_pending_purge(split_dir: Path) -> List[str]:
    path = split_dir / PENDING_PURGE_FILENAME
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sorted(set(json.load(f)))
    except (OSError, ValueError):
        return []


def clear_pending_purge(split_dir) -> None:
    """Appelé par data_pipeline.py APRÈS une purge aval réussie (voir run_split)."""
    path = Path(split_dir) / PENDING_PURGE_FILENAME
    if path.exists():
        path.unlink()


def _raw_label_digest(label_path: Path) -> Optional[str]:
    if not label_path.exists():
        return None
    return hashlib.sha1(label_path.read_bytes()).hexdigest()


def run_split(
    force: bool = False,
    raw_dir: str = RAW_DIR,
    split_dir: str = SPLIT_DIR,
    run_confirmation=None,
    class_config_path: Union[str, Path] = DEFAULT_CLASS_CONFIG_PATH,
    balance_by: str = "items",
    site_filter: Optional[List[str]] = None,
    inpaint_excluded: Optional[List[str]] = None,
    dry_run: bool = False,
    allow_mass_removal: bool = False,
):
    """Voir la docstring du module.

    `dry_run` (23/09/2026) : calcule et AFFICHE ce que ce run ferait (nouveaux parents,
    parents mis à jour, fichiers périmés à retirer, reconstruction complète éventuelle)
    sans rien écrire ni supprimer. À lancer une première fois après une mise à jour du
    code ou une campagne de corrections d'annotations.

    `allow_mass_removal` : lève le garde-fou qui refuse de retirer plus de 20 % des parents
    présents d'un coup (volontairement distinct de `force`, utilisé en routine).

    `inpaint_excluded` (23/09/2026, défaut : aucun) : noms de classes BRUTES exclues
    (ex: ["Inconnu", "à déterminer"]) à effacer des images par inpainting plutôt que de
    les laisser visibles comme du fond - voir `_inpaint_excluded_copy`. À utiliser avec un
    `split_dir` distinct (variante, voir data_pipeline.py --inpaint-excluded/--suffix).

    `class_config_path` : référentiel de classes à utiliser - permet de faire
    tourner cette étape sur une taxonomie alternative sans toucher au fichier
    principal, à condition de fournir aussi un `split_dir` distinct (aucun
    garde-fou de ce script n'empêche d'écraser le dataset principal sinon).

    `balance_by` ("items", défaut depuis le 24/09/2026, ou "images") : à l'intérieur de
    chaque lot (stratification par lot inchangée dans les deux modes), "images" répartit
    au prorata du nombre d'images parentes ; "items" au prorata du nombre total
    d'annotations brutes, via `_greedy_item_balanced_assignment`. Dans les deux modes,
    l'unité répartie est le GROUPE SPATIAL (voir SPLIT_GROUPS_PATH), une image isolée
    étant son propre groupe.

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
    inpaint_names = sorted({normalize_class_name(n) for n in (inpaint_excluded or [])})
    for n in inpaint_names:
        # Refus explicite d'une classe qui n'est pas exclue dans CE référentiel : effacer un
        # objet RETENU supprimerait de vrais exemples positifs.
        matches = [k for k in class_taxonomy if normalize_class_name(k) == n]
        if not matches or any(class_taxonomy[k] != EXCLUDE for k in matches):
            raise RuntimeError(
                f"--inpaint-excluded '{n}' : ce nom n'est pas une classe EXCLUE de {class_config_path} "
                f"(seules les classes marquées 'exclude' peuvent être effacées des images)."
            )

    # Manifeste du run précédent : taille/date de chaque image SOURCE au moment de sa copie
    # (voir parent_manifest ci-dessous) - nécessaire pour détecter une image source modifiée
    # quand la copie n'est pas byte-identique (variante --inpaint-excluded).
    previous_manifest: Dict[str, Dict] = {}
    prev_path = Path(split_dir) / PARENT_MANIFEST_FILENAME
    if prev_path.exists():
        try:
            with open(prev_path, "r", encoding="utf-8") as f:
                previous_manifest = json.load(f)
        except (OSError, ValueError):
            previous_manifest = {}

    split_groups = _load_split_groups()
    params = _split_params(class_taxonomy, target_names, frozen_test_ids, balance_by, site_filter, inpaint_names,
                           split_groups)
    if dry_run:
        from src.data.utils.pipeline_utils import diff_params, fingerprint
        mpath = Path(split_dir) / MANIFEST_FILENAME
        if mpath.exists():
            with open(mpath, "r", encoding="utf-8") as f:
                old = json.load(f)
            if old.get("fingerprint") != fingerprint(params):
                print(f"🔎 [dry-run] {split_dir} serait ENTIÈREMENT reconstruit (config changée) :")
                print("\n".join(diff_params(old.get("params", {}), params)))
                return {"dry_run": True, "would_rebuild": True}
        else:
            print(f"🔎 [dry-run] {split_dir} n'existe pas encore : il serait créé intégralement.")
            return {"dry_run": True, "would_rebuild": True}
    else:
        ensure_cache_is_safe(
            split_dir,
            params,
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

    # Chargé tôt (avant l'allocation) : nombre d'objets RETENUS par la taxonomie de chaque
    # image (affiché, pour lire le résultat dans l'espace de classes du run) et nombre
    # TOTAL d'annotations brutes (poids d'équilibrage du mode "items", indépendant de la
    # taxonomie - voir _count_raw_instances).
    local_names_by_batch: Dict[str, Dict[int, str]] = {}
    item_count_by_parent: Dict[str, int] = {}
    raw_count_by_parent: Dict[str, int] = {}
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
        raw_count_by_parent[item["parent_id"]] = _count_raw_instances(Path(item["label_path"]))

    frozen_test_id_set = set(frozen_test_ids)
    split_map: Dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0, "test_frozen": 0}
    item_counts = {"train": 0, "val": 0, "test": 0, "test_frozen": 0}
    small_batch_warnings: List[str] = []
    group_warnings: List[str] = []

    # Groupes spatiaux à l'échelle du dataset (08/10/2026) : un groupe peut couvrir deux lots
    # d'un même site (deux missions survolant le même terrain). Chaque groupe est rattaché à
    # UN lot - celui de son membre le plus annoté - où il est réparti comme une seule unité.
    # Un groupe qui contient une image du banc figé : ses autres membres sont mis en
    # QUARANTAINE (ni train, ni val, ni test) - le test reste EXACTEMENT le banc figé, et
    # aucune image qui montre le même terrain qu'une image test n'est vue à l'entraînement.
    members_by_group: Dict[str, List[Dict]] = defaultdict(list)
    for it in unique_parents:
        members_by_group[split_groups.get(it["parent_id"], it["parent_id"])].append(it)
    home_batch: Dict[str, str] = {}
    quarantined: Dict[str, str] = {}  # parent_id -> image figée qu'il recouvre
    for gid, members in members_by_group.items():
        frozen_members = sorted(m["parent_id"] for m in members if m["parent_id"] in frozen_test_id_set)
        if frozen_members:
            for m in members:
                if m["parent_id"] not in frozen_test_id_set:
                    quarantined[m["parent_id"]] = frozen_members[0]
        home_batch[gid] = max(members, key=lambda m: (raw_count_by_parent[m["parent_id"]], m["parent_id"]))["batch"]
    units_by_batch: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    for gid, members in members_by_group.items():
        kept = [m for m in members if m["parent_id"] not in quarantined]
        if kept:
            units_by_batch[home_batch[gid]][gid] = kept
    for pid, frozen_pid in sorted(quarantined.items()):
        group_warnings.append(f"  🚧 Quarantaine : {pid} recouvre l'image du banc figé {frozen_pid} - "
                              f"exclue de train/val/test.")

    for batch_name in sorted(parents_by_batch):
        # Unités de répartition = groupes spatiaux rattachés à ce lot (une image isolée est
        # son propre groupe, d'identifiant = son parent_id).
        units = units_by_batch.get(batch_name, {})

        # Le banc de test figé (s'il existe) est retiré du tirage AVANT toute
        # stratification : ces images sont en test de façon permanente, quel
        # que soit le lot, et ne participent plus au calcul train/val/test.
        remaining_units: Dict[str, List[Dict]] = {}
        for uid, members in units.items():
            if any(m["parent_id"] in frozen_test_id_set for m in members):
                for m in members:  # ici, uniquement des images figées (les autres sont en quarantaine)
                    split_map[m["parent_id"]] = "test"
                    counts["test"] += 1
                    item_counts["test"] += item_count_by_parent[m["parent_id"]]
                    counts["test_frozen"] += 1
                    item_counts["test_frozen"] += item_count_by_parent[m["parent_id"]]
            else:
                remaining_units[uid] = members
        remaining_items = [m for uid in sorted(remaining_units) for m in remaining_units[uid]]
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
            unit_items = [{"parent_id": uid} for uid in sorted(remaining_units)]
            unit_weights = {uid: sum(raw_count_by_parent[m["parent_id"]] for m in members)
                            for uid, members in remaining_units.items()}
            unit_split = _greedy_item_balanced_assignment(
                unit_items, unit_weights, val_share, test_share, f"{SEED}:{batch_name}"
            )
        else:
            # Assignation par hash stable de chaque unité (= du parent_id pour une image
            # isolée, donc identique au comportement d'avant les groupes) : déterminée
            # uniquement par (SEED, nom de lot, identifiant), jamais par le nombre ou
            # l'ordre des autres images du lot. `train_share` absorbe le reliquat.
            train_share = 1.0 - val_share - test_share
            unit_split = {}
            for uid in remaining_units:
                u = _stable_unit_interval(f"{SEED}:{batch_name}:{uid}")
                unit_split[uid] = "train" if u < train_share else ("val" if u < train_share + val_share else "test")
        for uid, members in remaining_units.items():
            for m in members:
                split_map[m["parent_id"]] = unit_split[uid]
                counts[unit_split[uid]] += 1
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
    unique_parents = [it for it in unique_parents if it["parent_id"] not in quarantined]
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
    # Détail par site (24/09/2026) : c'est à cette échelle que le déséquilibre du mode
    # "images" se voyait (69 % des objets d'Aldabra en val) - à relire à chaque split.
    site_items: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for it in unique_parents:
        site_items[site_of_batch(it["batch"])][split_map[it["parent_id"]]] += item_count_by_parent[it["parent_id"]]
    for site in sorted(site_items):
        c = site_items[site]
        t = sum(c.values()) or 1
        print(f"      {site:<3} objets : train {c['train']:>5} ({100 * c['train'] / t:4.0f}%) | "
              f"val {c['val']:>5} ({100 * c['val'] / t:4.0f}%) | test {c['test']:>5} ({100 * c['test'] / t:4.0f}%)")
    for w in group_warnings + small_batch_warnings:
        print(w)
    print()

    # ---- 1) PLAN : rien n'est écrit ni supprimé tant que tout n'a pas été évalué ----
    # (23/09/2026) Une interruption, une erreur ou le garde-fou de suppression massive ne
    # doivent jamais laisser une copie mise à jour sans que l'aval soit invalidé.
    parent_manifest: Dict[str, Dict] = {}
    plan = []  # (action, item, img_dst, lab_dst, out_lines, regenerate_image)
    for item in unique_parents:
        parent_id = item["parent_id"]
        split = split_map[parent_id]
        img_src = Path(item["img_path"])
        batch_name = item["batch"]
        src_stat = img_src.stat()
        raw_digest = _raw_label_digest(Path(item["label_path"]))
        parent_manifest[parent_id] = {
            "batch": batch_name,
            "split": split,
            "raw_img_path": str(img_src),
            "raw_label_path": item["label_path"],
            # Taille/date de l'image SOURCE et empreinte du label BRUT (23/09/2026) - servent
            # à détecter une source modifiée quand la copie n'est pas byte-identique
            # (variante --inpaint-excluded).
            "src_size": src_stat.st_size,
            "src_mtime": src_stat.st_mtime,
            "raw_label_sha1": raw_digest,
        }

        img_dst = Path(split_dir) / "images" / split / f"{parent_id}{img_src.suffix.lower()}"
        lab_dst = Path(split_dir) / "labels" / split / f"{parent_id}.txt"

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

        # Traduit TOUJOURS (coût négligeable, simple lecture de texte) - y compris pour un
        # parent déjà présent : c'est ce qui permet de détecter une annotation corrigée ou
        # complétée sur une image DÉJÀ répartie (23/09/2026). Avant, un parent déjà copié
        # était sauté sans rien relire : une correction de label dans 1_annotated_dataset
        # n'atteignait JAMAIS 2_split_dataset ni les tuiles, et `--force` n'y changeait
        # rien (il ne reconstruit que si la CONFIG a changé).
        out_lines = _translate_and_filter_label(
            Path(item["label_path"]), local_names_by_batch[batch_name], class_taxonomy, batch_name
        )

        if not img_dst.exists():
            plan.append(("new", item, img_dst, lab_dst, out_lines, True))
            continue
        label_unchanged = lab_dst.exists() and _same_label_content(lab_dst, out_lines)
        if inpaint_names:
            # Copie non byte-identique : comparaison à l'état de la source mémorisé au run
            # précédent. Un label BRUT modifié (même sur une classe effacée, absente du label
            # traduit) change les zones à effacer -> image régénérée.
            prev = previous_manifest.get(parent_id) or {}
            regenerate = not (prev.get("src_size") == src_stat.st_size
                              and abs(prev.get("src_mtime", -1e9) - src_stat.st_mtime) <= 2.0
                              and prev.get("raw_label_sha1") == raw_digest)
        else:
            # Copie par copy2 : même taille et même date que la source (même disque, donc
            # insensible au fuseau horaire des systèmes FAT).
            regenerate = not _same_file_stat(img_src, img_dst)
        if label_unchanged and not regenerate:
            plan.append(("unchanged", item, img_dst, lab_dst, out_lines, False))
        else:
            plan.append(("updated", item, img_dst, lab_dst, out_lines, regenerate))

    stale_files, n_present = _plan_stale_split_files(Path(split_dir), parent_manifest)
    stale_ids = sorted({f.stem for f in stale_files})
    updated_ids = [p[1]["parent_id"] for p in plan if p[0] == "updated"]
    n_new = sum(1 for p in plan if p[0] == "new")

    if dry_run:
        print(f"🔎 [dry-run] {split_dir} : {n_new} nouveau(x) parent(s), {len(updated_ids)} mis à jour "
              f"(label et/ou image source modifiés), {len(stale_ids)} périmé(s) à retirer - RIEN n'a été écrit.")
        for pid in updated_ids[:20]:
            print(f"      mis à jour : {pid}")
        for pid in stale_ids[:20]:
            print(f"      à retirer  : {pid}")
        return {"dry_run": True, "would_rebuild": False, "new": n_new, "updated": updated_ids, "removed": stale_ids}

    if n_present and len(stale_ids) > 0.2 * n_present and not allow_mass_removal:
        raise RuntimeError(
            f"🛑 {len(stale_ids)}/{n_present} parents présents dans {split_dir} ne correspondent plus au "
            f"dataset brut actuel - suppression massive refusée par sécurité (disque source incomplet ? "
            f"mauvais --raw-dir ?). RIEN n'a été modifié. Si c'est voulu : --allow-mass-removal."
        )

    # ---- 2) Journal des invalidations aval, écrit AVANT toute modification ----------
    # Cumulé avec un éventuel reliquat d'un run précédent interrompu (ou d'un split lancé
    # seul) ; data_pipeline.py le vide seulement après une purge aval réussie.
    pending = sorted(set(_read_pending_purge(Path(split_dir))) | set(updated_ids) | set(stale_ids))
    if pending:
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(split_dir) / PENDING_PURGE_FILENAME, "w", encoding="utf-8") as f:
            json.dump(pending, f, indent=1, ensure_ascii=False)

    # ---- 3) APPLICATION ------------------------------------------------------------
    for action, item, img_dst, lab_dst, out_lines, regenerate in plan:
        if action == "unchanged":
            continue
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        if regenerate:
            if inpaint_names:
                _inpaint_excluded_copy(
                    Path(item["img_path"]), Path(item["label_path"]), local_names_by_batch[item["batch"]],
                    class_taxonomy, set(inpaint_names), img_dst,
                )
            else:
                shutil.copy2(item["img_path"], img_dst)
        with open(lab_dst, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
    for f in stale_files:
        f.unlink()
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (Path(split_dir) / kind / split).mkdir(parents=True, exist_ok=True)

    with open(Path(split_dir) / PARENT_MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(parent_manifest, f, indent=2, ensure_ascii=False)

    print(f"[SUCCÈS] Split terminé.")
    print(f"  • Nouvelles images copiées : {n_new}")
    print(f"  • Images ignorées (déjà là, inchangées) : {sum(1 for p in plan if p[0] == 'unchanged')}")
    if updated_ids:
        print(f"  • ⚠️  {len(updated_ids)} parent(s) déjà présent(s) MIS À JOUR (label et/ou image source "
              f"modifiés depuis leur première copie) : {', '.join(updated_ids[:5])}"
              f"{'...' if len(updated_ids) > 5 else ''}")
    if stale_ids:
        print(f"  • 🧹 {len(stale_ids)} parent(s) périmé(s) retiré(s) (disparu du brut ou changé de split) : "
              f"{', '.join(stale_ids[:5])}{'...' if len(stale_ids) > 5 else ''}")
    print(f"  • Manifeste parent -> brut : {PARENT_MANIFEST_FILENAME} ({len(parent_manifest)} entrées)")
    # `pending_purge` (mis à jour + retirés + reliquat éventuel) : parents dont les copies et
    # tuiles en aval doivent être invalidées - consommé par data_pipeline.py, qui appelle
    # ensuite clear_pending_purge().
    return {"updated": updated_ids, "removed": stale_ids, "pending_purge": pending}


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
        "--balance-by", choices=["images", "items"], default="items",
        help="Critère d'équilibrage train/val/test à l'intérieur de chaque lot. "
             "'items' (défaut depuis le 24/09/2026) : vise la proportion en nombre "
             "d'annotations brutes (toutes classes), par groupe spatial. Ajouter des images "
             "à un lot existant peut redistribuer train/val de CE lot (jamais le test, figé "
             "par le banc v2). 'images' : assignation stable par hash, au prorata du nombre "
             "d'images - déséquilibrée en objets sur ce dataset (54/33/13 %% mesuré).",
    )
    parser.add_argument(
        "--site", default=None,
        help="Filtre par code de site (ex: SL, SB, A), un ou plusieurs séparés par "
             "virgule (ex: SL,A). Défaut : aucun filtre, tous les sites. Fournis "
             "AUSSI --split-dir pour ne jamais écraser le dataset complet.",
    )
    parser.add_argument(
        "--inpaint-excluded", default=None,
        help="Noms de classes BRUTES exclues à effacer des images (inpainting) plutôt que de les "
             "laisser visibles comme du fond, séparés par des virgules (ex: \"Inconnu,à déterminer\"). "
             "Variante expérimentale : fournis AUSSI --split-dir.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche ce que ce run ferait (nouveaux/mis à jour/retirés) sans rien écrire.")
    parser.add_argument("--allow-mass-removal", action="store_true",
                        help="Autorise le retrait de plus de 20 %% des parents présents d'un coup.")
    args = parser.parse_args()
    site_filter = [s.strip() for s in args.site.split(",")] if args.site else None
    inpaint_excluded = [s.strip() for s in args.inpaint_excluded.split(",")] if args.inpaint_excluded else None
    try:
        run_split(
            force=args.force, split_dir=args.split_dir, class_config_path=args.config,
            balance_by=args.balance_by, site_filter=site_filter, inpaint_excluded=inpaint_excluded,
            dry_run=args.dry_run, allow_mass_removal=args.allow_mass_removal,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
