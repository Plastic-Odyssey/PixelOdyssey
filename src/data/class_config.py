#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Chargement du référentiel unique de classes.

Chaque lot garde son propre `data.yaml` local tel quel (peu importe combien
de classes il déclare, dans quel ordre, avec quels IDs - voir
`load_batch_local_names`). Pour traduire un label, on lit le NOM de la classe
via CE data.yaml local, on le normalise (casse/accents/underscores - voir
`normalize_class_name`), puis on le cherche dans `class_taxonomy`
(config/data_config.yaml) qui dit soit "exclue", soit "va vers telle
super-classe". Une poignée de vraies divergences d'orthographe entre lots
(fautes de frappe, singulier/pluriel - ex: "fliflops"/"flipflops",
"bouteille pet"/"bouteilles pet") ne sont PAS résolues par la normalisation
automatique et sont déclarées explicitement dans `class_aliases`.

On ne bloque plus jamais sur un simple écart de nombre/liste de classes entre
lots : on bloque UNIQUEMENT si un nom réellement utilisé dans un label
n'apparaît, après normalisation et résolution des alias, dans AUCUNE entrée
de `class_taxonomy` (voir raw_dataset_checker.py, qui fait cette vérification
sur les données réelles).
"""

import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import yaml

# src/data/class_config.py -> parents[2] = racine du repo.
DEFAULT_CLASS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "data_config.yaml"

EXCLUDE = "exclude"  # valeur spéciale dans class_taxonomy : classe toujours ignorée.

ClassTaxonomy = Dict[str, Union[int, str]]  # nom normalisé -> ID de super-classe, ou EXCLUDE


def normalize_class_name(name: str) -> str:
    """Casse, accents et underscores/espaces uniformisés, pour que "Bouées",
    "bouees" et "Bou_ees" désignent la même entrée sans rien déclarer en plus.

    Ne corrige PAS les vraies divergences d'orthographe (fautes de frappe,
    singulier/pluriel) - c'est le rôle de `class_aliases`, volontairement
    explicite plutôt que deviné.
    """
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.replace("_", " ")
    return " ".join(name.lower().split())


def load_batch_local_names(yaml_path: Union[str, Path], key: str = "names") -> Dict[int, str]:
    """Charge le dict {id_local: nom} du data.yaml D'UN LOT (pas le référentiel
    projet). `key` vaut "names" pour un data.yaml de lot classique.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get(key, {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def resolve_class_name(name: str, taxonomy: ClassTaxonomy) -> Optional[Union[int, str]]:
    """Normalise `name` et le cherche dans `taxonomy` (déjà résolue - alias fondus dedans,
    voir `load_class_config`). Retourne l'ID de super-classe cible, `EXCLUDE`, ou None si
    ce nom (même normalisé) n'apparaît dans aucune entrée - LE cas qui doit bloquer le
    pipeline plutôt que d'être ignoré en silence.
    """
    return taxonomy.get(normalize_class_name(name))


def pick_placeholder_local_id(
    local_names: Dict[int, str], taxonomy: ClassTaxonomy, target_id: int
) -> Optional[int]:
    """
    Usage très spécifique : relecture des des annotations assistée par le modèle (src/review/).
    
    Choisit, PARMI LES CLASSES QUE CE LOT DÉCLARE LUI-MÊME (son propre
    data.yaml local), la première (par ID croissant) qui se résout vers
    `target_id`. Utilisé par l'outil de relecture assistée (src/review/) pour
    injecter une nouvelle détection dans un `.txt` brut : le modèle ne prédit
    qu'en espace SUPER-CLASSE (3 classes), jamais en espace fin (~20 classes
    par lot) - il ne peut donc jamais dire QUELLE sous-classe précise il a
    vue, seulement sa famille. On injecte donc un nom de sous-classe
    "placeholder" - toujours une sous-classe RÉELLEMENT déclarée par CE lot
    (pas une sous-classe empruntée à un autre lot qui n'existerait pas dans
    son data.yaml) - à charge pour la relecture humaine dans CVAT de corriger
    le sous-type exact. Retourne None si ce lot ne déclare AUCUNE sous-classe
    qui pointe vers `target_id` (rare, mais possible) - à charge de l'appelant
    de le signaler plutôt que d'injecter n'importe quoi.
    """
    for local_id in sorted(local_names):
        if resolve_class_name(local_names[local_id], taxonomy) == target_id:
            return local_id
    return None


def load_class_config(
    config_path: Union[str, Path] = DEFAULT_CLASS_CONFIG_PATH,
) -> Tuple[ClassTaxonomy, Dict[int, str]]:
    """Charge le référentiel de classes unique (config/data_config.yaml).

    Retourne (class_taxonomy, target_names) :
      - class_taxonomy : dict {nom_normalisé: ID_super_classe ou EXCLUDE} - alias de
        `class_aliases` déjà fondus dedans, donc UNE SEULE table à interroger
        (via `resolve_class_name`).
      - target_names : dict {ID_super_classe: nom} (le "names" du data.yaml YOLO).

    Émet un avertissement si une super-classe déclarée dans `names` n'est jamais
    atteinte par `class_taxonomy` (classe "morte" - jamais d'exemples d'entraînement
    pour elle).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Référentiel de classes introuvable : {config_path}. "
            "Ce fichier doit exister et contenir 'class_taxonomy' et 'names'."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_taxonomy = data.get("class_taxonomy") or {}
    class_taxonomy: ClassTaxonomy = {}
    for name, target in raw_taxonomy.items():
        key = normalize_class_name(str(name))
        value = target if target == EXCLUDE else int(target)
        class_taxonomy[key] = value

    raw_aliases = data.get("class_aliases") or {}
    for alias, canonical in raw_aliases.items():
        alias_key = normalize_class_name(str(alias))
        canonical_key = normalize_class_name(str(canonical))
        if canonical_key not in class_taxonomy:
            raise ValueError(
                f"class_aliases : '{alias}' pointe vers '{canonical}' (normalisé "
                f"'{canonical_key}'), qui n'existe pas dans class_taxonomy. Un alias doit "
                f"toujours pointer vers une entrée déjà présente dans class_taxonomy."
            )
        class_taxonomy[alias_key] = class_taxonomy[canonical_key]

    target_names = data.get("names", {})
    if isinstance(target_names, list):
        target_names = {i: n for i, n in enumerate(target_names)}
    target_names = {int(k): v for k, v in target_names.items()}

    reachable_targets = {v for v in class_taxonomy.values() if v != EXCLUDE}
    dead_targets = set(target_names) - reachable_targets
    if dead_targets:
        dead_str = ", ".join(f"{cid} ({target_names[cid]})" for cid in sorted(dead_targets))
        print(
            f"⚠️  [class_config] Classe(s) cible déclarée(s) dans '{config_path.name}' mais jamais "
            f"atteinte par class_taxonomy : {dead_str}. Tant que le mapping n'est pas complété, "
            f"ces classes ne recevront jamais d'exemple d'entraînement."
        )

    return class_taxonomy, target_names
