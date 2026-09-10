#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Registre des modèles "opérationnels" (pipeline application).

Source de vérité : config/models_registry.yaml, rempli À LA MAIN par Jame -
volontairement pas un scan automatique de TRAINING_RUNS_DIR (voir sa docstring :
ce dossier contient aussi des essais de réglage expérimentaux qu'on ne veut
jamais proposer par erreur pour un usage terrain).

Exemple :
    from src.application.model_registry import load_operational_models, resolve_model_choice
    models = load_operational_models()
    chosen = resolve_model_choice(models, requested_name=args.model)
"""

from pathlib import Path
from typing import List, NamedTuple, Optional

import yaml

from src.paths_config import TRAINING_RUNS_DIR

DEFAULT_MODELS_REGISTRY_PATH = Path("config/models_registry.yaml")


class ModelEntry(NamedTuple):
    name: str
    weights_path: str
    description: str
    date: str
    # Fichier de config taxonomie (config/data_config*.yaml) utilisé À
    # L'ENTRAÎNEMENT de ce modèle - nécessaire pour le garde-fou
    # `assert_model_matches_taxonomy` (même vérification que partout ailleurs
    # dans src/review/) : un modèle mono-classe et un modèle 7-classes n'ont
    # PAS le même `target_names` de référence, donc pas de config unique
    # possible à coder en dur ici. Si absent du registre, retombe sur
    # `class_config.DEFAULT_CLASS_CONFIG_PATH` (7-classes, config principale).
    taxonomy_config: Optional[str] = None


def load_operational_models(path: Path = DEFAULT_MODELS_REGISTRY_PATH) -> List[ModelEntry]:
    """Charge la liste des modèles opérationnels déclarés. Lève une erreur
    explicite si le registre est vide - un registre vide veut dire que
    personne n'a encore désigné de modèle "prêt terrain", pas un cas à
    contourner en silence (ex : en retombant sur un modèle au hasard dans
    TRAINING_RUNS_DIR)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registre modèles introuvable : {path}.")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw_models = data.get("models") or []

    if not raw_models:
        raise ValueError(
            f"{path} : aucun modèle opérationnel déclaré. Ajoute au moins une entrée "
            f"(voir l'exemple commenté dans le fichier) avant d'utiliser le pipeline "
            f"application - ce n'est pas un scan automatique de TRAINING_RUNS_DIR, un choix "
            f"explicite est nécessaire."
        )

    models = []
    for entry in raw_models:
        # `entry["weights_path"]` est relatif à TRAINING_RUNS_DIR (ex: "<run>/weights/best.pt",
        # voir src/paths_config.py) - Path.__truediv__ laisse passer tel quel un chemin déjà
        # absolu fourni dans le registre (rétrocompatible avec une entrée qui préciserait un
        # emplacement hors TRAINING_RUNS_DIR).
        weights_path = TRAINING_RUNS_DIR / entry["weights_path"]
        if not weights_path.exists():
            raise FileNotFoundError(
                f"{path} : l'entrée {entry.get('name')!r} pointe vers un fichier de poids "
                f"introuvable ({weights_path}) - run supprimé/déplacé ? Corrige le registre."
            )
        models.append(ModelEntry(
            name=entry["name"],
            weights_path=str(weights_path),
            description=entry.get("description", ""),
            date=entry.get("date", ""),
            taxonomy_config=entry.get("taxonomy_config"),
        ))
    return models


def resolve_model_choice(models: List[ModelEntry], requested_name: Optional[str] = None) -> ModelEntry:
    """Retourne le modèle choisi.

    - `requested_name` fourni : recherche exacte, erreur explicite listant
      les noms disponibles si absent du registre.
    - `requested_name` absent, un seul modèle déclaré : le retourne
      directement (pas besoin de demander).
    - `requested_name` absent, plusieurs modèles déclarés : invite
      interactive (input()) - à éviter dans un contexte non-interactif/
      scripté (appeler avec `requested_name` explicite dans ce cas).
    """
    if not models:
        raise ValueError("Aucun modèle disponible (registre vide).")

    if requested_name is not None:
        for m in models:
            if m.name == requested_name:
                return m
        raise ValueError(
            f"Modèle {requested_name!r} introuvable dans le registre. "
            f"Modèles disponibles : {[m.name for m in models]}."
        )

    if len(models) == 1:
        return models[0]

    print("Plusieurs modèles opérationnels disponibles :")
    for i, m in enumerate(models):
        print(f"  [{i}] {m.name} - {m.description} ({m.date})")
    while True:
        choice = input(f"Choisis un modèle [0-{len(models) - 1}] : ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(models):
            return models[int(choice)]
        print("Choix invalide, réessaie.")
