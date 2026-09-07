#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Résolution des specs capteur pour la géolocalisation directe
(pipeline "application" - voir src/application/geolocation.py).

Source de vérité : config/sensor_specs.yaml (jamais codé en dur ici - même
principe que class_config.py pour la taxonomie de classes : ajouter un
capteur = éditer le YAML, pas ce module).

Exemple :
    from src.application.sensor_config import load_sensor_registry, resolve_sensor
    registry = load_sensor_registry()
    spec = resolve_sensor("DJI", "FC3411", registry)
"""

from pathlib import Path
from typing import Any, Dict, NamedTuple

import yaml

DEFAULT_SENSOR_REGISTRY_PATH = Path("config/sensor_specs.yaml")


class SensorSpec(NamedTuple):
    """Specs physiques d'un capteur, telles que nécessaires au calcul de GSD
    et à la projection d'empreinte au sol (voir geolocation.py)."""
    make: str
    model: str
    label: str
    sensor_width_mm: float
    sensor_height_mm: float
    max_pitch_deviation_deg: float


def load_sensor_registry(path: Path = DEFAULT_SENSOR_REGISTRY_PATH) -> Dict[str, Any]:
    """Charge le registre brut (dict Make -> Model -> specs) depuis le YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Registre capteur introuvable : {path}. Ce fichier est la source de vérité "
            f"unique des specs capteur (voir sa docstring) - il doit exister avant tout "
            f"calcul de géolocalisation."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("sensors", {})


def resolve_sensor(make: str, model: str, registry: Dict[str, Any]) -> SensorSpec:
    """Retrouve les specs d'un capteur par (Make, Model) EXIF exacts.

    Échec explicite (pas de valeur par défaut ni d'approximation silencieuse)
    si le couple est inconnu - une GSD calculée sur de mauvaises specs
    capteur serait fausse de façon indétectable a posteriori, contrairement à
    une exception immédiate qui pointe exactement quoi ajouter et où.
    """
    make_entries = registry.get(make)
    if make_entries is None or model not in make_entries:
        known = {m: list(models.keys()) for m, models in registry.items()}
        raise ValueError(
            f"Capteur inconnu : Make={make!r} Model={model!r}. "
            f"Capteurs connus dans config/sensor_specs.yaml : {known}. "
            f"Ajoute une entrée pour ce capteur avant de continuer - voir la docstring "
            f"du fichier YAML pour la méthode (vérifier la fiche technique réelle, "
            f"jamais deviner)."
        )

    entry = make_entries[model]
    required_keys = ("label", "sensor_width_mm", "sensor_height_mm", "max_pitch_deviation_deg")
    missing = [k for k in required_keys if k not in entry]
    if missing:
        raise ValueError(
            f"Entrée capteur Make={make!r} Model={model!r} incomplète dans "
            f"config/sensor_specs.yaml : champs manquants {missing}."
        )

    return SensorSpec(
        make=make,
        model=model,
        label=str(entry["label"]),
        sensor_width_mm=float(entry["sensor_width_mm"]),
        sensor_height_mm=float(entry["sensor_height_mm"]),
        max_pitch_deviation_deg=float(entry["max_pitch_deviation_deg"]),
    )
