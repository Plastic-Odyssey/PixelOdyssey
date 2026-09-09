#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Convention de dossier de sortie PARTAGÉE par les deux modes du
pipeline application (batch et orthomosaïque) : les résultats des deux
pipelines doivent atterrir au même endroit, sous `4. Results/2_prediction/`
(sous-dossier numéroté, même convention que
`4. Results/1_assisted_annotation/` déjà en place pour `bootstrap_annotate.py`).

Un seul dossier plat (pas de sous-dossier séparé par mode) : chaque run garde
un préfixe dans son nom (`batch_<horodatage>` ou `ortho_<horodatage>`, voir
run_application.py/geo_density_map.py) pour rester identifiable au premier
coup d'œil en parcourant le dossier, sans dupliquer l'arborescence.
"""

DEFAULT_PREDICTION_DIR = r"E:\PixelOdyssey\4. Results\2_prediction"
