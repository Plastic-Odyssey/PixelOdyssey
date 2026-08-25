#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Géométrie de fenêtre glissante partagée.

Extrait de `slicer.py` (2026-08-21) pour que la géométrie de tuilage utilisée
à l'ENTRAÎNEMENT (découpage de 1_annotated_dataset -> 4_sliced_dataset) et
celle utilisée à l'INFÉRENCE par le futur outil de relecture assistée par
modèle (src/review/) restent identiques par construction, plutôt que
dupliquées dans deux fichiers qui pourraient diverger silencieusement.

Si les fenêtres vues par le modèle en inférence ne correspondent pas
EXACTEMENT à celles vues pendant l'entraînement (même tile_size, même stride,
même règle de dernière fenêtre en butée de bord), ses prédictions se
dégradent sans que rien ne le signale - d'où l'intérêt d'une seule fonction
source de vérité, importée des deux côtés.
"""

from typing import Iterator, Tuple


def iter_tile_windows(img_w: int, img_h: int, tile_size: int, stride: int) -> Iterator[Tuple[int, int, int, int]]:
    """Génère les fenêtres (x_start, y_start, x_end, y_end) d'un balayage par
    fenêtre glissante sur une image de dimensions (img_w, img_h).

    Règle de bord : une fenêtre qui dépasserait l'image est ramenée en butée
    (x_start/y_start ajustés pour que x_end/y_end restent dans l'image), pas
    tronquée - la fenêtre garde toujours exactement `tile_size` x `tile_size`.
    C'est la même règle que la boucle historique de `PlasticImageSlicer`
    (slicer.py) : ne JAMAIS la faire diverger sans mettre à jour LOGIC_VERSION
    là-bas, sous peine de mélanger deux géométries différentes en silence.
    """
    for y_offset in range(0, img_h, stride):
        for x_offset in range(0, img_w, stride):
            x_start = x_offset
            y_start = y_offset

            if x_start + tile_size > img_w:
                x_start = max(0, img_w - tile_size)
            if y_start + tile_size > img_h:
                y_start = max(0, img_h - tile_size)

            yield x_start, y_start, x_start + tile_size, y_start + tile_size
