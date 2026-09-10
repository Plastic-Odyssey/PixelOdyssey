#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Géométrie de fenêtre glissante partagée.

Fonction unique de calcul des fenêtres de tuilage, utilisée à la fois à
l'ENTRAÎNEMENT (découpage de 1_annotated_dataset -> 4_sliced_dataset) et à
l'INFÉRENCE par tuile (src/review/), pour garantir que les deux voient
exactement les mêmes fenêtres (même tile_size, même stride, même règle de
bord) plutôt que deux implémentations qui pourraient diverger silencieusement.

Contient aussi `most_square_grid`/`iter_grid_windows` : découpage en grille
SANS chevauchement en un nombre EXACT de morceaux (contrairement à
`iter_tile_windows`, pensé pour un balayage glissant AVEC recouvrement à
l'entraînement) - utilisé par src/review/split_for_cvat.py pour fragmenter
une orthomosaïque trop lourde pour CVAT en plusieurs morceaux gérables.
`min_pieces_for_pixel_cap` complète ces deux fonctions : calcule le nombre
minimal de morceaux nécessaire pour respecter un plafond de pixels par
morceau (ex : la limite d'import CVAT) - à appeler AVANT `most_square_grid`
pour combiner les deux contraintes ("le plus carré possible" ET "sous le
plafond CVAT") sans qu'elles se contredisent.

Exemple :
    from src.data.utils.tiling_geometry import iter_tile_windows
    for x0, y0, x1, y1 in iter_tile_windows(img_w=4000, img_h=3000, tile_size=640, stride=512):
        tile = img[y0:y1, x0:x1]

Entrée : dimensions de l'image (img_w, img_h) et paramètres de tuilage
(tile_size, stride).
Sortie : itérateur de tuples (x_start, y_start, x_end, y_end).
"""

import math
from typing import Iterator, Tuple


def iter_tile_windows(img_w: int, img_h: int, tile_size: int, stride: int) -> Iterator[Tuple[int, int, int, int]]:
    """Génère les fenêtres (x_start, y_start, x_end, y_end) d'un balayage par
    fenêtre glissante sur une image de dimensions (img_w, img_h).

    Règle de bord : une fenêtre qui dépasserait l'image est ramenée en butée
    (x_start/y_start ajustés pour que x_end/y_end restent dans l'image), pas
    tronquée - la fenêtre garde toujours exactement `tile_size` x `tile_size`.
    Cette règle doit rester identique à celle de `PlasticImageSlicer`
    (slicer.py) : toute modification ici doit s'accompagner d'une mise à jour
    de LOGIC_VERSION côté slicer.py, pour ne pas mélanger deux géométries
    différentes en silence.

    Entrée : img_w, img_h (dimensions de l'image en pixels), tile_size
    (taille de tuile), stride (pas de balayage).
    Sortie : itérateur de tuples (x_start, y_start, x_end, y_end).
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


def min_pieces_for_pixel_cap(img_w: int, img_h: int, max_pixels: int) -> int:
    """Nombre MINIMAL de morceaux nécessaires pour qu'une grille non
    chevauchante (voir `most_square_grid`/`iter_grid_windows`) garde chaque
    morceau sous `max_pixels` pixels.

    Une grille rows x cols partitionne l'image en morceaux de surface
    UNIFORME (largeur/cols x hauteur/rows, identique pour toutes les
    cellules) - `img_w * img_h / (rows * cols)`. Le nombre de pixels par
    morceau ne dépend donc que de `rows * cols` (= n_pieces), jamais de la
    forme (rows, cols) précise retenue par `most_square_grid` pour ce
    nombre : ce calcul est donc indépendant du choix "le plus carré
    possible" fait ensuite - les deux contraintes (carré + sous le plafond)
    se combinent sans se contredire, en résolvant d'abord celle-ci puis en
    appelant `most_square_grid` avec le résultat.

    Entrée : dimensions de l'image, plafond de pixels par morceau (ex :
    limite d'import CVAT).
    Sortie : n_pieces minimal (>= 1) tel que img_w*img_h / n_pieces <=
    max_pixels.
    """
    if max_pixels <= 0:
        raise ValueError(f"max_pixels doit être > 0 (reçu {max_pixels}).")
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Dimensions d'image invalides : {img_w}x{img_h}.")
    return max(1, math.ceil((img_w * img_h) / max_pixels))


def most_square_grid(n_pieces: int, img_w: int, img_h: int) -> Tuple[int, int]:
    """Trouve la grille (rows, cols) avec rows*cols == n_pieces dont les
    cellules résultantes sont les PLUS PROCHES D'UN CARRÉ, compte tenu du
    ratio largeur/hauteur réel de l'image (pas juste rows≈cols - une image
    très large a besoin de plus de colonnes que de lignes pour que CHAQUE
    morceau soit carré).

    Méthode : parcourt tous les couples diviseurs de n_pieces, calcule la
    distorsion de chaque cellule résultante (|log(largeur/hauteur)|, nul
    quand carré, symétrique entre "trop large" et "trop haut"), garde le
    couple qui minimise cette distorsion.

    Limite inhérente : si n_pieces est premier (ou n'a que des diviseurs très
    déséquilibrés), la seule grille possible est 1×n_pieces ou n_pieces×1 -
    aucune grille plus carrée n'existe pour ce compte exact de morceaux.
    L'appelant est libre de comparer `n_pieces` à des valeurs voisines si le
    résultat est trop allongé (voir l'avertissement émis par
    split_for_cvat.py).

    Entrée : nombre de morceaux voulu (>= 1), dimensions de l'image.
    Sortie : (rows, cols) - toujours rows*cols == n_pieces.
    """
    if n_pieces < 1:
        raise ValueError(f"n_pieces doit être >= 1 (reçu {n_pieces}).")
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Dimensions d'image invalides : {img_w}x{img_h}.")

    best: Tuple[float, int, int] = None
    for rows in range(1, n_pieces + 1):
        if n_pieces % rows != 0:
            continue
        cols = n_pieces // rows
        piece_w = img_w / cols
        piece_h = img_h / rows
        distortion = abs(math.log(piece_w / piece_h))
        if best is None or distortion < best[0]:
            best = (distortion, rows, cols)

    _, rows, cols = best
    return rows, cols


def iter_grid_windows(img_w: int, img_h: int, rows: int, cols: int) -> Iterator[Tuple[int, int, int, int]]:
    """Génère les fenêtres (x_start, y_start, x_end, y_end) d'une grille
    EXACTE rows x cols couvrant l'image entière, SANS chevauchement entre
    cellules (contrairement à `iter_tile_windows`) et sans zone non couverte.

    Les bornes sont calculées par interpolation linéaire puis arrondies : si
    `img_w / cols` (ou `img_h / rows`) n'est pas un entier exact, les
    dernières cellules d'une ligne/colonne absorbent le reste en pixels -
    aucun trou, aucun chevauchement, quelle que soit la combinaison de
    dimensions/grille.

    Entrée : dimensions de l'image, nombre de lignes/colonnes de la grille.
    Sortie : itérateur de tuples (x_start, y_start, x_end, y_end), ordre
    ligne par ligne puis colonne par colonne (comme iter_tile_windows).
    """
    xs = [round(c * img_w / cols) for c in range(cols + 1)]
    ys = [round(r * img_h / rows) for r in range(rows + 1)]
    for r in range(rows):
        for c in range(cols):
            yield xs[c], ys[r], xs[c + 1], ys[r + 1]
