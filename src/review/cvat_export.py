#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Écriture d'un lot au format d'import CVAT (Ultralytics YOLO
Segmentation : `images/<subset>/` + `labels/<subset>/` + `data.yaml`),
factorisé pour être appelé par plusieurs outils qui produisent chacun ce même
format à partir d'une source différente (une image de revue assistée dans
`assisted_annotate.py`, une orthomosaïque à découper dans `split_for_cvat.py`,
des prédictions du pipeline application dans `export_predictions_to_cvat.py`).

Contient : le calcul du nombre de morceaux nécessaire pour rester sous la
limite d'import CVAT (`CVAT_MAX_PIXELS`, importée depuis `split_for_cvat.py` -
SOURCE UNIQUE, jamais redéfinie ici), et le recadrage d'un polygone sur une
fenêtre de grille (intersection shapely + filtre d'aire résiduelle).

Point de vigilance, assumé plutôt que masqué : `assisted_annotate.py` a sa
PROPRE copie privée de ce calcul de morceaux et de ce recadrage
(`_resolve_n_pieces`/`_clip_one_to_window`), non encore migrée vers ce module
au moment de sa création - elle reste fonctionnellement équivalente (même
logique, même raisonnement, voir sa docstring), mais une 3e implémentation
divergente serait exactement le genre d'écart silencieux qui a causé le bug
CVAT_MAX_PIXELS (150M vs 50M, voir journal de décisions) : une migration de
`assisted_annotate.py` vers ce module reste un suivi possible, pas fait ici
pour ne pas toucher un outil de revue interactif déjà en place sans nécessité
directe.

Entrée : dimensions d'image/fenêtre, polygones (coordonnées PIXEL, repère de
l'image COMPLÈTE - jamais déjà normalisées).
Sortie : nombre de morceaux, fenêtres de grille, polygones recadrés, lot CVAT
écrit sur disque (images + labels + data.yaml).
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from shapely.geometry import MultiPolygon, Polygon, box

from src.data.utils.tiling_geometry import iter_grid_windows, min_pieces_for_pixel_cap, most_square_grid
from src.review.split_for_cvat import CVAT_MAX_PIXELS

# Ratio d'aire minimal conservé pour un fragment de polygone recadré au bord
# d'une fenêtre de grille - même valeur et même raisonnement que
# `assisted_annotate.GRID_MIN_AREA_RATIO` (voir sa docstring) : écarte un
# fragment résiduel négligeable plutôt que de polluer le label d'une fenêtre
# voisine avec un confetti sans valeur.
GRID_MIN_AREA_RATIO = 0.05


def resolve_n_pieces(img_w: int, img_h: int, n_pieces: Optional[int] = None,
                      max_pixels: int = CVAT_MAX_PIXELS) -> int:
    """Résout le nombre de morceaux à utiliser pour qu'une grille non
    chevauchante (voir `most_square_grid`) garde chaque morceau sous
    `max_pixels` pixels.

    `n_pieces=None` (défaut) : calcule le minimum requis
    (`min_pieces_for_pixel_cap`) - 1 si l'image entière tient déjà dedans.
    `n_pieces` fourni : REFUSE si insuffisant pour respecter le plafond,
    plutôt que d'écrire silencieusement un lot que CVAT rejettera à l'import.
    Ne refuse jamais une valeur plus grande que le minimum requis."""
    min_required = min_pieces_for_pixel_cap(img_w, img_h, max_pixels)
    total_px = img_w * img_h

    if n_pieces is None:
        if min_required > 1:
            print(f"    Image/fenêtre {img_w}x{img_h} = {total_px:,} px > limite CVAT "
                  f"({max_pixels:,} px) - découpage en {min_required} morceau(x) minimum.")
        return min_required

    if n_pieces < 1:
        raise RuntimeError(f"n_pieces doit être >= 1 (reçu {n_pieces}).")
    if n_pieces < min_required:
        raise RuntimeError(
            f"n_pieces={n_pieces} insuffisant ({img_w}x{img_h} = {total_px:,} px) : chaque morceau "
            f"ferait encore ~{total_px // n_pieces:,} px, au-dessus de la limite CVAT ({max_pixels:,} "
            f"px) - utilise au moins {min_required}."
        )
    return n_pieces


def clip_polygon_to_window(
    pixel_coords: List[Tuple[float, float]], class_id: int,
    x0: int, y0: int, x1: int, y1: int,
    min_area_ratio: float = GRID_MIN_AREA_RATIO,
) -> List[str]:
    """Recadre UN polygone (coordonnées PIXEL, repère image COMPLÈTE) sur une
    fenêtre de grille, retourne des lignes de label YOLO-seg (coordonnées
    normalisées LOCALES à cette fenêtre, `class_id x1 y1 x2 y2 ...\\n`).

    Même logique que `PlasticImageSlicer.slice_single_pair` (src/data/slicer.py)
    et `assisted_annotate._clip_one_to_window` : intersection shapely avec la
    fenêtre, filtre `min_area_ratio` pour écarter un fragment négligeable en
    bord de découpe (une grille SANS chevauchement ne rattrape pas ailleurs
    l'objet coupé - le fragment gardé est la seule trace de cet objet dans ce
    morceau). Une intersection multi-parties (objet à cheval sur un bord
    concave de la fenêtre, rare mais possible) produit une ligne de label par
    partie valide.

    Entrée : polygone source (liste de points pixel, image complète),
    identifiant de classe, bornes de la fenêtre.
    Sortie : liste de lignes de label (peut être vide si le polygone ne
    recoupe pas la fenêtre ou si le résidu est trop petit)."""
    if len(pixel_coords) < 3:
        return []
    geom = Polygon(pixel_coords)
    if not geom.is_valid or geom.area <= 0:
        return []

    window = box(x0, y0, x1, y1)
    if not window.intersects(geom):
        return []

    intersection = window.intersection(geom)
    if intersection.is_empty or intersection.area <= 0:
        return []
    if (intersection.area / geom.area) < min_area_ratio:
        return []

    if isinstance(intersection, Polygon):
        parts = [intersection]
    elif isinstance(intersection, MultiPolygon):
        parts = list(intersection.geoms)
    else:
        return []

    piece_w, piece_h = x1 - x0, y1 - y0
    lines: List[str] = []
    for part in parts:
        if part.area <= 0:
            continue
        local: List[float] = []
        for x, y in part.exterior.coords:
            local.extend([
                max(0.0, min(1.0, (x - x0) / piece_w)),
                max(0.0, min(1.0, (y - y0) / piece_h)),
            ])
        if len(local) >= 6:
            coords_str = " ".join(f"{c:.6f}" for c in local)
            lines.append(f"{class_id} {coords_str}\n")
    return lines


def write_grid_lot(
    lot_dir: Path,
    image_stem: str,
    img_w: int, img_h: int,
    detections: List[Tuple[List[Tuple[float, float]], int]],
    read_window_fn: Callable[[int, int, int, int], np.ndarray],
    names: Dict[int, str],
    n_pieces: Optional[int] = None,
) -> int:
    """Écrit un lot CVAT complet (images/train + labels/train + data.yaml) à
    partir d'une SEULE source raster (une photo batch ou une orthomosaïque),
    découpée en grille si nécessaire pour rester sous `CVAT_MAX_PIXELS`.

    `read_window_fn(x0, y0, x1, y1) -> np.ndarray BGR` abstrait la lecture de
    la source (ouverture PIL/cv2 d'une photo déjà sur disque, lecture
    `rasterio.windows.Window` d'un GeoTIFF) - le calcul de grille, le
    recadrage des polygones et l'écriture des fichiers sont ensuite
    identiques dans les deux cas, appelés une fois par morceau.

    `detections` : polygones PIXEL (repère de l'image COMPLÈTE, PAS encore
    recadrés) + leur class_id - typiquement lu depuis la propriété
    `pixel_polygon` de `detections.geojson` (voir
    `export_predictions_to_cvat.py`).

    Entrée : dossier du lot, nom de base (sans extension), dimensions de la
    source, détections, lecteur de fenêtre, noms de classe, nombre de
    morceaux souhaité (None = calculé automatiquement, voir `resolve_n_pieces`).
    Sortie : nombre de morceaux effectivement écrits (>= 1)."""
    n = resolve_n_pieces(img_w, img_h, n_pieces)
    img_dir = lot_dir / "images" / "train"
    lab_dir = lot_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    if n <= 1:
        windows = [(0, 0, img_w, img_h)]
    else:
        rows, cols = most_square_grid(n, img_w, img_h)
        windows = list(iter_grid_windows(img_w, img_h, rows, cols))

    for (x0, y0, x1, y1) in windows:
        piece_name = image_stem if n <= 1 else f"{image_stem}_y{y0}_x{x0}"
        piece_img = read_window_fn(x0, y0, x1, y1)
        cv2.imwrite(str(img_dir / f"{piece_name}.png"), piece_img)

        lines: List[str] = []
        for pixel_polygon, class_id in detections:
            lines.extend(clip_polygon_to_window(pixel_polygon, class_id, x0, y0, x1, y1))
        (lab_dir / f"{piece_name}.txt").write_text("".join(lines), encoding="utf-8")

    write_data_yaml(lot_dir, names)
    return len(windows)


def write_data_yaml(lot_dir: Path, names: Dict[int, str]) -> None:
    """Écrit `data.yaml` à la racine du lot - même convention que
    `assisted_annotate._init_lot` : `path`/`train` ne sont pas lus par le
    pipeline d'ingestion PixelOdyssey (seule `names` l'est) mais sont requis
    par l'import CVAT pour localiser les images associées aux labels. Un seul
    "split" (`train`, nom arbitraire) : un lot d'export n'a pas de notion
    train/val/test."""
    data_yaml_path = lot_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "names": {int(k): v for k, v in names.items()},
            },
            f, allow_unicode=True, sort_keys=False,
        )
