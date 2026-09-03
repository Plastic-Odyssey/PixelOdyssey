#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Inférence par fenêtre glissante sur une image PARENTE.

Le modèle est entraîné sur des tuiles 640x640 (4_sliced_dataset) : le faire
tourner directement sur une orthomosaïque/image parente entière donnerait de
mauvaises prédictions (objets bien trop petits dans le champ de vue du
réseau). Ce module répète donc EXACTEMENT le même découpage que
l'entraînement (voir src/data/tiling_geometry.py - source de vérité commune),
fait tourner le modèle tuile par tuile, recolle chaque prédiction dans les
coordonnées de l'image parente, puis fusionne les doublons créés par les
zones de recouvrement entre tuiles adjacentes (une même perle de déchet peut
être vue - et détectée - par deux tuiles qui se chevauchent).

La fonction de prédiction par tuile (`predict_tile_fn`) est injectée plutôt
que codée en dur sur l'API Ultralytics : ça permet de tester toute la
géométrie (fenêtrage, recollage, fusion) avec un faux prédicteur, sans modèle
ni poids réels - voir `make_ultralytics_predict_fn` pour le branchement réel.
"""

from pathlib import Path
from typing import Callable, List, Union

import numpy as np
from shapely.geometry import Polygon

from src.data.image_io import load_image_bgr
from src.data.tiling_geometry import iter_tile_windows
from src.review.matching import LabeledPolygon

# Une prédiction de tuile brute : (class_id, confidence, polygon_xy_tile_local_pixels)
TilePrediction = tuple  # (int, float, List[Tuple[float, float]])
PredictTileFn = Callable[[np.ndarray], List[TilePrediction]]


def make_ultralytics_predict_fn(model_path: Union[str, Path], conf_threshold: float = 0.25) -> PredictTileFn:
    """Construit une fonction de prédiction par tuile à partir d'un modèle
    Ultralytics YOLO-seg entraîné (best.pt). `conf_threshold` ici est un
    filtre de PREMIER NIVEAU large (25% par défaut) - le seuil de confiance
    métier utilisé pour décider "annotation potentiellement oubliée" (60% par
    défaut, voir label_review.py) est appliqué APRÈS la fusion inter-tuiles,
    pas ici, pour ne pas perdre une détection dont la confiance grimperait
    après fusion de deux tuiles qui la voient chacune partiellement.

    La fonction retournée porte un attribut `.model_names` (dict id -> nom, tel
    qu'embarqué dans le modèle au moment de son entraînement) - à vérifier contre
    la taxonomie ACTUELLE avant d'utiliser les class_id prédits pour quoi que ce
    soit (voir `class_config.assert_model_matches_taxonomy`) : un modèle plus
    ancien peut avoir été entraîné avec des ID de classe différents.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model_names = {int(k): str(v) for k, v in model.names.items()}

    def predict_tile(tile_img: np.ndarray) -> List[TilePrediction]:
        results = model.predict(tile_img, conf=conf_threshold, verbose=False)
        out: List[TilePrediction] = []
        if not results:
            return out
        result = results[0]
        if result.masks is None:
            return out
        classes = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
        # .xy (pas .xyn) : coordonnées en PIXELS de la tuile, pas normalisées -
        # on a besoin des pixels tuile pour les recaler ensuite dans l'image parente.
        polygons_xy = result.masks.xy
        for cls, conf, poly_xy in zip(classes, confidences, polygons_xy):
            out.append((int(cls), float(conf), [(float(x), float(y)) for x, y in poly_xy]))
        return out

    predict_tile.model_names = model_names
    return predict_tile


def nms_merge(predictions: List[LabeledPolygon], iou_threshold: float = 0.5) -> List[LabeledPolygon]:
    """Fusionne les détections dupliquées dans les zones de recouvrement entre
    tuiles adjacentes : deux prédictions de MÊME classe avec une IoU élevée
    dans l'espace de l'image parente sont considérées comme le même objet vu
    deux fois - on garde seulement celle de plus haute confiance.

    Public (pas de `_`) : réutilisé aussi par le futur module d'inférence
    géo-consciente sur orthomosaïque complète (src/review/geo_density_map.py) -
    la logique de fusion des recouvrements de tuiles est
    identique, que l'image source soit chargée entièrement en mémoire
    (predict_parent_image ci-dessous) ou lue fenêtre par fenêtre via rasterio
    pour une orthomosaïque trop grande pour tenir en RAM.
    """
    from src.review.matching import polygon_iou

    by_class: dict = {}
    for p in predictions:
        by_class.setdefault(p.class_id, []).append(p)

    kept: List[LabeledPolygon] = []
    for class_id, items in by_class.items():
        items = sorted(items, key=lambda p: p.confidence or 0.0, reverse=True)
        taken = [False] * len(items)
        for i, p in enumerate(items):
            if taken[i]:
                continue
            kept.append(p)
            for j in range(i + 1, len(items)):
                if taken[j]:
                    continue
                if polygon_iou(p.geom, items[j].geom) >= iou_threshold:
                    taken[j] = True
    return kept


def predict_parent_image(
    img_path: Union[str, Path],
    predict_tile_fn: PredictTileFn,
    tile_size: int = 640,
    overlap: int = 256,
    nms_iou_threshold: float = 0.5,
) -> List[LabeledPolygon]:
    """Fait tourner `predict_tile_fn` sur chaque fenêtre de l'image parente
    (même géométrie que l'entraînement), recolle les polygones en coordonnées
    image, fusionne les doublons de recouvrement, et retourne la liste finale
    de détections pour CETTE image parente entière.
    """
    img = load_image_bgr(img_path)
    if img is None:
        raise RuntimeError(f"Impossible de charger l'image : {img_path}")
    img_h, img_w = img.shape[:2]
    stride = tile_size - overlap

    all_predictions: List[LabeledPolygon] = []

    # Cas d'une image déjà au format tuile (pas de fenêtrage nécessaire).
    windows = (
        [(0, 0, img_w, img_h)]
        if img_h == tile_size and img_w == tile_size
        else list(iter_tile_windows(img_w, img_h, tile_size, stride))
    )

    for x_start, y_start, x_end, y_end in windows:
        tile_img = img[y_start:y_end, x_start:x_end]
        for class_id, confidence, poly_xy_tile in predict_tile_fn(tile_img):
            if len(poly_xy_tile) < 3:
                continue
            # Recollage tuile -> image parente : simple translation, aucune
            # remise à l'échelle nécessaire puisque la tuile est déjà à la
            # résolution native (pas de resize dans predict_tile_fn attendu).
            poly_xy_img = [(x + x_start, y + y_start) for x, y in poly_xy_tile]
            geom = Polygon(poly_xy_img)
            if not geom.is_valid or geom.area <= 0:
                continue
            all_predictions.append(LabeledPolygon(class_id=class_id, geom=geom, confidence=confidence))

    return nms_merge(all_predictions, iou_threshold=nms_iou_threshold)
