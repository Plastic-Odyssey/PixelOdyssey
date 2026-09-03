#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Appariement GT <-> prédictions par classe + IoU.

Décide, pour chaque objet prédit par le modèle sur une image parente, s'il
correspond à une annotation existante (GT) et à quel point la correspondance
est bonne (IoU). Utilisé par label_review.py comme base des 3 paniers de
triage :
  - prédiction sans GT correspondant (au-delà du seuil de confiance)
    -> annotation potentiellement oubliée.
  - GT et prédiction appariées mais IoU faible -> masque potentiellement
    à retoucher.
  - GT sans prédiction correspondante -> pour information seulement (pas
    de proposition à faire, rien à injecter).

Algorithme glouton (pas d'appariement optimal type Hongrois) : pour chaque
classe, les paires (GT, prédiction) qui se recoupent sont triées par IoU
décroissante et acceptées tant que ni la GT ni la prédiction n'ont déjà été
prises. Suffisant car les objets annotés sur une image de déchets ne se
chevauchent quasiment jamais au point de créer une ambiguïté d'appariement.

Entrée : listes de `LabeledPolygon` (GT et prédictions), seuil `min_iou`.
Sortie : liste de `MatchResult` (appariements et non-appariements des deux côtés).

Exemple :
    from src.review.matching import match_gt_to_predictions, LabeledPolygon
    results = match_gt_to_predictions(gt_objects, predictions, min_iou=0.1)
    for r in results:
        if r.gt is not None and r.pred is None:
            ...  # annotation potentiellement oubliée
"""

from typing import Dict, List, NamedTuple, Optional

from shapely.geometry import Polygon


class LabeledPolygon(NamedTuple):
    """Un objet (GT ou prédiction) avec sa classe et sa géométrie en coordonnées
    de l'image PARENTE (pas de tuile) - voir tiled_inference.py pour comment on
    y arrive côté prédictions."""
    class_id: int
    geom: Polygon
    confidence: Optional[float] = None  # None pour un GT (n'a pas de confiance)


class MatchResult(NamedTuple):
    gt: Optional[LabeledPolygon]
    pred: Optional[LabeledPolygon]
    iou: float  # 0.0 si gt ou pred est None (rien à comparer)


def polygon_iou(a: Polygon, b: Polygon) -> float:
    if not a.is_valid or not b.is_valid or a.area <= 0 or b.area <= 0:
        return 0.0
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def match_gt_to_predictions(
    gt_objects: List[LabeledPolygon],
    predictions: List[LabeledPolygon],
    min_iou: float = 0.1,
) -> List[MatchResult]:
    """Retourne la liste complète des résultats d'appariement :
    - une entrée par paire (gt, pred) appariée (iou >= min_iou) ;
    - une entrée par GT non appariée (pred=None, iou=0.0) ;
    - une entrée par prédiction non appariée (gt=None, iou=0.0).

    Deux objets de classes différentes ne sont JAMAIS candidats à un
    appariement, même si leurs géométries se recoupent - cf. label_review.py,
    qui a besoin de savoir si le modèle a bien reconnu la même famille
    d'objet, pas seulement "quelque chose au même endroit".

    `min_iou` : en dessous de ce seuil, un léger effleurement fortuit entre
    deux géométries n'est pas considéré comme "le même objet" - la GT et la
    prédiction sont alors traitées comme deux entrées non appariées
    distinctes (plutôt qu'une paire appariée à IoU quasi nulle, qui se
    confondrait à tort avec un vrai cas de masque à retoucher - panier C dans
    label_review.py).
    """
    candidates = []  # (iou, gt_idx, pred_idx)
    for gi, gt in enumerate(gt_objects):
        for pi, pred in enumerate(predictions):
            if gt.class_id != pred.class_id:
                continue
            iou = polygon_iou(gt.geom, pred.geom)
            if iou >= min_iou:
                candidates.append((iou, gi, pi))

    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gt: Dict[int, int] = {}   # gt_idx -> pred_idx
    matched_pred: Dict[int, int] = {}  # pred_idx -> gt_idx
    results: List[MatchResult] = []

    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt[gi] = pi
        matched_pred[pi] = gi
        results.append(MatchResult(gt=gt_objects[gi], pred=predictions[pi], iou=iou))

    for gi, gt in enumerate(gt_objects):
        if gi not in matched_gt:
            results.append(MatchResult(gt=gt, pred=None, iou=0.0))

    for pi, pred in enumerate(predictions):
        if pi not in matched_pred:
            results.append(MatchResult(gt=None, pred=pred, iou=0.0))

    return results
