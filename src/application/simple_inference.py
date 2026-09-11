#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Inférence simple sur une image UNIQUE isolée (jpg ou tif).

Rôle : un utilitaire volontairement minimal pour tester rapidement un modèle
sur UNE image (photo drone brute isolée, extrait recadré d'orthomosaïque,
capture quelconque) sans passer par le pipeline complet de
`run_inference.py` (qui exige un DOSSIER en mode batch ou un GeoTIFF
géoréférencé en mode orthomosaïque - voir sa docstring). Ce module ne
remplace ni l'un ni l'autre mode : il répond à un besoin différent, ponctuel
et exploratoire (« qu'est-ce que ce modèle voit sur CETTE image précise ? »),
pas à une session de collecte terrain complète.

Ce qui est RÉUTILISÉ du pipeline application (rien n'est réimplémenté) :
  - `model_registry.py` : résolution du modèle par nom depuis
    config/models_registry.yaml, comme run_inference.py.
  - `src.data.class_config` : référentiel de classes du modèle choisi
    (`model.taxonomy_config`, avec repli sur le référentiel 7-classes par
    défaut) + garde-fou `assert_model_matches_taxonomy` - même vérification
    que partout ailleurs dans src/review/.
  - `src.data.image_io.load_image_bgr` : lit indifféremment un .jpg ou un
    .tif/.tiff (bascule automatique sur rasterio pour le TIF).
  - `src.review.tiled_inference.predict_parent_image` : même géométrie de
    tuilage qu'à l'entraînement (voir tiling_geometry.py), MÊME fusion des
    recouvrements entre tuiles adjacentes (`nms_merge`, indexée
    spatialement - voir sa docstring) que dans les deux modes existants et
    que visualize_predictions.py. Cette fusion N'EST PAS le dédoublonnage
    volontairement exclu ici (voir plus bas) : elle est purement
    géométrique (une même détection vue deux fois par deux tuiles qui se
    chevauchent DANS CETTE IMAGE), nécessaire dès que l'image dépasse
    tile_size, qu'il y ait une ou mille images en entrée.
  - `src.application.stats_panel.compute_aggregate_stats`/`render_stats_html` :
    mêmes statistiques agrégées (nombre de détections, surface, poids, par
    classe) que les deux cartes web du pipeline application - voir docstring
    de stats_panel.py.

Ce qui est délibérément EXCLU (décisions prises avec Jame) :
  - Le dédoublonnage INTER-PHOTOS (`dedup.py`, mode batch uniquement) : n'a
    de sens que sur plusieurs photos qui se recouvrent dans l'espace réel -
    une image unique n'a par définition rien avec quoi être dédupliquée.
  - Toute géolocalisation/géoréférencement, MÊME quand l'image fournie est
    un TIF géoréférencé (GeoTIFF avec CRS projeté en mètres) : décision
    prise le 2026-09-09 de toujours ignorer ce cas plutôt que de réutiliser
    `geo_density_map._pixel_area_m2` de façon opportuniste - un comportement
    unique quel que soit le fichier (jpg ou tif, géoréférencé ou non) a été
    préféré à une fonction à deux vitesses, pour rester fidèle à l'esprit
    « toute simple » de cet utilitaire. Conséquence directe et volontaire :
    `area_m2`/`weight_kg` valent TOUJOURS None dans les enregistrements
    produits ici - le panneau de stats affiche donc systématiquement
    « surface non calculable » plutôt qu'un 0.00 m² trompeur (voir le
    correctif apporté à stats_panel.py le même jour, dont ce module est le
    premier appelant à dépendre réellement du nouveau cas
    `n_area_not_estimable == n_detections`).

Sortie : l'image annotée (tableau numpy BGR, ET sauvegardée sur disque à
côté de l'image source par défaut), la liste des détections, et les
statistiques agrégées (dict + fragment HTML), sans écrire nulle part dans
4. Results/ ni dans detections.geojson - un usage ponctuel n'a pas besoin de
la structure d'un run complet.

Exemple :
    from src.application.simple_inference import run_simple_inference
    result = run_simple_inference("photo_isolee.jpg", model_name="mono_class_v1")
    print(result["stats"]["n_detections"], "détection(s)")
    # result["annotated_image"] : tableau numpy BGR, déjà écrit dans
    # result["annotated_image_path"] (photo_isolee_annotated.jpg par défaut)

    python -m src.application.simple_inference --image photo_isolee.jpg --model mono_class_v1
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.application.model_registry import load_operational_models, resolve_model_choice
from src.application.stats_panel import compute_aggregate_stats, render_stats_html
from src.data.utils.class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    assert_model_matches_taxonomy,
    load_class_config,
)
from src.data.utils.image_io import load_image_bgr
from src.review.tiled_inference import make_ultralytics_predict_fn, predict_parent_image

# Une couleur par ID de classe (cycle si plus de classes que de couleurs) -
# PAS la palette à 4 couleurs de visualize_predictions.py (qui encode un
# STATUT GT/prédiction, sans objet ici : il n'y a pas de vérité terrain à
# comparer, seulement des prédictions à afficher).
_PALETTE: List[Tuple[int, int, int]] = [
    (60, 180, 60), (230, 130, 20), (30, 30, 220), (200, 160, 20),
    (180, 40, 180), (0, 200, 200), (140, 90, 40),
]


def _color_for_class(class_id: int) -> Tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _draw_predictions(img: np.ndarray, predictions, target_names: Dict[int, str]) -> np.ndarray:
    """Dessine chaque prédiction (polygone rempli semi-transparent + contour
    + étiquette classe/confiance) sur une COPIE de `img` (jamais l'original)."""
    out = img.copy()
    for p in predictions:
        color = _color_for_class(p.class_id)
        coords = [(int(round(x)), int(round(y))) for x, y in p.geom.exterior.coords]
        pts = np.array(coords, dtype=np.int32).reshape(-1, 1, 2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, dst=out)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)

        name = target_names.get(p.class_id, str(p.class_id))
        conf = p.confidence if p.confidence is not None else 0.0
        label = f"{name} {conf:.2f}"
        x, y = coords[0]
        org = (max(0, x), max(14, y - 8))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out


def run_simple_inference(
    image_path: Union[str, Path],
    model_name: Optional[str] = None,
    conf_threshold: float = 0.25,
    tile_size: int = 640,
    overlap: int = 256,
    output_path: Optional[Union[str, Path]] = None,
    predict_tile_fn=None,
) -> Dict:
    """Lance l'inférence sur UNE image isolée et retourne l'image annotée +
    les statistiques - voir docstring du module pour ce qui est réutilisé et
    ce qui est délibérément exclu.

    `model_name` : nom exact dans config/models_registry.yaml (invite
    interactive si omis et plusieurs modèles déclarés - voir
    model_registry.resolve_model_choice). Ignoré si `predict_tile_fn` est
    fourni (voir plus bas) - `model_name` sert alors uniquement à choisir le
    référentiel de classes (`model.taxonomy_config`) si un modèle de ce nom
    existe dans le registre, sinon repli sur le 7-classes par défaut.
    `output_path` : chemin de sauvegarde de l'image annotée (défaut :
    <image_path>_annotated.jpg, à côté de la source).
    `predict_tile_fn` : même rôle qu'en tiled_inference.py/
    visualize_predictions.py - permet de tester toute la géométrie
    (tuilage, recollage, fusion, dessin, stats) avec un faux prédicteur, sans
    modèle ni poids réels. Si fourni, `conf_threshold` n'est PAS appliqué ici
    (c'est la responsabilité de `predict_tile_fn` lui-même) et aucun modèle
    n'est chargé depuis le registre pour l'inférence (seul son
    `taxonomy_config` est encore utilisé, si `model_name` est fourni et
    résolu avec succès).

    Retourne un dict :
      - annotated_image : tableau numpy BGR (l'image annotée).
      - annotated_image_path : chemin où elle a été sauvegardée.
      - detections : liste de dicts {class_name, confidence, area_m2 (None),
        weight_kg (None)} - voir docstring du module pour pourquoi ces deux
        derniers champs sont toujours None ici.
      - stats / stats_html : sortie de stats_panel.compute_aggregate_stats /
        render_stats_html sur ces détections.
      - model_name : nom du modèle effectivement utilisé (utile quand
        `model_name` était omis et résolu par défaut/invite interactive,
        ou None si `predict_tile_fn` était injecté sans `model_name`).
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    img = load_image_bgr(image_path)
    if img is None:
        raise RuntimeError(
            f"Impossible de charger l'image : {image_path} (jpg ou tif/tiff attendu)."
        )

    resolved_model_name = model_name
    class_config_path = DEFAULT_CLASS_CONFIG_PATH

    if predict_tile_fn is None:
        models = load_operational_models()
        model = resolve_model_choice(models, requested_name=model_name)
        resolved_model_name = model.name
        # Même repli que geo_density_map.py/run_inference.py : le référentiel
        # de classes déclaré pour CE modèle dans le registre, sinon le
        # 7-classes par défaut - jamais codé en dur, voir class_config.py.
        class_config_path = model.taxonomy_config or DEFAULT_CLASS_CONFIG_PATH
        predict_tile_fn = make_ultralytics_predict_fn(model.weights_path, conf_threshold=conf_threshold)
    elif model_name is not None:
        # predict_tile_fn injecté (test), mais un nom de modèle est quand
        # même fourni : on récupère seulement son taxonomy_config si ce nom
        # existe dans le registre (permet de tester la résolution de
        # taxonomie sans charger de vrais poids) - jamais bloquant si le nom
        # est inconnu du registre en contexte de test.
        try:
            models = load_operational_models()
            model = resolve_model_choice(models, requested_name=model_name)
            class_config_path = model.taxonomy_config or DEFAULT_CLASS_CONFIG_PATH
        except (FileNotFoundError, ValueError):
            pass

    class_taxonomy, target_names = load_class_config(class_config_path)

    model_names = getattr(predict_tile_fn, "model_names", None)
    if model_names is not None:
        assert_model_matches_taxonomy(model_names, target_names, model_label=str(resolved_model_name or ""))

    predictions = predict_parent_image(
        image_path, predict_tile_fn, tile_size=tile_size, overlap=overlap
    )

    annotated = _draw_predictions(img, predictions, target_names)

    records: List[Dict] = []
    for p in predictions:
        records.append({
            "class_name": target_names.get(p.class_id, str(p.class_id)),
            "confidence": p.confidence,
            # Toujours None ici - voir docstring du module (décision du
            # 2026-09-09 : pas de géoréférencement, même opportuniste).
            "area_m2": None,
            "weight_kg": None,
        })
    stats = compute_aggregate_stats(records)

    if output_path is not None:
        out_p = Path(output_path)
    else:
        out_p = image_path.with_name(f"{image_path.stem}_annotated.jpg")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "annotated_image": annotated,
        "annotated_image_path": str(out_p),
        "detections": records,
        "stats": stats,
        "stats_html": render_stats_html(stats),
        "model_name": resolved_model_name,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inférence simple sur une image unique (jpg ou tif isolé)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Chemin vers l'image (jpg ou tif).")
    parser.add_argument(
        "--model", default=None,
        help="Nom du modèle dans config/models_registry.yaml (invite interactive si omis "
             "et plusieurs modèles disponibles - voir model_registry.py).",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument(
        "--output", default=None,
        help="Chemin de sauvegarde de l'image annotée (défaut : <image>_annotated.jpg).",
    )
    args = parser.parse_args()

    result = run_simple_inference(
        image_path=args.image,
        model_name=args.model,
        conf_threshold=args.conf_threshold,
        tile_size=args.tile_size,
        overlap=args.overlap,
        output_path=args.output,
    )

    stats = result["stats"]
    total_area_str = (
        "non calculable (pas de géoréférencement - voir docstring du module)"
        if stats["total_area_m2"] is None
        else f"{stats['total_area_m2']:.2f} m²"
    )
    total_weight_str = (
        f"{stats['total_weight_kg']:.2f} kg" if stats["total_weight_kg"] else "0 kg"
    )

    print(f"--- 🔎 Inférence simple sur {args.image} (modèle : {result['model_name']}) ---")
    print(f"  • Détections : {stats['n_detections']}")
    print(f"  • Surface totale : {total_area_str}")
    print(f"  • Poids total estimé : {total_weight_str}"
          + ("" if stats["calibrated"] else " (formule PROVISOIRE, non calibrée)"))
    for name, entry in stats["per_class"].items():
        area_str = f"{entry['area_m2']:.2f} m²" if entry["area_m2"] is not None else "non calculable"
        weight_str = f"{entry['weight_kg']:.2f} kg" if entry["weight_kg"] is not None else "non estimable"
        print(f"      - {name} : {entry['n']} | surface {area_str} | poids {weight_str}")
    print(f"  • Image annotée : {result['annotated_image_path']}")


if __name__ == "__main__":
    main()
