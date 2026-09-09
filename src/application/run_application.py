#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Pipeline "application" : orchestrateur de bout en bout.

Entrée : un dossier de photos drone brutes (jpg/png, recouvrement de vol
~80%), un capteur (auto-détecté via EXIF Make/Model - voir sensor_config.py),
un modèle opérationnel (registre config/models_registry.yaml), un seuil de
confiance final.

Étapes :
1. Inventaire + validation du batch (inventory.py) - géolocalisation directe
   NAÏVE (GPS+altitude+cap+GSD), PAS une orthorectification WebODM - compromis
   accepté explicitement pour ce cas d'usage (carte de densité à l'échelle
   d'une plage).
2. Inférence tuilée par photo (réutilise tiled_inference.py::predict_parent_image,
   qui gère DÉJÀ le dédoublonnage INTRA-photo des tuiles 640px - rien à
   reconstruire ici pour ce niveau).
3. Reprojection de chaque détection en coordonnées sol (mètres) via
   geolocation.py.
4. Dédoublonnage INTER-photos (dedup.py) : IoU géométrique toujours actif,
   plus une composante optionnelle de similarité visuelle (voir
   `--dedup-method` plus bas). Les imagettes de TOUTES les détections
   brutes sont générées EN MÉMOIRE avant le dédoublonnage (pas seulement
   celles des gagnantes), car la similarité visuelle se calcule dessus ;
   seules les imagettes des détections finalement retenues sont écrites
   sur disque (voir `_build_crop_images`/`_save_winner_crops`).
5. Seuil de confiance final appliqué APRÈS le dédoublonnage (pas avant) :
   l'inférence tuilée utilise un seuil large en 1ère passe pour ne pas
   perdre une détection dont la version la plus complète, ailleurs dans
   le batch, aurait une meilleure confiance.
6. Export GeoJSON (une Feature par détection retenue, ouvrable dans QGIS -
   voir guide_mesure_surface_bache_qgis.md) + imagettes JPEG avec contour de
   masque (mêmes conventions visuelles que
   `geo_density_map.build_detection_crops`, adaptées à des photos brutes
   plutôt qu'à un GeoTIFF). Chaque détection retenue reçoit une surface au
   sol (`area_m2`, aire de `local_polygon` - déjà en mètres réels via
   geolocation.py, pas de GSD séparé nécessaire ici contrairement au mode
   orthomosaïque) et un poids estimé (`weight_kg`, voir
   weight_estimation.py - ESTIMATION PROVISOIRE, aucune table de conversion
   calibrée n'existe encore, question ouverte), pour alimenter le panneau
   de statistiques agrégées de la carte web (voir web_map.py/stats_panel.py).

Rendu carte interactive : voir `src/application/web_map.py`, qui réutilise
`src/review/geo_density_map.py` (GeoJSON + crops produits ci-dessus).

`--dedup-method` - question ouverte, tranchée pour l'instant en faveur du
100% local :
- `iou` (défaut) : géométrie seule, 100% local, aucun appel réseau - mais sur
  données réelles ne fusionne quasiment aucun doublon (bruit de
  géoréférencement direct de la géolocalisation NAÏVE du point 1) : le
  dédoublonnage inter-photos reste un problème ouvert avec ce choix,
  accepté explicitement pour privilégier l'exécution terrain 100% locale.
- `visual` : IoU + similarité visuelle (DINOv2 via `transformers`/HuggingFace,
  voir dedup.py::deduplicate_detections_visual). Résout mieux le
  dédoublonnage mais dépend d'un appel réseau (même modèle déjà en cache)
  qui ralentit trop l'exécution en conditions de terrain (connexion
  faible/absente), sauf à imposer `HF_HUB_OFFLINE=1` - alternative écartée
  au profit d'une exécution garantie 100% locale. Reste disponible en
  opt-in pour un contexte avec connexion fiable.

Point d'entrée recommandé pour un usage normal : `run_inference.py` (pas ce
module directement) - il détecte automatiquement le mode batch/orthomosaïque
depuis l'entrée fournie et enchaîne, après la carte, les exports additifs
(stats .xlsx, lot CVAT compressé) que CE module seul ne produit pas. Cette
CLI directe reste utile pour un diagnostic ciblé sur le mode batch seul.

Exemple :
    python -m src.application.run_application --batch-dir "chemin/vers/photos" \\
        --model mono_class_v1 --conf-threshold 0.3 --output-dir "E:\\PixelOdyssey\\4. Results\\2_prediction\\essai1"
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, mapping

from src.application.dedup import PhotoDetection, deduplicate_detections, deduplicate_detections_visual
from src.application.geolocation import (
    local_xy_to_lonlat,
    pixel_polygon_to_local_polygon,
)
from src.application.inventory import scan_batch
from src.application.model_registry import load_operational_models, resolve_model_choice
from src.application.paths import DEFAULT_PREDICTION_DIR
from src.application.weight_estimation import estimate_weight_kg
from src.data.class_config import DEFAULT_CLASS_CONFIG_PATH, assert_model_matches_taxonomy, load_class_config
from src.review.tiled_inference import make_ultralytics_predict_fn, predict_parent_image

# Seuil de confiance "large" pour la 1ère passe d'inférence tuilée - PAS le
# seuil métier final (voir docstring du module : le seuil final s'applique
# après dédoublonnage, pas ici). Plus bas que le défaut générique de
# `make_ultralytics_predict_fn` (0.25) car ici on a explicitement besoin de
# garder les détections partielles/basse confiance jusqu'à ce que le
# dédoublonnage ait pu comparer toutes les vues d'un même objet.
DEFAULT_TILE_CONF_THRESHOLD = 0.10

# Seuil IoU (au sol) de clustering inter-photos - pas encore calibré
# empiriquement (voir dedup.py), exposé en CLI pour faciliter les essais.
DEFAULT_DEDUP_IOU_THRESHOLD = 0.3
# Idem pour la similarité visuelle et le rayon de recherche de candidats -
# voir dedup.py::deduplicate_detections_visual pour le détail de calibration
# (un seul exemple manuel à ce stade, pas une valeur éprouvée).
DEFAULT_DEDUP_SIMILARITY_THRESHOLD = 0.5
DEFAULT_DEDUP_DISTANCE_THRESHOLD_M = 2.0

CROP_MARGIN_PX = 60
CROP_TARGET_DIM = 300
CROP_OUTLINE_COLOR = (255, 235, 0)  # jaune vif, même convention que geo_density_map.py


def _build_detections_for_photo(
    entry, predict_tile_fn, origin: Tuple[float, float], nms_iou_threshold: float,
) -> List[PhotoDetection]:
    """Inférence tuilée sur UNE photo (dédoublonnage intra-photo déjà géré
    par `predict_parent_image`) puis reprojection au sol de chaque détection."""
    labeled_polygons = predict_parent_image(
        entry.metadata.path, predict_tile_fn, nms_iou_threshold=nms_iou_threshold,
    )
    detections = []
    for lp in labeled_polygons:
        local_poly = pixel_polygon_to_local_polygon(
            entry.metadata, entry.sensor, origin, list(lp.geom.exterior.coords)
        )
        detections.append(PhotoDetection(
            photo_id=entry.metadata.path,
            class_id=lp.class_id,
            confidence=lp.confidence,
            pixel_polygon=lp.geom,
            local_polygon=local_poly,
            photo_width_px=entry.metadata.width_px,
            photo_height_px=entry.metadata.height_px,
        ))
    return detections


def _make_crop_image(img: Image.Image, img_w: int, img_h: int, det: PhotoDetection) -> Image.Image:
    """Construit UNE imagette (contour du masque dessiné, agrandie si besoin)
    à partir d'une photo déjà ouverte - factorisé hors de
    `_build_crop_images` pour être réutilisable détection par détection."""
    minx, miny, maxx, maxy = det.pixel_polygon.bounds
    x0 = max(0, int(minx) - CROP_MARGIN_PX)
    y0 = max(0, int(miny) - CROP_MARGIN_PX)
    x1 = min(img_w, int(maxx) + CROP_MARGIN_PX)
    y1 = min(img_h, int(maxy) + CROP_MARGIN_PX)
    crop = img.crop((x0, y0, x1, y1)).copy()

    draw = ImageDraw.Draw(crop)
    local_coords = [(x - x0, y - y0) for x, y in det.pixel_polygon.exterior.coords]
    if len(local_coords) >= 2:
        draw.line(local_coords, fill=CROP_OUTLINE_COLOR, width=2, joint="curve")

    scale = CROP_TARGET_DIM / max(crop.width, crop.height, 1)
    if scale > 1.0:
        crop = crop.resize(
            (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS
        )
    return crop


def _build_crop_images(detections: List[PhotoDetection]) -> List[Image.Image]:
    """Génère une imagette EN MÉMOIRE (pas encore écrite sur disque) pour
    CHAQUE détection brute passée en entrée - nécessaire car le dédoublonnage
    par similarité visuelle a besoin d'un chip pour TOUTES les vues
    candidates, pas seulement pour les gagnantes (qu'on ne connaît qu'après).
    Regroupe par photo source pour n'ouvrir chaque photo qu'une seule fois.

    Point de vigilance mémoire (pas encore rencontré en pratique, à surveiller
    si un batch produit beaucoup plus que quelques milliers de détections
    brutes) : toutes ces imagettes restent en mémoire simultanément jusqu'à la
    fin du dédoublonnage - si ça devient un problème, la parade serait de
    calculer les embeddings (voir visual_similarity.py) photo par photo, au
    fil de l'eau, plutôt que de garder toutes les images PIL."""
    by_photo: Dict[str, List[int]] = defaultdict(list)
    for i, d in enumerate(detections):
        by_photo[d.photo_id].append(i)

    images: List[Optional[Image.Image]] = [None] * len(detections)
    for photo_path, idxs in by_photo.items():
        img = Image.open(photo_path).convert("RGB")
        img_w, img_h = img.size
        for i in idxs:
            images[i] = _make_crop_image(img, img_w, img_h, detections[i])
    return images


def _save_winner_crops(winners: List[PhotoDetection], winner_images: List[Image.Image], output_dir: Path) -> Dict[int, str]:
    """Écrit sur disque les imagettes déjà construites en mémoire des
    détections FINALEMENT retenues (voir `_build_crop_images`) - ne rouvre
    aucune photo source, puisque l'image existe déjà."""
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    filenames: Dict[int, str] = {}
    for i, (det, crop) in enumerate(zip(winners, winner_images)):
        filename = f"det_{i:05d}_class{det.class_id}.jpg"
        crop.save(crops_dir / filename, format="JPEG", quality=85, optimize=True)
        filenames[i] = f"crops/{filename}"
    return filenames


def run_application(
    batch_dir: str,
    model_name: str = None,
    conf_threshold: float = 0.25,
    tile_conf_threshold: float = DEFAULT_TILE_CONF_THRESHOLD,
    dedup_method: str = "iou",
    dedup_iou_threshold: float = DEFAULT_DEDUP_IOU_THRESHOLD,
    dedup_similarity_threshold: float = DEFAULT_DEDUP_SIMILARITY_THRESHOLD,
    dedup_distance_threshold_m: float = DEFAULT_DEDUP_DISTANCE_THRESHOLD_M,
    nms_iou_threshold: float = 0.5,
    output_dir: str = None,
) -> Path:
    if dedup_method not in ("iou", "visual"):
        raise ValueError(f"dedup_method doit être 'iou' ou 'visual', reçu {dedup_method!r}.")
    if output_dir is None:
        # Sortie dans `4. Results/2_prediction/`, partagée avec le mode
        # orthomosaïque (voir paths.py). Préfixe "batch_" pour rester
        # identifiable dans le dossier commun, sans sous-dossier dédié par mode.
        output_dir = f"{DEFAULT_PREDICTION_DIR}/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- 📦 1/5 - Inventaire du batch ---")
    entries, inv_report = scan_batch(batch_dir)
    print(f"    {inv_report.n_usable}/{inv_report.n_total_files} photo(s) exploitable(s) "
          f"({inv_report.n_skipped_metadata_error} exclue(s) métadonnées, "
          f"{inv_report.n_skipped_pitch_tolerance} exclue(s) tangage hors tolérance)")
    for path, reason in inv_report.skipped_details:
        print(f"    ⚠️  exclue : {path} - {reason}")

    print("--- 🧠 2/5 - Résolution du modèle ---")
    models = load_operational_models()
    model = resolve_model_choice(models, requested_name=model_name)
    print(f"    Modèle choisi : {model.name} ({model.weights_path})")

    class_taxonomy, target_names = load_class_config(model.taxonomy_config or DEFAULT_CLASS_CONFIG_PATH)
    predict_tile_fn = make_ultralytics_predict_fn(model.weights_path, conf_threshold=tile_conf_threshold)
    model_names = getattr(predict_tile_fn, "model_names", None)
    if model_names is not None:
        assert_model_matches_taxonomy(model_names, target_names, model_label=model.weights_path)

    # Origine du plan tangent local = centroïde GPS du batch (voir
    # geolocation.py) - arbitraire tant que la même origine sert pour tout le
    # batch, choisi comme le centroïde plutôt que la 1ère photo pour rester
    # numériquement centré sur la zone réelle.
    origin_lon = sum(e.metadata.lon for e in entries) / len(entries)
    origin_lat = sum(e.metadata.lat for e in entries) / len(entries)
    origin = (origin_lon, origin_lat)

    print("--- 🔍 3/5 - Inférence tuilée par photo + reprojection sol ---")
    all_detections: List[PhotoDetection] = []
    for i, entry in enumerate(entries):
        dets = _build_detections_for_photo(entry, predict_tile_fn, origin, nms_iou_threshold)
        all_detections.extend(dets)
        print(f"    [{i + 1}/{len(entries)}] {Path(entry.metadata.path).name} : {len(dets)} détection(s)")
    print(f"    Total détections brutes (avant dédoublonnage inter-photos) : {len(all_detections)}")

    print("--- 🧹 4/5 - Dédoublonnage inter-photos ---")
    # Imagettes en mémoire pour TOUTES les détections brutes - nécessaire pour
    # la méthode "visual" (comparaison d'apparence), voir docstring du module
    # et de _build_crop_images. Construites même en méthode "iou" pour éviter
    # deux chemins de code différents à maintenir - coût négligeable comparé
    # à l'inférence YOLO qui vient d'avoir lieu.
    all_crop_images = _build_crop_images(all_detections)
    crop_by_id = {id(d): img for d, img in zip(all_detections, all_crop_images)}

    if dedup_method == "visual":
        print("    Méthode : similarité visuelle (DINOv2) + IoU (--dedup-method visual, opt-in) - ne "
              "résout le dédoublonnage QUE si le modèle est déjà en cache local ET HF_HUB_OFFLINE=1 est "
              "positionné ; sinon appel réseau à HuggingFace à chaque exécution, inadapté au terrain. "
              "Voir dedup.py et le journal (07/09/2026) pour le détail.")
        from src.application.visual_similarity import compute_embeddings, load_similarity_model
        processor, sim_model = load_similarity_model()
        embeddings = compute_embeddings(all_crop_images, processor, sim_model)
        for det, emb in zip(all_detections, embeddings):
            det.embedding = emb
        winners, dedup_report = deduplicate_detections_visual(
            all_detections,
            iou_threshold=dedup_iou_threshold,
            similarity_threshold=dedup_similarity_threshold,
            distance_threshold_m=dedup_distance_threshold_m,
            return_report=True,
        )
    else:
        print("    Méthode : IoU géométrique seul (--dedup-method iou, défaut depuis le 07/09/2026) - "
              "100% local, aucun appel réseau. ATTENTION : sur données réelles (essai1), cette méthode "
              "ne fusionne quasiment aucune détection dupliquée (bruit de géoréférencement direct, voir "
              "dedup.py et journal 06/09/2026) - le dédoublonnage inter-photos reste un problème ouvert "
              "avec ce choix, accepté explicitement pour privilégier l'exécution 100% locale.")
        winners, dedup_report = deduplicate_detections(
            all_detections, iou_threshold=dedup_iou_threshold, return_report=True
        )
    print(f"    {dedup_report.n_input} détections -> {dedup_report.n_clusters} objets réels distincts "
          f"({dedup_report.n_border_disqualified} cluster(s) avec disqualification de bord appliquée)")

    winners_final = [w for w in winners if w.confidence >= conf_threshold]
    print(f"    Seuil de confiance final ({conf_threshold}) : {len(winners_final)}/{len(winners)} conservées")

    print("--- 🗺️  5/5 - Export GeoJSON + imagettes ---")
    winners_final_images = [crop_by_id[id(w)] for w in winners_final]
    crop_filenames = _save_winner_crops(winners_final, winners_final_images, output_dir)

    features = []
    for i, det in enumerate(winners_final):
        lonlat_coords = [local_xy_to_lonlat(x, y, origin_lon, origin_lat) for x, y in det.local_polygon.exterior.coords]
        centroid_lon, centroid_lat = local_xy_to_lonlat(
            det.local_polygon.centroid.x, det.local_polygon.centroid.y, origin_lon, origin_lat
        )
        # Surface au sol (m²) : `local_polygon` est déjà exprimé en mètres
        # réels par geolocation.py (plan tangent local) - contrairement au
        # mode orthomosaïque (geo_density_map.py), aucun GSD séparé n'est
        # nécessaire ici, l'aire shapely est directement en m². Poids estimé
        # à partir de cette aire - voir weight_estimation.py (PROVISOIRE, non
        # calibré, voir sa docstring) ; None pour les classes où la surface
        # n'est pas un prédicteur exploitable (Cordage_Filet, Debris_Divers).
        class_name = target_names.get(det.class_id, str(det.class_id))
        area_m2 = det.local_polygon.area
        weight_kg = estimate_weight_kg(class_name, area_m2)
        features.append({
            "type": "Feature",
            "geometry": mapping(Polygon(lonlat_coords)),
            "properties": {
                "class_id": det.class_id,
                "class_name": class_name,
                "confidence": round(det.confidence, 4),
                "source_photo": Path(det.photo_id).name,
                "crop_file": crop_filenames.get(i),
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "area_m2": round(area_m2, 4),
                "weight_kg": round(weight_kg, 4) if weight_kg is not None else None,
                # Champs additifs (non consommés par web_map.py) - persistent
                # la géométrie PIXEL dans le repère de la photo source, pour
                # que predictions_diagnostic.py/export_predictions_to_cvat.py
                # puissent reconstruire un lot d'import CVAT plus tard SANS
                # relancer l'inférence (même philosophie que
                # rapport_metrics.json pour training_report.py/compare_runs.py).
                "source_photo_path": str(det.photo_id),
                "photo_width_px": det.photo_width_px,
                "photo_height_px": det.photo_height_px,
                "pixel_polygon": [[round(x, 2), round(y, 2)] for x, y in det.pixel_polygon.exterior.coords],
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    geojson_path = output_dir / "detections.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"    {len(features)} détection(s) exportée(s) : {geojson_path}")
    print(f"    Imagettes : {output_dir / 'crops'}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", required=True, help="Dossier de photos drone brutes (jpg/png).")
    parser.add_argument("--model", default=None, help="Nom du modèle dans config/models_registry.yaml (invite interactive si omis et plusieurs modèles disponibles).")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Seuil de confiance final, appliqué APRÈS dédoublonnage.")
    parser.add_argument("--tile-conf-threshold", type=float, default=DEFAULT_TILE_CONF_THRESHOLD, help="Seuil large de 1ère passe (avant dédoublonnage) - laisser bas, voir docstring du module.")
    parser.add_argument("--dedup-iou-threshold", type=float, default=DEFAULT_DEDUP_IOU_THRESHOLD, help="Seuil d'IoU au sol pour regrouper des détections comme le même objet réel - pas encore calibré empiriquement, à ajuster.")
    parser.add_argument("--dedup-method", choices=["visual", "iou"], default="iou",
                         help="'iou' (défaut depuis le 07/09/2026) : géométrie seule, 100% local, aucun "
                              "appel réseau - mais ne fusionne quasiment aucun doublon sur données réelles "
                              "(diagnostic du 06/09/2026 dans dedup.py), décision assumée pour l'usage "
                              "terrain sans connexion fiable. 'visual' : IoU + similarité visuelle DINOv2, "
                              "opt-in - nécessite un modèle en cache + HF_HUB_OFFLINE=1 pour rester local, "
                              "voir dedup.py et le journal (07/09/2026).")
    parser.add_argument("--dedup-similarity-threshold", type=float, default=DEFAULT_DEDUP_SIMILARITY_THRESHOLD,
                         help="Seuil de similarité cosinus (méthode 'visual' uniquement) pour fusionner deux "
                              "détections proches mais géométriquement disjointes - pas calibré au-delà d'un "
                              "seul exemple manuel (voir dedup.py), à affiner.")
    parser.add_argument("--dedup-distance-threshold-m", type=float, default=DEFAULT_DEDUP_DISTANCE_THRESHOLD_M,
                         help="Rayon (mètres, méthode 'visual' uniquement) de recherche de candidats autour de "
                              "chaque détection - au-delà, deux détections ne sont jamais comparées visuellement.")
    parser.add_argument("--output-dir", default=None, help="Dossier de sortie (défaut : 4. Results/2_prediction/batch_<horodatage>, voir paths.py).")
    args = parser.parse_args()

    run_application(
        batch_dir=args.batch_dir,
        model_name=args.model,
        conf_threshold=args.conf_threshold,
        tile_conf_threshold=args.tile_conf_threshold,
        dedup_method=args.dedup_method,
        dedup_iou_threshold=args.dedup_iou_threshold,
        dedup_similarity_threshold=args.dedup_similarity_threshold,
        dedup_distance_threshold_m=args.dedup_distance_threshold_m,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
