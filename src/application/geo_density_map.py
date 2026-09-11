#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Carte de densité interactive sur orthomosaïque GeoTIFF complète.

Génère une carte web (Leaflet) affichant les détections du modèle sur une
orthomosaïque GeoTIFF, avec la position géoréférencée (lon/lat WGS84) de
chaque déchet détecté. Contrairement à `visualize_predictions.py` (image
chargée entièrement en mémoire), ce module lit l'orthomosaïque FENÊTRE PAR
FENÊTRE via `rasterio` - nécessaire pour des fichiers de plusieurs centaines
de Mo à plusieurs Go - et conserve le géoréférencement du GeoTIFF (CRS +
transformation affine) pour convertir les détections en coordonnées réelles.

Ce module ne modifie jamais `1_annotated_dataset` et ne produit lui-même
aucun export CVAT - c'est avant tout une visualisation, à partir des
prédictions du modèle seul (pas de comparaison à une vérité terrain,
contrairement à `visualize_predictions.py`). Écrit tout de même
`detections.geojson` à côté de la carte (même convention que
`run_application.py`, mode batch) - lu ensuite par
`predictions_diagnostic.py` (export .xlsx) et
`export_predictions_to_cvat.py` (lot CVAT, avec découpage en grille sous
`CVAT_MAX_PIXELS` pour ce mode, l'orthomosaïque entière étant presque
toujours trop lourde pour un import direct).

Géométrie de tuilage : réutilise `tiling_geometry.iter_tile_windows` (même
source de vérité que l'entraînement) et `tiled_inference.nms_merge` pour
fusionner les doublons de recouvrement entre fenêtres adjacentes.

Vit sous `src/application/`, avec le reste du pipeline application (mode
batch et mode orthomosaïque, plus leurs modules de support) - notamment
`src/application/vendor/` (Leaflet vendorisé), partagé avec `web_map.py`.

Le gabarit HTML/JS de la carte (panneau, popup, légende+filtre par classe,
curseur de confiance, statistiques agrégées, bascule Points/Densité) est
PARTAGÉ avec `web_map.py` via `map_builder.render_map_page` - la SEULE
différence visuelle avec la carte du mode batch est le fond : ici,
l'orthomosaïque entière (pyramide de tuiles locale, voir
`generate_tile_pyramid`) par-dessus le satellite, à opacité FIXE (1.0 -
curseur de transparence remplacé par le curseur de confiance, présent dans
les deux modes). La surface au sol (`area_m2`) est ici dérivée du GSD natif
du GeoTIFF (aire en pixels du masque x aire d'un pixel en m² dans le CRS
natif, voir `_pixel_area_m2`) - différent de run_application.py, où
`local_polygon` est déjà exprimé en mètres réels par geolocation.py.

Point de vigilance sur la taxonomie si ce module est appelé DIRECTEMENT
(plutôt que via `run_inference.py`) : sa propre CLI attend un chemin de
poids brut (`--model`) et un `--class-config` optionnel qui retombe
SILENCIEUSEMENT sur la taxonomie 7-classes par défaut si omis - un modèle
mono-classe (ou toute autre variante) choisi sans préciser `--class-config`
serait alors comparé à tort à cette taxonomie par défaut. `run_inference.py`
résout ce risque en résolvant le modèle (et son `taxonomy_config`) via le
registre commun `config/models_registry.yaml` AVANT d'appeler ce module -
c'est le point d'entrée recommandé pour un usage normal ; cette CLI directe
reste utile pour un diagnostic ponctuel avec un poids hors registre.

Entrée : --tif (orthomosaïque GeoTIFF), --model (modèle entraîné, best.pt).
Sortie : une page HTML autonome (carte Leaflet + tuiles locales de
l'orthomosaïque) sous `4. Results/2_prediction/ortho_<horodatage>/index.html`
par défaut (voir paths.py).

Exemple :
    python -m src.application.geo_density_map --tif "chemin/vers/orthomosaique.tif" --model chemin/vers/best.pt
"""

import argparse
import base64
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import calculate_default_transform
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds as warp_transform_bounds
from shapely.geometry import Polygon, mapping

from src.application.map_builder import render_map_page
from src.application.paths import DEFAULT_PREDICTION_DIR
from src.application.stats_panel import compute_aggregate_stats
from src.application.weight_estimation import estimate_weight_kg
from src.data.utils.class_config import DEFAULT_CLASS_CONFIG_PATH, assert_model_matches_taxonomy, load_class_config
from src.data.utils.tiling_geometry import iter_tile_windows
from src.review.matching import LabeledPolygon
from src.review.tiled_inference import PredictTileFn, make_ultralytics_predict_fn, nms_merge

WGS84 = "EPSG:4326"

MODE_LABEL = "Orthomosaïque (GeoTIFF unique)"


def predict_geotiff_windowed(
    tif_path: str,
    predict_tile_fn: PredictTileFn,
    tile_size: int = 640,
    overlap: int = 256,
    nms_iou_threshold: float = 0.5,
) -> Tuple[List[LabeledPolygon], "rasterio.crs.CRS", "rasterio.Affine", Tuple[int, int]]:
    """Fait tourner `predict_tile_fn` fenêtre par fenêtre sur un GeoTIFF, SANS
    jamais charger l'image entière en mémoire (lecture rasterio par fenêtre
    uniquement) - condition nécessaire pour une orthomosaïque de plusieurs
    centaines de Mo à plusieurs Go. Retourne les détections fusionnées en
    coordonnées PIXEL de l'image pleine résolution, plus le CRS et la
    transformation affine nécessaires pour les convertir en coordonnées
    réelles ensuite (voir `pixels_to_lonlat`).
    """
    with rasterio.open(tif_path) as src:
        img_w, img_h = src.width, src.height
        crs = src.crs
        transform = src.transform
        band_count = src.count
        # On ne lit que les 3 premières bandes (R,G,B) - une éventuelle 4e
        # bande alpha (cas standard des exports WebODM, voir image_io.py pour
        # le même constat sur les .tif individuels) ne sert à rien au modèle.
        bands = [1, 2, 3] if band_count >= 3 else [1]
        stride = tile_size - overlap

        all_predictions: List[LabeledPolygon] = []
        windows = list(iter_tile_windows(img_w, img_h, tile_size, stride))
        print(f"  → {len(windows)} fenêtre(s) {tile_size}x{tile_size} (stride {stride}) à traiter...")

        for i, (x_start, y_start, x_end, y_end) in enumerate(windows):
            win = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
            tile = src.read(bands, window=win)  # (bands, h, w)
            tile = np.transpose(tile, (1, 2, 0))  # (h, w, bands)
            if len(bands) == 1:
                tile = np.repeat(tile, 3, axis=2)
            # rasterio lit en ordre R,G,B - le reste du pipeline (image_io,
            # make_ultralytics_predict_fn) travaille toujours en BGR (convention
            # OpenCV) : on aligne ici plutôt que de laisser un décalage de canaux
            # silencieux entrer dans le modèle.
            tile_bgr = np.ascontiguousarray(tile[:, :, ::-1])

            for class_id, confidence, poly_xy_tile in predict_tile_fn(tile_bgr):
                if len(poly_xy_tile) < 3:
                    continue
                poly_xy_img = [(x + x_start, y + y_start) for x, y in poly_xy_tile]
                geom = Polygon(poly_xy_img)
                if not geom.is_valid or geom.area <= 0:
                    continue
                all_predictions.append(LabeledPolygon(class_id=class_id, geom=geom, confidence=confidence))

            if (i + 1) % 100 == 0 or (i + 1) == len(windows):
                print(f"    ... {i + 1}/{len(windows)} fenêtres traitées, "
                      f"{len(all_predictions)} détection(s) brute(s) avant fusion.")

    merged = nms_merge(all_predictions, iou_threshold=nms_iou_threshold)
    print(f"  → {len(all_predictions)} détection(s) brute(s) → {len(merged)} après fusion des recouvrements.")
    return merged, crs, transform, (img_w, img_h)


def pixels_to_lonlat(
    pixel_xy: List[Tuple[float, float]], transform, crs
) -> List[Tuple[float, float]]:
    """Convertit une liste de coordonnées PIXEL (x, y) de l'image pleine
    résolution en (lon, lat) WGS84 - la transformation affine du GeoTIFF donne
    les coordonnées dans le CRS natif (ex: UTM), puis rasterio.warp.transform
    reprojette vers WGS84 (le CRS attendu par Leaflet/tout affichage web).
    """
    if not pixel_xy:
        return []
    xs_native, ys_native = [], []
    for px, py in pixel_xy:
        x_geo, y_geo = transform * (px, py)
        xs_native.append(x_geo)
        ys_native.append(y_geo)
    lons, lats = warp_transform(crs, WGS84, xs_native, ys_native)
    return list(zip(lons, lats))


def _pixel_area_m2(transform, crs) -> Optional[float]:
    """Aire (m²) d'UN pixel de l'orthomosaïque, dérivée directement de la
    transformation affine du GeoTIFF - nécessaire pour convertir l'aire en
    pixels d'un masque de détection en une surface au sol réelle (voir
    run_geo_density_map, `area_m2` de chaque détection).

    Suppose un CRS PROJETÉ en mètres (le cas standard des orthomosaïques
    WebODM du projet, ex: UTM zone 26N/EPSG:32626) et une transformation sans
    rotation (`transform.b == transform.d == 0`, vrai pour un export WebODM
    standard) - dans ce cas, l'aire d'un pixel est simplement `abs(a * e)`
    (a = largeur de pixel, e = hauteur de pixel, signée car l'axe image
    pointe vers le bas). Statut : hypothèse tranchée pour les orthomosaïques
    WebODM du projet.

    Retourne None (plutôt qu'un nombre silencieusement faux) si le CRS est
    GÉOGRAPHIQUE (degrés, pas mètres - ex: EPSG:4326 brut) : dans ce cas
    `transform.a`/`transform.e` seraient en degrés, et le produit ne serait pas
    une aire en m². Aucune orthomosaïque du projet n'est actuellement dans ce
    cas, mais mieux vaut ne pas estimer de surface/poids du tout que
    d'afficher un chiffre faux de plusieurs ordres de grandeur.
    """
    if crs is not None and crs.is_geographic:
        return None
    return abs(transform.a * transform.e)


def build_detection_crops(
    tif_path: str,
    detections: List[LabeledPolygon],
    margin_px: int = 60,
    target_dim: int = 300,
    outline_color: Tuple[int, int, int] = (255, 235, 0),
) -> List[str]:
    """Découpe, pour CHAQUE détection, un petit chip à résolution NATIVE
    (marge autour du masque, jamais sous-échantillonné) avec le contour du
    masque dessiné dessus, encodé en JPEG base64 - c'est ce qui s'affiche
    dans le popup au clic sur une détection, pour juger si le masque colle
    vraiment à la forme du déchet réel. Coûte proportionnellement au nombre
    de détections, pas à la surface totale de l'orthomosaïque.

    Entrée : chemin du GeoTIFF, liste de détections (LabeledPolygon).
    Sortie : liste de chips encodés en data URI JPEG base64.
    """
    from PIL import Image, ImageDraw

    crops: List[str] = []
    with rasterio.open(tif_path) as src:
        img_w, img_h = src.width, src.height
        band_count = src.count
        bands = [1, 2, 3] if band_count >= 3 else [1]

        for det in detections:
            minx, miny, maxx, maxy = det.geom.bounds
            x0 = max(0, int(minx) - margin_px)
            y0 = max(0, int(miny) - margin_px)
            x1 = min(img_w, int(maxx) + margin_px)
            y1 = min(img_h, int(maxy) + margin_px)
            w, h = max(1, x1 - x0), max(1, y1 - y0)
            win = rasterio.windows.Window(x0, y0, w, h)
            data = src.read(bands, window=win)
            if len(bands) == 1:
                data = np.repeat(data, 3, axis=0)
            arr = np.transpose(data, (1, 2, 0))
            img = Image.fromarray(np.ascontiguousarray(arr), mode="RGB")

            # Contour du masque en coordonnées locales au chip (jaune vif,
            # lisible sur sable comme sur plastique coloré).
            draw = ImageDraw.Draw(img)
            local_coords = [(x - x0, y - y0) for x, y in det.geom.exterior.coords]
            if len(local_coords) >= 2:
                draw.line(local_coords, fill=outline_color, width=2, joint="curve")

            # Agrandissement (LANCZOS) pour l'affichage si le chip natif est petit.
            scale = target_dim / max(img.width, img.height)
            if scale > 1.0:
                img = img.resize(
                    (max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS
                )

            # JPEG plutôt que PNG : chip photographique, meilleure
            # compression pour une perte de qualité invisible à l'usage.
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            crops.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))

    return crops


WEBMERCATOR = "EPSG:3857"
_EARTH_CIRCUMFERENCE_M = 2 * math.pi * 6378137.0  # ≈ 40 075 016,686 m (rayon équatorial WGS84)


def _native_zoom_for_resolution(res_m: float, cap: int = 21) -> int:
    """Zoom de la grille slippy-map standard (OSM/Leaflet/Esri, EPSG:3857)
    correspondant à la résolution native de l'orthomosaïque - au-delà, une
    tuile serait plus fine que ce que la source contient vraiment.

    `cap` borne volontairement le zoom généré pour la carte de fond (vue
    d'ensemble, pas l'inspection fine d'un masque - voir
    `build_detection_crops` pour ça).

    Entrée : `res_m` (taille de pixel en mètres), `cap` (zoom max, défaut 21).
    Sortie : niveau de zoom entier."""
    if res_m <= 0:
        return cap
    z = math.log2(_EARTH_CIRCUMFERENCE_M / (256 * res_m))
    return max(0, min(cap, math.ceil(z)))


def _tile_bounds_3857(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """Bornes (minx, miny, maxx, maxy) en EPSG:3857 de la tuile (z, x, y) dans
    la grille slippy-map standard (origine nord-ouest, y croissant vers le
    sud) - exactement la convention attendue par `L.tileLayer` côté Leaflet.
    """
    n = 2 ** z
    tile_m = _EARTH_CIRCUMFERENCE_M / n
    minx = -_EARTH_CIRCUMFERENCE_M / 2 + x * tile_m
    maxx = minx + tile_m
    maxy = _EARTH_CIRCUMFERENCE_M / 2 - y * tile_m
    miny = maxy - tile_m
    return minx, miny, maxx, maxy


def generate_tile_pyramid(
    tif_path: str,
    tiles_dir: Path,
    tile_px: int = 256,
    zoom_margin: int = 5,
    max_zoom_cap: int = 21,
) -> Dict:
    """Génère une PYRAMIDE DE TUILES XYZ (grille standard EPSG:3857) à partir
    du GeoTIFF, comme fond de carte pour la navigation d'ensemble sur la
    plage - reste nette à chaque niveau de zoom (le navigateur ne charge que
    les tuiles visibles), contrairement à une unique image sous-échantillonnée.

    `max_zoom_cap` (défaut 21) borne volontairement la pyramide à une vue
    d'ensemble nette plutôt que la résolution vraiment native de
    l'orthomosaïque ; le jugement fin de la qualité d'un masque passe par le
    chip natif de `build_detection_crops`, affiché au clic sur la détection.

    Chaque tuile est reprojetée directement depuis le GeoTIFF source (une
    bande à la fois, `rasterio.warp.reproject`) - rasterio ne lit que la
    fenêtre source nécessaire à chaque tuile, jamais l'image entière.

    Entrée : chemin du GeoTIFF, dossier de sortie, taille de tuile en px.
    Sortie : dict (min_zoom, max_zoom, n_tiles, bornes sw/ne) ; écrit les PNG
    sous `tiles_dir/{z}/{x}/{y}.png`."""
    from PIL import Image
    from rasterio.warp import reproject

    with rasterio.open(tif_path) as src:
        band_count = src.count
        bands = [1, 2, 3, 4] if band_count >= 4 else [1, 2, 3]
        mode = "RGBA" if band_count >= 4 else "RGB"

        dst_transform_full, _, _ = calculate_default_transform(
            src.crs, WEBMERCATOR, src.width, src.height, *src.bounds
        )
        native_res_m = abs(dst_transform_full.a)
        max_zoom = _native_zoom_for_resolution(native_res_m, cap=max_zoom_cap)
        min_zoom = max(0, max_zoom - zoom_margin)

        left, bottom, right, top = warp_transform_bounds(src.crs, WEBMERCATOR, *src.bounds)

        print(f"  → Résolution native ≈ {native_res_m * 100:.2f} cm/px → zoom max {max_zoom} "
              f"(pyramide {min_zoom}→{max_zoom}).")

        n_tiles = 0
        for z in range(min_zoom, max_zoom + 1):
            n = 2 ** z
            tile_m = _EARTH_CIRCUMFERENCE_M / n
            x0 = int(math.floor((left + _EARTH_CIRCUMFERENCE_M / 2) / tile_m))
            x1 = int(math.floor((right + _EARTH_CIRCUMFERENCE_M / 2) / tile_m))
            y0 = int(math.floor((_EARTH_CIRCUMFERENCE_M / 2 - top) / tile_m))
            y1 = int(math.floor((_EARTH_CIRCUMFERENCE_M / 2 - bottom) / tile_m))
            level_count = 0
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    minx, miny, maxx, maxy = _tile_bounds_3857(z, x, y)
                    dst_transform = transform_from_bounds(minx, miny, maxx, maxy, tile_px, tile_px)
                    dst = np.zeros((len(bands), tile_px, tile_px), dtype=np.uint8)
                    for bi, b in enumerate(bands):
                        reproject(
                            source=rasterio.band(src, b),
                            destination=dst[bi],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=WEBMERCATOR,
                            resampling=Resampling.bilinear,
                            dst_nodata=0,
                        )
                    arr = np.transpose(dst, (1, 2, 0))
                    tile_dir = tiles_dir / str(z) / str(x)
                    tile_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr, mode=mode).save(tile_dir / f"{y}.png", optimize=False)
                    level_count += 1
            n_tiles += level_count
            print(f"    ... zoom {z} : {level_count} tuile(s) ({x1 - x0 + 1}x{y1 - y0 + 1}).")

        (sw_lon, sw_lat), (ne_lon, ne_lat) = pixels_to_lonlat(
            [(0, src.height), (src.width, 0)], src.transform, src.crs
        )

    print(f"  → Pyramide de tuiles : {n_tiles} tuile(s) écrite(s) sous {tiles_dir}.")
    return {
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "n_tiles": n_tiles,
        "sw": (sw_lon, sw_lat),
        "ne": (ne_lon, ne_lat),
    }


def run_geo_density_map(
    tif_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    tile_size: int = 640,
    overlap: int = 256,
    tile_conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    ortho_max_zoom_cap: int = 21,
    output_dir: "Optional[str]" = None,
    run_id: Optional[str] = None,
    class_config_path: "Optional[str]" = None,
) -> Dict:
    """`output_path` explicite prend le pas s'il est fourni (utile pour un
    test ponctuel) ; sinon la sortie va dans
    `<output_dir>/<run_id>/index.html` - `output_dir` par défaut
    `DEFAULT_PREDICTION_DIR` (`4. Results/2_prediction/`, voir paths.py,
    partagé avec le mode batch) ; `run_id` par défaut `ortho_<horodatage>`
    (préfixe qui distingue ce mode dans le dossier commun, voir paths.py).

    `class_config_path` (pour le dispatch unifié de `run_inference.py`) :
    PAR DÉFAUT None -> `DEFAULT_CLASS_CONFIG_PATH`
    (7-classes, comportement inchangé pour un appel direct de ce module en
    diagnostic). Un modèle choisi via le registre `config/models_registry.yaml`
    (voir `model_registry.py`) peut avoir été entraîné sous une AUTRE
    taxonomie (ex: mono-classe, `taxonomy_config: config/data_config_mono_class.yaml`) -
    sans ce paramètre, `assert_model_matches_taxonomy` ci-dessous lèverait à
    tort une erreur de mismatch (le modèle serait comparé à la mauvaise
    config), pour un modèle pourtant valide."""
    class_config_path = class_config_path or DEFAULT_CLASS_CONFIG_PATH
    class_taxonomy, target_names = load_class_config(class_config_path)

    print(f"--- 🗺️  CARTE DE DENSITÉ GÉORÉFÉRENCÉE ---")
    print(f"    Orthomosaïque : {tif_path}")
    print(f"    Modèle        : {model_path}")

    predict_tile_fn = make_ultralytics_predict_fn(model_path, conf_threshold=tile_conf_threshold)

    # Garde-fou : voir la même vérification dans label_review.py. Absent pour un
    # predict_tile_fn injecté en test (pas d'attribut model_names).
    model_names = getattr(predict_tile_fn, "model_names", None)
    if model_names is not None:
        assert_model_matches_taxonomy(model_names, target_names, model_label=str(model_path))

    detections, crs, transform, (img_w, img_h) = predict_geotiff_windowed(
        tif_path, predict_tile_fn, tile_size=tile_size, overlap=overlap, nms_iou_threshold=nms_iou_threshold
    )

    # Centroïde (pour le mode densité, qui a besoin d'un point unique par
    # détection) + contour COMPLET du masque de segmentation (pour dessiner
    # le vrai polygone en mode "Détections"). Un seul appel groupé à
    # pixels_to_lonlat pour l'ensemble plutôt qu'un par détection - évite des
    # centaines de petits appels à rasterio.warp.transform.
    centroids_px = [(d.geom.centroid.x, d.geom.centroid.y) for d in detections]
    vertex_counts = [len(d.geom.exterior.coords) for d in detections]
    all_polygon_px = [pt for d in detections for pt in d.geom.exterior.coords]
    all_lonlat = pixels_to_lonlat(centroids_px + all_polygon_px, transform, crs)
    centroid_lonlat = all_lonlat[: len(detections)]
    polygon_lonlat_flat = all_lonlat[len(detections):]

    # Surface au sol (m²) et poids estimé (voir weight_estimation.py,
    # PROVISOIRE) - `pixel_area_m2` est None si le CRS n'est pas projeté en
    # mètres (voir `_pixel_area_m2`), auquel cas aucune surface/poids n'est
    # calculée plutôt qu'un chiffre faux.
    pixel_area_m2 = _pixel_area_m2(transform, crs)
    if pixel_area_m2 is None:
        print("⚠️  [geo_density_map] CRS géographique (pas de mètres natifs) - surface/poids non "
              "calculés pour ce run (voir _pixel_area_m2).")

    print(f"  → Découpe de {len(detections)} chip(s) natif(s) autour de chaque masque "
          f"(voir build_detection_crops)...")
    crops = build_detection_crops(tif_path, detections)

    detection_records = []
    # Features au format GeoJSON (voir plus bas, export detections.geojson) -
    # tenues SÉPARÉES de detection_records à dessein : detection_records est
    # sérialisé tel quel dans le HTML (voir render_map_page) et n'a besoin
    # que des coordonnées lon/lat déjà calculées ; y ajouter le polygone
    # PIXEL pleine résolution (nécessaire pour reconstruire un lot CVAT plus
    # tard, voir export_predictions_to_cvat.py) gonflerait inutilement chaque
    # page HTML générée.
    geojson_features = []
    idx = 0
    for det, (lon, lat), n_verts, crop in zip(detections, centroid_lonlat, vertex_counts, crops):
        verts = polygon_lonlat_flat[idx: idx + n_verts]
        idx += n_verts
        class_name = target_names.get(det.class_id, f"classe_{det.class_id}")
        area_m2 = det.geom.area * pixel_area_m2 if pixel_area_m2 is not None else None
        weight_kg = estimate_weight_kg(class_name, area_m2)
        confidence = round(float(det.confidence or 0.0), 4)
        area_m2_rounded = round(area_m2, 4) if area_m2 is not None else None
        weight_kg_rounded = round(weight_kg, 4) if weight_kg is not None else None
        detection_records.append({
            "lon": lon,
            "lat": lat,
            "confidence": confidence,
            "class_id": det.class_id,
            "class_name": class_name,
            # [lat, lon] par sommet - convention attendue par L.polygon côté JS.
            "polygon": [[la, lo] for lo, la in verts],
            "crop": crop,
            "area_m2": area_m2_rounded,
            "weight_kg": weight_kg_rounded,
        })
        geojson_features.append({
            "type": "Feature",
            "geometry": mapping(Polygon([(lo, la) for lo, la in verts])),
            "properties": {
                "class_id": det.class_id,
                "class_name": class_name,
                "confidence": confidence,
                "area_m2": area_m2_rounded,
                "weight_kg": weight_kg_rounded,
                "centroid_lon": lon,
                "centroid_lat": lat,
                # Polygone PIXEL dans le repère de l'orthomosaïque COMPLÈTE
                # (pas encore recadré à un morceau de grille) - voir
                # ortho_source ci-dessous pour les dimensions/chemin
                # nécessaires à sa réutilisation (export_predictions_to_cvat.py).
                "pixel_polygon": [[round(x, 2), round(y, 2)] for x, y in det.geom.exterior.coords],
            },
        })

    stats = compute_aggregate_stats(detection_records)

    if output_path is None:
        if run_id is None:
            from datetime import datetime

            run_id = datetime.now().strftime("ortho_%Y%m%d_%H%M%S")
        out_p = Path(output_dir or DEFAULT_PREDICTION_DIR) / run_id / "index.html"
    else:
        out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Export detections.geojson - même convention que run_application.py
    # (mode batch), absent jusqu'ici du mode orthomosaïque (qui ne produisait
    # que la carte HTML). `ortho_source` (membre GeoJSON additionnel, en
    # dehors de `type`/`features` - autorisé par la spec GeoJSON) persiste le
    # nécessaire pour recadrer les polygones PIXEL ci-dessus en morceaux
    # d'une grille CVAT plus tard (voir export_predictions_to_cvat.py), sans
    # avoir à rouvrir le modèle ni relancer l'inférence.
    geojson = {
        "type": "FeatureCollection",
        "ortho_source": {"tif_path": str(tif_path), "width_px": img_w, "height_px": img_h},
        "features": geojson_features,
    }
    geojson_path = out_p.parent / "detections.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"  → {len(geojson_features)} détection(s) exportée(s) : {geojson_path}")

    print("  → Génération de la pyramide de tuiles de l'orthomosaïque (voir generate_tile_pyramid)...")
    tiles_dir = out_p.parent / "tiles"
    pyramid = generate_tile_pyramid(tif_path, tiles_dir, max_zoom_cap=ortho_max_zoom_cap)

    meta_extra_html = (
        f"Modèle : {Path(model_path).name}<br>"
        f"{len(detection_records)} détection(s) après fusion des recouvrements de tuiles."
    )
    html = render_map_page(
        detection_records, stats,
        bounds=(pyramid["sw"], pyramid["ne"]),
        source_name=Path(tif_path).name,
        mode_label=MODE_LABEL,
        map_max_zoom=pyramid["max_zoom"],
        ortho_layer={"tiles_rel_path": "tiles", "min_zoom": pyramid["min_zoom"], "max_zoom": pyramid["max_zoom"]},
        meta_extra_html=meta_extra_html,
    )
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[SUCCÈS] Carte écrite : {out_p} ({len(detection_records)} détection(s), "
          f"{pyramid['n_tiles']} tuile(s) sous {tiles_dir}/).")
    return {
        "n_detections": len(detection_records),
        "output_path": str(out_p),
        "tiles_dir": str(tiles_dir),
        "n_tiles": pyramid["n_tiles"],
        "crs": str(crs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carte de densité géoréférencée PixelOdyssey")
    parser.add_argument("--tif", required=True, help="Chemin vers l'orthomosaïque GeoTIFF.")
    parser.add_argument("--model", required=True, help="Chemin vers le modèle entraîné (best.pt).")
    parser.add_argument("--output", default=None,
                         help="Fichier HTML de sortie. Optionnel : si omis, écrit dans "
                              "4. Results/2_prediction/ortho_<horodatage>/index.html (voir paths.py).")
    parser.add_argument("--output-dir", default=None,
                         help="Dossier parent des runs (défaut : 4. Results/2_prediction/, voir paths.py) "
                              "- ignoré si --output est fourni.")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--tile-conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--ortho-max-zoom-cap", type=int, default=21,
                         help="Zoom max (grille slippy-map EPSG:3857) de la pyramide de tuiles de "
                              "fond de carte (vue d'ensemble, PAS l'inspection fine d'un masque - "
                              "voir build_detection_crops pour ça). Défaut 21 : vue nette suffisante "
                              "pour naviguer sans faire exploser la taille du dossier de sortie.")
    parser.add_argument("--class-config", default=None,
                         help="Référentiel de classes (config/data_config*.yaml) utilisé À "
                              "L'ENTRAÎNEMENT du modèle passé à --model. Optionnel : défaut "
                              "config/data_config.yaml (7-classes) - à préciser si le modèle est "
                              "mono-classe ou une autre variante (voir model_registry.py).")
    args = parser.parse_args()
    run_geo_density_map(
        tif_path=args.tif,
        model_path=args.model,
        output_path=args.output,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        tile_conf_threshold=args.tile_conf_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        ortho_max_zoom_cap=args.ortho_max_zoom_cap,
        class_config_path=args.class_config,
    )
