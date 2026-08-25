#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Carte de densité interactive sur orthomosaïque GeoTIFF complète.

Contrairement à `visualize_predictions.py` (image parente entière chargée en
mémoire via `image_io.load_image_bgr`, pensé pour des photos de terrain de
quelques Mo), ce module cible directement les ORTHOMOSAÏQUES GeoTIFF issues de
WebODM - potentiellement plusieurs centaines de Mo à plusieurs Go, bien trop
grandes pour tenir en RAM d'un coup. Il lit donc l'image FENÊTRE PAR FENÊTRE
via `rasterio` (jamais l'image entière en mémoire, sauf la version très
sous-échantillonnée utilisée comme simple fond de carte visuel), et - point
qui n'a jamais existé ailleurs dans le pipeline - conserve le géoréférencement
(CRS + transformation affine) du GeoTIFF pour convertir chaque détection en
coordonnées réelles (lon/lat WGS84), condition nécessaire à une vraie carte de
densité géolocalisée plutôt qu'un simple schéma en coordonnées pixel.

Voir le journal de décisions du projet (24/08/2026, section "Géolocalisation,
orthomosaïques et anticipation d'un changement de matériel drone") pour le
raisonnement complet derrière ce choix d'architecture.

Ce que ce module NE fait PAS (par design, comme `visualize_predictions.py`) :
il ne modifie jamais rien dans `1_annotated_dataset`, ne produit aucun export
CVAT - PUREMENT une visualisation, à partir des prédictions du modèle seul
(pas de comparaison à une vérité terrain ici, contrairement à
`visualize_predictions.py` qui, lui, appareille aux annotations existantes).

Géométrie de tuilage : réutilise `tiling_geometry.iter_tile_windows` (même
source de vérité que l'entraînement et que `tiled_inference.py`) et
`tiled_inference.nms_merge` pour fusionner les doublons de recouvrement entre
fenêtres adjacentes - aucune logique de fusion dupliquée.

Usage :
    python -m src.review.geo_density_map --tif "chemin/vers/orthomosaique.tif" --model chemin/vers/best.pt
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
from shapely.geometry import Polygon

from src.data.class_config import DEFAULT_CLASS_CONFIG_PATH, load_class_config
from src.data.tiling_geometry import iter_tile_windows
from src.review.matching import LabeledPolygon
from src.review.tiled_inference import PredictTileFn, make_ultralytics_predict_fn, nms_merge

# Même racine que les autres étapes de sortie du pipeline (5_review_dataset,
# 6_prediction_viewer) - garder la même convention numérotée plutôt qu'un
# chemin de sortie ad hoc, pour que ce nouvel outil s'intègre visuellement
# au reste de l'arborescence plutôt que de faire bande à part.
BASE_DIR = r"E:\PixelOdyssey\3. Processed dataset"
DENSITY_MAPS_DIR = os.path.join(BASE_DIR, "7_density_maps")

WGS84 = "EPSG:4326"

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def _read_vendor(filename: str) -> str:
    """Lit une librairie JS/CSS tierce vendorisée dans src/review/vendor/
    (Leaflet + plugin leaflet.heat) pour l'embarquer directement dans la page
    plutôt que de la charger depuis un CDN au chargement. Ce n'est PAS pour
    l'auto-suffisance (la page a de toute façon besoin d'Internet pour les
    tuiles satellite) mais pour la ROBUSTESSE : un CDN qui répond lentement,
    est bloqué par un pare-feu d'entreprise, ou a un souci ponctuel ferait
    échouer la carte entière (`L is not defined`) sans que rien ne l'indique
    clairement à l'utilisateur. Vendorisées une fois ici, ces deux petites
    librairies (~150 Ko au total) ne dépendent plus de rien d'externe."""
    path = VENDOR_DIR / filename
    if not path.exists():
        raise RuntimeError(
            f"Librairie vendorisée manquante : {path}. Voir la docstring de "
            f"_read_vendor() - à télécharger une fois depuis cdnjs (leaflet "
            f"1.9.4 et leaflet.heat 0.2.0) et à conserver dans le repo."
        )
    return path.read_text(encoding="utf-8")


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


def build_detection_crops(
    tif_path: str,
    detections: List[LabeledPolygon],
    margin_px: int = 60,
    target_dim: int = 300,
    outline_color: Tuple[int, int, int] = (255, 235, 0),
) -> List[str]:
    """Découpe, pour CHAQUE détection, un petit chip à résolution NATIVE
    (marge autour du masque, jamais sous-échantillonné) avec le contour du
    masque dessiné dessus, encodé en PNG base64 - c'est ce qui s'affiche dans
    le popup au clic sur une détection, pour juger si le masque colle
    vraiment à la forme du déchet réel.

    Choix retenu le 24/08/2026 (voir journal de décisions) à la place d'une
    pyramide de tuiles couvrant TOUTE l'orthomosaïque à résolution native :
    une telle pyramide (testée, voir `generate_tile_pyramid`) pèse plusieurs
    centaines de Mo sur ce fichier de test - largement au-delà de ce qui peut
    être livré via une pièce jointe de conversation ou le pont vers le disque
    de l'utilisateur (limites de taille de ces deux canaux). Un chip par
    détection coûte, lui, proportionnellement au nombre de détections et non
    à la surface totale de la plage - largement suffisant pour l'objectif
    "juger la qualité d'un masque", qui ne demande la résolution native QUE
    localement, autour de chaque déchet.
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

            # Contour du masque en coordonnées LOCALES au chip (translation
            # simple depuis les coordonnées pixel image entière) - jaune vif
            # choisi pour rester lisible sur du sable comme sur du plastique
            # coloré, plutôt que d'essayer de faire correspondre la couleur
            # par classe assignée dynamiquement côté JS.
            draw = ImageDraw.Draw(img)
            local_coords = [(x - x0, y - y0) for x, y in det.geom.exterior.coords]
            if len(local_coords) >= 2:
                draw.line(local_coords, fill=outline_color, width=2, joint="curve")

            # Agrandissement pour l'affichage (le chip natif est souvent tout
            # petit - un déchet de 10 cm à 0.5 cm/px + marge ne fait que
            # quelques dizaines de pixels de côté) - LANCZOS car c'est pour
            # l'œil humain, pas une entrée modèle.
            scale = target_dim / max(img.width, img.height)
            if scale > 1.0:
                img = img.resize(
                    (max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS
                )

            # JPEG plutôt que PNG : ce chip est une PHOTO (pas un graphique à
            # aplats de couleur comme les tuiles de fond), le gain de
            # compression est énorme (~5x mesuré) pour une perte de qualité
            # invisible à l'usage - déterminant ici car ce chip est répété
            # une fois par détection (jusqu'à plusieurs centaines) et
            # embarqué directement dans le HTML.
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            crops.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))

    return crops


WEBMERCATOR = "EPSG:3857"
_EARTH_CIRCUMFERENCE_M = 2 * math.pi * 6378137.0  # ≈ 40 075 016,686 m (rayon équatorial WGS84)


def _native_zoom_for_resolution(res_m: float, cap: int = 21) -> int:
    """Zoom de la grille slippy-map standard (celle d'OSM/Leaflet/Esri, en
    EPSG:3857) correspondant à la résolution native de l'orthomosaïque -
    au-delà, une tuile serait plus fine que ce que la source contient
    vraiment, donc pur flou d'interpolation. `res_m` = taille de pixel en
    mètres, dans la même unité que la grille (Web Mercator).

    `cap` borne délibérément le zoom généré pour LA CARTE DE FOND (pas pour
    l'inspection détaillée d'un masque - voir `build_detection_crops`) : une
    pyramide couvrant toute l'orthomosaïque jusqu'au zoom vraiment natif
    (25 sur ce fichier de test) pèse plusieurs centaines de Mo, bien au-delà
    de ce qu'une pièce jointe de conversation ou le pont vers le disque de
    l'utilisateur peuvent transporter (essayé puis abandonné le 24/08/2026,
    voir le journal de décisions). Le défaut 21 donne une vue d'ensemble
    nette pour naviguer sur la plage (~1.9 cm à 3.8 cm/px selon la latitude)
    à un coût de génération négligeable ; le jugement fin "ce masque colle-t-
    il au déchet ?" passe par le chip natif affiché au clic sur la
    détection."""
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
    du GeoTIFF, pour remplacer l'ancien `build_basemap_overlay` (une unique
    image sous-échantillonnée draper sur la carte via `L.imageOverlay`) comme
    fond de carte pour la NAVIGATION D'ENSEMBLE sur la plage.

    Pourquoi une pyramide plutôt qu'une image unique : un `imageOverlay` est
    figé à sa résolution d'export - zoomer au-delà ne révèle plus aucun
    détail réel, juste l'agrandissement flou de pixels déjà là. Une pyramide
    de tuiles reste nette à chaque niveau de zoom qu'elle couvre (le
    navigateur ne charge que les tuiles visibles, à la résolution du niveau
    courant), exactement comme n'importe quel outil web-GIS.

    Important - ce que cette fonction NE fait PLUS depuis le 24/08/2026 (voir
    journal de décisions) : couvrir toute l'orthomosaïque jusqu'à la
    résolution VRAIMENT native (0.5 cm/px, zoom ≈25 sur ce fichier). Essayé,
    ça pèse plusieurs centaines de Mo - bien au-delà de ce qu'une pièce
    jointe de conversation ou le pont vers le disque de l'utilisateur peuvent
    transporter. `max_zoom_cap` (défaut 21) borne donc volontairement la
    pyramide à un niveau "vue d'ensemble nette", suffisant pour naviguer sur
    la plage ; le jugement fin de la qualité d'un masque sur UN déchet précis
    passe par le chip natif de `build_detection_crops`, affiché au clic sur
    la détection - bien moins coûteux car proportionnel au nombre de
    détections, pas à la surface totale de l'image.

    Chaque tuile est reprojetée directement depuis le GeoTIFF source (une
    bande à la fois, via `rasterio.warp.reproject`) vers la grille EPSG:3857 -
    rasterio ne lit que la fenêtre source nécessaire à cette tuile, jamais
    l'image entière. Cette lecture-fenêtre-par-fenêtre reste la seule
    approche praticable sur `SL W1.tif` (~1 Go), quel que soit le zoom max
    choisi. Écrit les PNG sous `tiles_dir/{z}/{x}/{y}.png`.
    """
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


def _build_html(
    tile_min_zoom: int,
    tile_max_zoom: int,
    sw: Tuple[float, float],
    ne: Tuple[float, float],
    detections: List[Dict],
    source_name: str,
    model_name: str,
) -> str:
    """Page autonome (Leaflet + plugin leaflet.heat vendorisés, voir
    `_read_vendor`). Volontairement PAS publiée comme Artifact claude.ai : un
    Artifact interdit toute requête réseau externe hors polices Google (CSP
    stricte), ce qui bloquerait le fond de carte satellite ET les tuiles de
    l'orthomosaïque - cette page est prévue pour être ouverte directement
    dans un navigateur normal, sans cette contrainte.

    Le fond orthomosaïque est un `L.tileLayer` pointant vers le dossier local
    `tiles/{z}/{x}/{y}.png` (chemin RELATIF à ce fichier HTML - les deux
    doivent rester ensemble) plutôt qu'un `L.imageOverlay` unique : voir
    `generate_tile_pyramid` pour le raisonnement complet (zoom jusqu'au pixel
    natif pour juger la qualité d'un masque). `minNativeZoom`/`maxNativeZoom`
    indiquent à Leaflet de ne jamais demander de tuile hors de la pyramide
    générée - en dehors de cette plage, il agrandit/réduit la tuile la plus
    proche disponible plutôt que de laisser un trou.

    Chaque détection est dessinée comme un VRAI polygone (`d.polygon`, le
    contour du masque de segmentation reprojeté en lon/lat) et non plus comme
    un simple point - situe la FORME et la POSITION du masque sur la carte
    d'ensemble. Un petit point centré est ajouté en complément : à faible
    zoom un masque de quelques cm devient un polygone de quelques pixels
    écran, quasi impossible à cliquer sans lui. Le JUGEMENT FIN de la
    qualité du masque (colle-t-il vraiment au déchet ?) se fait dans le
    popup au clic, via le chip natif `d.crop` (voir `build_detection_crops`) -
    pas en zoomant la carte elle-même, plafonnée à un niveau de vue
    d'ensemble (voir `generate_tile_pyramid`).
    """
    detections_json = json.dumps(detections, ensure_ascii=False)
    center_lat = (sw[1] + ne[1]) / 2
    center_lon = (sw[0] + ne[0]) / 2
    leaflet_css = _read_vendor("leaflet.min.css")
    leaflet_js = _read_vendor("leaflet.min.js")
    leaflet_heat_js = _read_vendor("leaflet-heat.js")

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>PixelOdyssey — Carte de densité</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{leaflet_css}
</style>
<style>
  html, body {{ margin:0; padding:0; height:100%; background:#111; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  #map {{ position:absolute; top:0; bottom:0; left:0; right:0; }}
  #panel {{ position:absolute; top:12px; right:12px; z-index:1000; background:rgba(20,20,20,0.92); color:#eee;
            padding:14px 16px; border-radius:8px; width:250px; box-shadow:0 2px 10px rgba(0,0,0,0.4); font-size:13px; }}
  #panel h1 {{ font-size:14px; margin:0 0 10px; color:#fff; }}
  #panel .row {{ margin-bottom:10px; }}
  #panel label {{ display:block; margin-bottom:4px; color:#bbb; }}
  .modebtn {{ flex:1; padding:6px 8px; border:1px solid #444; background:#2a2a2a; color:#eee; border-radius:4px;
              cursor:pointer; font-size:12px; }}
  .modebtn.active {{ background:#3a7dff; border-color:#3a7dff; color:#fff; }}
  #modebtns {{ display:flex; gap:6px; }}
  input[type=range] {{ width:100%; }}
  #legend {{ margin-top:10px; padding-top:10px; border-top:1px solid #333; }}
  #legend .swatch {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
  #legend div {{ margin-bottom:4px; }}
  #meta {{ margin-top:10px; padding-top:10px; border-top:1px solid #333; color:#999; font-size:11px; line-height:1.5; }}
  .leaflet-popup-content {{ font-size:12px; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>PixelOdyssey — Densité de déchets</h1>
  <div class="row">
    <label>Affichage</label>
    <div id="modebtns">
      <button class="modebtn active" id="btnPoints">Détections</button>
      <button class="modebtn" id="btnHeat">Densité</button>
    </div>
  </div>
  <div class="row">
    <label>Opacité orthomosaïque : <span id="opacityVal">85%</span></label>
    <input type="range" id="opacitySlider" min="0" max="100" value="85">
  </div>
  <div id="legend"></div>
  <div id="meta">
    Source : {source_name}<br>
    Modèle : {model_name}<br>
    {len(detections)} détection(s) après fusion des recouvrements de tuiles.
  </div>
</div>

<script>
{leaflet_js}
</script>
<script>
{leaflet_heat_js}
</script>
<script>
const DETECTIONS = {detections_json};
const SW = [{sw[1]}, {sw[0]}];
const NE = [{ne[1]}, {ne[0]}];
const BOUNDS = L.latLngBounds(SW, NE);

const map = L.map('map', {{ zoomControl: true, maxZoom: {tile_max_zoom} }}).fitBounds(BOUNDS, {{ padding: [40, 40] }});

const satellite = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: 'Fond satellite : Esri, Maxar, Earthstar Geographics', maxZoom: {tile_max_zoom}, maxNativeZoom: 19 }}
).addTo(map);

// Pyramide de tuiles locales (voir generate_tile_pyramid) - chemin relatif à
// ce fichier HTML, donc "tiles/" doit rester dans le même dossier que lui.
// minNativeZoom/maxNativeZoom : hors de cette plage, Leaflet agrandit la
// tuile la plus proche au lieu de laisser un trou (aucune tuile générée).
const ortho = L.tileLayer('tiles/{{z}}/{{x}}/{{y}}.png', {{
  opacity: 0.85,
  minZoom: 0,
  maxZoom: {tile_max_zoom},
  minNativeZoom: {tile_min_zoom},
  maxNativeZoom: {tile_max_zoom},
  bounds: BOUNDS,
  noWrap: true,
  tms: false,
}}).addTo(map);

// Rectangle discret montrant l'emprise exacte de l'orthomosaïque, même si
// l'opacité est baissée à 0 - repère utile pour situer la zone étudiée.
L.rectangle(BOUNDS, {{ color: '#3a7dff', weight: 1.5, fill: false, dashArray: '4,4' }}).addTo(map);

const CLASS_COLORS = {{}};
const PALETTE = ['#ff5252', '#ffb300', '#3a7dff', '#26c281', '#c77dff', '#ff8a5c', '#5cd6ff', '#ff5cbb'];
let colorIdx = 0;
function colorForClass(name) {{
  if (!(name in CLASS_COLORS)) {{
    CLASS_COLORS[name] = PALETTE[colorIdx % PALETTE.length];
    colorIdx++;
  }}
  return CLASS_COLORS[name];
}}

const pointsLayer = L.layerGroup();
DETECTIONS.forEach(d => {{
  const color = colorForClass(d.class_name);
  // Chip natif (voir build_detection_crops côté Python) avec le contour du
  // masque dessiné dessus - c'est ce qui permet de juger si le masque colle
  // vraiment au déchet réel, sans avoir besoin de zoomer sur la carte.
  const cropHtml = d.crop
    ? `<img src="${{d.crop}}" style="display:block;margin-top:6px;max-width:260px;border-radius:4px;">`
    : '';
  const popupHtml = `<b>${{d.class_name}}</b><br>Confiance : ${{(d.confidence * 100).toFixed(0)}}%${{cropHtml}}`;

  // Le VRAI contour du masque prédit (reprojeté en lon/lat) - c'est ce qui
  // permet de zoomer sur un déchet et de juger si le masque colle à sa forme
  // réelle sous l'orthomosaïque, plutôt qu'un simple point sans épaisseur.
  if (d.polygon && d.polygon.length >= 3) {{
    const mask = L.polygon(d.polygon, {{
      color: color,
      weight: 2,
      fillColor: color,
      fillOpacity: 0.35,
    }});
    mask.bindPopup(popupHtml);
    pointsLayer.addLayer(mask);
  }}

  // Petit point centré en complément : à faible zoom, un masque de
  // quelques cm devient un polygone de quelques pixels écran - quasi
  // impossible à cliquer sans ce repère toujours visible.
  const marker = L.circleMarker([d.lat, d.lon], {{
    radius: 4,
    color: '#111',
    weight: 1,
    fillColor: color,
    fillOpacity: 0.9,
  }});
  marker.bindPopup(popupHtml);
  pointsLayer.addLayer(marker);
}});
pointsLayer.addTo(map);

const heatPoints = DETECTIONS.map(d => [d.lat, d.lon, 0.4 + d.confidence * 0.6]);
const heatLayer = L.heatLayer(heatPoints, {{ radius: 28, blur: 22, maxZoom: 21 }});

function renderLegend() {{
  const el = document.getElementById('legend');
  el.innerHTML = Object.entries(CLASS_COLORS).map(([name, color]) =>
    `<div><span class="swatch" style="background:${{color}}"></span>${{name}}</div>`
  ).join('');
}}
renderLegend();

document.getElementById('btnPoints').addEventListener('click', () => {{
  map.removeLayer(heatLayer);
  pointsLayer.addTo(map);
  document.getElementById('btnPoints').classList.add('active');
  document.getElementById('btnHeat').classList.remove('active');
  document.getElementById('legend').style.display = 'block';
}});
document.getElementById('btnHeat').addEventListener('click', () => {{
  map.removeLayer(pointsLayer);
  heatLayer.addTo(map);
  document.getElementById('btnHeat').classList.add('active');
  document.getElementById('btnPoints').classList.remove('active');
  document.getElementById('legend').style.display = 'none';
}});
document.getElementById('opacitySlider').addEventListener('input', (e) => {{
  const v = parseInt(e.target.value, 10);
  ortho.setOpacity(v / 100);
  document.getElementById('opacityVal').textContent = v + '%';
}});
</script>
</body>
</html>
"""


def run_geo_density_map(
    tif_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    tile_size: int = 640,
    overlap: int = 256,
    tile_conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    ortho_max_zoom_cap: int = 21,
    output_dir: str = DENSITY_MAPS_DIR,
    run_id: Optional[str] = None,
) -> Dict:
    """`output_path` explicite prend le pas s'il est fourni (utile pour un
    test ponctuel) ; sinon la sortie va dans
    `7_density_maps/<run_id>/index.html` (run_id = horodatage par défaut),
    même convention que `visualize_predictions.py` (6_prediction_viewer/<run_id>/)."""
    class_taxonomy, target_names = load_class_config(DEFAULT_CLASS_CONFIG_PATH)

    print(f"--- 🗺️  CARTE DE DENSITÉ GÉORÉFÉRENCÉE ---")
    print(f"    Orthomosaïque : {tif_path}")
    print(f"    Modèle        : {model_path}")

    predict_tile_fn = make_ultralytics_predict_fn(model_path, conf_threshold=tile_conf_threshold)

    detections, crs, transform, (img_w, img_h) = predict_geotiff_windowed(
        tif_path, predict_tile_fn, tile_size=tile_size, overlap=overlap, nms_iou_threshold=nms_iou_threshold
    )

    # Centroïde (pour le mode densité, qui a besoin d'un point unique par
    # détection) + contour COMPLET du masque de segmentation (pour dessiner
    # le vrai polygone en mode "Détections", voir _build_html). Un seul appel
    # groupé à pixels_to_lonlat pour l'ensemble plutôt qu'un par détection -
    # évite des centaines de petits appels à rasterio.warp.transform.
    centroids_px = [(d.geom.centroid.x, d.geom.centroid.y) for d in detections]
    vertex_counts = [len(d.geom.exterior.coords) for d in detections]
    all_polygon_px = [pt for d in detections for pt in d.geom.exterior.coords]
    all_lonlat = pixels_to_lonlat(centroids_px + all_polygon_px, transform, crs)
    centroid_lonlat = all_lonlat[: len(detections)]
    polygon_lonlat_flat = all_lonlat[len(detections):]

    print(f"  → Découpe de {len(detections)} chip(s) natif(s) autour de chaque masque "
          f"(voir build_detection_crops)...")
    crops = build_detection_crops(tif_path, detections)

    detection_records = []
    idx = 0
    for det, (lon, lat), n_verts, crop in zip(detections, centroid_lonlat, vertex_counts, crops):
        verts = polygon_lonlat_flat[idx: idx + n_verts]
        idx += n_verts
        detection_records.append({
            "lon": lon,
            "lat": lat,
            "confidence": round(float(det.confidence or 0.0), 4),
            "class_id": det.class_id,
            "class_name": target_names.get(det.class_id, f"classe_{det.class_id}"),
            # [lat, lon] par sommet - convention attendue par L.polygon côté JS.
            "polygon": [[la, lo] for lo, la in verts],
            "crop": crop,
        })

    if output_path is None:
        if run_id is None:
            from datetime import datetime

            run_id = datetime.now().strftime("density_%Y%m%d_%H%M%S")
        out_p = Path(output_dir) / run_id / "index.html"
    else:
        out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("  → Génération de la pyramide de tuiles de l'orthomosaïque (voir generate_tile_pyramid)...")
    tiles_dir = out_p.parent / "tiles"
    pyramid = generate_tile_pyramid(tif_path, tiles_dir, max_zoom_cap=ortho_max_zoom_cap)

    html = _build_html(
        pyramid["min_zoom"], pyramid["max_zoom"], pyramid["sw"], pyramid["ne"], detection_records,
        source_name=Path(tif_path).name, model_name=Path(model_path).name,
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
                              "7_density_maps/<horodatage>/index.html (même convention que "
                              "6_prediction_viewer/).")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--tile-conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--ortho-max-zoom-cap", type=int, default=21,
                         help="Zoom max (grille slippy-map EPSG:3857) de la pyramide de tuiles de "
                              "fond de carte (vue d'ensemble, PAS l'inspection fine d'un masque - "
                              "voir build_detection_crops pour ça). Défaut 21 : vue nette suffisante "
                              "pour naviguer sans faire exploser la taille du dossier de sortie.")
    args = parser.parse_args()
    run_geo_density_map(
        tif_path=args.tif,
        model_path=args.model,
        output_path=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        tile_conf_threshold=args.tile_conf_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        ortho_max_zoom_cap=args.ortho_max_zoom_cap,
    )
