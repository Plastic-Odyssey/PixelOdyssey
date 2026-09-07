#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Géolocalisation directe d'une photo drone brute (pipeline
"application" - géoréférencement NAÏF par GPS+altitude+capteur, PAS une
orthorectification photogrammétrique type WebODM - voir le compromis discuté
et accepté le 04/09/2026, journal_decisions_pipeline.md).

Simplifications assumées explicitement (décisions du 04/09/2026, PAS des
oublis) :
- Terrain supposé PLAT (altitude AGL uniforme sur tout le batch) - pas de
  correction de relief/marée. Confirmé acceptable par Jame pour ce cas
  d'usage (carte de densité à l'échelle d'une plage, pas un relevé
  topographique).
- Caméra supposée NADIR PARFAIT pour la projection (pas de correction de
  perspective liée au tangage réel) - MAIS `pitch_within_tolerance()` permet
  de détecter/exclure les photos dont l'écart au nadir dépasse la tolérance
  du capteur (voir config/sensor_specs.yaml), plutôt que d'appliquer
  aveuglément l'approximation partout. Écart réel observé sur l'échantillon
  du 04/09/2026 : 0° à 5° selon la photo.
- Terre plate localement (plan tangent, pas de projection cartographique
  formelle type UTM/pyproj) - valide à l'échelle d'une plage (centaines de
  mètres), pas d'une région entière. Si le besoin s'étend un jour à une zone
  de plusieurs dizaines de km, remplacer `lonlat_to_local_xy`/
  `local_xy_to_lonlat` par une vraie projection (pyproj, déjà disponible via
  la dépendance rasterio) sans toucher au reste du pipeline.

Convention du cap caméra (`gimbal_yaw_deg`) - À VALIDER EMPIRIQUEMENT dès que
possible, voir note plus bas : 0° = nord vrai, angle croissant dans le sens
HORAIRE, plage [-180, 180]. Cette convention est celle documentée
officiellement par DJI pour l'attitude de l'appareil (DJIAttitude.yaw, SDK
Mobile DJI : "0 corresponds to a True North heading" / "Yawing clockwise will
increase yaw value" - voir source en bas de fichier) - appliquée ICI par
extension à `GimbalYawDegree` (même écosystème DJI, valeurs observées sur
l'échantillon du 04/09/2026 cohérentes avec cette plage), PAS une
confirmation officielle publiée spécifiquement pour ce tag XMP précis.
**Validation empirique recommandée dès qu'un vrai batch de photos
séquentielles avec recouvrement est disponible** : vérifier que deux photos
consécutives projettent bien des empreintes adjacentes/superposées comme
attendu du plan de vol réel - si l'orientation est inversée, le signe de
`gimbal_yaw_deg` (ou l'ordre des vecteurs `right`/`up` ci-dessous) devra être
inversé.

Exemple :
    from src.application.exif_metadata import extract_photo_metadata
    from src.application.sensor_config import load_sensor_registry, resolve_sensor
    from src.application.geolocation import compute_gsd_cm_per_px, photo_footprint_local_polygon

    meta = extract_photo_metadata("lot1/DJI_0001.JPG")
    registry = load_sensor_registry()
    sensor = resolve_sensor(meta.make, meta.model, registry)
    gsd = compute_gsd_cm_per_px(meta, sensor)
    footprint = photo_footprint_local_polygon(meta, sensor, origin=(meta.lon, meta.lat))

Source DJI (convention d'attitude, appliquée par extension au cap caméra) :
https://developer.dji.com/api-reference/android-api/Components/FlightController/DJIFlightController_DJIAttitude.html
"""

import math
from typing import List, Tuple

from shapely.geometry import Polygon

from src.application.exif_metadata import PhotoMetadata
from src.application.sensor_config import SensorSpec

# Constantes de conversion degré -> mètres, plan tangent local (terre plate) -
# suffisant à l'échelle d'une plage, voir docstring du module.
_METERS_PER_DEG_LAT = 110_540.0
_METERS_PER_DEG_LON_AT_EQUATOR = 111_320.0


def lonlat_to_local_xy(lon: float, lat: float, origin_lon: float, origin_lat: float) -> Tuple[float, float]:
    """Convertit (lon, lat) en coordonnées locales (x=Est, y=Nord) en mètres,
    par rapport à une origine (plan tangent local, pas une vraie projection
    cartographique - voir docstring du module)."""
    x_m = (lon - origin_lon) * _METERS_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(origin_lat))
    y_m = (lat - origin_lat) * _METERS_PER_DEG_LAT
    return x_m, y_m


def local_xy_to_lonlat(x_m: float, y_m: float, origin_lon: float, origin_lat: float) -> Tuple[float, float]:
    """Inverse de `lonlat_to_local_xy` - reconversion en lon/lat WGS84 (utile
    pour l'affichage carte, voir le futur branchement sur geo_density_map.py)."""
    lon = origin_lon + x_m / (_METERS_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(origin_lat)))
    lat = origin_lat + y_m / _METERS_PER_DEG_LAT
    return lon, lat


def compute_gsd_cm_per_px(photo: PhotoMetadata, sensor: SensorSpec) -> float:
    """GSD (résolution sol, cm/pixel) = (altitude_cm x largeur_capteur_mm) /
    (focale_mm x largeur_image_px). Pixels supposés carrés (vrai pour ce
    capteur : sensor_width_mm/width_px == sensor_height_mm/height_px == pixel
    pitch - voir config/sensor_specs.yaml) - donc la GSD calculée sur la
    largeur est valable aussi en hauteur, pas besoin d'un calcul séparé.

    Utilise `photo.width_px` (dimension RÉELLE du fichier), pas la résolution
    native du capteur du registre - reste correct même si le fichier a été
    redimensionné (voir docstring de sensor_specs.yaml).
    """
    altitude_cm = photo.relative_altitude_m * 100.0
    return (altitude_cm * sensor.sensor_width_mm) / (photo.focal_length_mm * photo.width_px)


def pitch_within_tolerance(photo: PhotoMetadata, sensor: SensorSpec) -> bool:
    """Vrai si l'écart au nadir parfait (-90°) de `gimbal_pitch_deg` reste
    dans la tolérance du capteur (`sensor.max_pitch_deviation_deg`). Ne lève
    aucune exception ici : c'est à l'appelant (inventory.py) de décider quoi
    faire d'une photo hors tolérance (exclure/flaguer pour revue) - cette
    fonction est un simple prédicat, pas une politique."""
    return abs(photo.gimbal_pitch_deg - (-90.0)) <= sensor.max_pitch_deviation_deg


def photo_center_local_xy(photo: PhotoMetadata, origin: Tuple[float, float]) -> Tuple[float, float]:
    """Position du CENTRE de la photo en coordonnées locales (mètres), par
    rapport à `origin` = (lon, lat) de référence du batch (voir inventory.py -
    typiquement la première photo ou le centroïde du batch, arbitraire tant
    que la même origine est utilisée pour tout le batch)."""
    origin_lon, origin_lat = origin
    return lonlat_to_local_xy(photo.lon, photo.lat, origin_lon, origin_lat)


def pixel_to_local_xy(
    photo: PhotoMetadata, sensor: SensorSpec, origin: Tuple[float, float], px: float, py: float,
) -> Tuple[float, float]:
    """Projette un pixel (px, py) de l'image (origine haut-gauche, x vers la
    droite, y vers le bas - convention image standard) vers sa position sol
    en coordonnées locales (mètres, Est/Nord) par rapport à `origin`.

    Nadir supposé parfait (pas de correction de tangage réel - voir
    `pitch_within_tolerance` pour flaguer les photos où cette approximation
    est la plus risquée). Le cap `gimbal_yaw_deg` oriente l'empreinte -
    convention documentée dans la docstring du module.
    """
    gsd_m_per_px = compute_gsd_cm_per_px(photo, sensor) / 100.0

    # Offset par rapport au centre image, en pixels puis en mètres au sol.
    # dy_px inversé (haut = positif) car l'axe image Y pointe vers le bas.
    dx_px = px - photo.width_px / 2.0
    dy_px = photo.height_px / 2.0 - py
    right_m = dx_px * gsd_m_per_px   # distance vers la DROITE de l'image, au sol
    up_m = dy_px * gsd_m_per_px      # distance vers le HAUT de l'image, au sol

    # "Haut image" pointe vers le cap gimbal_yaw_deg (0=nord, horaire positif -
    # voir docstring). "Droite image" = ce vecteur tourné de 90° horaire.
    theta = math.radians(photo.gimbal_yaw_deg)
    up_east, up_north = math.sin(theta), math.cos(theta)
    right_east, right_north = math.cos(theta), -math.sin(theta)

    d_east = right_m * right_east + up_m * up_east
    d_north = right_m * right_north + up_m * up_north

    center_x, center_y = photo_center_local_xy(photo, origin)
    return center_x + d_east, center_y + d_north


def pixel_polygon_to_local_polygon(
    photo: PhotoMetadata, sensor: SensorSpec, origin: Tuple[float, float],
    pixel_coords: List[Tuple[float, float]],
) -> Polygon:
    """Reprojette un polygone (ex : contour d'un masque de détection) en
    coordonnées pixel de la photo vers un polygone en coordonnées locales
    (mètres) - c'est cette version qui doit être utilisée pour le
    dédoublonnage inter-photos (voir dedup.py), l'IoU en pixels d'une photo
    n'ayant aucun sens comparé à celui d'une autre photo."""
    local_coords = [pixel_to_local_xy(photo, sensor, origin, px, py) for px, py in pixel_coords]
    return Polygon(local_coords)


def photo_footprint_local_polygon(
    photo: PhotoMetadata, sensor: SensorSpec, origin: Tuple[float, float],
) -> Polygon:
    """Empreinte au sol de la photo ENTIÈRE (rectangle des 4 coins), en
    coordonnées locales (mètres). Utile pour visualiser la couverture du
    batch ou pour un filtrage grossier avant dédoublonnage (ne comparer que
    des photos dont les empreintes se recoupent)."""
    corners_px = [
        (0.0, 0.0),
        (photo.width_px, 0.0),
        (photo.width_px, photo.height_px),
        (0.0, photo.height_px),
    ]
    return pixel_polygon_to_local_polygon(photo, sensor, origin, corners_px)
