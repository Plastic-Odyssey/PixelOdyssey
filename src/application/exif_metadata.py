#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Extraction des métadonnées de vol (GPS, altitude, cap, tangage)
depuis une photo drone brute, pour la géolocalisation directe (pipeline
"application" - voir geolocation.py).

Découverte empirique du 04/09/2026 (3 photos réelles DJI Air 2S fournies par
l'utilisateur, voir journal_decisions_pipeline.md) qui structure ce module :

1. Le tag EXIF standard `GPSAltitude`/`GPSAltitudeRef` N'EST PAS FIABLE sur ce
   drone : sur les 3 photos testées, `GPSAltitudeRef` valait tantôt 0
   (au-dessus du niveau mer) tantôt 1 (EN DESSOUS), avec une `AbsoluteAltitude`
   XMP négative (-4.17m) sur l'une d'elles - inexploitable comme altitude de
   vol. La vraie source utilisable est le champ **XMP propriétaire DJI**
   `drone-dji:RelativeAltitude` (altitude relative au point de décollage,
   AGL) - c'est ce que ce module lit, jamais le GPS EXIF standard pour
   l'altitude.
2. Le cap de la caméra (`drone-dji:GimbalYawDegree`) et son tangage
   (`drone-dji:GimbalPitchDegree`) sont dans ce même bloc XMP - PAS dans
   l'EXIF standard (qui n'a pas de tag caméra pour un drone à gimbal
   orientable). `GimbalPitchDegree` observé : -85.00°/-89.90°/-89.90° - PAS
   toujours exactement -90° (nadir parfait) : geolocation.py doit vérifier
   cette valeur, pas la supposer.
3. Ce bloc XMP est un espace de noms PROPRIÉTAIRE DJI (`drone-dji:*`) - un
   autre constructeur (Delair, annoncé le 04/09/2026 mais pas encore
   disponible) n'a aucune raison d'utiliser le même schéma. Ce module
   structure donc l'extraction par ADAPTATEUR DE CONSTRUCTEUR (dispatch sur
   le tag EXIF `Make`) plutôt qu'un parsing DJI codé en dur partout - ajouter
   un constructeur = ajouter une fonction `_extract_flight_tags_<make>()`,
   jamais modifier `extract_photo_metadata()`. Tant qu'aucun fichier Delair
   réel n'a été fourni pour vérification, AUCUN adaptateur n'est ajouté à
   l'aveugle (même principe que config/sensor_specs.yaml).

Exemple :
    from src.application.exif_metadata import extract_photo_metadata
    meta = extract_photo_metadata("lot1/DJI_0001.JPG")
    print(meta.make, meta.model, meta.lat, meta.lon, meta.relative_altitude_m)
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from PIL import Image
from PIL.ExifTags import TAGS

_GPS_IFD_TAG = 0x8825
_EXIF_IFD_TAG = 0x8769

# Tolérance de désaccord entre GPS EXIF standard et GPS XMP (degrés décimaux,
# ~0.001° ~= 100m) - purement un signal d'alerte (`print`), jamais bloquant :
# les deux sources viennent du même GNSS, un léger écart est normal (arrondi,
# horodatage légèrement différent entre l'écriture des deux blocs).
_GPS_CROSS_CHECK_TOLERANCE_DEG = 0.001


@dataclass
class PhotoMetadata:
    """Métadonnées de vol d'une photo, telles que nécessaires à la
    géolocalisation directe. Un champ manquant lève une exception à
    l'extraction plutôt que d'être laissé `None` en silence - une détection
    sans position n'a aucune valeur pour ce pipeline (contrairement à d'autres
    outils du projet où un champ manquant est juste signalé pour revue)."""
    path: str
    make: str
    model: str
    width_px: int
    height_px: int
    focal_length_mm: float
    lat: float                    # degrés décimaux, WGS84
    lon: float                    # degrés décimaux, WGS84
    relative_altitude_m: float     # AGL, XMP drone-dji:RelativeAltitude
    gimbal_yaw_deg: float           # cap de la caméra, convention drone-dji (à valider empiriquement - voir geolocation.py)
    gimbal_pitch_deg: float         # -90 = nadir parfait
    gimbal_roll_deg: float


def _dms_to_decimal(dms, ref: str) -> float:
    """Convertit un triplet EXIF (degrés, minutes, secondes) + référence
    (N/S/E/W) en degrés décimaux signés."""
    degrees, minutes, seconds = (float(v) for v in dms)
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        value = -value
    return value


def _extract_xmp_attributes(img_path: Path) -> Dict[str, str]:
    """Extrait tous les attributs `namespace:Cle="Valeur"` du bloc XMP brut
    d'un JPEG (PIL n'expose pas le XMP DJI - il faut le lire dans les octets
    bruts du fichier). Retourne un dict {"namespace:Cle": "Valeur"} - le
    namespace est conservé dans la clé pour éviter toute collision entre
    constructeurs différents qui réutiliseraient un nom d'attribut générique.
    """
    raw = img_path.read_bytes()
    start = raw.find(b"<x:xmpmeta")
    end = raw.find(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return {}
    xmp_text = raw[start:end + len(b"</x:xmpmeta>")].decode("utf-8", errors="replace")
    return dict(re.findall(r'([\w-]+:[\w-]+)="([^"]*)"', xmp_text))


def _extract_flight_tags_dji(xmp_attrs: Dict[str, str], img_path: Path) -> Dict[str, float]:
    """Adaptateur DJI : traduit le bloc XMP `drone-dji:*` vers les champs
    génériques attendus par `PhotoMetadata`. Lève une erreur explicite et
    nommant le fichier si un champ essentiel manque - un XMP DJI incomplet
    est un signal que le fichier a été modifié/recompressé par un outil tiers
    (perte de métadonnées), pas un cas à ignorer silencieusement.
    """
    required = ["drone-dji:RelativeAltitude", "drone-dji:GimbalYawDegree", "drone-dji:GimbalPitchDegree"]
    missing = [k for k in required if k not in xmp_attrs]
    if missing:
        raise ValueError(
            f"{img_path} : champs XMP DJI manquants {missing} - fichier possiblement "
            f"recompressé/modifié par un outil tiers (perte de métadonnées de vol). "
            f"Impossible de géolocaliser cette photo sans ces champs."
        )
    return {
        "relative_altitude_m": float(xmp_attrs["drone-dji:RelativeAltitude"]),
        "gimbal_yaw_deg": float(xmp_attrs["drone-dji:GimbalYawDegree"]),
        "gimbal_pitch_deg": float(xmp_attrs["drone-dji:GimbalPitchDegree"]),
        "gimbal_roll_deg": float(xmp_attrs.get("drone-dji:GimbalRollDegree", 0.0)),
    }


# Dispatch par constructeur (tag EXIF `Make`) - ajouter un constructeur =
# ajouter une entrée ici + la fonction `_extract_flight_tags_<make>`
# correspondante, jamais modifier `extract_photo_metadata`.
_FLIGHT_TAG_EXTRACTORS = {
    "DJI": _extract_flight_tags_dji,
}


def extract_photo_metadata(img_path: Union[str, Path]) -> PhotoMetadata:
    """Extrait toutes les métadonnées de vol nécessaires d'une photo. Lève
    une exception explicite (nommant le fichier et le champ en cause) pour
    toute donnée manquante ou un constructeur non supporté - voir la
    docstring du module pour le raisonnement (pas de valeur par défaut
    silencieuse acceptable ici)."""
    img_path = Path(img_path)
    img = Image.open(img_path)
    width_px, height_px = img.size

    exif = img.getexif()
    orientation = exif.get(274, 1)
    if orientation != 1:
        raise ValueError(
            f"{img_path} : tag EXIF Orientation={orientation} (pas 1/normal). Le calcul de "
            f"projection (geolocation.py) suppose une image non pivotée par rotation EXIF - "
            f"une photo avec une orientation différente donnerait un cap/une empreinte faux. "
            f"À gérer explicitement si ce cas se présente vraiment (pas encore observé sur "
            f"l'échantillon DJI Air 2S du 04/09/2026)."
        )

    make = str(exif.get(271, "")).strip()
    model = str(exif.get(272, "")).strip()
    if not make or not model:
        raise ValueError(f"{img_path} : tags EXIF Make/Model manquants ou vides.")

    exif_ifd = exif.get_ifd(_EXIF_IFD_TAG)
    if 37386 not in exif_ifd:  # FocalLength
        raise ValueError(f"{img_path} : tag EXIF FocalLength manquant.")
    focal_length_mm = float(exif_ifd[37386])

    gps_ifd = exif.get_ifd(_GPS_IFD_TAG)
    lat_exif = lon_exif = None
    if all(k in gps_ifd for k in (1, 2, 3, 4)):
        lat_exif = _dms_to_decimal(gps_ifd[2], str(gps_ifd[1]))
        lon_exif = _dms_to_decimal(gps_ifd[4], str(gps_ifd[3]))

    xmp_attrs = _extract_xmp_attributes(img_path)

    lat_xmp = lon_xmp = None
    if "drone-dji:GpsLatitude" in xmp_attrs and "drone-dji:GpsLongitude" in xmp_attrs:
        lat_xmp = float(xmp_attrs["drone-dji:GpsLatitude"])
        lon_xmp = float(xmp_attrs["drone-dji:GpsLongitude"])

    if lat_exif is not None:
        lat, lon = lat_exif, lon_exif
        if lat_xmp is not None and (abs(lat_exif - lat_xmp) > _GPS_CROSS_CHECK_TOLERANCE_DEG
                                     or abs(lon_exif - lon_xmp) > _GPS_CROSS_CHECK_TOLERANCE_DEG):
            print(
                f"⚠️  [exif_metadata] {img_path} : désaccord GPS EXIF ({lat_exif},{lon_exif}) "
                f"vs XMP ({lat_xmp},{lon_xmp}) > {_GPS_CROSS_CHECK_TOLERANCE_DEG}° - EXIF utilisé, "
                f"à vérifier si ça se reproduit souvent."
            )
    elif lat_xmp is not None:
        lat, lon = lat_xmp, lon_xmp
    else:
        raise ValueError(f"{img_path} : position GPS introuvable (ni EXIF, ni XMP).")

    extractor = _FLIGHT_TAG_EXTRACTORS.get(make)
    if extractor is None:
        raise NotImplementedError(
            f"{img_path} : constructeur Make={make!r} non supporté - aucun adaptateur "
            f"d'extraction XMP écrit pour lui (voir _FLIGHT_TAG_EXTRACTORS dans "
            f"exif_metadata.py). Constructeurs supportés : {list(_FLIGHT_TAG_EXTRACTORS)}. "
            f"Fournir un exemple de fichier réel de ce constructeur avant d'écrire son "
            f"adaptateur - ne pas deviner son schéma de métadonnées."
        )
    flight_tags = extractor(xmp_attrs, img_path)

    return PhotoMetadata(
        path=str(img_path),
        make=make,
        model=model,
        width_px=width_px,
        height_px=height_px,
        focal_length_mm=focal_length_mm,
        lat=lat,
        lon=lon,
        relative_altitude_m=flight_tags["relative_altitude_m"],
        gimbal_yaw_deg=flight_tags["gimbal_yaw_deg"],
        gimbal_pitch_deg=flight_tags["gimbal_pitch_deg"],
        gimbal_roll_deg=flight_tags["gimbal_roll_deg"],
    )
