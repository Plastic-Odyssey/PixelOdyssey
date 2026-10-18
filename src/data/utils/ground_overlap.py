#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Terrain partagé entre images parentes de splits différents (protocole v4, 12/10/2026).

Pourquoi : séparer train / val / test par image d'origine ne suffit pas quand deux images
différentes montrent le MÊME terrain (donc les mêmes déchets). Mesuré le 12/10/2026 :
  - imagettes Saint-Brandon : 1024 px sur une grille de 924 px, chaque imagette partage une
    bande de 100 px avec ses voisines (88 paires train/test, 63 val/test, 152 train/val) ;
  - orthomosaïques Santa Luzia : transects voisins qui se recouvrent de 0,5 à 1 % à leurs
    extrémités (sous le seuil de regroupement de 2 %) ;
  - photos drone et orthomosaïques d'un même site : SL W4 n / DJI_0550 (train) couvre 18 % du
    transect 22 (banc de test) - recouvrement jamais contrôlé jusqu'ici.

Règle : pour chaque paire d'images de splits différents qui partagent du terrain, la zone
partagée est NOIRCIE dans l'image du split le moins prioritaire (test > val > train) au moment
de la copie dans 2_split_dataset, et déclarée comme masque de nodata explicite (voir
`write_nodata_mask`). Le tuilage v4 écarte toute tuile qui en contient un seul pixel. Le test
n'est donc jamais modifié, et aucun pixel de terrain du test n'est vu en train ou en val.

Géométrie :
  - imagettes (nom "<prefix>_y<Y>_x<X>") : positions exactes sur la grille de leur ortho
    d'origine -> bandes partagées exactes, au pixel près ;
  - orthomosaïques GeoTIFF : zone valide (canal alpha) géoréférencée ;
  - photos drone : empreinte au sol calculée depuis l'EXIF (GPS, altitude, cap, capteur),
    avec une marge de `GPS_MARGIN_M` (erreur GPS : 1 m en médiane, 6,5 m au maximum mesurés
    sur SL W1 lors du recalage de l'application) ;
  - entre deux orthos, marge `ORTHO_MARGIN_M` (géoréférencement commun, cohérent au cm).
Une image sans géoréférence (transects 26 et 27 en PNG) ne peut pas être contrôlée ici : elle
est listée comme « non vérifiable » dans le rapport (vérifiée à la main le 12/10/2026 par
appariement d'images : aucun terrain commun avec les transects 6 à 10 ni entre elles).

Tout est exprimé dans un repère métrique local par site (plan tangent, comme
src/application/geolocation.py), suffisant à l'échelle d'une plage.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.data.utils.raw_dataset import site_of_batch

GPS_MARGIN_M = 5.0
ORTHO_MARGIN_M = 0.25
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
TILE_NAME_RE = re.compile(r"^(?P<prefix>.+)_y(?P<y>\d+)_x(?P<x>\d+)$")
LOGIC_VERSION = 1

_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQ = 111_320.0


def _to_local(lon, lat, origin):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return ((lon - origin[0]) * _M_PER_DEG_LON_EQ * math.cos(math.radians(origin[1])),
            (lat - origin[1]) * _M_PER_DEG_LAT)


def _from_local(x, y, origin):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (origin[0] + x / (_M_PER_DEG_LON_EQ * math.cos(math.radians(origin[1]))),
            origin[1] + y / _M_PER_DEG_LAT)


@dataclass
class Footprint:
    parent_id: str
    site: str
    kind: str                      # "photo" ou "ortho"
    poly: object                   # shapely, repère local du site (mètres)
    width: int
    height: int
    to_px: Callable = field(repr=False)   # (N,2) local -> (N,2) pixels

    def zone_pixel_polygons(self, zone) -> List[List[List[float]]]:
        """Polygones (contours extérieurs) de `zone` dans les pixels de CETTE image, bornés à l'image."""
        out = []
        geoms = getattr(zone, "geoms", [zone])
        for g in geoms:
            if g.is_empty or g.geom_type != "Polygon" or g.area <= 0:
                continue
            px = self.to_px(np.asarray(g.exterior.coords, float))
            px[:, 0] = np.clip(px[:, 0], 0, self.width)
            px[:, 1] = np.clip(px[:, 1], 0, self.height)
            out.append([[round(float(a), 1), round(float(b), 1)] for a, b in px])
        return out


def _ortho_footprint(parent: Dict, origin_holder: Dict) -> Optional[Footprint]:
    import rasterio
    from rasterio import features
    from rasterio.warp import transform as wtransform
    from shapely.geometry import shape
    from shapely.ops import unary_union

    path = Path(parent["img_path"])
    with rasterio.open(path) as src:
        if src.crs is None or src.transform.is_identity:
            return None
        crs, T, W, H = src.crs, src.transform, src.width, src.height
        f = max(1, int(round(0.05 / abs(T.a))))  # grille de ~5 cm
        shp = (max(1, H // f), max(1, W // f))
        if src.count >= 4:
            valid = src.read(4, out_shape=shp) > 0
        else:
            valid = src.read([1, 2, 3], out_shape=shp).max(axis=0) > 10
        Ts = T * T.scale(W / shp[1], H / shp[0])
    polys = [shape(g) for g, v in features.shapes(valid.astype(np.uint8), mask=valid, transform=Ts) if v == 1]
    if not polys:
        return None
    geo = unary_union(polys).buffer(0)
    site = site_of_batch(parent["batch"])
    if site not in origin_holder:
        cx, cy = geo.centroid.x, geo.centroid.y
        lon, lat = wtransform(crs, "EPSG:4326", [cx], [cy])
        origin_holder[site] = (lon[0], lat[0])
    origin = origin_holder[site]

    def crs_to_local(coords):
        lon, lat = wtransform(crs, "EPSG:4326", list(coords[:, 0]), list(coords[:, 1]))
        x, y = _to_local(lon, lat, origin)
        return np.column_stack([x, y])

    from shapely.geometry import Polygon, MultiPolygon
    parts = []
    for g in getattr(geo, "geoms", [geo]):
        if g.geom_type != "Polygon":
            continue
        ext = crs_to_local(np.asarray(g.exterior.coords))
        holes = [crs_to_local(np.asarray(h.coords)) for h in g.interiors]
        parts.append(Polygon(ext, holes))
    local = (MultiPolygon(parts) if len(parts) > 1 else parts[0]).buffer(0).simplify(0.05)
    inv = ~T

    def to_px(local_xy):
        lon, lat = _from_local(local_xy[:, 0], local_xy[:, 1], origin)
        xs, ys = wtransform("EPSG:4326", crs, list(lon), list(lat))
        cols, rows = inv * (np.asarray(xs), np.asarray(ys))
        return np.column_stack([cols, rows])

    return Footprint(parent["parent_id"], site, "ortho", local, W, H, to_px)


def _photo_footprint(parent: Dict, origin_holder: Dict, registry) -> Optional[Footprint]:
    import cv2
    from shapely.geometry import Polygon
    from src.application.exif_metadata import extract_photo_metadata
    from src.application.geolocation import pixel_to_local_xy
    from src.application.sensor_config import resolve_sensor

    path = Path(parent["img_path"])
    try:
        meta = extract_photo_metadata(path)
        sensor = resolve_sensor(meta.make, meta.model, registry)
    except Exception:  # noqa: BLE001 - photo sans GPS/XMP exploitable : non vérifiable
        return None
    site = site_of_batch(parent["batch"])
    origin = origin_holder.setdefault(site, (meta.lon, meta.lat))
    W, H = meta.width_px, meta.height_px
    corners_px = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    corners_loc = np.float32([pixel_to_local_xy(meta, sensor, origin, float(x), float(y)) for x, y in corners_px])
    M = cv2.getPerspectiveTransform(corners_loc, corners_px)

    def to_px(local_xy):
        pts = np.asarray(local_xy, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, M).reshape(-1, 2)

    return Footprint(parent["parent_id"], site, "photo", Polygon(corners_loc), W, H, to_px)


def build_footprints(parents: List[Dict], sensor_registry_path: Path):
    """parent_id -> Footprint (photos et orthos géoréférencées), + liste des images non vérifiables."""
    from src.application.sensor_config import load_sensor_registry

    registry = load_sensor_registry(sensor_registry_path)
    origin_holder: Dict[str, Tuple[float, float]] = {}
    fps: Dict[str, Footprint] = {}
    unverifiable: List[str] = []
    # Photos d'abord : l'origine locale d'un site est sa première photo (orthos ensuite).
    ordered = sorted(parents, key=lambda p: (Path(p["img_path"]).suffix.lower() in (".tif", ".tiff"), p["parent_id"]))
    for p in ordered:
        path = Path(p["img_path"])
        ext = path.suffix.lower()
        if TILE_NAME_RE.match(path.stem):
            continue  # imagettes : traitées par imagette_bands (géométrie exacte de la grille)
        fp = None
        if ext in (".tif", ".tiff"):
            fp = _ortho_footprint(p, origin_holder)
        elif ext in (".jpg", ".jpeg"):
            fp = _photo_footprint(p, origin_holder, registry)
        if fp is None:
            unverifiable.append(p["parent_id"])
        else:
            fps[p["parent_id"]] = fp
    return fps, unverifiable


def imagette_geometry(parents: List[Dict]):
    """parent_id -> (clé de source, x, y, largeur, hauteur) pour les imagettes pré-découpées."""
    from PIL import Image

    out = {}
    for p in parents:
        m = TILE_NAME_RE.match(Path(p["img_path"]).stem)
        if not m:
            continue
        with Image.open(p["img_path"]) as im:
            w, h = im.size
        key = (p["batch"], str(Path(p["img_path"]).parent), m.group("prefix"))
        out[p["parent_id"]] = (key, int(m.group("x")), int(m.group("y")), w, h)
    return out


def compute_cross_split_zones(parents: List[Dict], split_map: Dict[str, str], sensor_registry_path: Path):
    """Zones de terrain partagé avec une image d'un split PLUS prioritaire, par image parente.

    Entrée : images parentes retenues (hors quarantaine) et leur split.
    Sortie : (zones, rapport) - zones = {parent_id: [polygone en pixels, ...]} (seulement les
    images à masquer), rapport = paires détectées, aires, images non vérifiables."""
    zones: Dict[str, List] = {}
    pairs_report = []
    in_scope = [p for p in parents if p["parent_id"] in split_map]

    # 1) Imagettes : bandes exactes entre voisines de la même source.
    geo = imagette_geometry(in_scope)
    by_src: Dict[tuple, List[str]] = {}
    for pid, (key, *_r) in geo.items():
        by_src.setdefault(key, []).append(pid)
    for key, pids in by_src.items():
        for i, a in enumerate(pids):
            _, xa, ya, wa, ha = geo[a]
            for b in pids[i + 1:]:
                _, xb, yb, wb, hb = geo[b]
                sa, sb = split_map[a], split_map[b]
                if sa == sb:
                    continue
                x0, x1 = max(xa, xb), min(xa + wa, xb + wb)
                y0, y1 = max(ya, yb), min(ya + ha, yb + hb)
                if x1 <= x0 or y1 <= y0:
                    continue
                low, high = (a, b) if SPLIT_PRIORITY[sa] < SPLIT_PRIORITY[sb] else (b, a)
                _, xl, yl, _, _ = geo[low]
                rect = [[x0 - xl, y0 - yl], [x1 - xl, y0 - yl], [x1 - xl, y1 - yl], [x0 - xl, y1 - yl]]
                zones.setdefault(low, []).append([[float(u), float(v)] for u, v in rect])
                pairs_report.append({"type": "imagettes", "masquee": low, "split_masquee": split_map[low],
                                     "prioritaire": high, "split_prioritaire": split_map[high],
                                     "aire_px": int((x1 - x0) * (y1 - y0))})

    # 2) Photos et orthos géoréférencées : par site.
    fps, unverifiable = build_footprints(in_scope, sensor_registry_path)
    items = sorted(fps.values(), key=lambda f: f.parent_id)
    for i, fa in enumerate(items):
        for fb in items[i + 1:]:
            if fa.site != fb.site:
                continue
            sa, sb = split_map[fa.parent_id], split_map[fb.parent_id]
            if sa == sb:
                continue
            margin = ORTHO_MARGIN_M if fa.kind == fb.kind == "ortho" else GPS_MARGIN_M
            low, high = (fa, fb) if SPLIT_PRIORITY[sa] < SPLIT_PRIORITY[sb] else (fb, fa)
            zone = low.poly.intersection(high.poly.buffer(margin))
            if zone.is_empty or zone.area <= 0.01:
                continue
            px = low.zone_pixel_polygons(zone)
            if not px:
                continue
            zones.setdefault(low.parent_id, []).extend(px)
            raw_overlap = low.poly.intersection(high.poly).area
            pairs_report.append({"type": f"{low.kind}/{high.kind}", "masquee": low.parent_id,
                                 "split_masquee": split_map[low.parent_id], "prioritaire": high.parent_id,
                                 "split_prioritaire": split_map[high.parent_id],
                                 "recouvrement_m2": round(raw_overlap, 2), "zone_masquee_m2": round(zone.area, 2),
                                 "part_image_masquee": round(zone.area / max(low.poly.area, 1e-9), 4),
                                 "marge_m": margin})
    report = {"logic_version": LOGIC_VERSION, "gps_margin_m": GPS_MARGIN_M, "ortho_margin_m": ORTHO_MARGIN_M,
              "paires": pairs_report, "non_verifiables": sorted(unverifiable),
              "n_images_masquees": len(zones)}
    return zones, report


def zones_digest(zone_polys: List) -> str:
    return hashlib.sha1(json.dumps(zone_polys, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def rasterize_zones(zone_polys: List, width: int, height: int) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), np.uint8)
    for poly in zone_polys:
        pts = np.round(np.asarray(poly, float)).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


NODATA_MASK_DIR = "nodata_masks"


def nodata_mask_path(split_dir: Path, split: str, parent_id: str) -> Path:
    return Path(split_dir) / NODATA_MASK_DIR / split / f"{parent_id}.png"


def write_nodata_mask(mask: np.ndarray, path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Écriture impossible : {path}")


def read_nodata_mask(path: Path) -> Optional[np.ndarray]:
    import cv2

    if not Path(path).exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127
