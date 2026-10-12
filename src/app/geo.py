"""Conversions de coordonnées : WGS84 <-> UTM (mètres), tuiles web Mercator."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform as rio_transform

WGS84 = CRS.from_epsg(4326)
WEB_MERCATOR = CRS.from_epsg(3857)
EARTH_R = 6378137.0


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6.0) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def lonlat_to(crs: CRS, lons: Iterable[float], lats: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = rio_transform(WGS84, crs, list(lons), list(lats))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def to_lonlat(crs: CRS, xs: Iterable[float], ys: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    lons, lats = rio_transform(crs, WGS84, list(xs), list(ys))
    return np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64)


def tile_bounds_3857(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    size = 2 * math.pi * EARTH_R / (2 ** z)
    minx = -math.pi * EARTH_R + x * size
    maxy = math.pi * EARTH_R - y * size
    return minx, maxy - size, minx + size, maxy
