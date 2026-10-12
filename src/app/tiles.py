"""Tuiles web (XYZ, Web Mercator) servies à la demande depuis la mosaïque GeoTIFF.

Seule la portion utile est lue, au niveau d'aperçu adapté au zoom : la carte reste fluide même
avec une mosaïque de plusieurs gigaoctets.
"""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds

from . import geo

_EMPTY: Optional[bytes] = None
_REG_LOCK = threading.Lock()
_POOLS: dict = {}
POOL_SIZE = 4


class _Pool:
    """Quelques jeux de données rasterio ouverts sur le même fichier (non partageables entre threads)."""

    def __init__(self, path: str):
        self.path = path
        self.free: list = []
        self.count = 0
        self.closed = False
        self.cond = threading.Condition()

    @contextmanager
    def get(self):
        with self.cond:
            while not self.free and self.count >= POOL_SIZE:
                self.cond.wait()
            ds = self.free.pop() if self.free else None
            if ds is None:
                self.count += 1
        if ds is None:
            try:
                ds = rasterio.open(self.path)
            except Exception:
                with self.cond:
                    self.count -= 1
                    self.cond.notify()
                raise
        try:
            yield ds
        finally:
            with self.cond:
                if self.closed:
                    ds.close()
                    self.count -= 1
                else:
                    self.free.append(ds)
                self.cond.notify_all()

    def close(self, timeout: float = 10.0):
        with self.cond:
            self.closed = True
            for ds in self.free:
                ds.close()
                self.count -= 1
            self.free.clear()
            self.cond.wait_for(lambda: self.count <= 0, timeout=timeout)


def _key(path) -> str:
    return str(Path(path).resolve())


def _pool(path) -> _Pool:
    k = _key(path)
    with _REG_LOCK:
        pool = _POOLS.get(k)
        if pool is None:
            pool = _POOLS[k] = _Pool(k)
        return pool


def release(path) -> None:
    """Ferme les accès ouverts sur ce fichier (indispensable sous Windows avant de le remplacer)."""
    with _REG_LOCK:
        pool = _POOLS.pop(_key(path), None)
    if pool is not None:
        pool.close()


def empty_png() -> bytes:
    global _EMPTY
    if _EMPTY is None:
        ok, buf = cv2.imencode(".png", np.zeros((1, 1, 4), np.uint8))
        _EMPTY = buf.tobytes()
    return _EMPTY


def bounds_lonlat(path) -> tuple:
    with _pool(path).get() as ds:
        return transform_bounds(ds.crs, geo.WGS84, *ds.bounds)


def render_tile(path, z: int, x: int, y: int, size: int = 256) -> bytes:
    with _pool(path).get() as ds:
        return _render(ds, z, x, y, size)


def _render(ds, z: int, x: int, y: int, size: int) -> bytes:
    mb = geo.tile_bounds_3857(z, x, y)
    try:
        sb = transform_bounds(geo.WEB_MERCATOR, ds.crs, *mb, densify_pts=5)
    except Exception:  # noqa: BLE001
        return empty_png()
    b = ds.bounds
    if sb[2] <= b.left or sb[0] >= b.right or sb[3] <= b.bottom or sb[1] >= b.top:
        return empty_png()
    win = from_bounds(*sb, transform=ds.transform)
    col0 = max(0, int(math.floor(win.col_off)) - 1)
    row0 = max(0, int(math.floor(win.row_off)) - 1)
    col1 = min(ds.width, int(math.ceil(win.col_off + win.width)) + 1)
    row1 = min(ds.height, int(math.ceil(win.row_off + win.height)) + 1)
    wpx, hpx = col1 - col0, row1 - row0
    if wpx <= 0 or hpx <= 0:
        return empty_png()
    # taille de la portion lue rapportée à la tuile : lecture au niveau d'aperçu adapté
    frac = max(wpx / max(win.width, 1e-9), hpx / max(win.height, 1e-9))
    # résolution de lecture : pas plus fine que nécessaire (GDAL choisit l'aperçu adapté)
    factor = max(1.0, max(wpx, hpx) / (size * 1.5 * min(1.0, frac)))
    ow, oh = max(1, int(round(wpx / factor))), max(1, int(round(hpx / factor)))
    window = rasterio.windows.Window(col0, row0, wpx, hpx)
    data = ds.read(window=window, out_shape=(3, oh, ow), resampling=Resampling.bilinear)
    mask = ds.read_masks(1, window=window, out_shape=(oh, ow), resampling=Resampling.nearest)
    if not mask.any():
        return empty_png()
    wt = ds.window_transform(window) * rasterio.Affine.scale(wpx / ow, hpx / oh)
    dst_t = transform_from_bounds(*mb, size, size)
    rgb = np.zeros((3, size, size), np.uint8)
    reproject(data, rgb, src_transform=wt, src_crs=ds.crs, dst_transform=dst_t, dst_crs=geo.WEB_MERCATOR,
              resampling=Resampling.bilinear)
    alpha = np.zeros((size, size), np.uint8)
    reproject(mask, alpha, src_transform=wt, src_crs=ds.crs, dst_transform=dst_t, dst_crs=geo.WEB_MERCATOR,
              resampling=Resampling.nearest)
    if not alpha.any():
        return empty_png()
    bgra = np.dstack([rgb[2], rgb[1], rgb[0], alpha])
    ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()


def max_native_zoom(path) -> int:
    with _pool(path).get() as ds:
        res = ds.res[0]
    lon0, lat0, lon1, lat1 = bounds_lonlat(path)
    lat = (lat0 + lat1) / 2
    z = math.log2(2 * math.pi * geo.EARTH_R * math.cos(math.radians(lat)) / (256 * res))
    return int(math.ceil(z))
