"""Mosaïque géoréférencée (GeoTIFF tuilé, compressé, avec masque et aperçus = COG pratique).

Chaque pixel de la mosaïque est pris dans la photo « propriétaire » du point (la plus centrale),
exactement comme pour la déduplication des détections : coutures et comptage sont cohérents.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

from . import config, tiles
from .inventory import Photo
from .ownership import Ownership
from .registration import PhotoTransform, imread

Progress = Callable[[float, str], None]


class _ImageCache:
    def __init__(self, size: int = 16):
        self.size = size
        self.data: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

    def get(self, path: str, factor: int) -> Optional[np.ndarray]:
        key = (path, factor)
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                8: cv2.IMREAD_REDUCED_COLOR_8}[factor]
        img = imread(path, flag)
        self.data[key] = img
        if len(self.data) > self.size:
            self.data.popitem(last=False)
        return img


def choose_resolution(photos: List[Photo], transforms: Dict[int, PhotoTransform], own: Ownership) -> float:
    scales = [math.sqrt(abs(np.linalg.det(transforms[p.idx].A))) for p in photos]
    res = 2.0 * float(np.median(scales))           # ~1 cm/pixel pour des photos à 0,5 cm
    rr, cc = np.nonzero(own.owner >= 0)
    if len(rr) == 0:
        raise ValueError("Aucune zone couverte par les photos")
    # le fichier couvre le rectangle englobant (pas seulement la zone vue) : c'est lui qu'il faut plafonner
    bbox_m2 = (cc.max() - cc.min() + 1) * (rr.max() - rr.min() + 1) * own.res ** 2
    n_pix = bbox_m2 / res ** 2
    if n_pix > config.MOSAIC_MAX_PIXELS:
        res *= math.sqrt(n_pix / config.MOSAIC_MAX_PIXELS)
    return round(res, 4)


def render(photos: List[Photo], transforms: Dict[int, PhotoTransform], own: Ownership, crs, origin: np.ndarray,
           out_path: Path, progress: Progress, res: Optional[float] = None) -> Dict:
    res = res or choose_resolution(photos, transforms, own)
    mask_any = own.owner >= 0
    rr, cc = np.nonzero(mask_any)
    gx0 = own.x0 + cc.min() * own.res
    gx1 = own.x0 + (cc.max() + 1) * own.res
    gy0 = own.y0 + rr.min() * own.res
    gy1 = own.y0 + (rr.max() + 1) * own.res
    W = int(math.ceil((gx1 - gx0) / res))
    H = int(math.ceil((gy1 - gy0) / res))
    transform = Affine(res, 0, origin[0] + gx0, 0, -res, origin[1] + gy1)
    by_idx = {p.idx: p for p in photos}
    scales = {p.idx: math.sqrt(abs(np.linalg.det(transforms[p.idx].A))) for p in photos}
    bbox = {}
    for p in photos:
        g = own.fp[p.idx]
        bbox[p.idx] = (g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max())
    cache = _ImageCache(20)
    C = config.MOSAIC_CHUNK
    step = 8  # la correspondance sol -> pixel est calculée tous les 8 pixels puis interpolée
    profile = dict(driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs=crs, transform=transform,
                   tiled=True, blockxsize=512, blockysize=512, compress="JPEG", photometric="YCBCR",
                   jpeg_quality=85, BIGTIFF="IF_SAFER")
    chunks = [(r0, c0) for r0 in range(0, H, C) for c0 in range(0, W, C)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.tif")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(tmp, "w", **profile) as dst:
            for ci, (r0, c0) in enumerate(chunks):
                h, w = min(C, H - r0), min(C, W - c0)
                # coordonnées sol (repère local) du chunk
                X0 = gx0 + c0 * res
                Y1 = gy1 - r0 * res
                cbox = (X0, Y1 - h * res, X0 + w * res, Y1)
                cands = [i for i, b in bbox.items() if not (b[2] < cbox[0] or b[0] > cbox[2] or b[3] < cbox[1] or b[1] > cbox[3])]
                if not cands:
                    continue
                xs = X0 + (np.arange(w) + 0.5) * res
                ys = Y1 - (np.arange(h) + 0.5) * res
                xg, yg = np.meshgrid(xs, ys)
                cw, ch = int(math.ceil((w - 1) / step)) + 1, int(math.ceil((h - 1) / step)) + 1
                # interpolation exacte : le pixel fin (x, y) correspond au point grossier (x/step, y/step)
                fgx, fgy = np.meshgrid(np.arange(w, dtype=np.float32) / step, np.arange(h, dtype=np.float32) / step)
                xs_c = X0 + (np.arange(cw) * step + 0.5) * res
                ys_c = Y1 - (np.arange(ch) * step + 0.5) * res
                xc, yc = np.meshgrid(xs_c, ys_c)
                pts_c = np.stack([xc.ravel(), yc.ravel()], 1)
                best = np.full((h, w), np.inf, dtype=np.float32)
                ownmap = np.full((h, w), -1, dtype=np.int32)
                maps = {}
                for i in cands:
                    T = transforms[i]
                    p = by_idx[i]
                    pc = T.to_pixel(pts_c).astype(np.float32)
                    mx = cv2.remap(np.ascontiguousarray(pc[:, 0].reshape(ch, cw)), fgx, fgy, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                    my = cv2.remap(np.ascontiguousarray(pc[:, 1].reshape(ch, cw)), fgx, fgy, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                    inside = (mx >= 0) & (mx < p.width - 1) & (my >= 0) & (my < p.height - 1)
                    if not inside.any():
                        continue
                    d2 = ((xg - T.t[0]) ** 2 + (yg - T.t[1]) ** 2).astype(np.float32)
                    upd = inside & (d2 < best)
                    best[upd] = d2[upd]
                    ownmap[upd] = i
                    maps[i] = (mx, my)
                if not maps:
                    continue
                rgb = np.zeros((h, w, 3), np.uint8)
                for i, (mx, my) in maps.items():
                    sel = ownmap == i
                    if not sel.any():
                        continue
                    f = 1
                    while f < 8 and scales[i] * f * 2 <= res * 1.01:
                        f *= 2
                    img = cache.get(by_idx[i].path, f)
                    if img is None:
                        continue
                    fx = img.shape[1] / by_idx[i].width
                    fy = img.shape[0] / by_idx[i].height
                    warped = cv2.remap(img, mx * fx, my * fy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    rgb[sel] = warped[sel]
                valid = (ownmap >= 0).astype(np.uint8) * 255
                win = Window(c0, r0, w, h)
                dst.write(rgb[:, :, ::-1].transpose(2, 0, 1), window=win)
                dst.write_mask(valid, window=win)
                progress((ci + 1) / len(chunks) * 0.85, f"Mosaïque : bloc {ci + 1}/{len(chunks)}")
    progress(0.88, "Mosaïque : calcul des aperçus pour l'affichage")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, COMPRESS_OVERVIEW="JPEG", PHOTOMETRIC_OVERVIEW="YCBCR",
                      INTERLEAVE_OVERVIEW="PIXEL", JPEG_QUALITY_OVERVIEW="85"):
        with rasterio.open(tmp, "r+") as ds:
            factors = [f for f in (2, 4, 8, 16, 32, 64, 128) if max(W, H) / f >= 256]
            if factors:
                ds.build_overviews(factors, Resampling.average)
                ds.update_tags(ns="rio_overview", resampling="average")
    tiles.release(out_path)
    if out_path.exists():
        out_path.unlink()
    tmp.replace(out_path)
    preview = out_path.with_name(out_path.stem + "_apercu.jpg")
    with rasterio.open(out_path) as ds:
        f = max(1, int(math.ceil(max(W, H) / 3000)))
        oh, ow = max(1, H // f), max(1, W // f)
        data = ds.read(out_shape=(3, oh, ow), resampling=Resampling.average)
        m = ds.read_masks(1, out_shape=(oh, ow), resampling=Resampling.nearest)
        img = data.transpose(1, 2, 0)[:, :, ::-1].copy()
        img[m == 0] = (255, 255, 255)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        buf.tofile(str(preview))
    progress(1.0, "Mosaïque terminée")
    return {"chemin": str(out_path), "apercu": str(preview), "resolution_m": res, "largeur": W, "hauteur": H,
            "epsg": crs.to_epsg()}
