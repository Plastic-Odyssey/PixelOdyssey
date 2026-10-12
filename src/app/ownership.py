"""Attribution de chaque point du sol à « sa » photo : celle qui le voit le plus près de son centre.

Une carte de propriété au sol (grille de ~20 cm) sert à la fois :
  - à dédoublonner sans seuil : une détection n'est gardée que sur la photo propriétaire de son centre ;
  - à ne lancer le modèle que sur la partie utile de chaque photo (sa zone + une marge) ;
  - à placer les coutures de la mosaïque exactement aux mêmes endroits.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from . import config
from .inventory import Photo
from .registration import PhotoTransform


def footprint_ground(p: Photo, T: PhotoTransform, per_edge: int = 8) -> np.ndarray:
    """Contour de l'emprise au sol (bords échantillonnés : la distorsion les courbe légèrement)."""
    w, h = p.width, p.height
    edges = []
    for (x0, y0), (x1, y1) in [((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))]:
        for s in np.linspace(0, 1, per_edge, endpoint=False):
            edges.append((x0 + (x1 - x0) * s, y0 + (y1 - y0) * s))
    return T.to_ground(np.array(edges))


class Ownership:
    def __init__(self, photos: List[Photo], transforms: Dict[int, PhotoTransform], res: float = config.OWNER_GRID_M):
        self.photos = photos
        self.T = transforms
        self.fp = {p.idx: footprint_ground(p, transforms[p.idx]) for p in photos}
        allpts = np.vstack(list(self.fp.values()))
        self.x0, self.y0 = allpts.min(0) - 1.0
        x1, y1 = allpts.max(0) + 1.0
        # au plus ~16 millions de cases
        area = (x1 - self.x0) * (y1 - self.y0)
        self.res = max(res, float(np.sqrt(area / 16e6)))
        self.W = int(np.ceil((x1 - self.x0) / self.res))
        self.H = int(np.ceil((y1 - self.y0) / self.res))
        best = np.full((self.H, self.W), np.inf, dtype=np.float32)
        owner = np.full((self.H, self.W), -1, dtype=np.int32)
        for p in photos:
            g = self.fp[p.idx]
            gx = (g[:, 0] - self.x0) / self.res
            gy = (g[:, 1] - self.y0) / self.res
            c0, c1 = max(0, int(gx.min())), min(self.W, int(np.ceil(gx.max())) + 1)
            r0, r1 = max(0, int(gy.min())), min(self.H, int(np.ceil(gy.max())) + 1)
            if c1 <= c0 or r1 <= r0:
                continue
            mask = np.zeros((r1 - r0, c1 - c0), np.uint8)
            cv2.fillPoly(mask, [np.round(np.stack([gx - c0, gy - r0], 1) * 4).astype(np.int32)], 1, shift=2)
            cx, cy = transforms[p.idx].t
            xs = self.x0 + (np.arange(c0, c1) + 0.5) * self.res - cx
            ys = self.y0 + (np.arange(r0, r1) + 0.5) * self.res - cy
            d2 = (xs[None, :] ** 2 + ys[:, None] ** 2).astype(np.float32)
            win_best = best[r0:r1, c0:c1]
            upd = (mask > 0) & (d2 < win_best)
            win_best[upd] = d2[upd]
            owner[r0:r1, c0:c1][upd] = p.idx
        self.owner = owner

    def owner_of(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        c = ((pts[:, 0] - self.x0) / self.res).astype(int)
        r = ((pts[:, 1] - self.y0) / self.res).astype(int)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        out = np.full(len(pts), -1, dtype=np.int64)
        out[ok] = self.owner[r[ok], c[ok]]
        return out

    def owned_cells(self, idx: int) -> np.ndarray:
        rr, cc = np.nonzero(self.owner == idx)
        return np.stack([self.x0 + (cc + 0.5) * self.res, self.y0 + (rr + 0.5) * self.res], 1)

    def owned_area_m2(self, idx: int) -> float:
        return float(np.count_nonzero(self.owner == idx)) * self.res ** 2

    def covered_area_m2(self) -> float:
        return float(np.count_nonzero(self.owner >= 0)) * self.res ** 2

    def useful_windows(self, p: Photo, windows: List, margin: int = config.TILE_MARGIN_PX) -> List:
        """Fenêtres de tuilage de la photo qui touchent sa zone propre (plus une marge)."""
        cells = self.owned_cells(p.idx)
        if len(cells) == 0:
            return []
        px = self.T[p.idx].to_pixel(cells)
        keep = []
        for (x0, y0, x1, y1) in windows:
            inside = ((px[:, 0] >= x0 - margin) & (px[:, 0] < x1 + margin) &
                      (px[:, 1] >= y0 - margin) & (px[:, 1] < y1 + margin))
            if inside.any():
                keep.append((x0, y0, x1, y1))
        return keep
