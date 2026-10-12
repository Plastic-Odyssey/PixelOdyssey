"""Inventaire d'un dossier de photos de drone : métadonnées, résolution au sol, emprises, alertes.

Lecture volontairement légère (EXIF + bloc XMP en tête de fichier, sans décoder l'image) :
quelques millisecondes par photo, pour afficher la carte de contrôle presque immédiatement.
"""
from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from . import config, geo

_XMP_RE = re.compile(rb'drone-dji:(\w+)="([^"]*)"')


@dataclass
class Photo:
    idx: int
    name: str
    path: str
    width: int = 0
    height: int = 0
    lon: float = float("nan")
    lat: float = float("nan")
    alt: float = float("nan")
    yaw: float = float("nan")
    pitch: float = float("nan")
    focal_mm: float = float("nan")
    sensor_width_mm: float = float("nan")
    sensor_label: str = ""
    make: str = ""
    model: str = ""
    date: str = ""
    gsd: float = float("nan")              # m / pixel
    usable: bool = True
    alerts: List[str] = field(default_factory=list)
    # repère local (mètres, origine = centre du lot, axes UTM)
    A0: Optional[np.ndarray] = None        # 2x2 : pixel centré -> mètres, d'après altitude et cap
    t0: Optional[np.ndarray] = None        # position GPS du centre de la photo

    def corners_px(self) -> np.ndarray:
        return np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]], dtype=np.float64)


def _ratio(v) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        try:
            return float(v[0]) / float(v[1])
        except Exception:  # noqa: BLE001
            return float("nan")


def _dms(vals, ref) -> float:
    d, m, s = (_ratio(x) for x in vals)
    out = d + m / 60.0 + s / 3600.0
    return -out if str(ref).upper() in ("S", "W") else out


def read_photo(idx: int, path: Path, sensors: Dict) -> Photo:
    ph = Photo(idx=idx, name=path.name, path=str(path))
    try:
        with open(path, "rb") as f:
            head = f.read(200_000)
        xmp = {k.decode(): v.decode(errors="ignore") for k, v in _XMP_RE.findall(head)}
        with Image.open(path) as im:
            ph.width, ph.height = im.size
            exif = im.getexif()
            ph.make = str(exif.get(271, "")).strip("\x00 ")
            ph.model = str(exif.get(272, "")).strip("\x00 ")
            ex = exif.get_ifd(0x8769)
            gps = exif.get_ifd(0x8825)
        ph.focal_mm = _ratio(ex.get(37386, float("nan")))
        f35 = _ratio(ex.get(41989, float("nan")))
        dt = ex.get(36867) or exif.get(306)
        if dt:
            ph.date = str(dt)
        if gps and 2 in gps and 4 in gps:
            ph.lat = _dms(gps[2], gps.get(1, "N"))
            ph.lon = _dms(gps[4], gps.get(3, "E"))
        elif "GpsLatitude" in xmp:
            ph.lat, ph.lon = float(xmp["GpsLatitude"]), float(xmp["GpsLongitude"])
        if "RelativeAltitude" in xmp:
            ph.alt = float(xmp["RelativeAltitude"])
        if "GimbalYawDegree" in xmp:
            ph.yaw = float(xmp["GimbalYawDegree"])
        elif "FlightYawDegree" in xmp:
            ph.yaw = float(xmp["FlightYawDegree"])
        if "GimbalPitchDegree" in xmp:
            ph.pitch = float(xmp["GimbalPitchDegree"])
        spec = (sensors.get(ph.make) or {}).get(ph.model)
        if spec:
            ph.sensor_width_mm = float(spec["sensor_width_mm"])
            ph.sensor_label = spec.get("label", ph.model)
        elif math.isfinite(f35) and f35 > 0 and math.isfinite(ph.focal_mm):
            ph.sensor_width_mm = 36.0 * ph.focal_mm / f35
            ph.sensor_label = f"{ph.make} {ph.model} (capteur estimé)"
            ph.alerts.append("capteur inconnu : résolution au sol estimée")
    except Exception as e:  # noqa: BLE001
        ph.usable = False
        ph.alerts.append(f"lecture impossible ({e.__class__.__name__})")
        return ph
    if not (math.isfinite(ph.lat) and math.isfinite(ph.lon)):
        ph.usable = False
        ph.alerts.append("pas de position GPS")
    if not math.isfinite(ph.alt) or ph.alt <= 1:
        ph.usable = False
        ph.alerts.append("altitude de vol absente (photo d'un drone non DJI ?)")
    if not math.isfinite(ph.yaw):
        ph.usable = False
        ph.alerts.append("orientation (cap) absente")
    if not math.isfinite(ph.sensor_width_mm) or not math.isfinite(ph.focal_mm):
        ph.usable = False
        ph.alerts.append("capteur inconnu")
    if math.isfinite(ph.pitch) and abs(ph.pitch + 90.0) > config.MAX_PITCH_DEV_DEG:
        ph.usable = False
        ph.alerts.append(f"photo non verticale (inclinaison {ph.pitch:+.0f}°)")
    if ph.usable:
        ph.gsd = ph.alt * ph.sensor_width_mm / (ph.focal_mm * ph.width)
    return ph


def naive_affine(ph: Photo) -> np.ndarray:
    """Pixel centré (dx vers la droite, dy vers le bas) -> (Est, Nord) en mètres, d'après la
    résolution au sol et le cap (même convention que src/application/geolocation.py)."""
    th = math.radians(ph.yaw)
    c, s = math.cos(th), math.sin(th)
    return ph.gsd * np.array([[c, -s], [-s, -c]], dtype=np.float64)


class Inventory:
    def __init__(self, folder: str, photos: List[Photo], crs, origin: np.ndarray):
        self.folder = folder
        self.photos = photos
        self.crs = crs
        self.origin = origin  # (X, Y) UTM du repère local

    @property
    def usable(self) -> List[Photo]:
        return [p for p in self.photos if p.usable]

    def footprint_local(self, ph: Photo, A: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None) -> Polygon:
        A = ph.A0 if A is None else A
        t = ph.t0 if t is None else t
        c = np.array([ph.width / 2.0, ph.height / 2.0])
        pts = (ph.corners_px() - c) @ A.T + t
        return Polygon(pts)

    def to_lonlat(self, pts_local: np.ndarray) -> np.ndarray:
        lon, lat = geo.to_lonlat(self.crs, pts_local[:, 0] + self.origin[0], pts_local[:, 1] + self.origin[1])
        return np.stack([lon, lat], axis=1)


def scan(folder: str) -> Inventory:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in config.PHOTO_EXTS
                   and not any(part.startswith(".") for part in p.relative_to(root).parts))
    if not files:
        raise ValueError("Aucune photo JPG dans ce dossier (les PNG n'ont pas les métadonnées GPS du drone).")
    sensors = config.sensor_registry()
    with ThreadPoolExecutor(max_workers=8) as ex:
        photos = list(ex.map(lambda a: read_photo(a[0], a[1], sensors), enumerate(files)))
    ok = [p for p in photos if p.usable]
    if not ok:
        raise ValueError("Aucune photo exploitable : " + "; ".join(sorted({a for p in photos for a in p.alerts}))[:300])
    lon0 = float(np.median([p.lon for p in ok]))
    lat0 = float(np.median([p.lat for p in ok]))
    crs = geo.utm_crs_for(lon0, lat0)
    xs, ys = geo.lonlat_to(crs, [p.lon for p in ok], [p.lat for p in ok])
    origin = np.array([float(np.round(np.median(xs))), float(np.round(np.median(ys)))])
    for p, x, y in zip(ok, xs, ys):
        p.A0 = naive_affine(p)
        p.t0 = np.array([x, y]) - origin
    inv = Inventory(folder, photos, crs, origin)
    # altitude atypique
    alts = np.array([p.alt for p in ok])
    med_alt = float(np.median(alts))
    for p in ok:
        if abs(p.alt - med_alt) > 0.3 * med_alt:
            p.alerts.append(f"altitude atypique ({p.alt:.0f} m pour {med_alt:.0f} m en médiane)")
    # photos isolées : aucune voisine ne recouvre au moins 15 % de leur emprise
    polys = [inv.footprint_local(p) for p in ok]
    tree = STRtree(polys)
    best_overlap = []
    for i, (p, g) in enumerate(zip(ok, polys)):
        ov = 0.0
        for j in tree.query(g):
            j = int(j)
            if j != i:
                ov = max(ov, g.intersection(polys[j]).area / g.area)
        best_overlap.append(ov)
        if ov < 0.15:
            p.alerts.append("photo isolée : aucune voisine ne la recouvre")
    inv.best_overlap = best_overlap
    inv.union_area = float(unary_union(polys).area) if polys else 0.0
    return inv


def to_json(inv: Inventory) -> Dict:
    ok = inv.usable
    items = []
    for p in inv.photos:
        d = {"nom": p.name, "chemin": p.path, "utilisable": p.usable, "alertes": p.alerts,
             "lon": None if not math.isfinite(p.lon) else p.lon, "lat": None if not math.isfinite(p.lat) else p.lat,
             "alt": None if not math.isfinite(p.alt) else round(p.alt, 1),
             "gsd_cm": None if not math.isfinite(p.gsd) else round(p.gsd * 100, 3), "emprise": None,
             "etat": "exclue" if not p.usable else ("isolee" if any("isolée" in a for a in p.alerts) else
                                                     ("alerte" if p.alerts else "ok"))}
        if p.usable:
            ring = inv.to_lonlat(np.asarray(inv.footprint_local(p).exterior.coords)[:4])
            d["emprise"] = [[round(float(a), 7), round(float(b), 7)] for a, b in ring]
        items.append(d)
    gsds = [p.gsd for p in ok]
    dates = sorted(p.date for p in ok if p.date)
    n_iso = sum(1 for d in items if d["etat"] == "isolee")
    est = estimate_minutes(len(ok))
    resume = {
        "n_total": len(inv.photos), "n_utilisables": len(ok), "n_exclues": len(inv.photos) - len(ok),
        "n_isolees": n_iso, "surface_m2": round(inv.union_area, 1),
        "gsd_cm_median": round(float(np.median(gsds)) * 100, 3), "gsd_cm_min": round(min(gsds) * 100, 3),
        "gsd_cm_max": round(max(gsds) * 100, 3), "altitude_median": round(float(np.median([p.alt for p in ok])), 1),
        "recouvrement_median": round(float(np.median(inv.best_overlap)) * 100) if ok else 0,
        "capteurs": sorted({p.sensor_label for p in ok}), "duree_estimee_min": est,
        "date_debut": _fmt_date(dates[0]) if dates else None, "date_fin": _fmt_date(dates[-1]) if dates else None,
        "epsg": inv.crs.to_epsg(),
    }
    return {"dossier": inv.folder, "photos": items, "resume": resume}


def estimate_minutes(n: int) -> int:
    # poste avec GPU : ~0,6 s de points homologues, ~0,5 s d'appariements, ~1 s de détection
    # et ~0,4 s de mosaïque par photo, plus le chargement du modèle
    return max(1, int(round((n * 2.6 + 30) / 60)))


def _fmt_date(s: str) -> str:
    try:
        return datetime.strptime(s, "%Y:%m:%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
    except Exception:  # noqa: BLE001
        return s
