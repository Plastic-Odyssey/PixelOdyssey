"""Masque d'une ortho ODX : met à transparent tout ce qui est hors de l'emprise réelle des photos.

ODX (--crop 0) peut prolonger le maillage au-delà de ce que les photos voient : on obtient
alors des zones de pixels étirés, sans image réelle. Ce script calcule l'union des empreintes
au sol des photos (position GPS, cap de la nacelle, altitude relative, nacelle à -85 deg),
élargie d'une marge, et :
  --controle : mesure la surface d'ortho hors emprise (rien n'est écrit) ;
  --sortie F : écrit une copie de l'ortho avec l'alpha mis à 0 hors emprise.

Usage : python masque_emprise.py ortho.tif dossier_photos [--marge 2] [--exclure A.JPG,B.JPG]
                                 [--controle] [--sortie ortho_masquee.tif] [--geojson emprise.geojson]
"""
import argparse, glob, json, math, os, re
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

CAPTEUR_MM, FOCALE_MM, LARGEUR_PX, HAUTEUR_PX = 13.2, 8.38, 5472, 3648   # DJI Air 2S (FC3411)

ap = argparse.ArgumentParser()
ap.add_argument("ortho"); ap.add_argument("photos")
ap.add_argument("--marge", type=float, default=2.0)
ap.add_argument("--exclure", default="")
ap.add_argument("--controle", action="store_true")
ap.add_argument("--sortie"); ap.add_argument("--geojson")
a = ap.parse_args()
exclues = {x.strip().upper() for x in a.exclure.split(",") if x.strip()}

def dms(v, r):
    x = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return -x if r in ("S", "W") else x

with rasterio.open(a.ortho) as s:
    crs, T, W, H, res = s.crs, s.transform, s.width, s.height, s.res[0]
tr = Transformer.from_crs(4326, crs, always_xy=True)

empreintes = []
for f in sorted(glob.glob(os.path.join(a.photos, "*.JPG"))):
    if os.path.basename(f).upper() in exclues:
        continue
    ex = Image.open(f)._getexif() or {}
    g = ex.get(34853)
    raw = open(f, "rb").read(80000)
    def xmp(k):
        m = re.search(rb'drone-dji:' + k + rb'="([-+0-9.]+)"', raw)
        return float(m.group(1)) if m else None
    h = xmp(b"RelativeAltitude"); cap = xmp(b"GimbalYawDegree")
    if cap is None: cap = xmp(b"FlightYawDegree")
    pitch = xmp(b"GimbalPitchDegree") or -90.0
    if not g or h is None or h < 5:
        continue   # photo de décollage ou sans GPS
    x, y = tr.transform(dms(g[4], g[3]), dms(g[2], g[1]))
    gsd = CAPTEUR_MM * h / (FOCALE_MM * LARGEUR_PX)
    demi_l, demi_h = gsd * LARGEUR_PX / 2, gsd * HAUTEUR_PX / 2   # largeur perpendiculaire au cap
    avance = h * math.tan(math.radians(90 + pitch))                # décalage vers l'avant (nacelle inclinée)
    c = math.radians(cap)
    ax, ay = math.sin(c), math.cos(c)        # vecteur du cap (vers le haut de l'image)
    px_, py_ = math.cos(c), -math.sin(c)     # vecteur vers la droite de l'image
    cx, cy = x + ax * avance, y + ay * avance
    coins = [(cx + sx * demi_l * px_ + sy * demi_h * ax, cy + sx * demi_l * py_ + sy * demi_h * ay)
             for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    empreintes.append(Polygon(coins))

emprise = unary_union(empreintes).buffer(a.marge)
print(f"{len(empreintes)} empreintes ; emprise {emprise.area/1e4:.2f} ha (marge {a.marge} m)")
if a.geojson:
    json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(emprise)}]},
              open(a.geojson, "w"))

if a.controle:
    f = 16
    with rasterio.open(a.ortho) as s:
        alpha = s.read(4, out_shape=(H // f, W // f))
    Tf = T * T.scale(W / (W // f), H / (H // f))
    dedans = ~geometry_mask([mapping(emprise)], out_shape=alpha.shape, transform=Tf)
    px = (res * W / (W // f)) ** 2
    hors = (alpha > 0) & ~dedans
    print(f"ortho couverte {(alpha > 0).sum()*px/1e4:.2f} ha ; hors emprise {hors.sum()*px:.0f} m2 "
          f"({100*hors.sum()/max(1,(alpha>0).sum()):.1f} %)")

if a.sortie:
    with rasterio.open(a.ortho) as s:
        prof = s.profile.copy()
        prof.update(driver="GTiff", tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                    predictor=2, BIGTIFF="YES", num_threads="all_cpus")
        with rasterio.open(a.sortie, "w", **prof) as d:
            pas = 4096
            for r0 in range(0, H, pas):
                for c0 in range(0, W, pas):
                    w = Window(c0, r0, min(pas, W - c0), min(pas, H - r0))
                    bloc = s.read(window=w)
                    dedans = ~geometry_mask([mapping(emprise)], out_shape=(w.height, w.width),
                                            transform=s.window_transform(w))
                    bloc[3][~dedans] = 0
                    bloc[:3, :, :][:, ~dedans] = 0
                    d.write(bloc, window=w)
    print("ecrit", a.sortie)
