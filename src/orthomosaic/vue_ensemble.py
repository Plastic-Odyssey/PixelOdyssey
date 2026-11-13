"""Vue d'ensemble de toutes les mosaïques d'un dossier : image nord en haut, étiquettes, échelle, flèche du nord.

Usage : python vue_ensemble.py "D:\\...\\ODM Santa Luzia" "Vue ensemble Santa Luzia.jpg" [--res 0.5]
Chaque sous-dossier « Mosaic ... flight N\\Mosaic ... flight N.tif » est lu à la résolution demandée
(via ses aperçus internes) et collé sur un même canevas en UTM.
"""
import argparse, math, os, re
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pyproj import Transformer

ap = argparse.ArgumentParser()
ap.add_argument("dossier"); ap.add_argument("sortie")
ap.add_argument("--res", type=float, default=0.5)
a = ap.parse_args()
R = a.res

POLICE = r"C:\Windows\Fonts\segoeui.ttf"; GRAS = r"C:\Windows\Fonts\segoeuib.ttf"; LEGER = r"C:\Windows\Fonts\segoeuil.ttf"
ENCRE = (34, 40, 49); ACCENT = (0, 122, 140); FOND = (244, 242, 237)

couches = []
for nom in os.listdir(a.dossier):
    m = re.search(r"flight (\d+)$", nom)
    tif = os.path.join(a.dossier, nom, nom + ".tif")
    if m and os.path.isfile(tif):
        couches.append((int(m.group(1)), tif))
couches.sort()
bornes = []
for _, tif in couches:
    with rasterio.open(tif) as s:
        bornes.append(s.bounds); crs = s.crs

marge_m = 60
x0 = min(b.left for b in bornes) - marge_m; x1 = max(b.right for b in bornes) + marge_m
y0 = min(b.bottom for b in bornes) - marge_m; y1 = max(b.top for b in bornes) + marge_m
W, H = int((x1 - x0) / R), int((y1 - y0) / R)
bandeau = 150          # titre en haut
pied = 170             # échelle et légende en bas
carte = Image.new("RGB", (W, H), FOND)
masque_total = Image.new("L", (W, H), 0)

centres, surfaces = {}, {}
for (n, tif), b in zip(couches, bornes):
    w, h = int((b.right - b.left) / R), int((b.top - b.bottom) / R)
    with rasterio.open(tif) as s:
        d = s.read(out_shape=(4, h, w))
    rgb = Image.fromarray(np.dstack(d[:3]).astype(np.uint8)); al = Image.fromarray(d[3])
    pos = (int((b.left - x0) / R), int((y1 - b.top) / R))
    carte.paste(rgb, pos, al); masque_total.paste(al, pos, al)
    ys, xs = np.nonzero(d[3] > 0)
    # point d'ancrage : pixel couvert le plus proche du centre de gravité de la mosaïque
    cy, cx = ys.mean(), xs.mean()
    i = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
    centres[n] = (pos[0] + xs[i], pos[1] + ys[i])
    surfaces[n] = (d[3] > 0).sum() * R * R / 1e4
    print(f"flight {n}: {surfaces[n]:.2f} ha")

# ombre portée douce sous les mosaïques
ombre = masque_total.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.35))
fond = Image.new("RGB", (W, H), FOND)
fond.paste(Image.new("RGB", (W, H), (120, 115, 105)), (6, 8), ombre)
fond.paste(carte, (0, 0), masque_total)

S = W / 2000          # facteur d'échelle des textes et symboles (lisibles quelle que soit la taille de l'image)
bandeau, pied = int(bandeau * S * 0.9), int(pied * S * 0.9)
img = Image.new("RGB", (W, H + bandeau + pied), FOND)
img.paste(fond, (0, bandeau))
dr = ImageDraw.Draw(img, "RGBA")

def p(v):
    return int(round(v * S))

def police(chemin, taille):
    return ImageFont.truetype(chemin, p(taille))

# Étiquettes : pastille numérotée côté terre (sud-ouest de la côte), reliée à la mosaïque par un trait fin
f_num = police(GRAS, 30); f_lib = police(POLICE, 15)
dx, dy = -p(55), p(62)      # décalage vers le sud-ouest
for n, (cx, cy) in centres.items():
    cy += bandeau
    lx, ly = cx + dx, cy + dy
    dr.line([(cx, cy), (lx, ly)], fill=ENCRE + (220,), width=max(2, p(1.5)))
    dr.ellipse([cx - p(4), cy - p(4), cx + p(4), cy + p(4)], fill=(255, 255, 255, 255), outline=ENCRE + (255,), width=max(2, p(1.5)))
    r = p(21)
    dr.ellipse([lx - r + p(2), ly - r + p(3), lx + r + p(2), ly + r + p(3)], fill=(0, 0, 0, 45))          # ombre
    dr.ellipse([lx - r, ly - r, lx + r, ly + r], fill=ACCENT + (255,), outline=(255, 255, 255, 255), width=p(2.5))
    dr.text((lx, ly - p(1)), str(n), font=f_num, fill=(255, 255, 255), anchor="mm")
    dr.text((lx, ly + r + p(5)), "FLIGHT", font=f_lib, fill=ENCRE, anchor="mt")

# Titre
mx = p(40)
dr.text((mx, p(22)), "Santa Luzia — mosaïques drone", font=police(GRAS, 46), fill=ENCRE)
t = Transformer.from_crs(crs, 4326, always_xy=True)
lon, lat = t.transform((x0 + x1) / 2, (y0 + y1) / 2)
dr.text((mx + p(2), p(82)), f"Cap-Vert · {abs(lat):.3f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.3f}° {'O' if lon < 0 else 'E'} · "
        f"{len(couches)} vols DJI Air 2S à 17 m, du 24 au 26/02/2026 · résolution 0,49 cm/px · {crs.to_string()}",
        font=police(LEGER, 22), fill=(90, 96, 105))
dr.line([(mx, bandeau - p(10)), (W - mx, bandeau - p(10))], fill=ENCRE + (70,), width=max(2, p(1)))

# Encadré récapitulatif (en haut à droite, dans la mer)
f_cle, f_val, f_ligne = police(LEGER, 17), police(GRAS, 34), police(POLICE, 17)
lw, lh = p(330), p(124) + p(26) * len(surfaces)
ex, ey = W - mx - lw, bandeau + p(40)
dr.rounded_rectangle([ex + p(3), ey + p(4), ex + lw + p(3), ey + lh + p(4)], radius=p(14), fill=(0, 0, 0, 35))
dr.rounded_rectangle([ex, ey, ex + lw, ey + lh], radius=p(14), fill=(255, 255, 255, 235), outline=ENCRE + (40,), width=max(1, p(1)))
cx_ = ex + p(24)
for i, (cle, val) in enumerate((("VOLS", str(len(surfaces))), ("SURFACE", f"{sum(surfaces.values()):.1f} ha".replace(".", ",")))):
    x = cx_ + i * p(150)
    dr.text((x, ey + p(22)), cle, font=f_cle, fill=(110, 115, 122))
    dr.text((x, ey + p(42)), val, font=f_val, fill=ACCENT)
yy = ey + p(108)
dr.line([(cx_, yy - p(10)), (ex + lw - p(24), yy - p(10))], fill=ENCRE + (40,), width=max(1, p(1)))
smax = max(surfaces.values())
for n, s_ha in surfaces.items():
    dr.text((cx_, yy), f"Flight {n}", font=f_ligne, fill=ENCRE)
    bx0, bx1 = cx_ + p(90), ex + lw - p(90)
    dr.rounded_rectangle([bx0, yy + p(5), bx0 + (bx1 - bx0) * s_ha / smax, yy + p(15)], radius=p(4), fill=ACCENT + (190,))
    dr.text((ex + lw - p(24), yy), f"{s_ha:.2f} ha".replace(".", ","), font=f_ligne, fill=ENCRE, anchor="ra")
    yy += p(26)

# Échelle (alternance foncé / blanc) en bas à gauche
f_ech = police(POLICE, 18)
long_m = 10 ** math.floor(math.log10((x1 - x0) / 5)); long_m *= 2 if (x1 - x0) / 5 / long_m >= 2 else 1
pas = 4; seg = long_m / pas / R
bx, by = mx + p(4), H + bandeau + p(45)
for i in range(pas):
    dr.rectangle([bx + i * seg, by, bx + (i + 1) * seg, by + p(9)], fill=(ENCRE if i % 2 == 0 else (255, 255, 255)), outline=ENCRE, width=max(2, p(1)))
for i in (0, pas // 2, pas):
    dr.text((bx + i * seg, by + p(14)), f"{int(long_m * i / pas)}", font=f_ech, fill=ENCRE, anchor="mt")
dr.text((bx + pas * seg + p(10), by + p(4)), "m", font=f_ech, fill=ENCRE, anchor="lm")

# Flèche du nord en bas à droite (nord de la grille UTM ≈ nord géographique ici)
nx, ny = W - mx - p(20), H + bandeau + p(55)
dr.polygon([(nx, ny - p(30)), (nx - p(13), ny + p(14)), (nx, ny + p(5))], fill=ENCRE)
dr.polygon([(nx, ny - p(30)), (nx + p(13), ny + p(14)), (nx, ny + p(5))], fill=(255, 255, 255), outline=ENCRE)
dr.text((nx, ny - p(34)), "N", font=police(GRAS, 22), fill=ENCRE, anchor="mb")

# Source
dr.text((W // 2, H + bandeau + p(58)), "Plastic Odyssey · PixelOdyssey · orthomosaïques ODX (OpenDroneMap)",
        font=police(LEGER, 17), fill=(110, 115, 122), anchor="mm")

img.save(os.path.join(a.dossier, a.sortie), quality=92, optimize=True)
print("ecrit", os.path.join(a.dossier, a.sortie), img.size)
