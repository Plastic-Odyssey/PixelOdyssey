"""Contrôle d'une ortho finale : surface, trous intérieurs, photos reconstruites, aperçus.

Usage : python controle_ortho.py ortho.tif dossier_sortie [--projet C:\\odm_runs\\sl_wX] [--emprise emprise.geojson]
Écrit dans dossier_sortie : apercu_1sur32.png, zoom_centre.jpg, controle_emprise.jpg (si --emprise).
Affiche des lignes « cle: valeur » reprises dans FAIT.txt.
"""
import argparse, glob, json, os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("ortho"); ap.add_argument("sortie")
ap.add_argument("--projet"); ap.add_argument("--emprise")
a = ap.parse_args()

with rasterio.open(a.ortho) as s:
    T, W, H, res = s.transform, s.width, s.height, s.res[0]
    f = 16
    alpha = s.read(4, out_shape=(H // f, W // f))
    n = H // 32
    rgb = s.read((1, 2, 3), out_shape=(3, n, W // 32))
    a32 = s.read(4, out_shape=(n, W // 32))
    px = (res * W / (W // f)) ** 2
    h, w = alpha.shape
    m = Image.new("L", (w + 2, h + 2), 0)
    m.paste(Image.fromarray(np.where(alpha > 0, 255, 0).astype(np.uint8)), (1, 1))
    ImageDraw.floodfill(m, (0, 0), 128)
    trous = (np.array(m)[1:-1, 1:-1] == 0).sum() * px
    ys, xs = np.nonzero(alpha > 0)
    cy, cx = int(np.median(ys)) * f, int(np.median(xs)) * f
    z = s.read((1, 2, 3), window=Window(cx - 600, cy - 400, 1200, 800))
    print(f"taille_px: {W} x {H}")
    print(f"resolution_cm: {res*100:.2f}")
    print(f"crs: {s.crs}")
    print(f"apercus_internes: {s.overviews(1)}")
print(f"surface_ha: {(alpha > 0).sum() * px / 1e4:.2f}")
print(f"trous_interieurs_m2: {trous:.0f}")

os.makedirs(a.sortie, exist_ok=True)
Image.fromarray(z.transpose(1, 2, 0)).save(os.path.join(a.sortie, "zoom_centre.jpg"), quality=90)
ap_img = Image.fromarray(np.dstack([rgb.transpose(1, 2, 0), a32]).astype(np.uint8))
ap_img.save(os.path.join(a.sortie, "apercu_1sur32.png"))

if a.emprise:
    from shapely.geometry import shape
    c = ap_img.convert("RGB"); d = ImageDraw.Draw(c); k = W / c.size[0]
    g = shape(json.load(open(a.emprise))["features"][0]["geometry"])
    for p in ([g] if g.geom_type == "Polygon" else g.geoms):
        d.line([tuple(v / k for v in ~T * (x, y)) for x, y in p.exterior.coords], fill=(255, 0, 0), width=3)
    c.save(os.path.join(a.sortie, "controle_emprise.jpg"), quality=85)

if a.projet:
    # vol en un bloc : opensfm/reconstruction.json ; vol découpé : un reconstruction.json par sous-modèle
    fichiers = glob.glob(os.path.join(a.projet, "submodels", "submodel_*", "opensfm", "reconstruction.json"))
    if not fichiers:
        fichiers = [os.path.join(a.projet, "opensfm", "reconstruction.json")]
    photos, morceaux = set(), 0
    for fr in fichiers:
        for r in json.load(open(fr)):
            morceaux += 1
            photos.update(r["shots"])
    images = sorted(os.listdir(os.path.join(a.projet, "images")))
    perdues = [i for i in images if i not in photos]
    print(f"reconstruction: {morceaux} morceau(x) sur {len(fichiers)} sous-modele(s), {len(photos)} photos sur {len(images)}")
    print(f"photos_perdues: {', '.join(perdues) if perdues else 'aucune'}")
