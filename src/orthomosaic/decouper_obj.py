"""Découpe le modèle texturé 2,5D d'un projet ODX en tronçons le long du vol, pour calculer
l'ortho tronçon par tronçon avec odm_orthophoto (qui charge toute l'ortho de son modèle en RAM).

Tous les tronçons viennent du même modèle et des mêmes textures : ils se superposent exactement,
sans décalage ni écart de teinte. Les faces sont réparties selon la photo la plus proche de leur centre
(photos triées par heure de prise de vue), avec un recouvrement de quelques photos entre tronçons.

Usage : python decouper_obj.py C:\\odm_runs\\sl_w1b [--photos-par-troncon 45] [--recouvrement 5]
Écrit odm_texturing_25d\\troncon_NN.obj (même .mtl et mêmes textures que le modèle d'origine).
"""
import argparse, glob, os, re
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("projet")
ap.add_argument("--photos-par-troncon", type=int, default=45)
ap.add_argument("--recouvrement", type=int, default=5)
a = ap.parse_args()

tex = os.path.join(a.projet, "odm_texturing_25d")
obj = os.path.join(tex, "odm_textured_model_geo.obj")

# Décalage UTM du modèle (coords.txt : 1re ligne = système, 2e ligne = « est nord »)
lignes = open(os.path.join(a.projet, "odm_georeferencing", "coords.txt")).read().split("\n")
dx, dy = map(float, lignes[1].split()[:2])
zone = int(re.search(r"UTM\s*(\d+)", lignes[0]).group(1))
tr = Transformer.from_crs(4326, 32600 + zone, always_xy=True)

def dms(v, r):
    x = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return -x if r in ("S", "W") else x

photos = []
for f in glob.glob(os.path.join(a.projet, "images", "*.JPG")):
    ex = Image.open(f)._getexif() or {}
    g = ex.get(34853)
    if not g:
        continue
    x, y = tr.transform(dms(g[4], g[3]), dms(g[2], g[1]))
    photos.append((ex.get(36867, ""), os.path.basename(f), x - dx, y - dy))
photos.sort()
pos = np.array([(p[2], p[3]) for p in photos])
n = len(photos)
k = max(1, round(n / a.photos_par_troncon))
bornes = np.linspace(0, n, k + 1).astype(int)
print(f"{n} photos -> {k} troncons (environ {n // k} photos chacun, recouvrement {a.recouvrement} photos)")

# Lecture de l'OBJ
v, vt, vn, faces = [], [], [], []   # faces : (materiau, [(iv, ivt, ivn), ...])
mtllib, mat = None, None
for l in open(obj, encoding="utf-8", errors="replace"):
    if l.startswith("v "):
        v.append(l)
    elif l.startswith("vt "):
        vt.append(l)
    elif l.startswith("vn "):
        vn.append(l)
    elif l.startswith("f "):
        faces.append((mat, [tuple(int(t) if t else 0 for t in (c.split("/") + ["", ""])[:3]) for c in l.split()[1:]]))
    elif l.startswith("usemtl "):
        mat = l.split(None, 1)[1].strip()
    elif l.startswith("mtllib "):
        mtllib = l.split(None, 1)[1].strip()
xy = np.array([[float(t) for t in s.split()[1:3]] for s in v])
print(f"modele : {len(v)} sommets, {len(faces)} faces, {len(set(f[0] for f in faces))} materiaux")

# Photo la plus proche du centre de chaque face
centres = np.array([xy[[c[0] - 1 for c in f[1]]].mean(axis=0) for f in faces])
_, proche = cKDTree(pos).query(centres)

for t in range(k):
    debut, fin = max(0, bornes[t] - a.recouvrement), min(n, bornes[t + 1] + a.recouvrement)
    choix = [i for i in range(len(faces)) if debut <= proche[i] < fin]
    rv, rvt, rvn = {}, {}, {}
    sortie = [f"mtllib {mtllib}\n"] if mtllib else []
    corps, mat_cour = [], None
    for i in sorted(choix, key=lambda i: faces[i][0] or ""):
        m, sommets = faces[i]
        if m != mat_cour:
            corps.append(f"usemtl {m}\n"); mat_cour = m
        elems = []
        for iv, ivt, ivn in sommets:
            a1 = rv.setdefault(iv, len(rv) + 1)
            e = str(a1)
            if ivt or ivn:
                e += "/" + (str(rvt.setdefault(ivt, len(rvt) + 1)) if ivt else "")
            if ivn:
                e += "/" + str(rvn.setdefault(ivn, len(rvn) + 1))
            elems.append(e)
        corps.append("f " + " ".join(elems) + "\n")
    sortie += [v[i - 1] for i in sorted(rv, key=rv.get)]
    sortie += [vt[i - 1] for i in sorted(rvt, key=rvt.get)]
    sortie += [vn[i - 1] for i in sorted(rvn, key=rvn.get)]
    nom = os.path.join(tex, f"troncon_{t:02d}.obj")
    open(nom, "w", encoding="utf-8").writelines(sortie + corps)
    b = xy[[i - 1 for i in rv]]
    lx, ly = np.ptp(b[:, 0]), np.ptp(b[:, 1])
    print(f"  troncon {t:02d} : photos {debut}-{fin - 1}, {len(choix)} faces, emprise {lx:.0f} x {ly:.0f} m "
          f"-> environ {(lx / 0.0049) * (ly / 0.0049) * 7.5 / 1e9:.1f} Go de RAM pour l'ortho (si le rectangle est celui des sommets)")
