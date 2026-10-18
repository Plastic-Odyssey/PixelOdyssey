#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Revue du banc de test, sans CVAT (17/10/2026).

Pourquoi : le banc de test figé (82 images) plafonne la mesure. Sur les 6 images de Santa Luzia,
environ un tiers des « fausses alertes sur le fond » du modèle v4 sont des déchets que personne
n'a annotés : un meilleur modèle y est compté comme moins bon. Cette revue complète (et nettoie)
la vérité terrain du banc, directement dans les annotations brutes de 1_annotated_dataset
(source unique), avec sauvegarde des fichiers d'origine.

Trois commandes, depuis la racine du dépôt :

    python -m src.review.revue_banc preparer --campagne campagne_v4_20261012
        Passe les modèles de la campagne sur les 82 images du banc et prépare deux listes :
          A. « déchet oublié ? »  : endroits où au moins un modèle détecte un déchet (confiance
             >= 0,25) sans annotation, hors zones déjà annotées (Inconnu, bois, verre) et hors
             bordure noire. Les détections des différents modèles au même endroit sont fusionnées.
             Le contour proposé vient de SAM (Segment Anything) à partir du cadre de la détection :
             il ne favorise aucun des modèles comparés (un contour pris sur un modèle lui donnerait
             une correspondance parfaite). Mesuré : SAM 2.1 reproduit les annotations humaines du
             banc avec une IoU médiane de 0,84 (95 % au-dessus de 0,5).
          B. « annotation douteuse ? » : objets annotés qu'AUCUN modèle ne retrouve.
        Écrit candidats.json et les vignettes dans le dossier de revue.

    python -m src.review.revue_banc revue
        Ouvre la page de revue dans le navigateur (serveur local, rien n'est envoyé ailleurs).
        Chaque décision est enregistrée immédiatement dans decisions.json : on peut s'arrêter et
        reprendre quand on veut.

    python -m src.review.revue_banc appliquer [--essai]
        Écrit les décisions dans les annotations brutes des images du banc (sauvegarde des
        fichiers d'origine dans le dossier de revue, journal des modifications). Rejouable sans
        doublon. Ensuite :  python -m src.training.campagne_v4 --nom campagne_v4_20261012 --etapes data,eval,tableau --refaire

Décisions possibles :
  A (déchet oublié ?) : déchet -> ajouté avec la classe choisie ; pas un déchet -> rien ;
                        douteux, ou déchet mais contour faux -> ajouté en « Inconnu » (zone ignorée
                        à l'évaluation, ni bonne ni mauvaise réponse).
  B (annotation douteuse ?) : bien un déchet -> rien ; pas un déchet -> annotation retirée ;
                        douteux -> passée en « Inconnu » (zone ignorée).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from src.data.utils.class_config import load_batch_local_names, load_class_config, normalize_class_name
from src.paths_config import DATA_ROOT, PROCESSED_DATASET_DIR, TRAINING_RUNS_DIR, resolve_data_config_path_field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOSSIER = DATA_ROOT / "5. Tests" / "revue_banc_v3"
GT_CONFIG = PROJECT_ROOT / "config" / "data_config_v4.yaml"
RAW_DIR = PROCESSED_DATASET_DIR / "1_annotated_dataset"
CONF_CANDIDAT = 0.25
SAM_POIDS = "sam2.1_b.pt"
INCONNU = "Inconnu"
# Classe brute proposée par défaut pour une super-classe prédite par un modèle multi-classe.
SUPER_VERS_BRUT = {"Bouteille": "bouteille PET", "Flipflops": "fliflops", "Cagette": "cagette", "Bouee": "Bouées",
                   "Cordage_Filet": "Cordage", "Bidon": "Petit bidon", "Debris_Divers": "Fragments plastique rigide"}
# Classes proposées dans la page (raccourci clavier, nom brut, libellé).
CLASSES_REVUE = [
    ("b", "bouteille PET", "Bouteille"), ("d", "Petit bidon", "Bidon"), ("c", "cagette", "Cagette"),
    ("o", "Bouées", "Bouée / flotteur"), ("t", "fliflops", "Tong"), ("f", "Cordage", "Cordage / filet"),
    ("x", "Fragments plastique rigide", "Fragment rigide"), ("s", "Plastique souple (sac)", "Plastique souple"),
    ("m", "Polystyrene", "Mousse / polystyrène"), ("a", "à déterminer", "Déchet, type incertain"),
]


# ============================================================================ préparation
def _split_dir_of(config_path: Path) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        sliced = Path(resolve_data_config_path_field(yaml.safe_load(f) or {})["path"])
    return sliced.parent / sliced.name.replace("4_sliced_dataset", "2_split_dataset", 1)


def _rel_raw(path_str: str) -> str:
    norm = str(path_str).replace("\\", "/")
    marker = "1_annotated_dataset/"
    return norm.split(marker, 1)[1] if marker in norm else norm


def _models_from_campaign(name: str) -> List[Dict]:
    state_path = TRAINING_RUNS_DIR.parent / "campagnes" / name / "campagne.json"
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    out = []
    for label, ev in state.get("evals", {}).items():
        w = Path(ev["run_dir"]) / "weights" / "best.pt"
        if w.exists():
            out.append({"label": label, "poids": str(w), "tax": ev["tax"]})
    if not out:
        raise RuntimeError(f"Aucun modèle évalué trouvé dans {state_path}.")
    return out


def _predict(model, img, conf: float):
    from shapely.geometry import Polygon
    from src.data.utils.tiling_geometry import iter_tile_windows
    from src.review.matching import LabeledPolygon
    from src.review.tiled_inference import nms_merge
    h, w = img.shape[:2]
    windows = list(dict.fromkeys(iter_tile_windows(w, h, 640, 384))) if (w, h) != (640, 640) else [(0, 0, w, h)]
    preds = []
    for i in range(0, len(windows), 8):
        chunk = windows[i:i + 8]
        for (x0, y0, _, _), r in zip(chunk, model.predict([img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk],
                                                          conf=conf, verbose=False)):
            if r.masks is None:
                continue
            for c, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.xy):
                if len(xy) < 3:
                    continue
                g = Polygon([(float(x) + x0, float(y) + y0) for x, y in xy])
                if g.is_valid and g.area > 0:
                    preds.append(LabeledPolygon(int(c), g, float(cf)))
    return nms_merge(preds, iou_threshold=0.5)


def _sam_polygon(sam, img, box):
    """Contour SAM à partir d'un cadre (recadrage autour de l'objet, agrandi à 1024 px)."""
    import cv2
    from shapely.geometry import Polygon
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(160, 3 * max(x1 - x0, y1 - y0))
    cx0, cy0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    cx1, cy1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    crop = img[cy0:cy1, cx0:cx1]
    k = 1024 / max(crop.shape[:2])
    up = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    r = sam(up, bboxes=[[(x0 - cx0) * k, (y0 - cy0) * k, (x1 - cx0) * k, (y1 - cy0) * k]], verbose=False)[0]
    if r.masks is None or not len(r.masks.xy):
        return None
    cands = [a for a in r.masks.xy if len(a) >= 3]
    if not cands:
        return None
    g = Polygon(max(cands, key=lambda a: Polygon(a).area) / k + [cx0, cy0]).buffer(0)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    return g if g.area > 0 else None


def _draw_vignettes(img, item_poly, gts, zones, box, out_base: Path, other_polys=()):
    """Trois vignettes : zoom avec contours, zoom sans contour, contexte (640 px autour)."""
    import cv2
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def crop(size):
        a0, b0 = int(max(0, min(w - size, cx - size / 2))), int(max(0, min(h - size, cy - size / 2)))
        return a0, b0, min(w, a0 + size), min(h, b0 + size)

    def poly_pts(g, a0, b0, k):
        return [np.round((np.asarray(p.exterior.coords) - [a0, b0]) * k).astype(np.int32)
                for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]

    zs = int(max(160, 3 * max(x1 - x0, y1 - y0)))
    a0, b0, a1, b1 = crop(zs)
    z = img[b0:b1, a0:a1]
    k = 420 / max(z.shape[:2])
    plain = cv2.resize(z, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    lined = plain.copy()
    for g in gts:
        cv2.polylines(lined, poly_pts(g, a0, b0, k), True, (90, 220, 90), 1)
    for _, g in zones:
        cv2.polylines(lined, poly_pts(g, a0, b0, k), True, (40, 200, 255), 1)
    for g in other_polys:
        cv2.polylines(lined, poly_pts(g, a0, b0, k), True, (80, 80, 255), 1)
    cv2.polylines(lined, poly_pts(item_poly, a0, b0, k), True, (255, 255, 0), 2)
    cs = 640
    c0, d0, c1, d1 = crop(cs)
    ctx = img[d0:d1, c0:c1].copy()
    m = 0.5 * max(x1 - x0, y1 - y0) + 12
    cv2.rectangle(ctx, (int(x0 - c0 - m / 2), int(y0 - d0 - m / 2)), (int(x1 - c0 + m / 2), int(y1 - d0 + m / 2)),
                  (255, 255, 0), 2)
    ctx = cv2.resize(ctx, None, fx=420 / max(ctx.shape[:2]), fy=420 / max(ctx.shape[:2]), interpolation=cv2.INTER_AREA)
    names = {}
    for suffix, im in (("z", lined), ("n", plain), ("c", ctx)):
        p = out_base.parent / f"{out_base.name}_{suffix}.jpg"
        cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        names[suffix] = p.name
    return names


def preparer(campagne: str, dossier: Path = DOSSIER, sam_poids: str = SAM_POIDS, conf: float = CONF_CANDIDAT,
             models: Optional[List[Dict]] = None) -> Path:
    import cv2
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    from ultralytics import SAM, YOLO
    from src.data.utils.image_io import load_image_bgr
    from src.data.utils.raw_dataset import site_of_batch
    from src.data.utils.slicer import compute_nodata_mask
    from src.data.utils.split_dataset import PARENT_MANIFEST_FILENAME
    from src.review.matching import polygon_iou
    from src.training.object_level_eval import _excluded_zones, _frac_of_pred, _nodata_fraction

    models = models or _models_from_campaign(campagne)
    taxonomy, names = load_class_config(GT_CONFIG)
    split_dir = _split_dir_of(GT_CONFIG)
    with open(split_dir / PARENT_MANIFEST_FILENAME, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    parents = sorted(p for p, v in manifest.items() if v["split"] == "test")
    dossier = Path(dossier)
    (dossier / "vignettes").mkdir(parents=True, exist_ok=True)
    print(f"--- Préparation de la revue du banc : {len(parents)} images, {len(models)} modèles "
          f"({', '.join(m['label'] for m in models)}), confiance >= {conf} ---")
    yolo = {m["label"]: YOLO(m["poids"]) for m in models}
    sam_path = Path(sam_poids)
    if not sam_path.exists():
        cand = PROJECT_ROOT / "models" / "pretrained" / sam_poids
        sam_path = cand if cand.exists() else Path(sam_poids)
    sam = SAM(str(sam_path))
    items: List[Dict] = []
    for k_p, pid in enumerate(parents, 1):
        info = manifest[pid]
        ext = Path(str(info["raw_img_path"]).replace("\\", "/")).suffix.lower()
        img = load_image_bgr(split_dir / "images" / "test" / f"{pid}{ext}")
        h, w = img.shape[:2]
        # Annotations (vérité terrain v4) avec leur texte de coordonnées, pour retrouver la ligne brute.
        gts = []
        lab = split_dir / "labels" / "test" / f"{pid}.txt"
        for line in (lab.read_text(encoding="utf-8").splitlines() if lab.exists() else []):
            parts = line.split()
            if len(parts) < 7:
                continue
            xy = np.array([float(v) for v in parts[1:]]).reshape(-1, 2) * [w, h]
            g = Polygon(xy).buffer(0)
            if g.area > 0:
                gts.append({"cls": int(parts[0]), "geom": g, "coords": " ".join(parts[1:])})
        info_local = dict(info)
        info_local["raw_label_path"] = str(RAW_DIR / _rel_raw(info["raw_label_path"]))
        zones = _excluded_zones(info_local, taxonomy, w, h)
        nodata = compute_nodata_mask(img)
        tree = STRtree([g["geom"] for g in gts]) if gts else None
        clusters: List[Dict] = []
        gt_seen = [False] * len(gts)
        for m in models:
            for p in _predict(yolo[m["label"]], img, conf):
                if nodata.any() and _nodata_fraction(p.geom, nodata) >= 0.5:
                    continue
                near = [int(j) for j in tree.query(p.geom, predicate="intersects")] if tree is not None else []
                on_gt = False
                for j in near:
                    iou = polygon_iou(p.geom, gts[j]["geom"])
                    if iou >= 0.3:
                        gt_seen[j] = True
                    if iou >= 0.3 or _frac_of_pred(p.geom, gts[j]["geom"]) >= 0.3:
                        on_gt = True
                if on_gt or any(_frac_of_pred(p.geom, z) >= 0.5 for _, z in zones):
                    continue
                cl = next((c for c in clusters if polygon_iou(p.geom, c["geom"]) >= 0.3
                           or _frac_of_pred(p.geom, c["geom"]) >= 0.5 or _frac_of_pred(c["geom"], p.geom) >= 0.5), None)
                if cl is None:
                    cl = {"geom": p.geom, "conf": p.confidence, "modeles": {}, "classe": None, "classe_conf": 0.0}
                    clusters.append(cl)
                elif p.confidence > cl["conf"]:
                    cl["geom"], cl["conf"] = p.geom, p.confidence
                cl["modeles"][m["label"]] = round(max(cl["modeles"].get(m["label"], 0.0), p.confidence), 3)
                if m["tax"] == "multi" and p.confidence > cl["classe_conf"]:
                    cl["classe"], cl["classe_conf"] = names.get(p.class_id), p.confidence
        base = {"parent_id": pid, "lot": info["batch"], "site": site_of_batch(info["batch"]),
                "label_brut": _rel_raw(info["raw_label_path"]), "image_brute": _rel_raw(info["raw_img_path"]),
                "largeur": w, "hauteur": h}
        for k_c, cl in enumerate(sorted(clusters, key=lambda c: -c["conf"])):
            sam_g = _sam_polygon(sam, img, cl["geom"].bounds)
            ratio = sam_g.area / cl["geom"].area if sam_g is not None else 0
            poly, source = (sam_g, "SAM") if sam_g is not None and 0.25 <= ratio <= 4 else (cl["geom"], "modèle")
            iid = "A_" + hashlib.sha1(f"{pid}:{k_c}".encode()).hexdigest()[:10]
            box = poly.union(cl["geom"]).bounds
            vign = _draw_vignettes(img, poly, [g["geom"] for g in gts], zones, box, dossier / "vignettes" / iid,
                                   other_polys=[cl["geom"]] if source == "SAM" else [])
            x0, y0, x1, y1 = poly.bounds
            items.append({**base, "id": iid, "type": "A", "conf_max": round(cl["conf"], 3), "modeles": cl["modeles"],
                          "classe_suggeree": SUPER_VERS_BRUT.get(cl["classe"], "Fragments plastique rigide"),
                          "contour_source": source, "taille_px": round(max(x1 - x0, y1 - y0), 1),
                          "polygone": [[round(float(a), 1), round(float(b), 1)] for a, b in poly.exterior.coords],
                          "vignettes": vign})
        for j, g in enumerate(gts):
            if gt_seen[j]:
                continue
            key = f"{pid}:{g['coords'][:60]}:{j}"
            iid = "B_" + hashlib.sha1(key.encode()).hexdigest()[:10]
            vign = _draw_vignettes(img, g["geom"], [o["geom"] for i2, o in enumerate(gts) if i2 != j], zones,
                                   g["geom"].bounds, dossier / "vignettes" / iid)
            x0, y0, x1, y1 = g["geom"].bounds
            items.append({**base, "id": iid, "type": "B", "classe_annotee": names.get(g["cls"]),
                          "coords_texte": g["coords"], "taille_px": round(max(x1 - x0, y1 - y0), 1), "vignettes": vign})
        n_a = sum(1 for it in items if it["parent_id"] == pid and it["type"] == "A")
        n_b = sum(1 for it in items if it["parent_id"] == pid and it["type"] == "B")
        print(f"  [{k_p}/{len(parents)}] {pid} : {len(gts)} annotations, {n_a} déchet(s) oublié(s) ?, "
              f"{n_b} annotation(s) douteuse(s) ?", flush=True)
    # Ordre de revue : d'abord les endroits vus par le plus de modèles et avec la plus forte confiance (les
    # plus probablement de vrais déchets oubliés), puis les plus grandes annotations jamais retrouvées. On
    # peut s'arrêter quand la liste ne propose plus que du sable et des galets : un élément non revu ne
    # change rien au banc.
    items.sort(key=lambda it: (it["type"], -len(it.get("modeles", {})), -it.get("conf_max", 0.0), -it["taille_px"]))
    payload = {"cree": datetime.now().isoformat(timespec="seconds"), "campagne": campagne,
               "modeles": [m["label"] for m in models], "conf": conf, "sam": sam_path.name,
               "verite_terrain": str(GT_CONFIG.name), "classes": [{"touche": t, "brut": b, "libelle": l} for t, b, l in CLASSES_REVUE],
               "items": items}
    out = dossier / "candidats.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ {sum(1 for i in items if i['type'] == 'A')} déchets oubliés possibles, "
          f"{sum(1 for i in items if i['type'] == 'B')} annotations douteuses possibles -> {out}")
    return out


# ============================================================================ page de revue
PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revue du banc de test</title>
<style>
:root{--bg:#f4f5f2;--panel:#ffffff;--ink:#1f2420;--muted:#5d665f;--line:#d9ddd6;--accent:#0b6e69;--accent-ink:#ffffff;
--ok:#2f7d32;--no:#b3261e;--warn:#9a6700;--chip:#eef1ec}
@media (prefers-color-scheme:dark){:root{--bg:#141815;--panel:#1d221e;--ink:#e6ebe6;--muted:#9aa59c;--line:#323a34;
--accent:#3fb3aa;--accent-ink:#0d1211;--ok:#6cc070;--no:#ef7b72;--warn:#e0b54a;--chip:#262d28}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,"Segoe UI",sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 18px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:17px;margin:0;font-weight:650}
.bar{flex:1;min-width:160px;height:8px;background:var(--chip);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
select,button{font:inherit;color:inherit}
select{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
main{display:grid;grid-template-columns:auto 1fr;gap:18px;padding:16px 18px}
.imgs{display:flex;gap:12px;flex-wrap:wrap}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}
figure img{display:block;width:420px;height:420px;object-fit:contain;background:#000;border-radius:4px}
figcaption{font-size:12px;color:var(--muted);margin-top:6px}
.side{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;max-width:520px}
.q{font-size:18px;font-weight:650;margin:0 0 4px}
.meta{color:var(--muted);font-size:13px;margin-bottom:12px}
.actions{display:grid;gap:8px;margin:12px 0}
.actions button{display:flex;gap:10px;align-items:center;text-align:left;border:1px solid var(--line);background:var(--chip);
border-radius:7px;padding:9px 12px;cursor:pointer}
.actions button:hover,.actions button:focus-visible{border-color:var(--accent);outline:none}
.actions button.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
kbd{font:600 12px ui-monospace,Consolas,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;
padding:1px 6px;background:var(--panel);color:var(--ink)}
.classes{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 4px}
.classes button{border:1px solid var(--line);background:var(--chip);border-radius:999px;padding:4px 10px;cursor:pointer;font-size:13px}
.classes button.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.legend{font-size:12px;color:var(--muted);margin-top:10px;line-height:1.7}
.sw{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:4px}
.nav{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
.nav button{border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:5px 10px;cursor:pointer}
.status{font-weight:600}.status.dechet,.status.garder{color:var(--ok)}.status.pas_dechet,.status.retirer{color:var(--no)}
.status.ignorer{color:var(--warn)}
.saved{font-size:12px;color:var(--muted)}
@media (max-width:1000px){main{grid-template-columns:1fr}figure img{width:100%;height:auto;aspect-ratio:1}}
</style></head><body>
<header><h1>Revue du banc de test</h1>
<label>Liste <select id="filt"><option value="A">A. Déchets oubliés ?</option><option value="B">B. Annotations douteuses ?</option></select></label>
<label><input type="checkbox" id="todo" checked> non revus d'abord</label>
<div class="bar"><i id="prog"></i></div><span id="count"></span><span class="saved" id="saved"></span></header>
<main><div class="imgs">
<figure><img id="zoom" alt="Zoom sur l'objet"><figcaption>Zoom (<kbd>Espace</kbd> : contours on/off)</figcaption></figure>
<figure><img id="ctx" alt="Contexte autour de l'objet"><figcaption>Contexte : 640 px autour (~3 m), objet encadré</figcaption></figure>
</div>
<div class="side"><p class="q" id="q"></p><div class="meta" id="meta"></div>
<div id="clsbox"><div style="font-size:13px;color:var(--muted)">Classe si c'est un déchet (touche) :</div><div class="classes" id="classes"></div></div>
<div class="actions" id="actions"></div>
<div>Décision : <span class="status" id="status">-</span></div>
<div class="nav"><button id="prev"><kbd>←</kbd> précédent</button><button id="next">suivant <kbd>→</kbd></button>
<span class="saved">Chaque décision est enregistrée tout de suite.</span></div>
<div class="legend" id="legend"></div></div></main>
<script>
const st={items:[],dec:{},classes:[],i:0,lines:true,cls:null};
const $=id=>document.getElementById(id);
const ACT={A:[["1","dechet","Déchet oublié : l'ajouter (classe choisie)"],["2","pas_dechet","Pas un déchet : ne rien faire"],
["3","ignorer","Douteux, ou déchet au contour faux : zone ignorée"]],
B:[["1","garder","Bien un déchet : garder l'annotation"],["2","retirer","Pas un déchet : retirer l'annotation"],
["3","ignorer","Douteux : zone ignorée"]]};
const LAB={dechet:"déchet ajouté",pas_dechet:"pas un déchet",ignorer:"zone ignorée",garder:"annotation gardée",retirer:"annotation retirée"};
function list(){const t=$("filt").value;let L=st.items.filter(x=>x.type===t);
 if($("todo").checked){const a=L.filter(x=>!st.dec[x.id]),b=L.filter(x=>st.dec[x.id]);L=a.concat(b);}return L;}
function cur(){return list()[st.i];}
function show(){const L=list();if(st.i>=L.length)st.i=Math.max(0,L.length-1);const it=L[st.i];
 const all=st.items.filter(x=>x.type===$("filt").value),done=all.filter(x=>st.dec[x.id]).length;
 $("prog").style.width=(all.length?100*done/all.length:0)+"%";$("count").textContent=done+" / "+all.length+" revus";
 if(!it){$("q").textContent="Rien à revoir dans cette liste.";$("zoom").removeAttribute("src");$("ctx").removeAttribute("src");$("actions").innerHTML="";$("meta").textContent="";return;}
 $("zoom").src="vignettes/"+(st.lines?it.vignettes.z:it.vignettes.n);$("ctx").src="vignettes/"+it.vignettes.c;
 const d=st.dec[it.id];
 if(it.type==="A"){$("q").textContent="Un déchet non annoté ?";
  $("meta").innerHTML=it.parent_id.split("_images_")[0]+" · "+it.parent_id.split("_").slice(-2).join("_")+" · ~"+Math.round(it.taille_px)+" px<br>Vu par : "+
   Object.entries(it.modeles).map(([k,v])=>k+" ("+v.toFixed(2)+")").join(", ")+" · contour "+it.contour_source;
  $("clsbox").style.display="";st.cls=(d&&d.classe)||it.classe_suggeree;
  $("legend").innerHTML='<span class="sw" style="background:#00ffff"></span>contour proposé (à ajouter) · <span class="sw" style="background:#ff5050"></span>détection du modèle · <span class="sw" style="background:#5adc5a"></span>déjà annoté · <span class="sw" style="background:#ffc828"></span>zone exclue (Inconnu, bois, verre)';}
 else{$("q").textContent="Cette annotation est-elle un déchet ?";
  $("meta").innerHTML=it.parent_id.split("_images_")[0]+" · "+it.parent_id.split("_").slice(-2).join("_")+" · annotée « "+it.classe_annotee+" » · ~"+Math.round(it.taille_px)+" px<br>Aucun modèle ne la retrouve.";
  $("clsbox").style.display="none";
  $("legend").innerHTML='<span class="sw" style="background:#00ffff"></span>annotation examinée · <span class="sw" style="background:#5adc5a"></span>autres annotations · <span class="sw" style="background:#ffc828"></span>zone exclue';}
 $("classes").innerHTML=st.classes.map(c=>`<button data-b="${c.brut}" class="${c.brut===st.cls?"on":""}"><kbd>${c.touche}</kbd> ${c.libelle}</button>`).join("");
 $("classes").querySelectorAll("button").forEach(b=>b.onclick=()=>{st.cls=b.dataset.b;show();});
 $("actions").innerHTML=ACT[it.type].map(([k,v,t])=>`<button data-v="${v}" class="${d&&d.decision===v?"on":""}"><kbd>${k}</kbd>${t}</button>`).join("");
 $("actions").querySelectorAll("button").forEach(b=>b.onclick=()=>decide(b.dataset.v));
 $("status").className="status "+(d?d.decision:"");$("status").textContent=d?LAB[d.decision]+(d.decision==="dechet"?" ("+d.classe+")":""):"à revoir";}
async function decide(v){const it=cur();if(!it)return;const body={id:it.id,decision:v,classe:v==="dechet"?st.cls:null};
 const r=await fetch("api/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
 if(!r.ok){alert("Enregistrement impossible : "+await r.text());return;}
 st.dec[it.id]={decision:v,classe:body.classe};$("saved").textContent="enregistré "+new Date().toLocaleTimeString();
 if($("todo").checked){/* l'élément revu passe en fin de liste : on reste à la même position */}else st.i++;show();}
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT")return;const it=cur();
 if(e.key==="ArrowRight"){st.i=Math.min(list().length-1,st.i+1);show();}
 else if(e.key==="ArrowLeft"){st.i=Math.max(0,st.i-1);show();}
 else if(e.key===" "){e.preventDefault();st.lines=!st.lines;show();}
 else if(it&&["1","2","3"].includes(e.key)){decide(ACT[it.type][+e.key-1][1]);}
 else if(it&&it.type==="A"){const c=st.classes.find(c=>c.touche===e.key.toLowerCase());if(c){st.cls=c.brut;show();}}});
$("prev").onclick=()=>{st.i=Math.max(0,st.i-1);show();};$("next").onclick=()=>{st.i=Math.min(list().length-1,st.i+1);show();};
$("filt").onchange=()=>{st.i=0;show();};$("todo").onchange=()=>{st.i=0;show();};
fetch("api/donnees").then(r=>r.json()).then(d=>{st.items=d.items;st.dec=d.decisions;st.classes=d.classes;show();});
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    dossier: Path = DOSSIER
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _decisions(self) -> Dict:
        p = self.dossier / "decisions.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/donnees":
            cand = json.loads((self.dossier / "candidats.json").read_text(encoding="utf-8"))
            body = {"items": cand["items"], "classes": cand["classes"], "decisions": self._decisions()}
            return self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json")
        if path.startswith("/vignettes/"):
            f = (self.dossier / "vignettes" / Path(path).name)
            if f.exists():
                return self._send(200, f.read_bytes(), "image/jpeg")
        return self._send(404, b"introuvable")

    def do_POST(self):
        if urlparse(self.path).path != "/api/decision":
            return self._send(404, b"introuvable")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
        if body.get("decision") not in ("dechet", "pas_dechet", "ignorer", "garder", "retirer"):
            return self._send(400, b"decision inconnue")
        with self.lock:
            dec = self._decisions()
            dec[body["id"]] = {"decision": body["decision"], "classe": body.get("classe"),
                               "date": datetime.now().isoformat(timespec="seconds")}
            tmp = self.dossier / "decisions.json.tmp"
            tmp.write_text(json.dumps(dec, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.dossier / "decisions.json")
        return self._send(200, b"ok")


def revue(dossier: Path = DOSSIER, port: int = 8766, ouvrir: bool = True) -> None:
    dossier = Path(dossier)
    if not (dossier / "candidats.json").exists():
        raise SystemExit(f"{dossier / 'candidats.json'} introuvable : lancer d'abord la préparation.")
    _Handler.dossier = dossier
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Page de revue : {url}\nDécisions enregistrées dans {dossier / 'decisions.json'}\nCtrl+C pour arrêter.")
    if ouvrir:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt. Pour écrire les décisions dans les annotations : python -m src.review.revue_banc appliquer")


# ============================================================================ application
def _local_id(names: Dict[int, str], raw_name: str, aliases: Dict[str, str]) -> Optional[int]:
    target = normalize_class_name(raw_name)
    for i, n in names.items():
        nn = normalize_class_name(str(n))
        if nn == target or aliases.get(nn) == target:
            return i
    return None


def appliquer(dossier: Path = DOSSIER, essai: bool = False) -> Dict:
    dossier = Path(dossier)
    cand = json.loads((dossier / "candidats.json").read_text(encoding="utf-8"))
    dec_path = dossier / "decisions.json"
    decisions = json.loads(dec_path.read_text(encoding="utf-8")) if dec_path.exists() else {}
    done_path = dossier / "applique.json"
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    with open(GT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    aliases = {normalize_class_name(str(a)): normalize_class_name(str(c)) for a, c in (cfg.get("class_aliases") or {}).items()}
    items = {it["id"]: it for it in cand["items"]}
    todo = [(iid, d) for iid, d in decisions.items() if iid in items and iid not in done
            and d["decision"] in ("dechet", "ignorer", "retirer")]
    changed = [iid for iid, d in decisions.items() if iid in done and done[iid]["decision"] != d["decision"]]
    if changed:
        print(f"⚠️  {len(changed)} décision(s) modifiée(s) APRÈS application ({', '.join(changed[:3])}...) : non "
              f"rejouées automatiquement ; restaurer la sauvegarde d'origine puis supprimer applique.json pour tout réappliquer.")
    print(f"--- Application de la revue : {len(todo)} modification(s) à écrire "
          f"({len(decisions)} décisions, {len(done)} déjà appliquées){' - ESSAI, rien n écrit' if essai else ''} ---")
    by_file: Dict[str, List] = {}
    for iid, d in todo:
        by_file.setdefault(items[iid]["label_brut"], []).append((iid, d, items[iid]))
    backup = dossier / "sauvegarde_annotations_origine"
    journal = []
    stats = {"ajouts": 0, "zones_ignorees": 0, "retraits": 0}
    for rel, entries in sorted(by_file.items()):
        lab = RAW_DIR / rel
        lot_dir = RAW_DIR / rel.split("/", 1)[0]
        yaml_path = lot_dir / "data.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data_yaml = yaml.safe_load(f) or {}
        names = {int(k): v for k, v in (data_yaml.get("names") or {}).items()}
        yaml_changed = False

        def class_id(raw_name):
            nonlocal yaml_changed
            cid = _local_id(names, raw_name, aliases)
            if cid is None:
                cid = max(names) + 1 if names else 0
                names[cid] = raw_name
                yaml_changed = True
            return cid

        lines = lab.read_text(encoding="utf-8").splitlines() if lab.exists() else []
        for iid, d, it in entries:
            if it["type"] == "A":
                raw = d.get("classe") or it["classe_suggeree"] if d["decision"] == "dechet" else INCONNU
                cid = class_id(raw)
                w, h = it["largeur"], it["hauteur"]
                coords = " ".join(f"{min(1, max(0, x / w)):.6f} {min(1, max(0, y / h)):.6f}" for x, y in it["polygone"][:-1])
                lines.append(f"{cid} {coords}")
                stats["ajouts" if d["decision"] == "dechet" else "zones_ignorees"] += 1
                journal.append({"id": iid, "fichier": rel, "action": "ajout" if d["decision"] == "dechet" else "zone ignorée",
                                "classe": raw})
            else:
                idx = next((k for k, l in enumerate(lines) if " ".join(l.split()[1:]) == it["coords_texte"]), None)
                if idx is None:
                    print(f"  ⚠️  {iid} : annotation introuvable dans {rel} (déjà modifiée ?) - ignorée.")
                    continue
                if d["decision"] == "retirer":
                    old = lines.pop(idx)
                    stats["retraits"] += 1
                    journal.append({"id": iid, "fichier": rel, "action": "retrait", "ligne": old[:80]})
                else:
                    parts = lines[idx].split()
                    old_cls = names.get(int(parts[0]))
                    lines[idx] = " ".join([str(class_id(INCONNU))] + parts[1:])
                    stats["zones_ignorees"] += 1
                    journal.append({"id": iid, "fichier": rel, "action": "zone ignorée", "ancienne_classe": old_cls})
            done[iid] = {"decision": d["decision"], "date": datetime.now().isoformat(timespec="seconds")}
        if essai:
            continue
        for src in [lab] + ([yaml_path] if yaml_changed else []):
            dst = backup / (rel if src == lab else f"{rel.split('/', 1)[0]}/data.yaml")
            if src.exists() and not dst.exists():  # garde la version d'ORIGINE (première application)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        lab.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if yaml_changed:
            data_yaml["names"] = {k: names[k] for k in sorted(names)}
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
    if not essai and todo:
        done_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        jpath = dossier / "journal_modifications.json"
        old = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else []
        jpath.write_text(json.dumps(old + [{"date": datetime.now().isoformat(timespec="seconds"), **stats,
                                            "modifications": journal}], ensure_ascii=False, indent=1), encoding="utf-8")
        _note_bench_version(stats)
    print(f"✅ {stats['ajouts']} déchet(s) ajouté(s), {stats['zones_ignorees']} zone(s) ignorée(s), "
          f"{stats['retraits']} annotation(s) retirée(s) dans {len(by_file)} fichier(s).")
    if not essai and todo:
        print(f"   Originaux sauvegardés dans {backup}.\n   Étape suivante (reconstruit les datasets, réévalue tous les "
              f"modèles, ~15 min) :\n   python -m src.training.campagne_v4 --nom campagne_v4_20261012 --etapes data,eval,tableau --refaire")
    return stats


def _note_bench_version(stats: Dict) -> None:
    """Trace la nouvelle version de la vérité terrain du banc dans config/frozen_test_parents.json (les
    identifiants des images ne changent pas, seules leurs annotations)."""
    from src.data.utils.split_dataset import FROZEN_TEST_PATH
    with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("verite_terrain", []).append({
        "date": datetime.now().isoformat(timespec="seconds"), "source": "revue sans CVAT (src/review/revue_banc.py)",
        "dossier": str(DOSSIER.relative_to(DATA_ROOT)), **stats})
    with open(FROZEN_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Revue du banc de test sans CVAT.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preparer")
    p.add_argument("--campagne", default="campagne_v4_20261012")
    p.add_argument("--dossier", default=str(DOSSIER))
    p.add_argument("--sam", default=SAM_POIDS)
    p.add_argument("--conf", type=float, default=CONF_CANDIDAT)
    r = sub.add_parser("revue")
    r.add_argument("--dossier", default=str(DOSSIER))
    r.add_argument("--port", type=int, default=8766)
    a = sub.add_parser("appliquer")
    a.add_argument("--dossier", default=str(DOSSIER))
    a.add_argument("--essai", action="store_true", help="Affiche ce qui serait fait, sans rien écrire.")
    args = ap.parse_args()
    if args.cmd == "preparer":
        preparer(args.campagne, Path(args.dossier), args.sam, args.conf)
    elif args.cmd == "revue":
        revue(Path(args.dossier), args.port)
    else:
        appliquer(Path(args.dossier), args.essai)


if __name__ == "__main__":
    main()
