#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Tuilage du protocole v4 (12/10/2026) : 2_split_dataset<suffixe> -> 4_sliced_dataset<suffixe>.

Remplace, pour les datasets v4 uniquement, les étapes « augmentation » (simple copie) et
« tuilage » (slice_dataset.py / slicer.py, inchangés pour les autres datasets). Règles :

1. Tuiles de `tile_size` px (640) avec un recouvrement de `tile_overlap` px (128, contre 256
   avant) : 95 % des objets annotés mesurent moins de 126 px, ils sont donc toujours ENTIERS
   dans au moins une tuile ; chaque objet est vu 1,6 fois par epoch au lieu de 2,9 (mesuré
   sur les 576 images le 12/10/2026).
2. Zéro noir artificiel : une tuile qui contient un seul pixel de nodata est écartée
   (`max_nodata_fraction` 0). Nodata = bordure noire des orthomosaïques / imagettes
   (slicer.compute_nodata_mask) + zones de terrain partagé noircies au split
   (nodata_masks/, voir ground_overlap.py).
3. Fenêtres de rattrapage : un objet qui n'est visible à plus de 50 % dans aucune tuile
   propre de la grille reçoit une fenêtre placée entièrement dans la zone valide, la plus
   centrée possible sur lui. Mesuré : sans elles, la tolérance 0 ferait perdre 6 % des objets
   SL (bords d'orthos) ; avec, 1,1 % (objets à cheval sur le bord noir lui-même).
4. Tuiles sans déchet : val et test gardent TOUTES leurs tuiles (distribution réelle, pour
   choisir la meilleure epoch et mesurer honnêtement). En train : toutes les tuiles avec
   déchet, toutes les tuiles avec un « contre-exemple » annoté (bois, verre), puis des tuiles
   vides tirées au hasard (graine fixe) pour que les tuiles sans déchet fassent
   `train_negative_fraction` (10 %) du train, PAR SITE (au prorata des tuiles avec déchet :
   avant, 96 % des tuiles vides venaient du sable de Saint-Brandon alors que 64 % des fausses
   alertes à Santa Luzia tombent sur du fond).

Pas de mode incrémental : si l'entrée (split) ou un paramètre change, tout est refait
(~5-10 min) ; sinon l'étape est sautée. Manifeste et statistiques : .slicing_v4_manifest.json.
"""

import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.strtree import STRtree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.utils.class_config import EXCLUDE, load_batch_local_names, load_class_config, normalize_class_name, resolve_class_name
from src.data.utils.ground_overlap import nodata_mask_path, read_nodata_mask
from src.data.utils.image_io import load_image_bgr
from src.data.utils.pipeline_utils import ensure_cache_is_safe, fingerprint
from src.data.utils.raw_dataset import VALID_IMG_EXTS, site_of_batch
from src.data.utils.slicer import compute_nodata_mask
from src.data.utils.split_dataset import MANIFEST_FILENAME as SPLIT_MANIFEST_FILENAME, PARENT_MANIFEST_FILENAME
from src.data.utils.tiling_geometry import iter_tile_windows

LOGIC_VERSION_V4 = 1
MANIFEST_FILENAME = ".slicing_v4_manifest.json"
SPLITS = ("train", "val", "test")
MIN_AREA_RATIO = 0.05      # fragment d'objet coupé au bord gardé s'il fait >= 5 % de l'objet (comme avant)
SEEN_FRACTION = 0.5        # objet « vu » dans une tuile s'il y est à >= 50 %
HARD_NEG_FRACTION = 0.5    # tuile « contre-exemple » si un objet bois/verre y est à >= 50 %


def _stable_rank(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:15], 16)


def _read_polys(label_path: Path, w: int, h: int, class_of=None) -> List[Dict]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        xy = np.array([float(v) for v in parts[1:]], float).reshape(-1, 2) * [w, h]
        g = Polygon(xy)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty or g.area <= 0:
            continue
        cls = int(parts[0]) if class_of is None else class_of(int(parts[0]))
        if cls is None:
            continue
        out.append({"cls": cls, "geom": g, "area": g.area, "bbox": g.bounds})
    return out


def _clip_lines(polys: List[Dict], tree: Optional[STRtree], win) -> List[str]:
    x0, y0, x1, y1 = win
    tw, th = x1 - x0, y1 - y0
    b = box(x0, y0, x1, y1)
    lines = []
    if tree is None:
        return lines
    for j in tree.query(b):
        p = polys[int(j)]
        inter = b.intersection(p["geom"])
        if inter.is_empty or inter.area <= 0 or inter.area / p["area"] < MIN_AREA_RATIO:
            continue
        parts = [inter] if isinstance(inter, Polygon) else (list(inter.geoms) if isinstance(inter, MultiPolygon) else [])
        for part in parts:
            if part.area <= 0:
                continue
            c = np.asarray(part.exterior.coords, float)
            c[:, 0] = np.clip((c[:, 0] - x0) / tw, 0, 1)
            c[:, 1] = np.clip((c[:, 1] - y0) / th, 0, 1)
            if len(c) >= 3:
                lines.append(f"{p['cls']} " + " ".join(f"{v:.6f}" for v in c.reshape(-1)) + "\n")
    return lines


def _frac_inside(p: Dict, win) -> float:
    bx0, by0, bx1, by1 = p["bbox"]
    x0, y0, x1, y1 = win
    if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
        return 0.0
    if bx0 >= x0 and by0 >= y0 and bx1 <= x1 and by1 <= y1:
        return 1.0
    return p["geom"].intersection(box(*win)).area / p["area"]


def plan_parent(img: np.ndarray, extra_nodata: Optional[np.ndarray], pos: List[Dict], hard: List[Dict],
                tile: int, overlap: int, max_nodata_fraction: float, rescue: bool) -> Dict:
    """Fenêtres retenues pour une image parente : [(fenêtre, lignes de label, catégorie, rattrapage)] + stats."""
    h, w = img.shape[:2]
    nd = compute_nodata_mask(img)
    if extra_nodata is not None and extra_nodata.shape == nd.shape:
        nd |= extra_nodata
    integ = np.pad(nd.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    tw, th = min(tile, w), min(tile, h)

    def n_nodata(x0, y0, x1, y1):
        return integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]

    def ok(win):
        x0, y0, x1, y1 = win
        return n_nodata(x0, y0, x1, y1) <= max_nodata_fraction * (x1 - x0) * (y1 - y0)

    if w <= tile and h <= tile:
        grid = [(0, 0, w, h)]
    else:
        grid = [(x0, y0, min(x1, w), min(y1, h)) for x0, y0, x1, y1 in
                dict.fromkeys(iter_tile_windows(w, h, tile, tile - overlap))]
    kept = [win for win in grid if ok(win)]
    stats = {"grid": len(grid), "grid_nodata": len(grid) - len(kept), "rescue": 0, "objects": len(pos),
             "lost": 0, "in_shared_zone": 0, "complete": 0}

    best = [0.0] * len(pos)
    tree_pos = STRtree([p["geom"] for p in pos]) if pos else None
    for win in kept:
        if tree_pos is None:
            break
        for j in tree_pos.query(box(*win)):
            j = int(j)
            best[j] = max(best[j], _frac_inside(pos[j], win))

    rescued = []
    lost_idx = [j for j in range(len(pos)) if best[j] < SEEN_FRACTION]
    if rescue and lost_idx and nd.any():
        cnt = integ[th:, tw:] - integ[:-th, tw:] - integ[th:, :-tw] + integ[:-th, :-tw]
        valid = cnt <= max_nodata_fraction * tw * th
        for j in lost_idx:
            if best[j] >= SEEN_FRACTION:
                continue  # rattrapé par une fenêtre ajoutée pour un autre objet
            bx0, by0, bx1, by1 = pos[j]["bbox"]
            xl, xh = int(max(0, math.ceil(bx1 - tw))), int(min(w - tw, math.floor(bx0)))
            yl, yh = int(max(0, math.ceil(by1 - th))), int(min(h - th, math.floor(by0)))
            if xl > xh or yl > yh:
                continue
            sub = valid[yl:yh + 1, xl:xh + 1]
            ys, xs = np.nonzero(sub)
            if len(xs) == 0:
                continue
            cx, cy = (bx0 + bx1) / 2 - tw / 2, (by0 + by1) / 2 - th / 2
            k = int(np.argmin((xs + xl - cx) ** 2 + (ys + yl - cy) ** 2))
            win = (int(xs[k] + xl), int(ys[k] + yl), int(xs[k] + xl + tw), int(ys[k] + yl + th))
            if win in kept or win in rescued:
                continue
            rescued.append(win)
            for jj in tree_pos.query(box(*win)):
                jj = int(jj)
                best[jj] = max(best[jj], _frac_inside(pos[jj], win))
    stats["rescue"] = len(rescued)
    # Objets non vus : dans une zone de terrain partagé noircie (vus dans l'image du split prioritaire)
    # ou réellement perdus (à cheval sur le bord noir d'une ortho / imagette).
    stats["lost"] = stats["in_shared_zone"] = 0
    for j, v in enumerate(best):
        if v >= SEEN_FRACTION:
            continue
        c = pos[j]["geom"].representative_point()
        cx, cy = min(w - 1, max(0, int(c.x))), min(h - 1, max(0, int(c.y)))
        if extra_nodata is not None and extra_nodata.shape == nd.shape and extra_nodata[cy, cx]:
            stats["in_shared_zone"] += 1
        else:
            stats["lost"] += 1
    stats["complete"] = sum(1 for v in best if v >= 0.99)

    tree_hard = STRtree([p["geom"] for p in hard]) if hard else None
    tiles = []
    for win, is_rescue in [(x, False) for x in kept] + [(x, True) for x in rescued]:
        lines = _clip_lines(pos, tree_pos, win)
        if lines:
            cat = "pos"
        elif tree_hard is not None and any(_frac_inside(hard[int(j)], win) >= HARD_NEG_FRACTION
                                          for j in tree_hard.query(box(*win))):
            cat = "hard"
        else:
            cat = "vide"
        tiles.append({"win": win, "lines": lines, "cat": cat, "rescue": is_rescue})
    return {"tiles": tiles, "stats": stats}


def _write_tile(img: np.ndarray, tile: Dict, prefix: str, img_dir: Path, lab_dir: Path) -> None:
    x0, y0, x1, y1 = tile["win"]
    name = f"{prefix}_{x0}_{y0}"
    if not cv2.imwrite(str(img_dir / f"{name}.png"), img[y0:y1, x0:x1]):
        raise RuntimeError(f"Écriture impossible : {img_dir / name}.png")
    with open(lab_dir / f"{name}.txt", "w", encoding="utf-8") as f:
        f.writelines(tile["lines"])


def _hard_negative_polys(info: Dict, raw_root_hint: Path, class_taxonomy, hard_names: set, w: int, h: int) -> List[Dict]:
    """Objets annotés « contre-exemples » (bois, verre...) d'une image parente, lus dans son label BRUT."""
    if not hard_names:
        return []
    label_path = Path(str(info["raw_label_path"]).replace("\\", "/"))
    if not label_path.exists():
        norm = str(info["raw_label_path"]).replace("\\", "/")
        marker = "1_annotated_dataset/"
        if marker in norm:
            label_path = raw_root_hint / norm.split(marker, 1)[1]
    if not label_path.exists():
        return []
    root = label_path
    while root.parent != root and root.name != info["batch"]:
        root = root.parent
    names = load_batch_local_names(root / "data.yaml") if (root / "data.yaml").exists() else {}

    def class_of(local_id):
        n = names.get(local_id)
        if n is None or normalize_class_name(n) not in hard_names:
            return None
        return 0 if resolve_class_name(n, class_taxonomy) == EXCLUDE else None

    return _read_polys(label_path, w, h, class_of)


def run_slice_v4(split_dir: str, sliced_dir: str, class_config_path: str, tile_size: int = 640, tile_overlap: int = 128,
                 max_nodata_fraction: float = 0.0, rescue_windows: bool = True, train_negative_fraction: float = 0.10,
                 hard_negative_classes: Optional[List[str]] = None, raw_dir: Optional[str] = None,
                 force: bool = False, run_confirmation=None) -> Dict:
    split_dir, sliced_dir = Path(split_dir), Path(sliced_dir)
    class_taxonomy, _ = load_class_config(class_config_path)
    hard_names = {normalize_class_name(n) for n in (hard_negative_classes or [])}
    with open(split_dir / PARENT_MANIFEST_FILENAME, "r", encoding="utf-8") as f:
        parent_manifest = json.load(f)
    split_fp = None
    if (split_dir / SPLIT_MANIFEST_FILENAME).exists():
        with open(split_dir / SPLIT_MANIFEST_FILENAME, "r", encoding="utf-8") as f:
            split_fp = json.load(f).get("fingerprint")
    pm_digest = hashlib.sha1(json.dumps(parent_manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    params = {"slicer_v4_logic_version": LOGIC_VERSION_V4, "tile_size": tile_size, "tile_overlap": tile_overlap,
              "max_nodata_fraction": max_nodata_fraction, "rescue_windows": rescue_windows,
              "train_negative_fraction": train_negative_fraction, "hard_negative_classes": sorted(hard_names),
              "min_area_ratio": MIN_AREA_RATIO, "upstream_split_fingerprint": split_fp,
              "upstream_parent_manifest_digest": pm_digest}
    mpath = sliced_dir / MANIFEST_FILENAME
    if mpath.exists():
        with open(mpath, "r", encoding="utf-8") as f:
            old = json.load(f)
        if old.get("fingerprint") == fingerprint(params) and old.get("complete"):
            print(f"✅ {sliced_dir.name} déjà à jour (même split, mêmes paramètres) - tuilage sauté.")
            return old.get("stats", {})
    ensure_cache_is_safe(sliced_dir, params, force=force, wipe_subdirs=["images", "labels"],
                         manifest_filename=MANIFEST_FILENAME, run_confirmation=run_confirmation)
    # Reconstruction complète (y compris après une interruption : le manifeste n'est marqué complet qu'à la fin).
    for sub in ("images", "labels"):
        shutil.rmtree(sliced_dir / sub, ignore_errors=True)
        for s in SPLITS:
            (sliced_dir / sub / s).mkdir(parents=True, exist_ok=True)
    raw_root_hint = Path(raw_dir) if raw_dir else split_dir.parent / "1_annotated_dataset"

    stats = {s: defaultdict(lambda: defaultdict(int)) for s in SPLITS}
    pending_empty: Dict[str, List] = defaultdict(list)   # site -> [(nom, parent_id, tuile)] (train)
    pos_tiles_site: Dict[str, int] = defaultdict(int)
    hard_tiles_site: Dict[str, int] = defaultdict(int)
    n_parents = sum(1 for v in parent_manifest.values() if v.get("split") in SPLITS)
    k = 0
    for split in SPLITS:
        img_src = split_dir / "images" / split
        lab_src = split_dir / "labels" / split
        img_dst, lab_dst = sliced_dir / "images" / split, sliced_dir / "labels" / split
        files = sorted(p for p in img_src.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS) \
            if img_src.exists() else []
        for img_path in files:
            pid = img_path.stem
            info = parent_manifest.get(pid)
            if info is None or info.get("split") != split:
                raise RuntimeError(f"{img_path} absent du manifeste du split ({split}) - split incohérent, tuilage arrêté.")
            k += 1
            img = load_image_bgr(img_path)
            if img is None:
                raise RuntimeError(f"Image illisible : {img_path}")
            h, w = img.shape[:2]
            pos = _read_polys(lab_src / f"{pid}.txt", w, h)
            hard = _hard_negative_polys(info, raw_root_hint, class_taxonomy, hard_names, w, h)
            plan = plan_parent(img, read_nodata_mask(nodata_mask_path(split_dir, split, pid)), pos, hard,
                               tile_size, tile_overlap, max_nodata_fraction, rescue_windows)
            site = site_of_batch(info["batch"])
            st = stats[split][site]
            for key in ("objects", "lost", "in_shared_zone", "complete", "rescue", "grid_nodata"):
                st[key] += plan["stats"][key]
            for t in plan["tiles"]:
                if split == "train" and t["cat"] == "vide":
                    pending_empty[site].append((f"{pid}_{t['win'][0]}_{t['win'][1]}", pid, t))
                    continue
                _write_tile(img, t, pid, img_dst, lab_dst)
                st[f"tuiles_{t['cat']}"] += 1
                st["instances"] += len(t["lines"])
                if split == "train":
                    if t["cat"] == "pos":
                        pos_tiles_site[site] += 1
                    elif t["cat"] == "hard":
                        hard_tiles_site[site] += 1
            if k % 50 == 0 or k == n_parents:
                print(f"  ✂️  {k}/{n_parents} images parentes tuilées", flush=True)
            del img

    # Tuiles vides du train : tirage par site (graine fixe = rang par hash du nom de tuile).
    selected: Dict[str, List] = defaultdict(list)
    f = train_negative_fraction
    for site, cands in pending_empty.items():
        target = int(math.ceil(pos_tiles_site[site] * f / max(1e-9, 1 - f)))
        n_take = max(0, min(len(cands), target - hard_tiles_site[site]))
        cands = sorted(cands, key=lambda c: _stable_rank(f"v4:{c[0]}"))[:n_take]
        for name, pid, t in cands:
            selected[pid].append(t)
        stats["train"][site]["tuiles_vides_disponibles"] = len(pending_empty[site])
    for pid, tiles in selected.items():
        img = load_image_bgr(split_dir / "images" / "train" / f"{pid}{Path(parent_manifest[pid]['raw_img_path'].replace(chr(92), '/')).suffix.lower()}")
        site = site_of_batch(parent_manifest[pid]["batch"])
        for t in tiles:
            _write_tile(img, t, pid, sliced_dir / "images" / "train", sliced_dir / "labels" / "train")
            stats["train"][site]["tuiles_vide"] += 1
        del img

    flat = {s: {site: dict(v) for site, v in d.items()} for s, d in stats.items()}
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": fingerprint(params), "params": params, "complete": True, "stats": flat},
                  fh, indent=1, ensure_ascii=False)
    _print_stats(flat)
    return flat


def _print_stats(flat: Dict) -> None:
    print("\n--- 📊 TUILAGE v4 ---")
    print(f"{'split':6s} {'site':4s} {'tuiles déchet':>13s} {'contre-ex.':>10s} {'vides':>6s} {'% sans déchet':>13s} "
          f"{'objets':>7s} {'entiers':>8s} {'z.partag.':>9s} {'perdus':>7s} {'rattrap.':>8s}")
    for s, d in flat.items():
        for site, v in sorted(d.items()):
            n_pos, n_hard, n_empty = v.get("tuiles_pos", 0), v.get("tuiles_hard", 0), v.get("tuiles_vide", 0)
            tot = max(1, n_pos + n_hard + n_empty)
            obj = max(1, v.get("objects", 0))
            print(f"{s:6s} {site:4s} {n_pos:13d} {n_hard:10d} {n_empty:6d} {100 * (n_hard + n_empty) / tot:12.1f}% "
                  f"{v.get('objects', 0):7d} {100 * v.get('complete', 0) / obj:7.1f}% "
                  f"{v.get('in_shared_zone', 0):9d} {v.get('lost', 0):7d} {v.get('rescue', 0):8d}")
