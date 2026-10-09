#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Groupes spatiaux d'images parentes pour le split (config/split_groups.json).

Pourquoi (24/09/2026) : le split répartit des IMAGES parentes entre train/val/test. Si
deux images montrent en partie le même terrain, un même déchet peut se retrouver à la
fois en train et en test - le test n'est plus indépendant. Mesuré sur le dataset du
24/09/2026 :
  - imagettes Saint Brandon (1024 px sur une grille de 924 px, donc 100 px de
    recouvrement avec chaque voisine) : 33 % des annotations sont dans une bande de
    recouvrement, 14 % avaient leur double dans un AUTRE split ;
  - orthomosaïques Santa Luzia : transects voisins recouvrant 1 à 7 % de leur surface ;
  - photos drone : une paire à Aldabra (A LEG1_15h46_16h02 n, DJI_0053 / DJI_0152)
    recouvrant 34 % de son empreinte.
Ce script calcule ces groupes une fois ; split_dataset.py place ensuite chaque groupe
ENTIER dans un seul split. À relancer après l'ajout d'un nouveau lot (sinon ses images
sont traitées comme isolées - signalé par split_dataset.py).

Trois règles, toutes calculées sur les images de 1_annotated_dataset :
  1. Imagettes pré-découpées (nom "<préfixe>_y<Y>_x<X>") : regroupées par blocs de
     BLOCK_CELLS x BLOCK_CELLS cellules de la grille d'origine. Seuls les recouvrements
     entre blocs voisins subsistent (compte affiché) - compromis entre étanchéité et
     nombre d'unités à répartir (~10 blocs par transect SB en 4x4).
  2. Orthomosaïques GeoTIFF : deux orthos sont regroupées si leurs zones VALIDES (canal
     alpha, sinon pixels non noirs) se recouvrent sur au moins MIN_ORTHO_OVERLAP de la
     surface valide de la plus petite.
  3. Photos drone avec GPS (EXIF/XMP, via src/application) : regroupées si leurs
     empreintes au sol se recouvrent sur au moins MIN_PHOTO_OVERLAP de la plus petite.
     Empreinte calculée comme dans l'application (terrain plat, nadir), une origine
     locale par site.
Les regroupements sont transitifs (union-find). Les règles 2 et 3 comparent toutes les
images d'un même SITE, y compris entre deux lots (08/10/2026 : deux missions de Santa Luzia
se recouvrent) ; split_dataset.py rattache alors le groupe au lot de son membre le plus
annoté.

Entrée : 1_annotated_dataset (RAW_DIR de split_dataset.py), config/sensor_specs.yaml.
Sortie : config/split_groups.json (groupes d'au moins 2 images + justification).

Exemple :
    python -m src.data.diagnostics.spatial_groups
    python -m src.data.diagnostics.spatial_groups --dry-run    # affiche sans écrire
"""

import argparse
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np

from src.data.utils.raw_dataset import collect_parent_images, site_of_batch
from src.data.utils.split_dataset import RAW_DIR, SPLIT_GROUPS_PATH

BLOCK_CELLS = 4  # 24/09/2026 : 50 blocs SB, 23 % des paires voisines entre deux blocs (32 % à 3x3, 50 % à 2x2)
MIN_ORTHO_OVERLAP = 0.02
MIN_PHOTO_OVERLAP = 0.05
ORTHO_GRID_M = 0.05  # résolution de la grille de comparaison des orthos (5 cm)
TILE_NAME_RE = re.compile(r"^(?P<prefix>.+)_y(?P<y>\d+)_x(?P<x>\d+)$")


class _UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, a: str) -> str:
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _imagette_blocks(parents: List[Dict]) -> Tuple[Dict[str, str], int, int]:
    """parent_id -> clé de bloc pour les imagettes pré-découpées, + nombre de paires
    d'imagettes voisines (recouvrement) dans le même bloc / dans deux blocs différents."""
    by_source: Dict[Tuple[str, str, str], List[Tuple[str, int, int]]] = defaultdict(list)
    for p in parents:
        m = TILE_NAME_RE.match(Path(p["img_path"]).stem)
        if m:
            key = (p["batch"], str(Path(p["img_path"]).parent), m.group("prefix"))
            by_source[key].append((p["parent_id"], int(m.group("y")), int(m.group("x"))))
    block_of: Dict[str, str] = {}
    same = cross = 0
    for (batch, _, prefix), items in by_source.items():
        coords = [v for _, y, x in items for v in (y, x) if v > 0]
        step = 0
        for v in coords:
            step = math.gcd(step, v)
        if step <= 0 or len(items) < 2:
            continue
        cell = {(y // step, x // step): pid for pid, y, x in items}
        for (gy, gx), pid in cell.items():
            block_of[pid] = f"tuiles:{batch}:{prefix}:{gy // BLOCK_CELLS}_{gx // BLOCK_CELLS}"
        for (gy, gx), pid in cell.items():
            for ny, nx in ((gy + 1, gx), (gy, gx + 1)):
                other = cell.get((ny, nx))
                if other is None:
                    continue
                if block_of[pid] == block_of[other]:
                    same += 1
                else:
                    cross += 1
    return block_of, same, cross


def _ortho_valid_mask(path: Path):
    """(x0, y0, pas en m, masque booléen sous-échantillonné) d'une GeoTIFF, ou None."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    tags = im.tag_v2 if hasattr(im, "tag_v2") else {}
    scale, tie = tags.get(33550), tags.get(33922)
    if not scale or not tie:
        return None
    w, h = im.size
    factor = max(1, int(round(ORTHO_GRID_M / scale[0])))
    small = im.reduce(factor) if factor > 1 else im
    arr = np.asarray(small)
    if arr.ndim == 3 and arr.shape[2] == 4:
        valid = arr[..., 3] > 0
    elif arr.ndim == 3:
        valid = arr[..., :3].max(axis=2) > 10
    else:
        valid = arr > 10
    x0 = tie[3] - tie[0] * scale[0]
    y0 = tie[4] + tie[1] * scale[1]
    return x0, y0, scale[0] * factor, valid


def _ortho_pairs(parents: List[Dict]) -> List[Tuple[str, str, float]]:
    orthos = {}
    for p in parents:
        if Path(p["img_path"]).suffix.lower() in (".tif", ".tiff"):
            try:
                info = _ortho_valid_mask(Path(p["img_path"]))
            except Exception as e:  # noqa: BLE001 - une ortho illisible ne bloque pas les autres
                print(f"  ⚠️  ortho illisible, ignorée : {p['img_path']} ({e})")
                continue
            if info is not None:
                orthos[p["parent_id"]] = (p["batch"], info)
    pairs = []
    for (pa, (ba, ia)), (pb, (bb, ib)) in itertools.combinations(sorted(orthos.items()), 2):
        # Même SITE (plus seulement même lot, 08/10/2026) : deux missions d'un même site peuvent
        # survoler le même terrain à quelques jours d'intervalle.
        if site_of_batch(ba) != site_of_batch(bb):
            continue
        xa, ya, sa, va = ia
        xb, yb, sb, vb = ib
        gx0, gx1 = max(xa, xb), min(xa + va.shape[1] * sa, xb + vb.shape[1] * sb)
        gy0, gy1 = min(ya, yb), max(ya - va.shape[0] * sa, yb - vb.shape[0] * sb)
        if gx1 <= gx0 or gy0 <= gy1:
            continue
        nx, ny = int((gx1 - gx0) / ORTHO_GRID_M) + 1, int((gy0 - gy1) / ORTHO_GRID_M) + 1

        def raster(x0, y0, s, v):
            m = np.zeros((ny, nx), dtype=bool)
            yy, xx = np.nonzero(v)
            ix = ((x0 + (xx + 0.5) * s - gx0) / ORTHO_GRID_M).astype(int)
            iy = ((gy0 - (y0 - (yy + 0.5) * s)) / ORTHO_GRID_M).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            m[iy[ok], ix[ok]] = True
            return m

        inter = (raster(xa, ya, sa, va) & raster(xb, yb, sb, vb)).sum() * ORTHO_GRID_M ** 2
        smaller = min(va.sum() * sa * sa, vb.sum() * sb * sb)
        if smaller > 0 and inter / smaller >= MIN_ORTHO_OVERLAP:
            pairs.append((pa, pb, float(inter / smaller)))
    return pairs


def _photo_pairs(parents: List[Dict], sensor_registry_path: Path) -> List[Tuple[str, str, float]]:
    try:
        from src.application.exif_metadata import extract_photo_metadata
        from src.application.geolocation import photo_footprint_local_polygon
        from src.application.sensor_config import load_sensor_registry, resolve_sensor
    except ImportError as e:
        print(f"  ⚠️  modules de géolocalisation indisponibles ({e}) : recouvrement des photos non vérifié.")
        return []
    registry = load_sensor_registry(sensor_registry_path)
    photos_by_site: Dict[str, List] = defaultdict(list)
    for p in parents:
        path = Path(p["img_path"])
        if path.suffix.lower() not in (".jpg", ".jpeg") or TILE_NAME_RE.match(path.stem):
            continue
        try:
            meta = extract_photo_metadata(path)
            sensor = resolve_sensor(meta.make, meta.model, registry)
        except Exception:  # noqa: BLE001 - photo sans GPS/XMP exploitable : traitée comme isolée
            continue
        photos_by_site[site_of_batch(p["batch"])].append((p["parent_id"], p["batch"], meta, sensor))
    pairs = []
    for site, photos in photos_by_site.items():
        origin = (photos[0][2].lon, photos[0][2].lat)
        polys = [(pid, b, photo_footprint_local_polygon(m, s, origin)) for pid, b, m, s in photos]
        for (pa, ba, ga), (pb, bb, gb) in itertools.combinations(polys, 2):
            # Toutes les paires du site, y compris entre deux lots (08/10/2026) : SL W3 n / DJI_0092
            # (banc de test) et SL W4 n / DJI_0564 se recouvrent à 31 %.
            if not ga.intersects(gb):
                continue
            ratio = ga.intersection(gb).area / max(1e-9, min(ga.area, gb.area))
            if ratio >= MIN_PHOTO_OVERLAP:
                pairs.append((pa, pb, float(ratio)))
    return pairs


def build_split_groups(raw_dir: Path = Path(RAW_DIR), output: Path = SPLIT_GROUPS_PATH,
                       dry_run: bool = False) -> Dict:
    parents = list({p["parent_id"]: p for p in collect_parent_images(Path(raw_dir))}.values())
    print(f"--- Groupes spatiaux : {len(parents)} images parentes ({raw_dir}) ---")
    uf = _UnionFind()
    evidence = []

    block_of, same, cross = _imagette_blocks(parents)
    by_block: Dict[str, List[str]] = defaultdict(list)
    for pid, blk in block_of.items():
        by_block[blk].append(pid)
    for members in by_block.values():
        for m in members[1:]:
            uf.union(members[0], m)
    print(f"  • Imagettes pré-découpées : {len(block_of)} images en {len(by_block)} blocs de "
          f"{BLOCK_CELLS}x{BLOCK_CELLS} ; paires voisines dans le même bloc : {same}, "
          f"entre deux blocs (recouvrement résiduel possible entre splits) : {cross}.")

    ortho = _ortho_pairs(parents)
    for a, b, r in ortho:
        uf.union(a, b)
        evidence.append({"rule": "ortho", "a": a, "b": b, "overlap": round(r, 3)})
    print(f"  • Orthomosaïques : {len(ortho)} paire(s) recouvrante(s) (>= {MIN_ORTHO_OVERLAP:.0%}).")

    sensor_path = Path(__file__).resolve().parents[3] / "config" / "sensor_specs.yaml"
    photo = _photo_pairs(parents, sensor_path)
    for a, b, r in photo:
        uf.union(a, b)
        evidence.append({"rule": "photo", "a": a, "b": b, "overlap": round(r, 3)})
    print(f"  • Photos drone : {len(photo)} paire(s) recouvrante(s) (>= {MIN_PHOTO_OVERLAP:.0%}).")

    members_by_root: Dict[str, List[str]] = defaultdict(list)
    for p in parents:
        members_by_root[uf.find(p["parent_id"])].append(p["parent_id"])
    groups = {f"grp:{min(m)}": sorted(m) for m in members_by_root.values() if len(m) >= 2}
    payload = {
        "note": ("Groupes d'images parentes qui se recouvrent sur le terrain : split_dataset.py place "
                 "chaque groupe entier dans un seul split. Généré par "
                 "src/data/diagnostics/spatial_groups.py - ne pas éditer à la main, relancer après "
                 "l'ajout d'un lot."),
        "generated": date.today().isoformat(),
        "rules": {"block_cells": BLOCK_CELLS, "min_ortho_overlap": MIN_ORTHO_OVERLAP,
                  "min_photo_overlap": MIN_PHOTO_OVERLAP},
        "imagette_neighbour_pairs": {"same_block": same, "cross_block": cross},
        "n_groups": len(groups),
        "n_grouped_images": sum(len(v) for v in groups.values()),
        "groups": groups,
        "overlap_evidence": evidence,
    }
    print(f"  => {len(groups)} groupe(s), {payload['n_grouped_images']} image(s) regroupée(s).")
    if dry_run:
        print("  (dry-run : rien n'est écrit)")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
        print(f"📄 {output}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calcule config/split_groups.json (groupes spatiaux).")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build_split_groups(Path(args.raw_dir), dry_run=args.dry_run)
