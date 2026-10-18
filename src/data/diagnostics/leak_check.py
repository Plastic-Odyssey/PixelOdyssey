#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Contrôle d'étanchéité train / val / test d'un dataset tuilé (protocole v4, 12/10/2026).

Appelé à la fin de data_pipeline.py pour les datasets v4 (bloquant), ou à la main :

    python -m src.data.diagnostics.leak_check --config config/data_config_mono_v4.yaml

Vérifie, sans faire confiance aux étapes précédentes (tout est recalculé) :
  1. chaque image parente est dans un seul split, et le test est exactement le banc figé ;
  2. chaque fichier de 2_split_dataset et chaque tuile de 4_sliced_dataset appartient à une
     image parente du MÊME split (une tuile s'appelle "<parent_id>_<x>_<y>") ;
  3. terrain partagé : les zones communes à deux images de splits différents (recalculées par
     ground_overlap.py) ne figurent dans AUCUNE tuile de l'image la moins prioritaire ;
  4. les images sans géoréférence (non vérifiables) sont listées.
Sortie : console + <4_sliced_dataset>/.leak_check.json. Code de sortie 1 en cas de fuite.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import yaml

from src.data.utils.ground_overlap import compute_cross_split_zones, rasterize_zones
from src.data.utils.raw_dataset import VALID_IMG_EXTS
from src.data.utils.split_dataset import (
    FROZEN_TEST_BENCH_ENABLED, FROZEN_TEST_PATH, PARENT_MANIFEST_FILENAME, SHARED_GROUND_REPORT_FILENAME,
)
from src.paths_config import resolve_data_config_path_field

REPORT_FILENAME = ".leak_check.json"


def _tile_parent(stem: str) -> str:
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit() else stem


def run_leak_check(split_dir, sliced_dir, sensor_registry_path=None) -> dict:
    split_dir, sliced_dir = Path(split_dir), Path(sliced_dir)
    sensor_registry_path = sensor_registry_path or Path(__file__).resolve().parents[3] / "config" / "sensor_specs.yaml"
    problems = []
    with open(split_dir / PARENT_MANIFEST_FILENAME, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    split_of = {pid: v["split"] for pid, v in manifest.items()}

    # 1) Banc figé et quarantaine.
    bad = [pid for pid, s in split_of.items() if s not in ("train", "val", "test")]
    if bad:
        problems.append(f"{len(bad)} image(s) avec un split inconnu : {bad[:3]}")
    test = {pid for pid, s in split_of.items() if s == "test"}
    if FROZEN_TEST_BENCH_ENABLED and FROZEN_TEST_PATH.exists():
        with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
            frozen = set(json.load(f)["parent_ids"])
        if test != frozen:
            problems.append(f"test différent du banc figé : {len(test - frozen)} en trop, {len(frozen - test)} manquante(s) "
                            f"({sorted(test ^ frozen)[:3]})")
    quarantined = {}
    rep_path = split_dir / SHARED_GROUND_REPORT_FILENAME
    if rep_path.exists():
        with open(rep_path, "r", encoding="utf-8") as f:
            quarantined = json.load(f).get("quarantaine", {})
    leaked_q = sorted(set(quarantined) & set(split_of))
    if leaked_q:
        problems.append(f"{len(leaked_q)} image(s) en quarantaine présentes dans le split : {leaked_q[:3]}")

    # 2) Fichiers et tuiles.
    n_files = defaultdict(int)
    for base, is_tiles in ((split_dir, False), (sliced_dir, True)):
        for kind in ("images", "labels"):
            for s in ("train", "val", "test"):
                d = base / kind / s
                if not d.exists():
                    continue
                for f in d.iterdir():
                    if not f.is_file() or f.name.startswith("."):
                        continue
                    if kind == "images" and f.suffix.lower() not in VALID_IMG_EXTS:
                        continue
                    pid = _tile_parent(f.stem) if is_tiles else f.stem
                    n_files[(base.name, kind, s)] += 1
                    if split_of.get(pid) != s:
                        problems.append(f"{base.name}/{kind}/{s}/{f.name} : image parente {pid} "
                                        f"{'absente du manifeste' if pid not in split_of else 'dans le split ' + split_of[pid]}")

    # 3) Terrain partagé : recalculé indépendamment du split.
    parents = [{"parent_id": pid, "img_path": str(Path(str(v["raw_img_path"]).replace("\\", "/"))), "batch": v["batch"]}
               for pid, v in manifest.items()]
    for p in parents:  # chemins bruts ré-ancrés si le disque a changé de lettre
        if not Path(p["img_path"]).exists():
            norm = p["img_path"]
            marker = "1_annotated_dataset/"
            if marker in norm:
                p["img_path"] = str(split_dir.parent / "1_annotated_dataset" / norm.split(marker, 1)[1])
    zones, report = compute_cross_split_zones(parents, split_of, Path(sensor_registry_path))
    tiles_by_parent = defaultdict(list)
    for s in ("train", "val", "test"):
        d = sliced_dir / "images" / s
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() == ".png":
                    pid = _tile_parent(f.stem)
                    x, y = (int(v) for v in f.stem.rsplit("_", 2)[1:])
                    tiles_by_parent[pid].append((x, y, f))
    from PIL import Image
    leaked_px = 0
    for pid, polys in zones.items():
        info = next(p for p in parents if p["parent_id"] == pid)
        with Image.open(info["img_path"]) as im:
            w, h = im.size
        zm = rasterize_zones(polys, w, h)
        integ = np.pad(zm.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        for x, y, f in tiles_by_parent.get(pid, []):
            with Image.open(f) as t:
                tw, th = t.size
            x1, y1 = min(w, x + tw), min(h, y + th)
            n = int(integ[y1, x1] - integ[y, x1] - integ[y1, x] + integ[y, x])
            if n > 0:
                leaked_px += n
                problems.append(f"tuile {f.name} : {n} pixel(s) de terrain partagé avec un split plus prioritaire")

    result = {"ok": not problems, "problemes": problems[:200], "n_problemes": len(problems),
              "images": {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "val", "test")},
              "fichiers": {f"{a}/{b}/{c}": n for (a, b, c), n in sorted(n_files.items())},
              "paires_terrain_partage": len(report["paires"]), "images_masquees": len(zones),
              "pixels_partages_dans_tuiles": leaked_px, "non_verifiables": report["non_verifiables"],
              "quarantaine": quarantined}
    with open(sliced_dir / REPORT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("\n--- 🔒 CONTRÔLE D'ÉTANCHÉITÉ ---")
    print(f"  • Images parentes : train {result['images']['train']} | val {result['images']['val']} | "
          f"test {result['images']['test']} (banc figé) | quarantaine {len(quarantined)}")
    print(f"  • Terrain partagé entre splits : {len(report['paires'])} paire(s), {len(zones)} image(s) masquée(s), "
          f"{leaked_px} pixel(s) partagé(s) dans les tuiles")
    if report["non_verifiables"]:
        print(f"  • Non vérifiables (sans géoréférence) : {', '.join(report['non_verifiables'])}")
    if problems:
        print(f"  🛑 {len(problems)} problème(s) :")
        for p in problems[:15]:
            print(f"     - {p}")
    else:
        print("  ✅ Étanche : aucune image, aucune tuile, aucun pixel de terrain commun entre deux splits.")
    print(f"  📄 {sliced_dir / REPORT_FILENAME}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Contrôle d'étanchéité train/val/test d'un dataset tuilé.")
    ap.add_argument("--config", required=True, help="data_config*.yaml du dataset (son `path` = 4_sliced_dataset...).")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = resolve_data_config_path_field(yaml.safe_load(f) or {})
    sliced = Path(cfg["path"])
    split = sliced.parent / sliced.name.replace("4_sliced_dataset", "2_split_dataset", 1)
    res = run_leak_check(split, sliced)
    sys.exit(0 if res["ok"] else 1)
