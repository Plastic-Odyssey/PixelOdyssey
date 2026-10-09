#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Évaluation au niveau OBJET (image parente entière), complémentaire de
training_report.py / evaluation_matrix.py (ajouté le 23/09/2026).

Pourquoi un second outil d'évaluation : les métriques Ultralytics de training_report.py
sont calculées TUILE PAR TUILE sur des tuiles qui se recouvrent (640 px, recouvrement
256 px). Mesuré le 23/09/2026 sur le test actuel : chaque objet réel y est compté ~3 fois
(905 objets réels -> 2 678 "instances"), ~20 % de ces instances sont des morceaux d'objet
coupés au bord d'une tuile, et le F1 est pris au seuil de confiance optimal choisi SUR LE
TEST LUI-MÊME. Ce script mesure au contraire ce que produit réellement l'application :

  - inférence tuilée de production (même géométrie que l'entraînement, voir
    tiling_geometry.iter_tile_windows) puis fusion des doublons inter-tuiles
    (tiled_inference.nms_merge, ou variante --merge ios) ;
  - comparaison aux polygones de l'image PARENTE entière (2_split_dataset/labels/<split>),
    chaque objet réel compté une seule fois ;
  - appariement type COCO (prédictions par confiance décroissante, IoU polygone >= 0,5,
    même classe) ;
  - P/R/F1 au seuil de PRODUCTION (--conf, 0,25 par défaut, celui de run_inference.py),
    F1 max et AP50, par site et au total ;
  - intervalle de confiance à 95 % du F1 macro par bootstrap sur les IMAGES PARENTES
    (les tuiles d'une même image ne sont pas indépendantes) ;
  - ventilation des faux positifs (part de la prédiction couverte >= 50 %) : sur un objet
    annoté retenu (doublon, fragment, confusion de classe), sur un objet annoté puis EXCLU
    de la taxonomie (Inconnu, à déterminer, bois, verre - lus dans le label brut), ou sur
    du fond.

Résultats de référence (23/09/2026, test actuel = tirage par hash, 49 images parentes, CPU) :
  - reference_monoclasse_v2 : F1 objet @0,25 = 0,531 [IC95 0,44-0,61], P 0,58 / R 0,49,
    AP50 0,47. Faux positifs : 61 % sur fond, 22 % sur un objet annoté (doublons/fragments
    de bord de tuile), 15 % sur un objet "Inconnu"/"à déterminer" - 11 sur 14 à Saint Brandon.
    Avec --merge ios : F1 0,543, P 0,62, faux positifs "doublon/fragment" divisés par 2.
  - reference_multiclasse_v2 : F1 macro objet @0,25 = 0,435 [IC95 0,31-0,48] (F1 max 0,50 :
    le seuil 0,25 n'est pas optimal en multi-classe) ; à Santa Luzia, 40 % des faux positifs
    tombent sur un objet annoté (surtout des confusions de classe).

Options de comparaison (24/09/2026) :
  - --gt-config CONFIG : vérité terrain lue dans le split d'une AUTRE configuration que
    celle du run (mêmes images test grâce au banc figé). Sert à comparer deux taxonomies
    sur une même définition de l'objet, ex : mono avec Inconnu vs mono sans Inconnu, tous
    deux évalués sur la vérité terrain de data_config_mono_class.yaml.
  - --ignore-zones "Inconnu,à déterminer" : une prédiction dont au moins la moitié de la
    surface tombe sur un objet annoté de ces classes (exclues dans la config de vérité
    terrain) est retirée - ni vrai ni faux positif. Métrique neutre vis-à-vis de ces
    classes ambiguës.
  - --unseen-by RUN : n'évalue que les images test sur lesquelles RUN ne s'est jamais
    entraîné (absentes de son train, d'après son training_parent_manifest.json ; sa val, qui
    n'a servi qu'à choisir l'epoch, est acceptée) - pour comparer un ancien modèle et un
    nouveau sur un terrain propre aux deux.

Exemple :
    python -m src.training.object_level_eval --run "E:\\PixelOdyssey\\6. Model outputs\\runs\\<run>"
    python -m src.training.object_level_eval --run "<run>" --merge ios --conf 0.25
    python -m src.training.object_level_eval --run "<run mono Inconnu>" --gt-config config/data_config_mono_class.yaml --ignore-zones "Inconnu,à déterminer"
    python -m src.training.object_level_eval --run "<nouveau run>" --unseen-by "<ancien run>"

Entrée : dossier d'un run (weights/best.pt + args.yaml), split à évaluer (défaut test).
Sortie : résumé console + `<run>/rapport_objet_<split>.json` (`rapport_objet_<split>_ios.json`
avec --merge ios, pour ne pas écraser le premier) (+ détail par image parente).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import yaml
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from src.data.utils.class_config import (
    EXCLUDE, load_batch_local_names, load_class_config, normalize_class_name, resolve_class_name,
)
from src.data.utils.image_io import load_image_bgr
from src.data.utils.raw_dataset import site_of_batch
from src.data.utils.split_dataset import PARENT_MANIFEST_FILENAME
from src.data.utils.tiling_geometry import iter_tile_windows
from src.paths_config import PROCESSED_DATASET_DIR, resolve_data_config_path_field
from src.review.matching import LabeledPolygon, polygon_iou
from src.review.tiled_inference import nms_merge
from src.training.training_report import resolve_run_class_config_path

CONF_FLOOR = 0.05  # plancher de confiance pour tracer la courbe P/R (AP50, F1 max)


def ios_merge(preds: List[LabeledPolygon], thr: float = 0.5) -> List[LabeledPolygon]:
    """Fusion inter-tuiles par intersection / aire de la PLUS PETITE géométrie (au lieu de
    l'IoU de nms_merge) : récupère le fragment qu'une tuile voisine détecte sur un objet
    coupé par son bord, que l'IoU laisse passer comme un second objet (principe du NMM de
    SAHI). Même classe uniquement, conserve la prédiction de plus haute confiance."""
    by_class: Dict[int, List[LabeledPolygon]] = {}
    for p in preds:
        by_class.setdefault(p.class_id, []).append(p)
    kept: List[LabeledPolygon] = []
    for items in by_class.values():
        items = sorted(items, key=lambda p: p.confidence or 0.0, reverse=True)
        tree = STRtree([p.geom for p in items])
        taken = [False] * len(items)
        for i, p in enumerate(items):
            if taken[i]:
                continue
            kept.append(p)
            for j in tree.query(p.geom, predicate="intersects"):
                j = int(j)
                if j <= i or taken[j]:
                    continue
                a, b = p.geom, items[j].geom
                try:
                    inter = a.intersection(b).area
                except Exception:  # noqa: BLE001 - géométrie dégénérée : on ne fusionne pas
                    continue
                if inter / max(1e-9, min(a.area, b.area)) >= thr:
                    taken[j] = True
    return kept


def _polys_from_yolo(label_path: Path, w: int, h: int, class_map=None) -> List[LabeledPolygon]:
    out: List[LabeledPolygon] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        c = int(parts[0]) if class_map is None else class_map(int(parts[0]))
        xy = [float(v) for v in parts[1:]]
        g = Polygon([(xy[i] * w, xy[i + 1] * h) for i in range(0, len(xy) - 1, 2)])
        if not g.is_valid:
            g = g.buffer(0)
        if g.area > 0 and g.geom_type == "Polygon":
            out.append(LabeledPolygon(c, g))
    return out


def _reanchor(path_str: str) -> Path:
    """Chemin brut enregistré dans le manifeste (ex: E:\\PixelOdyssey\\3. Processed dataset\\...),
    ré-ancré sur le PROCESSED_DATASET_DIR de CETTE machine s'il n'existe plus tel quel
    (disque monté sous une autre lettre)."""
    p = Path(path_str)
    if p.exists():
        return p
    norm = path_str.replace("\\", "/")
    marker = "3. Processed dataset/"
    if marker in norm:
        return PROCESSED_DATASET_DIR / norm.split(marker, 1)[1]
    return p


def _excluded_zones(info: Dict, class_taxonomy, w: int, h: int):
    """Polygones des objets annotés dont la classe brute est EXCLUE (label brut du lot)."""
    label_path = _reanchor(info["raw_label_path"])
    if not label_path.exists():
        return []
    batch_root = label_path
    while batch_root.parent != batch_root and batch_root.name != info["batch"]:
        batch_root = batch_root.parent
    yaml_path = batch_root / "data.yaml"
    local_names = load_batch_local_names(yaml_path) if yaml_path.exists() else {}
    zones = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        name = local_names.get(int(parts[0]))
        if name is None or resolve_class_name(name, class_taxonomy) != EXCLUDE:
            continue
        xy = [float(v) for v in parts[1:]]
        g = Polygon([(xy[i] * w, xy[i + 1] * h) for i in range(0, len(xy) - 1, 2)])
        if not g.is_valid:
            g = g.buffer(0)
        if g.area > 0:
            zones.append((normalize_class_name(name), g))
    return zones


def _frac_of_pred(pred_geom, other_geom) -> float:
    try:
        return pred_geom.intersection(other_geom).area / max(1e-9, pred_geom.area)
    except Exception:  # noqa: BLE001 - géométrie dégénérée
        return 0.0


def _metrics(records: List[Dict], conf: float) -> Dict:
    classes = sorted({c for r in records for c in r["n_gt"]})
    per_class = {}
    for c in classes:
        n = sum(r["n_gt"].get(c, 0) for r in records)
        preds = sorted([p for r in records for p in r["preds"] if p[1] == c], key=lambda x: -x[0])
        if n == 0:
            continue
        if not preds:
            per_class[c] = {"n_gt": n, "ap50": 0.0, "f1_max": 0.0, "f1": 0.0, "p": 0.0, "r": 0.0}
            continue
        tp = np.cumsum([p[2] for p in preds])
        fp = np.cumsum([1 - p[2] for p in preds])
        confs = np.array([p[0] for p in preds])
        rec, prec = tp / n, tp / np.maximum(tp + fp, 1)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.maximum.accumulate(np.concatenate([[1.0], prec, [0.0]])[::-1])[::-1]
        ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        k = int(np.searchsorted(-confs, -conf, side="right")) - 1
        per_class[c] = {
            "n_gt": n, "ap50": ap, "f1_max": float(f1.max()),
            "f1": float(f1[k]) if k >= 0 else 0.0,
            "p": float(prec[k]) if k >= 0 else 0.0, "r": float(rec[k]) if k >= 0 else 0.0,
        }
    macro = {m: float(np.mean([v[m] for v in per_class.values()])) if per_class else float("nan")
             for m in ("ap50", "f1_max", "f1", "p", "r")}
    return {"macro": macro, "per_class": per_class}


def run_object_level_eval(run_dir, split: str = "test", conf: float = 0.25, merge: str = "iou",
                          weights_name: str = "best.pt", n_boot: int = 300, gt_config=None,
                          ignore_zones: Optional[List[str]] = None, unseen_by=None) -> Path:
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    cfg_path = Path(gt_config) if gt_config else resolve_run_class_config_path(run_dir)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = resolve_data_config_path_field(yaml.safe_load(f) or {})
    sliced = Path(cfg["path"])
    split_dir = sliced.parent / sliced.name.replace("4_sliced_dataset", "2_split_dataset", 1)
    class_taxonomy, gt_names = load_class_config(cfg_path)
    ignore_set = {normalize_class_name(z) for z in (ignore_zones or [])}
    with open(split_dir / PARENT_MANIFEST_FILENAME, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    parents = sorted(p for p, v in manifest.items() if v["split"] == split)
    if unseen_by:
        snap = Path(unseen_by) / "training_parent_manifest.json"
        if not snap.exists():
            raise RuntimeError(f"--unseen-by : {snap} introuvable (instantané écrit par train.py depuis le "
                               f"23/09/2026) - impossible de savoir ce que ce run a vu.")
        with open(snap, "r", encoding="utf-8") as f:
            # Seul le TRAIN compte comme "vu" : la val n'a servi qu'à choisir l'epoch (best.pt), un
            # biais léger et commun aux deux modèles comparés - l'exclure aussi ne laisserait sur le
            # banc v2 que 6 images (40 objets) pour les modèles du 23/09.
            seen = {p for p, v in json.load(f).items() if v.get("split") == "train"}
        # Lots renommés depuis cet entraînement (08/10/2026) : traduit les anciens identifiants
        # vus par ce run vers les nouveaux, d'après l'historique de remappage du banc figé.
        from src.data.utils.split_dataset import FROZEN_TEST_PATH
        if FROZEN_TEST_PATH.exists():
            with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
                for entry in json.load(f).get("remapped", []):
                    seen |= {new for old, new in entry.get("renamed", {}).items() if old in seen}
        n_before = len(parents)
        parents = [p for p in parents if p not in seen]
        print(f"ℹ️  --unseen-by {Path(unseen_by).name} : {len(parents)}/{n_before} images test sur lesquelles ce run "
              f"ne s'est jamais entraîné.")
    if not parents:
        raise RuntimeError(f"Aucune image parente à évaluer dans le split '{split}' de {split_dir}.")
    model = YOLO(str(run_dir / "weights" / weights_name))
    if len(model.names) != len(gt_names):
        raise RuntimeError(f"Le modèle prédit {len(model.names)} classe(s) mais la vérité terrain "
                           f"({cfg_path.name}) en a {len(gt_names)} - comparaison impossible.")
    print(f"--- Évaluation niveau objet : {run_dir.name} | vérité terrain {split_dir.name}/{split} "
          f"({len(parents)} images parentes) | fusion {merge} | seuil {conf}"
          + (f" | zones ignorées : {', '.join(sorted(ignore_set))}" if ignore_set else "") + " ---")

    records = []
    for k_parent, pid in enumerate(parents, 1):
        info = manifest[pid]
        # Même convention de nom que split_dataset.py : "<parent_id><extension source en minuscules>".
        img_path = split_dir / "images" / split / f"{pid}{Path(info['raw_img_path'].replace(chr(92), '/')).suffix.lower()}"
        img = load_image_bgr(img_path) if img_path.exists() else None
        if img is None:
            print(f"  ⚠️  image illisible/absente, ignorée : {img_path}")
            continue
        h, w = img.shape[:2]
        gts = _polys_from_yolo(split_dir / "labels" / split / f"{pid}.txt", w, h)
        windows = list(dict.fromkeys(iter_tile_windows(w, h, 640, 384))) if (w, h) != (640, 640) else [(0, 0, w, h)]
        preds: List[LabeledPolygon] = []
        for i in range(0, len(windows), 8):
            chunk = windows[i:i + 8]
            results = model.predict([img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk], conf=CONF_FLOOR, verbose=False)
            for (x0, y0, _, _), r in zip(chunk, results):
                if r.masks is None:
                    continue
                for c, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.xy):
                    if len(xy) < 3:
                        continue
                    g = Polygon([(float(x) + x0, float(y) + y0) for x, y in xy])
                    # Même règle que la production (tiled_inference.predict_parent_image) :
                    # un polygone invalide est écarté, pas réparé.
                    if not g.is_valid or g.area <= 0:
                        continue
                    preds.append(LabeledPolygon(int(c), g, float(cf)))
        preds = ios_merge(preds) if merge == "ios" else nms_merge(preds, iou_threshold=0.5)
        preds.sort(key=lambda p: -p.confidence)
        zones = _excluded_zones(info, class_taxonomy, w, h)
        n_ignored = 0
        if ignore_set:
            ign = [g for name, g in zones if name in ignore_set]
            kept_preds = [p for p in preds if not any(_frac_of_pred(p.geom, g) >= 0.5 for g in ign)]
            n_ignored = len(preds) - len(kept_preds)
            preds = kept_preds

        used = [False] * len(gts)
        tree = STRtree([g.geom for g in gts]) if gts else None
        matched = []
        for p in preds:
            best, bj = 0.0, -1
            if tree is not None:
                for j in tree.query(p.geom, predicate="intersects"):
                    j = int(j)
                    if used[j] or gts[j].class_id != p.class_id:
                        continue
                    iou = polygon_iou(p.geom, gts[j].geom)
                    if iou > best:
                        best, bj = iou, j
            if best >= 0.5:
                used[bj] = True
                matched.append((p.confidence, p.class_id, 1))
            else:
                matched.append((p.confidence, p.class_id, 0))

        fp_zone: Dict[str, int] = {}
        for p, m in zip(preds, matched):
            if m[2] == 1 or m[0] < conf:
                continue
            # Part de la PRÉDICTION couverte (pas de la plus petite des deux géométries) : une
            # grande fausse détection qui effleure un petit objet n'est pas "sur" cet objet.
            if any(_frac_of_pred(p.geom, g.geom) >= 0.5 for g in gts):
                zone = "objet annoté (doublon, fragment ou autre classe)"
            else:
                zone = next((name for name, g in zones if _frac_of_pred(p.geom, g) >= 0.5), "fond")
            fp_zone[zone] = fp_zone.get(zone, 0) + 1
        n_gt: Dict[int, int] = {}
        for g in gts:
            n_gt[g.class_id] = n_gt.get(g.class_id, 0) + 1
        records.append({"parent_id": pid, "site": site_of_batch(info["batch"]), "preds": matched,
                        "n_gt": n_gt, "fp_by_zone": fp_zone, "n_ignored_preds": n_ignored})
        print(f"  [{k_parent}/{len(parents)}] {pid}: {sum(n_gt.values())} objets, {len(preds)} prédictions")

    if not records:
        raise RuntimeError("Aucune image parente évaluable (toutes illisibles ou absentes) - rien à mesurer.")
    summary = {"run": run_dir.name, "split_dir": str(split_dir), "split": split, "conf": conf, "merge": merge,
               "gt_config": str(cfg_path), "ignore_zones": sorted(ignore_set),
               "unseen_by": Path(unseen_by).name if unseen_by else None, "n_parents": len(records),
               "all": _metrics(records, conf)}
    for site in sorted({r["site"] for r in records}):
        sub = [r for r in records if r["site"] == site]
        summary[site] = _metrics(sub, conf)
        summary[site]["n_parents"] = len(sub)
    rng = np.random.default_rng(0)
    boot = [_metrics([records[i] for i in rng.integers(0, len(records), len(records))], conf)["macro"]["f1"]
            for _ in range(n_boot)]
    summary["f1_ci95_bootstrap_parents"] = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    fz: Dict[str, int] = {}
    for r in records:
        for z, n in r["fp_by_zone"].items():
            fz[z] = fz.get(z, 0) + n
    summary["fp_by_zone"] = fz

    tags = ([merge] if merge != "iou" else []) + ([f"gt-{cfg_path.stem}"] if gt_config else []) \
        + (["ign"] if ignore_set else []) + ([f"unseen-{Path(unseen_by).name[:40]}"] if unseen_by else [])
    out = run_dir / f"rapport_objet_{split}{''.join('_' + t for t in tags)}.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False), encoding="utf-8")
    a = summary["all"]["macro"]
    lo, hi = summary["f1_ci95_bootstrap_parents"]
    print(f"\nTOTAL : F1 macro @{conf} = {a['f1']:.3f} [IC95 {lo:.2f}-{hi:.2f}] | P {a['p']:.3f} R {a['r']:.3f} | "
          f"F1 max {a['f1_max']:.3f} | AP50 {a['ap50']:.3f}")
    for site in sorted({r["site"] for r in records}):
        m = summary[site]["macro"]
        print(f"  {site:<4} ({summary[site]['n_parents']} images) F1 {m['f1']:.3f} | P {m['p']:.3f} R {m['r']:.3f} | AP50 {m['ap50']:.3f}")
    print(f"Faux positifs @{conf} par zone : {fz}")
    print(f"📄 {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation au niveau objet (image parente entière).")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance de production (défaut 0.25).")
    parser.add_argument("--merge", default="iou", choices=["iou", "ios"],
                        help="Fusion inter-tuiles : 'iou' (nms_merge, production actuelle) ou 'ios' (variante SAHI).")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--gt-config", default=None,
                        help="Configuration dont le split fournit la vérité terrain (défaut : celle du run).")
    parser.add_argument("--ignore-zones", default=None,
                        help="Classes brutes exclues dont les zones neutralisent les prédictions, séparées par "
                             "des virgules (ex: \"Inconnu,à déterminer\").")
    parser.add_argument("--unseen-by", default=None,
                        help="Dossier d'un run : n'évalue que les images test absentes de son train.")
    args = parser.parse_args()
    run_object_level_eval(args.run, split=args.split, conf=args.conf, merge=args.merge, weights_name=args.weights,
                          gt_config=args.gt_config,
                          ignore_zones=[z.strip() for z in args.ignore_zones.split(",")] if args.ignore_zones else None,
                          unseen_by=args.unseen_by)
