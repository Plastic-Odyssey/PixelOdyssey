#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Campagne du protocole v4 (12/10/2026) : réentraînement mono + multi sur tout le dataset.

Un seul appel, depuis la racine du dépôt (terminal VS Code, environnement du projet actif) :

    python -m src.training.campagne_v4

Enchaîne, avec reprise automatique si la campagne est interrompue (relancer la même commande) :
  1. Vérifications : banc de test figé complet (identifiants suivis en cas de renommage), groupes
     spatiaux régénérés, contrôle d'alignement des annotations (bloquant sur un décalage avéré),
     espace disque, GPU.
  2. Datasets v4 : multi (config/data_config_v4.yaml -> *_v4) et mono (config/data_config_mono_v4.yaml
     -> *_mono_v4), chacun terminé par le contrôle d'étanchéité bloquant ; on vérifie ensuite que les
     deux datasets ont exactement le même découpage train/val/test.
  3. Entraînements (recette v4, 200 epochs, sans arrêt anticipé) :
       v4_mono_s  : yolo11s-seg, mono    (~5-7 h sur la RTX 3070)
       v4_multi_s : yolo11s-seg, multi   (~5-7 h)
       v4_mono_n  : yolo11n-seg, mono    (~2-3 h, témoin : le small vaut-il son coût dans l'application ?)
  4. Évaluations au niveau objet (object_level_eval.py, images parentes entières) :
       - seuil de confiance choisi sur la VAL pour chaque modèle (F1 max) ;
       - banc de test figé, vérité terrain v4 (« à déterminer » compté, Inconnu = zone ignorée) ;
       - mêmes mesures pour les modèles du 08/10 (références de l'application v1.1).
  5. Tableau : F1 [IC95 bootstrap par image] au seuil choisi et à 0,25, P/R, AP50, par site, par
     classe (multi), + durées et meilleure epoch. Terminal + CSV + Markdown dans
     <6. Model outputs>/campagnes/<nom>/.

Options :
    --nom campagne_v4_20261012     (défaut : campagne_v4_<date>)
    --etapes verifs,data,train,eval,tableau
    --modeles v4_mono_s,v4_multi_s,v4_mono_n
    --refaire                      (refait entraînements / évaluations déjà faits)
    --oui                          (pas de confirmation)
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from src.paths_config import DATA_ROOT, TRAINING_RUNS_DIR, resolve_data_config_path_field
from src.training.campaign import Campaign, _banner, _find_new_run, _load_records

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_PROD = 0.25
IGNORE_ZONES = ["Inconnu"]
MIN_FREE_GB = 40
STEPS = ["verifs", "data", "train", "eval", "tableau"]
DATASETS = {
    "multi": ("config/data_config_v4.yaml", "_v4"),
    "mono": ("config/data_config_mono_v4.yaml", "_mono_v4"),
}
# clé, taxonomie, poids pré-entraînés, durée indicative (h, RTX 3070 portable)
MODELS = [
    ("v4_mono_s", "mono", "yolo11s-seg.pt", 6.0),
    ("v4_multi_s", "multi", "yolo11s-seg.pt", 6.0),
    ("v4_mono_n", "mono", "yolo11n-seg.pt", 2.5),
]
MODEL_BY_KEY = {m[0]: m for m in MODELS}
# Modèles en service dans l'application v1.1 (campagne du 08/10/2026), évalués de la même façon.
REFERENCES = {
    "ref_mono_0810": ("mono", "campagne_20261008_full_mono_yolo11n-seg_20261008_211021"),
    "ref_multi_0810": ("multi", "campagne_20261008_full_multi_yolo11n-seg_20261009_183305"),
}
COLS = [("Tous sites", "all"), ("Santa Luzia", "SL"), ("Saint-Brandon", "SB"), ("Aldabra", "A")]
CONF_GRID = [round(x, 3) for x in np.arange(0.05, 0.701, 0.025)]


# ----------------------------------------------------------------------------- 1. vérifications
def _alignment_check(c: Campaign) -> None:
    """Contrôle d'alignement annotations / images (label_alignment_check) : bloque si une image montre
    la signature d'un décalage géométrique (score faible tel quel, nettement meilleur après un zoom ou
    un décalage de plus de 20 px) - le cas des transects 6-10 et de SL W6 n du 12/10/2026."""
    out = Path(c.dir) / "controle_alignement_annotations.csv"
    c.log("Contrôle d'alignement des annotations (1 à 3 min)...")
    subprocess.run([sys.executable, "-m", "src.data.diagnostics.label_alignment_check", "--output", str(out)],
                   cwd=str(PROJECT_ROOT), check=True, stdout=subprocess.DEVNULL)
    bad = []
    with open(out, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            ident, best, n = float(r["score_identite"]), float(r["score_meilleur"]), int(r["n_objets"])
            dx, dy = abs(int(r["dx_px"])), abs(int(r["dy_px"]))
            # Signature d'un décalage réel (mesurée le 12/10/2026 sur tout le jeu) : assez d'objets pour
            # que la corrélation ait un sens, score faible tel quel, nettement meilleur après un PETIT
            # zoom ou décalage (< 250 px). Les grands décalages sur les imagettes SB peu annotées et la
            # roche d'Aldabra sont des pics de corrélation fortuits, pas des décalages.
            moved = max(dx, dy) <= 250 and (dx + dy > 20 or abs(float(r["zoom"]) - 1) > 0.015)
            if n >= 40 and ident < 0.7 and best > 1.35 * max(ident, 0.05) and moved:
                bad.append(f"{r['lot']} / {r['image']} (tel quel {ident}, après recalage {best}, zoom {r['zoom']}, "
                           f"{r['dx_px']},{r['dy_px']} px)")
    if bad:
        raise RuntimeError("🛑 Annotations décalées par rapport à leur image (à corriger avant d'entraîner) :\n  - "
                           + "\n  - ".join(bad) + f"\nDétail : {out}. Corrige puis relance la campagne.")
    c.log("Alignement des annotations : aucun décalage détecté.")


def step_checks(c: Campaign) -> None:
    _banner("1/5 VÉRIFICATIONS")
    from src.data.diagnostics.spatial_groups import build_split_groups
    from src.data.utils import freeze_benchmark_test as fb
    from src.data.utils.raw_dataset import collect_parent_images
    from src.data.utils.split_dataset import FROZEN_TEST_BENCH_ENABLED, FROZEN_TEST_PATH, RAW_DIR

    if not FROZEN_TEST_BENCH_ENABLED or not FROZEN_TEST_PATH.exists():
        raise RuntimeError("Banc de test figé désactivé ou absent.")
    current = {p["parent_id"] for p in collect_parent_images(Path(RAW_DIR))}
    with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
        frozen = json.load(f)["parent_ids"]
    if any(pid not in current for pid in frozen):
        c.log("Banc figé : identifiant(s) introuvable(s) - recherche d'un renommage (lot ou fichier).")
        fb.remap_renamed_frozen()
        with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
            frozen = json.load(f)["parent_ids"]
        lost = [pid for pid in frozen if pid not in current]
        if lost:
            raise RuntimeError(f"{len(lost)} image(s) du banc figé introuvable(s) : {lost[:3]} - campagne arrêtée.")
    c.log(f"Banc figé : {len(frozen)} images, toutes présentes.")
    groups = build_split_groups()
    c.state["checks"]["groups"] = groups["n_groups"]
    _alignment_check(c)
    free_gb = shutil.disk_usage(DATA_ROOT).free / 1e9
    c.log(f"Espace libre sur {DATA_ROOT} : {free_gb:.0f} Go.")
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(f"Espace disque insuffisant ({free_gb:.0f} Go, minimum {MIN_FREE_GB}).")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        gpu = None
    c.log(f"GPU : {gpu or 'AUCUN - entraînements sur CPU, beaucoup trop lents'}")
    c.save()


# ----------------------------------------------------------------------------- 2. datasets
def _split_dir(cfg: str) -> Path:
    with open(PROJECT_ROOT / cfg, "r", encoding="utf-8") as f:
        sliced = Path(resolve_data_config_path_field(yaml.safe_load(f) or {})["path"])
    return sliced.parent / sliced.name.replace("4_sliced_dataset", "2_split_dataset", 1)


def step_data(c: Campaign, taxonomies: List[str]) -> None:
    _banner("2/5 DATASETS v4")
    from src.data.data_pipeline import run_full_pipeline
    for tax in taxonomies:
        cfg, suffix = DATASETS[tax]
        c.log(f"Dataset {tax} : {cfg} -> suffixe {suffix}")
        t0 = time.time()
        run_full_pipeline(force=True, config_path=str(PROJECT_ROOT / cfg), suffix=suffix)
        c.state["datasets"][tax] = {"config": cfg, "minutes": round((time.time() - t0) / 60, 1),
                                    "done": datetime.now().isoformat(timespec="seconds")}
        c.save()
    if len(taxonomies) == 2:
        m = {}
        for tax in taxonomies:
            with open(_split_dir(DATASETS[tax][0]) / ".parent_manifest.json", "r", encoding="utf-8") as f:
                m[tax] = {pid: v["split"] for pid, v in json.load(f).items()}
        if m["mono"] != m["multi"]:
            diff = sorted(set(m["mono"].items()) ^ set(m["multi"].items()))
            raise RuntimeError(f"🛑 Mono et multi n'ont pas le même découpage ({len(diff)} écart(s), ex. {diff[:2]}).")
        c.log(f"Mono et multi : même découpage ({len(m['mono'])} images parentes).")


# ----------------------------------------------------------------------------- 3. entraînements
def step_train(c: Campaign, keys: List[str], redo: bool) -> None:
    _banner("3/5 ENTRAÎNEMENTS (recette v4)")
    for i, key in enumerate(keys, 1):
        _, tax, weights, hours = MODEL_BY_KEY[key]
        cfg = DATASETS[tax][0]
        m = c.model(key)
        if not redo and m.get("run_dir") and (Path(m["run_dir"]) / "weights" / "best.pt").exists():
            c.log(f"[{i}/{len(keys)}] {key} : déjà entraîné ({Path(m['run_dir']).name}) - ignoré.")
            continue
        tag = f"{c.name}_{key}"
        c.log(f"[{i}/{len(keys)}] {key} : {weights} sur {cfg} (~{hours:g} h)...")
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "src.training.train", "--config", cfg, "--tag", tag,
                               "--recette", "v4", "--model", weights, "--skip-viz"], cwd=str(PROJECT_ROOT))
        run_dir = _find_new_run(tag, t0)
        ok = proc.returncode == 0 and run_dir is not None and (run_dir / "weights" / "best.pt").exists()
        m.update({"tag": tag, "run_dir": str(run_dir) if run_dir else None, "train_ok": ok,
                  "train_minutes": round((time.time() - t0) / 60, 1), "reports": {}})
        c.save()
        c.log(f"    -> {'OK' if ok else 'ÉCHEC (code ' + str(proc.returncode) + ')'} {run_dir.name if run_dir else ''} "
              f"en {m['train_minutes']} min")


# ----------------------------------------------------------------------------- 4. évaluations
def _best_conf(records) -> float:
    from src.training.object_level_eval import _metrics
    scores = [(_metrics(records, t)["macro"]["f1"], t) for t in CONF_GRID]
    return max(scores)[1]


def _evaluate(c: Campaign, label: str, run_dir: Path, tax: str, redo: bool) -> None:
    from src.training.object_level_eval import run_object_level_eval
    st = c.state.setdefault("evals", {}).setdefault(label, {"run_dir": str(run_dir), "tax": tax})
    gt = str(PROJECT_ROOT / DATASETS[tax][0])
    for split in ("val", "test"):
        key = f"rapport_{split}"
        if redo or not st.get(key) or not Path(st[key]).exists():
            c.log(f"{label} : évaluation niveau objet ({split}) de {run_dir.name}")
            st[key] = str(run_object_level_eval(run_dir, split=split, conf=CONF_PROD, gt_config=gt,
                                                ignore_zones=IGNORE_ZONES))
            c.save()
    _, recs_val = _load_full(st["rapport_val"])
    st["conf_val"] = _best_conf(recs_val)
    c.log(f"{label} : seuil choisi sur la val = {st['conf_val']}")
    c.save()


def step_eval(c: Campaign, keys: List[str], redo: bool) -> None:
    _banner("4/5 ÉVALUATIONS (niveau objet, seuil choisi sur la val, banc figé)")
    for key in keys:
        m = c.model(key)
        if not m.get("run_dir") or not Path(m["run_dir"], "weights", "best.pt").exists():
            c.log(f"{key} : pas de modèle entraîné - ignoré.")
            continue
        _evaluate(c, key, Path(m["run_dir"]), MODEL_BY_KEY[key][1], redo)
    taxes = {MODEL_BY_KEY[k][1] for k in keys}
    for label, (tax, run_name) in REFERENCES.items():
        run_dir = TRAINING_RUNS_DIR / run_name
        if tax in taxes and (run_dir / "weights" / "best.pt").exists():
            _evaluate(c, label, run_dir, tax, redo)


# ----------------------------------------------------------------------------- 5. tableau
def _cell(records, site: str, conf: float, n_boot: int = 300) -> Dict:
    from src.training.object_level_eval import _metrics
    sub = records if site == "all" else [r for r in records if r["site"] == site]
    if not sub:
        return {}
    mm = _metrics(sub, conf)
    m = mm["macro"]
    rng = np.random.default_rng(0)
    boot = [_metrics([sub[i] for i in rng.integers(0, len(sub), len(sub))], conf)["macro"]["f1"] for _ in range(n_boot)]
    return {"f1": m["f1"], "p": m["p"], "r": m["r"], "ap50": m["ap50"], "f1_max": m["f1_max"],
            "lo": float(np.nanpercentile(boot, 2.5)), "hi": float(np.nanpercentile(boot, 97.5)),
            "n_obj": int(sum(sum(r["n_gt"].values()) for r in sub)), "n_img": len(sub), "per_class": mm["per_class"]}


def _train_info(run_dir: Path) -> Dict:
    info = {}
    res = run_dir / "results.csv"
    if res.exists():
        with open(res, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            k = [c for c in rows[0] if "mAP50-95(M)" in c]
            if k:
                best = max(rows, key=lambda r: float(r[k[0]].strip() or 0))
                info = {"epochs": len(rows), "best_epoch": int(float(best["epoch"])), "val_map5095_m": float(best[k[0]])}
    return info


def _load_full(report: str):
    """Comme campaign._load_records, mais garde le détail ajouté le 17/10/2026 (surface et zone de chaque
    prédiction, taille / surface / confiance de détection de chaque objet annoté)."""
    with open(report, "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data["records"]:
        r["n_gt"] = {int(k): v for k, v in r["n_gt"].items()}
        r["preds"] = [(float(p[0]), int(p[1]), int(p[2]), *p[3:]) for p in r["preds"]]
        r["gts"] = [tuple(g) for g in r.get("gts", [])]
    return data["summary"], data["records"]


def _paired(recs_a, conf_a, recs_b, conf_b, n_boot: int = 500) -> Dict:
    """Écart de F1 macro A - B sur les MÊMES images, IC95 par bootstrap apparié (images tirées ensemble)."""
    from src.training.object_level_eval import _metrics
    a = {r["parent_id"]: r for r in recs_a}
    b = {r["parent_id"]: r for r in recs_b}
    ids = sorted(set(a) & set(b))
    d0 = _metrics([a[i] for i in ids], conf_a)["macro"]["f1"] - _metrics([b[i] for i in ids], conf_b)["macro"]["f1"]
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n_boot):
        s = [ids[k] for k in rng.integers(0, len(ids), len(ids))]
        diffs.append(_metrics([a[i] for i in s], conf_a)["macro"]["f1"] - _metrics([b[i] for i in s], conf_b)["macro"]["f1"])
    diffs = np.array(diffs)
    return {"ecart": d0, "lo": float(np.nanpercentile(diffs, 2.5)), "hi": float(np.nanpercentile(diffs, 97.5)),
            "p_gain": float(np.mean(diffs > 0))}


def step_table(c: Campaign) -> Path:
    from src.training.object_level_eval import SIZE_BINS, _metrics, size_recall, surface_metrics
    _banner("5/5 TABLEAU")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [f"# Campagne {c.name} - protocole v4\n",
             f"Généré le {stamp}. F1 macro au niveau objet sur le banc de test figé, au seuil choisi sur la val "
             f"(et à {CONF_PROD}), [IC95 bootstrap par image]. Vérité terrain v4 : « à déterminer » compté comme "
             f"déchet, Inconnu = zone ignorée, prédictions sur la bordure noire des orthos écartées. Les modèles du "
             f"08/10 n'ont jamais appris « à déterminer » (léger handicap) et ont été entraînés sur les annotations "
             f"fausses des transects 6-10 et de SL W6 n.\n"]
    rows_csv = []
    order = [k for k, *_ in MODELS] + list(REFERENCES)
    for tax, title in (("mono", "MONO-CLASSE"), ("multi", "MULTI-CLASSE (7 classes)")):
        labels = [l for l in order if c.state.get("evals", {}).get(l, {}).get("tax") == tax
                  and c.state["evals"][l].get("rapport_test") and Path(c.state["evals"][l]["rapport_test"]).exists()]
        if not labels:
            continue
        recs_by = {l: _load_full(c.state["evals"][l]["rapport_test"])[1] for l in labels}
        conf_by = {l: c.state["evals"][l]["conf_val"] for l in labels}
        print(f"\n{title}")
        lines.append(f"\n## {title}\n\n| Modèle | seuil val | " + " | ".join(col for col, _ in COLS)
                     + " | F1 @0,25 (tous) | AP50 (tous) | epochs (meilleure) |\n|" + "---|" * (len(COLS) + 5))
        for label in labels:
            ev = c.state["evals"][label]
            recs, conf = recs_by[label], conf_by[label]
            cells = {site: _cell(recs, site, conf) for _, site in COLS}
            f1_prod = _metrics(recs, CONF_PROD)["macro"]["f1"]
            ti = _train_info(Path(ev["run_dir"]))
            txt = [f"{cells[s]['f1']:.3f} [{cells[s]['lo']:.2f}-{cells[s]['hi']:.2f}]" if cells[s] else "n/a" for _, s in COLS]
            ep = f"{ti.get('epochs', '?')} ({ti.get('best_epoch', '?')})"
            print(f"  {label:<16} seuil {conf:<5} | " + " | ".join(f"{col} {t}" for (col, _), t in zip(COLS, txt))
                  + f" | @0,25 {f1_prod:.3f} | AP50 {cells['all']['ap50']:.3f} | epochs {ep}")
            lines.append(f"| {label} | {conf} | " + " | ".join(txt) + f" | {f1_prod:.3f} | {cells['all']['ap50']:.3f} | {ep} |")
            sizes = size_recall(recs, conf)
            surf = surface_metrics(recs, conf)
            for col, site in COLS:
                x = cells[site]
                if x:
                    row = {"campagne": c.name, "taxonomie": tax, "modele": label, "run": Path(ev["run_dir"]).name,
                           "evaluation": col, "seuil_val": conf, **{k: round(x[k], 4) for k in ("f1", "lo", "hi", "p", "r", "ap50", "f1_max")},
                           "n_objets": x["n_obj"], "n_images": x["n_img"]}
                    sub = recs if site == "all" else [r for r in recs if r["site"] == site]
                    for b in size_recall(sub, conf):
                        row[f"rappel {b['classe_taille']}"] = round(b["rappel"], 4) if b["n"] else ""
                    for k, v in surface_metrics(sub, conf).items():
                        row[k] = round(v, 4) if v == v else ""
                    rows_csv.append(row)
        recs0 = recs_by[labels[0]]
        counts = []
        for col, site in COLS:
            sub = recs0 if site == "all" else [r for r in recs0 if r["site"] == site]
            counts.append(f"{col} : {sum(sum(r['n_gt'].values()) for r in sub)} objets / {len(sub)} images")
        lines.append("\nBanc de test : " + " ; ".join(counts) + ".")

        # Rappel par taille et mesures en surface (17/10/2026).
        if all(r.get("gts") for r in recs0 if sum(r["n_gt"].values())):
            nb = size_recall(recs0, 0.0)
            lines.append("\n### Rappel par taille d'objet et surface (seuil choisi sur la val, tous sites)\n\n"
                         "Taille = plus grande dimension de l'objet annoté (~0,5 cm/px). Surface retrouvée = part de la "
                         "surface des déchets annotés retrouvée (approximation du poids). Fausses alertes = surface "
                         "prédite sur du fond, rapportée à la surface des vraies détections.\n\n| Modèle | "
                         + " | ".join(f"{b['classe_taille']} (n={b['n']})" for b in nb)
                         + " | surface retrouvée | fausses alertes / vraies (surface) | doublons, fragments / vraies |\n|"
                         + "---|" * (len(nb) + 4))
            print("  Rappel par taille / surface :")
            for label in labels:
                sr = size_recall(recs_by[label], conf_by[label])
                su = surface_metrics(recs_by[label], conf_by[label])
                cells = [f"{b['rappel']:.2f}" if b["n"] else "n/a" for b in sr]
                lines.append(f"| {label} | " + " | ".join(cells) + f" | {su['surface_retrouvee']:.2f} | "
                             f"{su['fausses_alertes_fond_sur_vrais']:.2f} | {su['doublons_fragments_sur_vrais']:.2f} |")
                print(f"    {label:<16} " + " ".join(f"{b['classe_taille']}: {c_}" for b, c_ in zip(sr, cells))
                      + f" | surface {su['surface_retrouvee']:.2f} | FA/vrais {su['fausses_alertes_fond_sur_vrais']:.2f}")
        else:
            lines.append("\n(Rappel par taille et surface : rapports d'évaluation antérieurs au 17/10/2026, relancer "
                         "`--etapes eval,tableau --refaire`.)")

        # Comparaisons appariées : chaque modèle v4 contre la référence de sa taxonomie, et small contre nano.
        ref = next((l for l in labels if l.startswith("ref_")), None)
        pairs = [(l, ref) for l in labels if ref and l != ref]
        if "v4_mono_s" in labels and "v4_mono_n" in labels and tax == "mono":
            pairs.append(("v4_mono_s", "v4_mono_n"))
        if pairs:
            lines.append("\n### Comparaisons appariées (mêmes images, bootstrap par image)\n\n"
                         "| Comparaison | Tous sites : écart F1 [IC95] | P(gain) | Santa Luzia : écart F1 [IC95] |\n|---|---|---|---|")
            for a, b in pairs:
                pa = _paired(recs_by[a], conf_by[a], recs_by[b], conf_by[b])
                sl_a = [r for r in recs_by[a] if r["site"] == "SL"]
                sl_b = [r for r in recs_by[b] if r["site"] == "SL"]
                ps = _paired(sl_a, conf_by[a], sl_b, conf_by[b]) if sl_a and sl_b else None
                lines.append(f"| {a} - {b} | {pa['ecart']:+.3f} [{pa['lo']:+.3f} ; {pa['hi']:+.3f}] | {pa['p_gain']:.2f} | "
                             + (f"{ps['ecart']:+.3f} [{ps['lo']:+.3f} ; {ps['hi']:+.3f}]" if ps else "n/a") + " |")
                print(f"  {a} - {b} : {pa['ecart']:+.3f} [{pa['lo']:+.3f} ; {pa['hi']:+.3f}], P(gain) {pa['p_gain']:.2f}")

        if tax == "multi":
            with open(PROJECT_ROOT / DATASETS["multi"][0], "r", encoding="utf-8") as f:
                names = (yaml.safe_load(f) or {}).get("names", {})
            lines.append("\n### Par classe (F1 au seuil choisi sur la val, tous sites)\n\n| Classe | objets test | "
                         + " | ".join(labels) + " |\n|" + "---|" * (len(labels) + 2))
            per = {l: _metrics(recs_by[l], conf_by[l])["per_class"] for l in labels}
            for cid in sorted({k for v in per.values() for k in v}):
                ngt = next(v[cid]["n_gt"] for v in per.values() if cid in v)
                lines.append(f"| {names.get(cid, cid)} | {ngt} | " + " | ".join(
                    f"{per[l][cid]['f1']:.3f}" if cid in per[l] else "n/a" for l in labels) + " |")
    out_md = Path(c.dir) / f"tableau_{stamp}.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rows_csv:
        fields = list(dict.fromkeys(k for r in rows_csv for k in r))
        with open(Path(c.dir) / f"tableau_{stamp}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
            w.writeheader()
            w.writerows(rows_csv)
    print(f"\n📄 {out_md}")
    return out_md


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Campagne du protocole v4 (mono + multi, tout le dataset).")
    ap.add_argument("--nom", default=f"campagne_v4_{datetime.now().strftime('%Y%m%d')}")
    ap.add_argument("--etapes", default=",".join(STEPS))
    ap.add_argument("--modeles", default=",".join(m[0] for m in MODELS))
    ap.add_argument("--refaire", action="store_true")
    ap.add_argument("--oui", action="store_true")
    args = ap.parse_args()
    keys = [k.strip() for k in args.modeles.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODEL_BY_KEY]
    if unknown:
        sys.exit(f"Modèle(s) inconnu(s) : {unknown}. Choix : {', '.join(MODEL_BY_KEY)}")
    steps = [s.strip() for s in args.etapes.split(",")]
    c = Campaign(args.nom)
    todo = [k for k in keys if args.refaire or not (c.model(k).get("run_dir") and Path(c.model(k)["run_dir"], "weights", "best.pt").exists())]
    hours = sum(MODEL_BY_KEY[k][3] for k in todo) if "train" in steps else 0
    print(f"Campagne '{c.name}' -> {c.dir}\nÉtapes : {', '.join(steps)}\nModèles : {', '.join(keys)}"
          f"\nÀ entraîner : {', '.join(todo) if 'train' in steps and todo else 'aucun'}"
          f"\nDurée indicative : ~{hours + (0.5 if 'data' in steps else 0) + (0.2 * (len(keys) + 2) if 'eval' in steps else 0):.0f} h")
    if not args.oui:
        if input("Lancer ? [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
            sys.exit("Annulé.")
    c.log(f"Début - étapes {steps}, modèles {keys}")
    t0 = time.time()
    try:
        if "verifs" in steps:
            step_checks(c)
        if "data" in steps:
            step_data(c, sorted({MODEL_BY_KEY[k][1] for k in keys}))
        if "train" in steps:
            step_train(c, keys, args.refaire)
        if "eval" in steps:
            step_eval(c, keys, args.refaire)
        if "tableau" in steps:
            step_table(c)
    finally:
        c.log(f"Fin - {(time.time() - t0) / 3600:.1f} h")
        c.save()


if __name__ == "__main__":
    main()
