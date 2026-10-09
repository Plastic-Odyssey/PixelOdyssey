#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Campagne complète : datasets -> entraînements -> évaluations -> grand tableau.

Un seul appel (08/10/2026), depuis la racine du repo dans le terminal VS Code :

    python -m src.training.campaign

enchaîne, sans autre intervention qu'une confirmation au départ :
  1. Vérifications : dataset brut, groupes spatiaux (config/split_groups.json, régénérés à
     chaque campagne), banc de test figé (identifiants suivis en cas de renommage de lot,
     arrêt si une image du banc a disparu), espace disque, GPU.
  2. Datasets : les 9 variantes (multi, mono, mono + Inconnu, et chaque site SL/SB/A en multi
     et en mono) via data_pipeline.py - reconstruites seulement si leur configuration a changé.
  3. Entraînements : 9 modèles (même recette que les références du 23/09 pour que l'effet
     des données et du protocole soit isolé), dans l'ordre d'importance (Full d'abord).
  4. Évaluations au niveau objet (object_level_eval.py) sur le banc figé, toutes sur la
     vérité terrain du dataset COMPLET de leur taxonomie : chaque modèle reçoit ainsi ses 4
     colonnes (Full, SL, SB, A), y compris les modèles dédiés à un site.
  5. Grand tableau : F1 objet @0,25 [IC95 bootstrap par image] pour chaque case, contrôle de
     fuite par case, tableau par classe, mono vs mono + Inconnu (métrique neutre), et
     évolution par rapport aux références du 23/09 (sur les images test qu'elles n'ont
     jamais vues). Sorties : terminal + CSV + Markdown + page HTML dans le dossier de la
     campagne, et une copie du CSV dans src/training/evaluation_matrix_history/.

Reprise : l'état est enregistré dans <6. Model outputs>/campagnes/<nom>/campagne.json. Relancer
la même commande reprend là où la campagne s'est arrêtée (un modèle déjà entraîné ou une
évaluation déjà faite n'est jamais refait, sauf --refaire). Une nouvelle campagne :
--nom <autre_nom>.

Options utiles :
    --etapes verifs,data,train,eval,tableau   (défaut : toutes)
    --modeles full_multi,full_mono            (sous-ensemble, défaut : les 9)
    --tableau-seul                            (raccourci : reconstruit seulement le tableau)
    --oui                                     (pas de confirmation au départ)

Remplace, pour le grand tableau, src/training/evaluation_matrix.py (métrique par tuile au seuil
choisi sur le test, qui reste disponible pour l'historique).
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
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.paths_config import DATA_ROOT, TRAINING_RUNS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS_DIR = TRAINING_RUNS_DIR.parent / "campagnes"
CONF = 0.25
IGNORE_ZONES = ["Inconnu", "à déterminer"]
MIN_FREE_GB = 60

GT_CONFIG = {
    "multi": "config/data_config.yaml",
    "mono": "config/data_config_mono_class.yaml",
}

# clé, taxonomie, ligne du tableau, config, suffixe des dossiers de données, filtre de site,
# durée indicative d'entraînement (heures, RTX 3070 portable, dataset du 08/10/2026)
MODELS = [
    ("full_multi", "multi", "Full", "config/data_config.yaml", "", None, 2.0),
    ("full_mono", "mono", "Full", "config/data_config_mono_class.yaml", "_mono_class", None, 2.0),
    ("full_mono_inconnu", "mono_inconnu", "Full + Inconnu", "config/data_config_mono_class_inconnu.yaml",
     "_mono_inconnu", None, 2.0),
    ("SL_multi", "multi", "Santa Luzia", "config/variants/data_config_SL.yaml", "_SL", ["SL"], 1.5),
    ("SL_mono", "mono", "Santa Luzia", "config/variants/data_config_SL_mono.yaml", "_SL_mono", ["SL"], 1.5),
    ("A_multi", "multi", "Aldabra", "config/variants/data_config_A.yaml", "_A", ["A"], 0.5),
    ("A_mono", "mono", "Aldabra", "config/variants/data_config_A_mono.yaml", "_A_mono", ["A"], 0.5),
    ("SB_multi", "multi", "Saint Brandon", "config/variants/data_config_SB.yaml", "_SB", ["SB"], 0.3),
    ("SB_mono", "mono", "Saint Brandon", "config/variants/data_config_SB_mono.yaml", "_SB_mono", ["SB"], 0.3),
]
MODEL_BY_KEY = {m[0]: m for m in MODELS}

# Références du 23/09/2026 : comparées aux nouveaux modèles Full sur les images test qu'elles
# n'ont jamais eues en entraînement (object_level_eval --unseen-by).
PREVIOUS_REFERENCES = {
    "multi": "reference_multiclasse_v2_yolo11n-seg_20260923_114634",
    "mono": "reference_monoclasse_v2_yolo11n-seg_20260923_134351",
}

ROWS = ["Full", "Santa Luzia", "Saint Brandon", "Aldabra"]
COLS = [("Full", "all"), ("Santa Luzia", "SL"), ("Saint Brandon", "SB"), ("Aldabra", "A")]
STEPS = ["verifs", "data", "train", "eval", "tableau"]
CORNER = "Entraînement \\ Évaluation"


# ----------------------------------------------------------------------------- état / journal
class Campaign:
    def __init__(self, name: str):
        self.name = name
        self.dir = CAMPAIGNS_DIR / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "campagne.json"
        self.log_path = self.dir / "campagne.log"
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        else:
            self.state = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
                          "models": {}, "datasets": {}, "checks": {}}

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=1, ensure_ascii=False)
        tmp.replace(self.state_path)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def model(self, key: str) -> Dict:
        return self.state["models"].setdefault(key, {})


def _banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


# ----------------------------------------------------------------------------- 1. vérifications
def step_checks(c: Campaign, keys: List[str]) -> None:
    _banner("1/5 VÉRIFICATIONS")
    from src.data.diagnostics.spatial_groups import build_split_groups
    from src.data.utils import freeze_benchmark_test as fb
    from src.data.utils.raw_dataset import collect_parent_images
    from src.data.utils.split_dataset import FROZEN_TEST_BENCH_ENABLED, FROZEN_TEST_PATH, RAW_DIR

    # Banc figé : obligatoire, complet, identifiants à jour.
    if not FROZEN_TEST_BENCH_ENABLED or not FROZEN_TEST_PATH.exists():
        raise RuntimeError("Banc de test figé désactivé ou absent : une campagne n'a de sens que sur le banc figé.")
    current = {p["parent_id"] for p in collect_parent_images(Path(RAW_DIR))}
    with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
        frozen = json.load(f)["parent_ids"]
    if any(pid not in current for pid in frozen):
        c.log("Banc figé : identifiant(s) introuvable(s) - recherche d'un renommage de lot.")
        fb.remap_renamed_frozen()
        with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
            frozen = json.load(f)["parent_ids"]
        lost = [pid for pid in frozen if pid not in current]
        if lost:
            raise RuntimeError(f"{len(lost)} image(s) du banc figé introuvable(s) : {lost[:3]}... - "
                               f"campagne arrêtée (le banc doit rester complet).")
    c.log(f"Banc figé : {len(frozen)} images, toutes présentes.")
    c.state["checks"]["bench_size"] = len(frozen)

    # Groupes spatiaux : régénérés (déterministe) pour couvrir les lots ajoutés.
    groups = build_split_groups()
    c.state["checks"]["groups"] = groups["n_groups"]

    # Disque et GPU.
    free_gb = shutil.disk_usage(DATA_ROOT).free / 1e9
    c.log(f"Espace libre sur {DATA_ROOT} : {free_gb:.0f} Go (minimum conseillé {MIN_FREE_GB} Go).")
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(f"Espace disque insuffisant ({free_gb:.0f} Go) pour reconstruire les datasets.")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        gpu = None
    c.log(f"GPU : {gpu or 'AUCUN - entraînements sur CPU, beaucoup trop lents pour une campagne'}")
    c.save()


# ----------------------------------------------------------------------------- 2. datasets
def step_data(c: Campaign, keys: List[str]) -> None:
    _banner("2/5 DATASETS")
    from src.data.data_pipeline import run_full_pipeline
    done = set()
    for key in keys:
        _, _, _, cfg, suffix, site, _ = MODEL_BY_KEY[key]
        ds = f"{cfg}|{suffix}"
        if ds in done:
            continue
        done.add(ds)
        c.log(f"Dataset {suffix or '(principal)'} - {cfg}{' - site ' + ','.join(site) if site else ''}")
        t0 = time.time()
        # force=True : reconstruit une étape SEULEMENT si sa configuration a changé (sinon
        # mise à jour incrémentale) - la confirmation a été donnée une fois au lancement.
        run_full_pipeline(force=True, config_path=str(PROJECT_ROOT / cfg), suffix=suffix, site_filter=site)
        c.state["datasets"][suffix or "principal"] = {"config": cfg, "done": datetime.now().isoformat(timespec="seconds"),
                                                       "minutes": round((time.time() - t0) / 60, 1)}
        c.save()


# ----------------------------------------------------------------------------- 3. entraînements
def _find_new_run(tag: str, since: float) -> Optional[Path]:
    """Dossier créé par train.py pour ce tag : "<tag>_<modèle>_<AAAAMMJJ_HHMMSS>" (le motif exact
    évite qu'un tag préfixe d'un autre - full_mono / full_mono_inconnu - ne s'y confonde)."""
    pattern = re.compile(re.escape(tag) + r"_[^_]+_\d{8}_\d{6}")
    cands = [d for d in TRAINING_RUNS_DIR.iterdir()
             if d.is_dir() and pattern.fullmatch(d.name) and d.stat().st_mtime >= since - 5]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def step_train(c: Campaign, keys: List[str], redo: bool) -> None:
    _banner("3/5 ENTRAÎNEMENTS")
    for i, key in enumerate(keys, 1):
        _, _, row, cfg, _, _, hours = MODEL_BY_KEY[key]
        m = c.model(key)
        if not redo and m.get("run_dir") and (Path(m["run_dir"]) / "weights" / "best.pt").exists():
            c.log(f"[{i}/{len(keys)}] {key} : déjà entraîné ({Path(m['run_dir']).name}) - ignoré.")
            continue
        tag = f"{c.name}_{key}"
        c.log(f"[{i}/{len(keys)}] {key} : entraînement ({cfg}, ~{hours:g} h)...")
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "src.training.train", "--config", cfg, "--tag", tag, "--skip-viz"],
                              cwd=str(PROJECT_ROOT))
        run_dir = _find_new_run(tag, t0)
        ok = proc.returncode == 0 and run_dir is not None and (run_dir / "weights" / "best.pt").exists()
        m.update({"tag": tag, "run_dir": str(run_dir) if run_dir else None, "train_ok": ok,
                  "train_minutes": round((time.time() - t0) / 60, 1), "reports": {}})
        c.save()
        c.log(f"    -> {'OK' if ok else 'ÉCHEC (code ' + str(proc.returncode) + ')'} "
              f"{run_dir.name if run_dir else ''} en {m['train_minutes']} min")


# ----------------------------------------------------------------------------- 4. évaluations
def _eval(c: Campaign, label: str, run_dir: Path, store: Dict, redo: bool, **kw) -> None:
    from src.training.object_level_eval import run_object_level_eval
    if not redo and store.get(label) and Path(store[label]).exists():
        return
    try:
        out = run_object_level_eval(run_dir, conf=CONF, **kw)
        store[label] = str(out)
        c.save()
    except Exception as e:  # noqa: BLE001 - une évaluation en échec ne bloque pas les autres
        c.log(f"    ÉCHEC évaluation {label} de {run_dir.name} : {e}")


def step_eval(c: Campaign, keys: List[str], redo: bool) -> None:
    _banner("4/5 ÉVALUATIONS (niveau objet, banc figé)")
    for key in keys:
        _, tax, _, cfg, _, _, _ = MODEL_BY_KEY[key]
        m = c.model(key)
        if not m.get("run_dir") or not Path(m["run_dir"], "weights", "best.pt").exists():
            c.log(f"{key} : pas de modèle entraîné - évaluation ignorée.")
            continue
        run_dir = Path(m["run_dir"])
        reports = m.setdefault("reports", {})
        c.log(f"{key} : évaluation de {run_dir.name}")
        if tax == "mono_inconnu":
            _eval(c, "natif", run_dir, reports, redo, gt_config=str(PROJECT_ROOT / cfg))
            _eval(c, "neutre", run_dir, reports, redo, gt_config=str(PROJECT_ROOT / GT_CONFIG["mono"]),
                  ignore_zones=IGNORE_ZONES)
        else:
            _eval(c, "tableau", run_dir, reports, redo, gt_config=str(PROJECT_ROOT / GT_CONFIG[tax]))
            if key == "full_mono":
                _eval(c, "neutre", run_dir, reports, redo, gt_config=str(PROJECT_ROOT / GT_CONFIG["mono"]),
                      ignore_zones=IGNORE_ZONES)
    # Évolution depuis le 23/09 : ancienne et nouvelle référence sur le même sous-ensemble.
    prev = c.state.setdefault("previous", {})
    for tax, run_name in PREVIOUS_REFERENCES.items():
        old_dir = TRAINING_RUNS_DIR / run_name
        new_key = f"full_{tax}"
        new = c.model(new_key)
        if not old_dir.exists() or new_key not in keys or not new.get("run_dir"):
            continue
        store = prev.setdefault(tax, {"old_run": run_name})
        c.log(f"Comparaison au 23/09 ({tax}) sur les images jamais vues par {run_name}")
        _eval(c, "ancien", old_dir, store, redo, gt_config=str(PROJECT_ROOT / GT_CONFIG[tax]), unseen_by=str(old_dir))
        _eval(c, "nouveau", Path(new["run_dir"]), store, redo, gt_config=str(PROJECT_ROOT / GT_CONFIG[tax]),
              unseen_by=str(old_dir))


# ----------------------------------------------------------------------------- 5. tableau
def _load_records(report: str):
    with open(report, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = data["records"]
    for r in recs:  # JSON : clés de classe en texte -> entiers (comme en mémoire dans object_level_eval)
        r["n_gt"] = {int(k): v for k, v in r["n_gt"].items()}
        r["preds"] = [(float(p[0]), int(p[1]), int(p[2])) for p in r["preds"]]
    return data["summary"], recs


def _cell(records, site: str, n_boot: int = 300) -> Dict:
    from src.training.object_level_eval import _metrics
    sub = records if site == "all" else [r for r in records if r["site"] == site]
    if not sub:
        return {}
    m = _metrics(sub, CONF)["macro"]
    rng = np.random.default_rng(0)
    boot = [_metrics([sub[i] for i in rng.integers(0, len(sub), len(sub))], CONF)["macro"]["f1"] for _ in range(n_boot)]
    return {"f1": m["f1"], "p": m["p"], "r": m["r"], "ap50": m["ap50"], "f1_max": m["f1_max"],
            "ci_lo": float(np.nanpercentile(boot, 2.5)), "ci_hi": float(np.nanpercentile(boot, 97.5)),
            "n_parents": len(sub), "n_objets": int(sum(sum(r["n_gt"].values()) for r in sub)),
            "parents": [r["parent_id"] for r in sub]}


def _seen(run_dir: Path):
    snap = run_dir / "training_parent_manifest.json"
    if not snap.exists():
        return None
    with open(snap, "r", encoding="utf-8") as f:
        return {p for p, v in json.load(f).items() if v.get("split") in ("train", "val")}


def _fmt(cell: Dict) -> str:
    if not cell:
        return "n/a"
    return f"{cell['f1']:.3f} [{cell['ci_lo']:.2f}-{cell['ci_hi']:.2f}]" + cell.get("mark", "")


def step_table(c: Campaign) -> Path:
    _banner("5/5 GRAND TABLEAU")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_csv, md, html_tables = [], [f"# Grand tableau - {c.name}\n", f"Généré le {stamp}. F1 objet @{CONF} "
                                     "[IC95 bootstrap par image], banc de test figé, vérité terrain du dataset complet "
                                     "de chaque taxonomie. « ! » = fuite (image test vue à l'entraînement).\n"], []
    for tax, title in (("multi", "MULTI-CLASSE (7 classes)"), ("mono", "MONO-CLASSE")):
        grid = {}
        for key, ktax, row, *_ in MODELS:
            if ktax != tax:
                continue
            m = c.model(key)
            rep = m.get("reports", {}).get("tableau")
            if not rep or not Path(rep).exists():
                continue
            _, recs = _load_records(rep)
            seen = _seen(Path(m["run_dir"]))
            for col, site in COLS:
                cell = _cell(recs, site)
                if cell:
                    leak = None if seen is None else len(set(cell["parents"]) & seen)
                    cell["mark"] = "!" if leak else ("?" if leak is None else "")
                    cell["leaked"] = leak
                grid[(row, col)] = cell
                if cell:
                    rows_csv.append({"campagne": c.name, "taxonomie": tax, "entrainement": row,
                                     "run": Path(m["run_dir"]).name, "evaluation": col,
                                     **{k: round(cell[k], 4) for k in ("f1", "ci_lo", "ci_hi", "p", "r", "ap50", "f1_max")},
                                     "n_objets": cell["n_objets"], "n_images": cell["n_parents"], "fuite": cell["leaked"]})
        print(f"\n{title}\n{CORNER:<26}" + "".join(f"{col:<22}" for col, _ in COLS))
        md.append(f"\n## {title}\n\n| Entraînement \\ Évaluation | " + " | ".join(col for col, _ in COLS) + " |\n|" + "---|" * 5)
        for row in ROWS:
            cells = [_fmt(grid.get((row, col), {})) for col, _ in COLS]
            print(f"{row:<26}" + "".join(f"{x:<22}" for x in cells))
            md.append(f"| {row} | " + " | ".join(cells) + " |")
        ns = [grid.get(("Full", col), {}) for col, _ in COLS]
        print(f"{'(objets / images test)':<26}" + "".join(f"{(str(n['n_objets']) + ' / ' + str(n['n_parents'])) if n else '':<22}" for n in ns))
        md.append("| objets / images test | " + " | ".join(f"{n['n_objets']} / {n['n_parents']}" if n else "" for n in ns) + " |")
        html_tables.append((title, grid, ns))

    # Par classe (Full multi)
    full = c.model("full_multi").get("reports", {}).get("tableau")
    per_class_rows = []
    if full and Path(full).exists():
        summ, _ = _load_records(full)
        from src.data.utils.class_config import load_class_config
        _, names = load_class_config(PROJECT_ROOT / GT_CONFIG["multi"])
        print("\nPAR CLASSE (modèle Full multi, toutes images test)")
        md.append("\n## Par classe (modèle Full multi)\n\n| Classe | objets test | F1 @0,25 | AP50 |\n|---|---|---|---|")
        for cls, v in sorted(summ["all"]["per_class"].items(), key=lambda kv: int(kv[0])):
            nm = names.get(int(cls), cls)
            per_class_rows.append((nm, v["n_gt"], v["f1"], v["ap50"]))
            print(f"  {nm:<16} {v['n_gt']:>5} objets  F1 {v['f1']:.3f}  AP50 {v['ap50']:.3f}")
            md.append(f"| {nm} | {v['n_gt']} | {v['f1']:.3f} | {v['ap50']:.3f} |")

    # Mono vs mono + Inconnu (métrique neutre)
    neutral = []
    for key, label in (("full_mono", "Mono (Inconnu exclu)"), ("full_mono_inconnu", "Mono + Inconnu")):
        rep = c.model(key).get("reports", {}).get("neutre")
        if rep and Path(rep).exists():
            _, recs = _load_records(rep)
            neutral.append((label, [_cell(recs, s) for _, s in COLS]))
    if neutral:
        print("\nMONO vs MONO + INCONNU (métrique neutre : prédictions sur Inconnu/à déterminer ignorées)")
        md.append("\n## Mono vs mono + Inconnu (métrique neutre)\n\n| Modèle | " + " | ".join(col for col, _ in COLS) + " |\n|" + "---|" * 5)
        for label, cells in neutral:
            print(f"  {label:<24}" + "".join(f"{_fmt(x):<22}" for x in cells))
            md.append(f"| {label} | " + " | ".join(_fmt(x) for x in cells) + " |")

    # Évolution depuis le 23/09
    evo = []
    for tax, store in c.state.get("previous", {}).items():
        if store.get("ancien") and store.get("nouveau") and Path(store["ancien"]).exists() and Path(store["nouveau"]).exists():
            _, ro = _load_records(store["ancien"])
            _, rn = _load_records(store["nouveau"])
            evo.append((tax, _cell(ro, "all"), _cell(rn, "all")))
    if evo:
        print("\nÉVOLUTION DEPUIS LE 23/09 (images test jamais vues en entraînement par la référence du 23/09)")
        md.append("\n## Évolution depuis le 23/09\n\n| Taxonomie | Référence 23/09 | Nouvelle campagne | objets / images |\n|---|---|---|---|")
        for tax, o, n in evo:
            print(f"  {tax:<6} 23/09 : {_fmt(o):<22} -> maintenant : {_fmt(n):<22} ({o.get('n_objets')} objets, {o.get('n_parents')} images)")
            md.append(f"| {tax} | {_fmt(o)} | {_fmt(n)} | {o.get('n_objets')} / {o.get('n_parents')} |")

    print("\n'!' = fuite (cellule non interprétable), '?' = contrôle impossible. Les colonnes SB et Aldabra "
          "reposent sur peu d'objets : regarder l'intervalle de confiance avant de conclure.")

    # Fichiers
    out_csv = c.dir / f"grand_tableau_{stamp}.csv"
    if rows_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
        hist = PROJECT_ROOT / "src" / "training" / "evaluation_matrix_history"
        hist.mkdir(exist_ok=True)
        shutil.copyfile(out_csv, hist / f"{c.name}_{stamp}.csv")
    (c.dir / f"grand_tableau_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    html_path = c.dir / f"grand_tableau_{stamp}.html"
    html_path.write_text(_html(c, stamp, html_tables, per_class_rows, neutral, evo), encoding="utf-8")
    c.log(f"Tableau écrit : {html_path}")
    return html_path


def _html(c, stamp, tables, per_class, neutral, evo) -> str:
    def color(cell):
        if not cell:
            return "#f3f4f6"
        f = max(0.0, min(1.0, cell["f1"]))
        return f"hsl({int(10 + 120 * f)}, 55%, {92 - int(25 * f)}%)"
    parts = [f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Grand tableau {c.name}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1d1e3a;background:#fff}}
h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px}}table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #d8dbe3;padding:8px 12px;text-align:center;font-variant-numeric:tabular-nums}}
th{{background:#eef0f4}}td.l{{text-align:left;font-weight:600}}small{{color:#6d7089}}.leak{{color:#b3262c;font-weight:700}}</style>
</head><body><h1>Grand tableau - {c.name}</h1><p><small>Généré le {stamp}. F1 objet au seuil {CONF} [IC95 bootstrap
par image], banc de test figé, vérité terrain du dataset complet de chaque taxonomie. « ! » = fuite.</small></p>"""]
    for title, grid, ns in tables:
        parts.append(f"<h2>{title}</h2><table><tr><th>Entraînement \\ Évaluation</th>" + "".join(f"<th>{col}</th>" for col, _ in COLS) + "</tr>")
        for row in ROWS:
            parts.append(f"<tr><td class='l'>{row}</td>")
            for col, _ in COLS:
                cell = grid.get((row, col), {})
                txt = (f"<b>{cell['f1']:.3f}</b><br><small>[{cell['ci_lo']:.2f}-{cell['ci_hi']:.2f}]</small>"
                       + (f" <span class='leak'>{cell['mark']}</span>" if cell.get("mark") else "")) if cell else "n/a"
                parts.append(f"<td style='background:{color(cell)}'>{txt}</td>")
            parts.append("</tr>")
        parts.append("<tr><td class='l'><small>objets / images test</small></td>" + "".join(
            f"<td><small>{n['n_objets']} / {n['n_parents']}</small></td>" if n else "<td></td>" for n in ns) + "</tr></table>")
    if per_class:
        parts.append("<h2>Par classe (modèle Full multi)</h2><table><tr><th>Classe</th><th>Objets test</th><th>F1</th><th>AP50</th></tr>"
                     + "".join(f"<tr><td class='l'>{n}</td><td>{k}</td><td>{f:.3f}</td><td>{a:.3f}</td></tr>" for n, k, f, a in per_class) + "</table>")
    if neutral:
        parts.append("<h2>Mono vs mono + Inconnu (métrique neutre)</h2><table><tr><th>Modèle</th>" + "".join(f"<th>{col}</th>" for col, _ in COLS) + "</tr>")
        for label, cells in neutral:
            parts.append(f"<tr><td class='l'>{label}</td>" + "".join(f"<td style='background:{color(x)}'>{_fmt(x)}</td>" for x in cells) + "</tr>")
        parts.append("</table>")
    if evo:
        parts.append("<h2>Évolution depuis le 23/09</h2><table><tr><th>Taxonomie</th><th>Référence 23/09</th><th>Nouvelle campagne</th><th>Objets / images</th></tr>"
                     + "".join(f"<tr><td class='l'>{t}</td><td>{_fmt(o)}</td><td>{_fmt(n)}</td><td>{o.get('n_objets')} / {o.get('n_parents')}</td></tr>" for t, o, n in evo) + "</table>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Campagne complète PixelOdyssey (datasets, entraînements, évaluations, grand tableau).")
    ap.add_argument("--nom", default=f"campagne_{datetime.now().strftime('%Y%m%d')}")
    ap.add_argument("--etapes", default=",".join(STEPS), help=f"Sous-ensemble de {','.join(STEPS)}.")
    ap.add_argument("--modeles", default=",".join(m[0] for m in MODELS), help="Sous-ensemble des clés de modèles.")
    ap.add_argument("--tableau-seul", action="store_true", help="Reconstruit seulement le grand tableau.")
    ap.add_argument("--refaire", action="store_true", help="Refait les entraînements/évaluations déjà faits.")
    ap.add_argument("--oui", action="store_true", help="Pas de confirmation au lancement.")
    args = ap.parse_args()

    keys = [k.strip() for k in args.modeles.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODEL_BY_KEY]
    if unknown:
        sys.exit(f"Modèle(s) inconnu(s) : {unknown}. Choix : {', '.join(MODEL_BY_KEY)}")
    steps = ["tableau"] if args.tableau_seul else [s.strip() for s in args.etapes.split(",")]
    c = Campaign(args.nom)

    todo = [k for k in keys if args.refaire or not (c.model(k).get("run_dir") and Path(c.model(k)["run_dir"], "weights", "best.pt").exists())]
    hours = sum(MODEL_BY_KEY[k][6] for k in todo) if "train" in steps else 0
    print(f"Campagne '{c.name}' -> {c.dir}\nÉtapes : {', '.join(steps)}\nModèles : {', '.join(keys)}"
          f"\nÀ entraîner : {', '.join(todo) if 'train' in steps and todo else 'aucun'}"
          f"\nDurée indicative : ~{hours + (1.5 if 'data' in steps else 0) + (0.1 * len(keys) if 'eval' in steps else 0):.0f} h")
    if not args.oui and steps != ["tableau"]:
        if input("Lancer ? Les datasets dont la configuration a changé seront reconstruits. [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
            sys.exit("Annulé.")
    c.log(f"Début - étapes {steps}, modèles {keys}")
    t0 = time.time()
    try:
        if "verifs" in steps:
            step_checks(c, keys)
        if "data" in steps:
            step_data(c, keys)
        if "train" in steps:
            step_train(c, keys, args.refaire)
        if "eval" in steps:
            step_eval(c, keys, args.refaire)
        if "tableau" in steps:
            out = step_table(c)
            print(f"\n✅ Grand tableau : {out}")
    finally:
        c.log(f"Fin - {(time.time() - t0) / 3600:.1f} h")
        c.save()


if __name__ == "__main__":
    main()
