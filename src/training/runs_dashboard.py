#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Tableau de bord comparatif de TOUS les runs d'entraînement.

Complète `compare_runs.py` plutôt que de le remplacer : celui-ci reste l'outil
de référence pour comparer 2-3 runs en détail dans le terminal (diff explicite
des hyperparamètres/du fingerprint de donnée entre UN run de référence et
chaque autre). Mais avec 16 runs et plus dans `output/runs/`, un tableau
terminal à 16 colonnes de 32 caractères devient illisible - ce script produit
à la place une vue d'ensemble scannable (CSV + page HTML triable) où CHAQUE
run est une ligne, avec ses paramètres ET ses performances côte à côte.

RÉUTILISE `rapport_metrics.json` (généré par training_report.py) pour les
métriques - jamais recalculé ici. Un run sans ce fichier est listé à part
avec la commande pour le générer, jamais silencieusement ignoré.

POINT IMPORTANT, vérifié dans le code avant d'écrire ce script (pas supposé) :
`training_report.py::_HYPERPARAM_KEYS` (la liste copiée dans le "hyperparams"
de rapport_metrics.json) ne contient PAS `cls_pw`, `mixup`, `overlap_mask` ni
`copy_paste_mode` - alors que ce sont exactement les 4 leviers testés dans les
ablations de la semaine du 28/08 (cls_pw05, mixup01, overlap_mask_off,
copy_paste_flip05). Ce script relit donc `args.yaml` DIRECTEMENT (liste de
clés élargie ci-dessous, BROAD_HYPERPARAM_KEYS) plutôt que de se fier au
sous-ensemble déjà curaté de rapport_metrics.json, pour ne pas reproduire ce
trou - args.yaml (écrit par Ultralytics) contient TOUJOURS la valeur réelle
de ces 4 paramètres, même sur les runs où rapport_metrics.json ne les liste
pas.

Entrée : `output/runs/*/` (best.pt + args.yaml + rapport_metrics.json).
Sortie : `output/runs_dashboard.html` (page triable, un coup d'oeil sur tout)
         + `output/runs_dashboard.csv` (format large, un run+split par ligne,
           pour analyse dans un tableur).

Usage :
    python -m src.training.runs_dashboard
    python -m src.training.runs_dashboard --runs-dir output/runs --out-html output/runs_dashboard.html
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "output" / "runs"
DEFAULT_OUT_HTML = PROJECT_ROOT / "output" / "runs_dashboard.html"
DEFAULT_OUT_CSV = PROJECT_ROOT / "output" / "runs_dashboard.csv"

# Liste ÉLARGIE par rapport à training_report._HYPERPARAM_KEYS (voir docstring
# ci-dessus) - inclut les 4 leviers d'ablation de la semaine du 28/08 qui
# manquent dans le sous-ensemble curaté de rapport_metrics.json. Lu
# directement depuis args.yaml, qui les contient toujours (écrit par
# Ultralytics avec TOUS les hyperparamètres résolus du run, pas seulement
# ceux que training_report.py a choisi de recopier).
BROAD_HYPERPARAM_KEYS = [
    "model", "epochs", "patience", "imgsz", "batch", "seed", "deterministic",
    "degrees", "flipud", "fliplr", "scale",
    "mosaic", "mixup", "copy_paste", "copy_paste_mode", "cls_pw", "overlap_mask",
]

# Colonnes affichées dans le tableau HTML (sous-ensemble de BROAD_HYPERPARAM_KEYS
# assez court pour rester lisible - les leviers d'ablation d'abord, ceux qui ne
# varient jamais entre runs de ce projet en dernier).
DISPLAY_HYPERPARAM_KEYS = [
    "model", "epochs_done", "patience", "batch",
    "degrees", "flipud", "scale",
    "copy_paste", "copy_paste_mode", "cls_pw", "mixup", "overlap_mask",
]


def _read_args_yaml(run_dir: Path) -> Dict:
    """Lit args.yaml en entier (best-effort) et ne garde que
    BROAD_HYPERPARAM_KEYS - jamais supposé, jamais recalculé."""
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return {}
    try:
        import yaml
        with open(args_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {k: data[k] for k in BROAD_HYPERPARAM_KEYS if k in data}
    except Exception:  # noqa: BLE001 - annexe optionnelle, ne doit jamais faire planter le tableau
        return {}


def _read_epochs_done(run_dir: Path) -> Optional[int]:
    """Epoch RÉELLEMENT atteinte (results.csv, dernière ligne) - distincte de
    args.yaml['epochs'] (le maximum demandé) car `patience` peut arrêter un run
    avant terme, ou un run peut avoir été interrompu manuellement (voir
    ablation_no_copy_paste_...015543, interrompu quasiment au lancement)."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None
    try:
        with open(results_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and rows[-1].get("epoch"):
            return int(float(rows[-1]["epoch"]))
    except Exception:  # noqa: BLE001 - idem, purement informatif
        pass
    return None


def discover_runs(runs_dir: Path) -> Dict[str, List[Path]]:
    """Partitionne les sous-dossiers de runs_dir (ceux qui ont un weights/best.pt,
    donc un vrai run entraîné) en 'avec rapport_metrics.json' / 'sans' - jamais
    un run entraîné ignoré silencieusement faute de rapport."""
    with_report: List[Path] = []
    without_report: List[Path] = []
    if not runs_dir.exists():
        return {"with_report": [], "without_report": []}
    for run_dir in sorted(runs_dir.iterdir()):
        if not (run_dir / "weights" / "best.pt").exists():
            continue
        if (run_dir / "rapport_metrics.json").exists():
            with_report.append(run_dir)
        else:
            without_report.append(run_dir)
    return {"with_report": with_report, "without_report": without_report}


def build_run_record(run_dir: Path) -> Optional[Dict]:
    metrics_path = run_dir / "rapport_metrics.json"
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:  # noqa: BLE001 - un run corrompu ne doit pas faire planter les autres
        print(f"  ⚠️  {metrics_path} illisible ({e}) - run ignoré.")
        return None

    hyperparams = _read_args_yaml(run_dir)
    hyperparams["epochs_done"] = _read_epochs_done(run_dir)

    return {
        "run_name": metrics.get("run_name", run_dir.name),
        "run_dir": str(run_dir),
        "hyperparams": hyperparams,
        "fingerprint": metrics.get("sliced_data_fingerprint"),
        "splits": metrics.get("splits", {}),
    }


def _assign_fingerprint_groups(records: List[Dict]) -> Dict[Optional[str], str]:
    """Une lettre courte par fingerprint distinct rencontré (A, B, C...) - pour
    repérer d'un coup d'oeil quels runs partagent EXACTEMENT la même donnée
    d'entraînement (seule base de comparaison de performance vraiment valide,
    voir compare_runs.py) sans avoir à comparer les hachages en entier."""
    seen: Dict[Optional[str], str] = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for r in records:
        fp = r["fingerprint"]
        if fp not in seen:
            seen[fp] = letters[len(seen)] if len(seen) < len(letters) else f"#{len(seen)}"
    return seen


def _all_class_names(records: List[Dict], split: str) -> List[str]:
    """Union ordonnée des classes rencontrées dans ce split, toutes runs
    confondues (une classe absente d'un run donné - ex: taxonomie différente
    comme no_debris_ref - s'affiche comme n/a pour ce run, jamais en erreur)."""
    seen: List[str] = []
    for r in records:
        split_data = r["splits"].get(split)
        if not split_data:
            continue
        for row in split_data.get("per_class", []):
            if row["class_name"] not in seen:
                seen.append(row["class_name"])
    return seen


def _fmt_pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _fmt_raw(x) -> str:
    return "—" if x is None else str(x)


def write_csv(records: List[Dict], out_path: Path) -> None:
    """Format LARGE : une ligne par (run, split), hyperparamètres +
    métriques agrégées + une colonne par classe (P/R/F1/n) - prêt pour un
    tableau croisé dynamique dans un tableur."""
    class_names = sorted({
        row["class_name"]
        for r in records
        for split_data in r["splits"].values()
        for row in split_data.get("per_class", [])
    })

    fieldnames = (
        ["run_name", "split", "fingerprint"]
        + DISPLAY_HYPERPARAM_KEYS
        + ["macro_p", "macro_r", "macro_f1", "map50", "map50_95",
           "global_p", "global_r", "global_f1"]
    )
    for c in class_names:
        fieldnames += [f"{c}_P", f"{c}_R", f"{c}_F1", f"{c}_n"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            for split, split_data in r["splits"].items():
                row_out = {
                    "run_name": r["run_name"], "split": split,
                    "fingerprint": r["fingerprint"] or "",
                }
                for k in DISPLAY_HYPERPARAM_KEYS:
                    row_out[k] = r["hyperparams"].get(k, "")
                macro = split_data.get("macro", {})
                glob = split_data.get("global", {})
                row_out.update({
                    "macro_p": macro.get("precision"), "macro_r": macro.get("recall"),
                    "macro_f1": macro.get("f1"),
                    "map50": macro.get("map50"), "map50_95": macro.get("map50_95"),
                    "global_p": glob.get("global_precision"), "global_r": glob.get("global_recall"),
                    "global_f1": glob.get("global_f1"),
                })
                per_class = {row["class_name"]: row for row in split_data.get("per_class", [])}
                for c in class_names:
                    row = per_class.get(c)
                    if row and row.get("instances"):
                        row_out[f"{c}_P"] = row.get("mask_p")
                        row_out[f"{c}_R"] = row.get("mask_r")
                        row_out[f"{c}_F1"] = row.get("mask_f1")
                        row_out[f"{c}_n"] = row.get("instances")
                    else:
                        row_out[f"{c}_P"] = row_out[f"{c}_R"] = row_out[f"{c}_F1"] = row_out[f"{c}_n"] = ""
                writer.writerow(row_out)

    print(f"📄 CSV (format large, un run+split par ligne) : {out_path}")


_HTML_TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>PixelOdyssey — tableau de bord des runs</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 2rem; font-size: 0.85rem; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.05rem; margin: 2rem 0 0.5rem; }}
  .meta {{ color: #888; font-size: 0.8rem; margin-bottom: 1.5rem; }}
  .warn {{ background: #a8532c22; border: 1px solid #a8532c88; border-radius: 6px; padding: 0.7rem 1rem; margin-bottom: 1.5rem; }}
  .warn code {{ font-size: 0.82rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }}
  th, td {{ border: 1px solid #8884; padding: 0.35rem 0.55rem; text-align: right; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; position: sticky; left: 0; background: Canvas; }}
  th {{ cursor: pointer; user-select: none; background: #8882; position: sticky; top: 0; }}
  th:hover {{ background: #8884; }}
  th.sorted-desc::after {{ content: " ▼"; }}
  th.sorted-asc::after {{ content: " ▲"; }}
  tbody tr:hover {{ background: #8881; }}
  .fp-badge {{ display: inline-block; width: 1.3em; height: 1.3em; border-radius: 50%; background: #4a7; color: white;
               text-align: center; line-height: 1.3em; font-size: 0.75em; font-weight: 600; }}
  .table-wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>PixelOdyssey — tableau de bord des runs</h1>
<p class="meta">{n_runs} run(s) avec rapport · généré le {generated_at} · clique un en-tête de colonne pour trier ·
   le badge coloré = groupe de fingerprint (même lettre = MÊME donnée d'entraînement, seule base de comparaison
   de performance vraiment valide — voir <code>compare_runs.py</code> pour un diff détaillé entre deux runs précis).</p>
{warning_block}
{sections}
<script>
function sortTable(table, colIdx, numeric) {{
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const th = table.tHead.rows[0].cells[colIdx];
  const asc = !th.classList.contains('sorted-asc');
  Array.from(table.tHead.rows[0].cells).forEach(c => c.classList.remove('sorted-asc', 'sorted-desc'));
  th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
  rows.sort((a, b) => {{
    let va = a.cells[colIdx].dataset.sort, vb = b.cells[colIdx].dataset.sort;
    if (numeric) {{ va = parseFloat(va); vb = parseFloat(vb);
      if (isNaN(va)) va = -Infinity; if (isNaN(vb)) vb = -Infinity; }}
    if (va < vb) return asc ? -1 : 1;
    if (va > vb) return asc ? 1 : -1;
    return 0;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
document.querySelectorAll('table').forEach(table => {{
  Array.from(table.tHead.rows[0].cells).forEach((th, idx) => {{
    th.addEventListener('click', () => sortTable(table, idx, th.dataset.numeric === '1'));
  }});
}});
</script>
</body>
</html>
"""


def _render_section(split: str, records: List[Dict], fp_groups: Dict[Optional[str], str]) -> str:
    class_names = _all_class_names(records, split)

    headers = ["Run", "Fingerprint"] + DISPLAY_HYPERPARAM_KEYS + [
        "P macro", "R macro", "F1 macro", "mAP50", "mAP50-95",
        "P global", "R global", "F1 global",
    ] + [f"Rappel {c}" for c in class_names]
    numeric_flags = [0, 0] + [0] * len(DISPLAY_HYPERPARAM_KEYS) + [1] * 8 + [1] * len(class_names)

    body_rows = []
    for r in records:
        split_data = r["splits"].get(split)
        if not split_data:
            continue
        macro = split_data.get("macro", {})
        glob = split_data.get("global", {})
        per_class = {row["class_name"]: row for row in split_data.get("per_class", [])}

        cells = []
        cells.append((r["run_name"], r["run_name"]))
        fp_letter = fp_groups.get(r["fingerprint"], "?")
        cells.append((f'<span class="fp-badge" title="{r["fingerprint"] or "inconnu"}">{fp_letter}</span>', fp_letter))
        for k in DISPLAY_HYPERPARAM_KEYS:
            v = r["hyperparams"].get(k)
            cells.append((_fmt_raw(v), _fmt_raw(v)))
        cells.append((_fmt_pct(macro.get("precision")), macro.get("precision")))
        cells.append((_fmt_pct(macro.get("recall")), macro.get("recall")))
        cells.append((_fmt_pct(macro.get("f1")), macro.get("f1")))
        cells.append((_fmt_pct(macro.get("map50")), macro.get("map50")))
        cells.append((_fmt_pct(macro.get("map50_95")), macro.get("map50_95")))
        cells.append((_fmt_pct(glob.get("global_precision")), glob.get("global_precision")))
        cells.append((_fmt_pct(glob.get("global_recall")), glob.get("global_recall")))
        cells.append((_fmt_pct(glob.get("global_f1")), glob.get("global_f1")))
        for c in class_names:
            row = per_class.get(c)
            if row and row.get("instances"):
                val = row.get("mask_r")
                cells.append((f'{_fmt_pct(val)} (n={row["instances"]})', val))
            else:
                cells.append(("n/a", ""))

        tds = "".join(f'<td data-sort="{sort_val if sort_val is not None else ""}">{disp}</td>' for disp, sort_val in cells)
        body_rows.append(f"<tr>{tds}</tr>")

    ths = "".join(
        f'<th data-numeric="{n}">{h}</th>' for h, n in zip(headers, numeric_flags)
    )
    return (
        f"<h2>Split {split.upper()}</h2>"
        f'<div class="table-wrap"><table><thead><tr>{ths}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'
    )


def write_html(records: List[Dict], missing: List[Path], out_path: Path) -> None:
    from datetime import datetime

    fp_groups = _assign_fingerprint_groups(records)

    warning_block = ""
    if missing:
        cmds = "".join(
            f'<li><code>python -m src.training.training_report --run "{m}"</code></li>' for m in missing
        )
        warning_block = (
            f'<div class="warn">⚠️ {len(missing)} run(s) entraîné(s) SANS rapport_metrics.json '
            f"(absents du tableau ci-dessous) - régénère leur rapport pour les inclure :"
            f"<ul>{cmds}</ul></div>"
        )

    sections = []
    for split in ("test", "val"):
        if not any(r["splits"].get(split) for r in records):
            continue  # aucun run n'a ce split évalué - ne pas afficher un tableau vide
        sections.append(_render_section(split, records, fp_groups))

    html = _HTML_TEMPLATE.format(
        n_runs=len(records),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        warning_block=warning_block,
        sections="".join(sections),
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"🖥️  Tableau de bord HTML : {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tableau de bord comparatif de tous les runs PixelOdyssey.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--out-html", default=str(DEFAULT_OUT_HTML))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    found = discover_runs(runs_dir)
    print(f"--- 📊 {len(found['with_report'])} run(s) avec rapport_metrics.json, "
          f"{len(found['without_report'])} sans (sous {runs_dir}) ---")

    records = []
    for run_dir in found["with_report"]:
        rec = build_run_record(run_dir)
        if rec is not None:
            records.append(rec)

    if not records:
        print("❌ Aucun run exploitable trouvé - rien à écrire.")
        return

    write_csv(records, Path(args.out_csv))
    write_html(records, found["without_report"], Path(args.out_html))


if __name__ == "__main__":
    main()
