#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Comparaison de plusieurs runs d'entraînement.

Lit le `rapport_metrics.json` de chaque run donné (généré par
training_report.py - à régénérer d'abord si absent) et affiche, pour chaque
split commun (val/test) : les métriques globales/macro puis un tableau par
classe, pour chaque run comparé côte à côte. Signale explicitement si les
hyperparamètres ou la donnée d'entraînement (fingerprint de
4_sliced_dataset) diffèrent entre les runs comparés : une comparaison de
performance n'a de sens que si UNE SEULE chose a changé entre deux runs
(plusieurs changements groupés dans un même run rendent la cause impossible
à isoler).

F1 (macro et global/pondéré, en plus de P et R déjà présents) répond au fait
qu'une comparaison P/R qui bouge en sens opposé (précision qui monte, rappel
qui baisse, ou l'inverse) est difficile à trancher visuellement classe par
classe. F1 macro = MOYENNE DES F1 PAR CLASSE (pas F1 recalculé depuis
précision/rappel macro - ce sont deux quantités différentes, voir
`_macro_f1` dans training_report.py) ; F1 global = moyenne harmonique de
précision/rappel globaux (déjà des taux pondérés par instances, donc pas
d'ambiguïté de calcul à ce niveau). Reste un résumé, pas un remplacement :
un F1 stable peut masquer un vrai changement de point de fonctionnement
(plus de rappel/moins de précision utile si la priorité du projet est
justement de rattraper des ratés) - lire aussi P et R séparément avant de
conclure. Absent des rapports générés avant l'ajout de ce champ (`.get(...)`
partout ci-dessous) - régénérer via `training_report.py --run <dossier>`
pour l'obtenir sur un ancien run.

Entrée : chemins d'au moins 2 dossiers de run, chacun contenant déjà
`rapport_metrics.json`.
Sortie : tableau comparatif affiché dans le terminal ; `--out-csv` optionnel
pour un export par classe/run/split exploitable dans un tableur.

Exemple :
    python -m src.training.compare_runs "E:\PixelOdyssey\6. Model outputs\runs\sans_source_X" "E:\PixelOdyssey\6. Model outputs\runs\avec_source_X"
    python -m src.training.compare_runs "E:\PixelOdyssey\6. Model outputs\runs\run_a" "E:\PixelOdyssey\6. Model outputs\runs\run_b" --out-csv comparaison.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.pipeline_utils import diff_params

NO_DIFF_PLACEHOLDER = ["  (impossible de déterminer le détail - manifest précédent incomplet)"]


def _load_run(run_dir: Path) -> Dict:
    metrics_path = run_dir / "rapport_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} introuvable - régénère d'abord le rapport de ce run :\n"
            f'  python -m src.training.training_report --run "{run_dir}"'
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("run_name", run_dir.name)
    return data


def _fmt(x, pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%" if pct else str(x)


def _print_metadata_diffs(runs: List[Dict]) -> None:
    baseline = runs[0]
    print(f"Référence : {baseline['run_name']}\n")
    for other in runs[1:]:
        print(f"--- {other['run_name']} vs {baseline['run_name']} ---")

        hp_diff = diff_params(baseline.get("hyperparams", {}), other.get("hyperparams", {}))
        if hp_diff and hp_diff != NO_DIFF_PLACEHOLDER:
            print("⚠️  Hyperparamètres différents :")
            print("\n".join(hp_diff))
        else:
            print("✅ Mêmes hyperparamètres (autant qu'on puisse le lire depuis args.yaml).")

        fp_a = baseline.get("sliced_data_fingerprint")
        fp_b = other.get("sliced_data_fingerprint")
        if fp_a is None or fp_b is None:
            print("ℹ️  Fingerprint de donnée indisponible pour au moins un des deux runs - "
                  "impossible de confirmer si la donnée d'entraînement était identique.")
        elif fp_a == fp_b:
            print(f"✅ Même donnée d'entraînement (fingerprint {fp_a}).")
        else:
            print(f"⚠️  Donnée d'entraînement DIFFÉRENTE ({fp_a} -> {fp_b}) - une partie (ou la "
                  f"totalité) du delta de performance ci-dessous peut venir de là, pas seulement "
                  f"d'un changement de code/hyperparamètres.")
        print()


def _class_order(runs: List[Dict], split: str) -> List[str]:
    """Ordre des classes = celui de la première run qui a ce split (les classes
    déclarées dans data_config.yaml sont censées être les mêmes pour tous les
    runs comparés - une classe manquante ailleurs s'affiche comme n/a)."""
    for r in runs:
        split_data = r.get("splits", {}).get(split)
        if split_data:
            return [row["class_name"] for row in split_data["per_class"]]
    return []


def _print_split_comparison(split: str, runs: List[Dict]) -> None:
    print(f"\n=== Split {split.upper()} ===\n")

    col_w = 32  # élargi (P/R/F1 + n= tient plus large que P/R seuls)
    header = f"{'Classe':<20}" + "".join(f"{r['run_name'][:col_w - 2]:>{col_w}}" for r in runs)
    print(header)
    print("-" * len(header))

    for class_name in _class_order(runs, split):
        cells = []
        for r in runs:
            split_data = r.get("splits", {}).get(split)
            row = None
            if split_data:
                row = next((x for x in split_data["per_class"] if x["class_name"] == class_name), None)
            if row is None or row["instances"] == 0:
                cell = "n/a"
            else:
                cell = (
                    f"{_fmt(row['mask_p'])}P/{_fmt(row['mask_r'])}R/"
                    f"{_fmt(row.get('mask_f1'))}F1 (n={row['instances']})"
                )
            cells.append(cell.rjust(col_w))
        print(f"{class_name:<20}" + "".join(cells))

    print()
    aggregate_rows = [
        ("Précision macro", lambda s: s["macro"]["precision"]),
        ("Rappel macro", lambda s: s["macro"]["recall"]),
        ("F1 macro", lambda s: s["macro"].get("f1")),  # .get : absent des rapports générés avant ce champ
        ("mAP50-95 (masque)", lambda s: s["macro"]["map50_95"]),
        ("Précision globale (pondérée)", lambda s: s["global"]["global_precision"]),
        ("Rappel global (pondéré)", lambda s: s["global"]["global_recall"]),
        ("F1 global (pondéré)", lambda s: s["global"].get("global_f1")),
    ]
    for label, getter in aggregate_rows:
        cells = []
        for r in runs:
            split_data = r.get("splits", {}).get(split)
            val = getter(split_data) if split_data else None
            cells.append(_fmt(val).rjust(col_w))
        print(f"{label:<20}" + "".join(cells))


def _write_csv(runs: List[Dict], out_path: Path) -> None:
    fieldnames = ["run_name", "split", "class_name", "instances", "precision", "recall", "f1", "map50", "map50_95"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            for split, split_data in r.get("splits", {}).items():
                for row in split_data["per_class"]:
                    writer.writerow({
                        "run_name": r["run_name"],
                        "split": split,
                        "class_name": row["class_name"],
                        "instances": row["instances"],
                        "precision": row["mask_p"],
                        "recall": row["mask_r"],
                        "f1": row.get("mask_f1"),
                        "map50": row["map50"],
                        "map50_95": row["map50_95"],
                    })
    print(f"\n📄 Export CSV écrit : {out_path}")


def compare_runs(run_dirs: List[str], out_csv: Optional[str] = None) -> None:
    if len(run_dirs) < 2:
        raise ValueError("Il faut au moins 2 runs à comparer.")
    runs = [_load_run(Path(d)) for d in run_dirs]

    print("=" * 70)
    print("COMPARAISON DE RUNS — " + " vs ".join(r["run_name"] for r in runs))
    print("=" * 70 + "\n")

    _print_metadata_diffs(runs)

    common_splits = set(runs[0].get("splits", {}).keys())
    for r in runs[1:]:
        common_splits &= set(r.get("splits", {}).keys())
    if not common_splits:
        print("⚠️  Aucun split commun entre ces runs (val/test) - rien à comparer.")
        return

    for split in ("val", "test"):
        if split in common_splits:
            _print_split_comparison(split, runs)

    if out_csv:
        _write_csv(runs, Path(out_csv))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare les métriques de plusieurs runs PixelOdyssey.")
    parser.add_argument("runs", nargs="+", help="≥2 dossiers de run (ex: \"E:\\PixelOdyssey\\6. Model outputs\\runs\\run_a\" ...)")
    parser.add_argument("--out-csv", default=None, help="Chemin d'un CSV optionnel (une ligne par classe/run/split)")
    args = parser.parse_args()
    try:
        compare_runs(args.runs, out_csv=args.out_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        sys.exit(1)
