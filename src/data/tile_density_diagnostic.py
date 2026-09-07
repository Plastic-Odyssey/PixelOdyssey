#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Diagnostic de densité : nombre d'objets par tuile, par split.

Complète `dataset_diagnostic.py` (qui travaille au niveau IMAGE PARENTE, sur
1_annotated_dataset) en descendant au niveau TUILE, sur `4_sliced_dataset` -
l'unité réellement vue par le modèle à chaque itération d'entraînement.

Pourquoi ce diagnostic, au-delà de la simple curiosité : la question de fond
qu'il permet de vérifier est celle de la REPRÉSENTATIVITÉ du split - train,
val et test ont-ils une distribution de densité comparable (même proportion
de tuiles vides, même proportion de tuiles très chargées), ou l'un des trois
est-il structurellement différent (ex: test surexposé à des tuiles denses
inhabituelles, ce qui biaiserait sa mesure) ? C'est la même préoccupation que
les 2 feuilles "Par split"/"Répartition classes x split" déjà ajoutées à
`dataset_diagnostic.py` (voir ce fichier), mais au niveau de la DENSITÉ
(nombre d'objets par tuile) plutôt que du DÉSÉQUILIBRE DE CLASSE - deux
questions différentes, un déséquilibre de classe correct peut coexister avec
une distribution de densité très différente entre splits.

Lien direct avec le chantier `overlap_mask` (voir journal 28/08/2026) :
c'est précisément dans les tuiles à forte densité (queue de distribution à
droite de cet histogramme) qu'`overlap_mask=True` (défaut Ultralytics) fusionne
les masques qui se chevauchent et efface les petits objets recouverts - ce
diagnostic chiffre combien de tuiles sont réellement concernées par ce
mécanisme, plutôt que de le supposer.

Une tuile écrit TOUJOURS un fichier de label (voir slicer.py,
`f.writelines(tile_labels)` même si `tile_labels` est vide) - une tuile de
fond a donc un .txt de 0 ligne, jamais un .txt absent. Compter les lignes non
vides de chaque .txt de `4_sliced_dataset/labels/<split>/` donne donc
directement et exactement le nombre d'objets de sa tuile associée.

Entrée : `4_sliced_dataset/{images,labels}/<split>/` (doit déjà exister, donc
le pipeline doit avoir tourné au moins une fois - voir data_pipeline.py).
Sortie : un histogramme .png (3 splits + 1 comparatif superposé) et,
optionnellement, un export .csv de la distribution brute par split.

Exemple :
    python -m src.data.tile_density_diagnostic
    python -m src.data.tile_density_diagnostic --sliced-dir "E:\\PixelOdyssey\\3. Processed dataset\\4_sliced_dataset_mono_class" --cap 15 --out-csv densite.csv
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # sauvegarde en fichier uniquement, jamais d'affichage interactif requis
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.slice_dataset import BASE_DIR, SLICED_DIR, SPLITS  # source partagée, pas de redéfinition locale

OUTPUT_PNG_DEFAULT = os.path.join(BASE_DIR, "tile_density_histogram.png")
DEFAULT_CAP = 20  # au-delà, les tuiles sont regroupées dans un dernier bucket "N+" pour rester lisible

SPLIT_COLORS = {"train": "#3b7dd8", "val": "#e0a72e", "test": "#c04a4a"}


def count_objects_per_tile(sliced_dir: Path, split: str) -> List[int]:
    """Nombre d'objets (lignes non vides du .txt) pour chaque tuile de ce
    split. Une tuile listée dans images/<split> mais sans .txt correspondant
    (ne devrait normalement jamais arriver, voir docstring du module) est
    comptée à 0 plutôt que d'ignorer silencieusement la tuile."""
    img_dir = sliced_dir / "images" / split
    lab_dir = sliced_dir / "labels" / split
    if not img_dir.exists():
        return []

    counts: List[int] = []
    for img_path in sorted(img_dir.glob("*.png")):
        label_path = lab_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            counts.append(0)
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        counts.append(n)
    return counts


def _bucket(counts: List[int], cap: int) -> Counter:
    """Regroupe tout compte > cap dans un seul bucket `cap + 1` (affiché
    'cap+') - sans ça, une poignée de tuiles extrêmes (ex: un amas de
    Debris_Divers) étire l'axe X et rend le reste du graphe illisible."""
    bucketed: Counter = Counter()
    for n in counts:
        bucketed[min(n, cap + 1)] += 1
    return bucketed


def _bucket_labels(cap: int) -> List[str]:
    return [str(i) for i in range(cap + 1)] + [f"{cap}+"]


def run_diagnostic(
    sliced_dir: str = SLICED_DIR,
    cap: int = DEFAULT_CAP,
    output_png: str = OUTPUT_PNG_DEFAULT,
    output_csv: str = None,
) -> Dict:
    sliced_dir_p = Path(sliced_dir)
    if not sliced_dir_p.exists():
        raise RuntimeError(
            f"Dossier introuvable : {sliced_dir_p}. Lance d'abord le pipeline complet "
            f"(python -m src.data.data_pipeline) ou au moins l'étape de slicing "
            f"(python -m src.data.slice_dataset)."
        )

    raw_counts: Dict[str, List[int]] = {}
    for split in SPLITS:
        counts = count_objects_per_tile(sliced_dir_p, split)
        if not counts:
            print(f"⚠️  Split [{split}] : aucune tuile trouvée sous {sliced_dir_p / 'images' / split}.")
        raw_counts[split] = counts

    if not any(raw_counts.values()):
        raise RuntimeError(f"Aucune tuile trouvée sous {sliced_dir_p} pour aucun split. Rien à diagnostiquer.")

    print(f"--- 📊 DENSITÉ D'OBJETS PAR TUILE ({sliced_dir_p}) ---\n")
    print(f"{'Split':<8}{'N tuiles':>10}{'N objets':>12}{'Moyenne':>10}{'Médiane':>10}{'% vides':>10}{'Max':>8}")

    summary: Dict[str, Dict] = {}
    for split in SPLITS:
        counts = raw_counts[split]
        n_tiles = len(counts)
        if n_tiles == 0:
            summary[split] = {"n_tiles": 0}
            continue
        sorted_counts = sorted(counts)
        mean = sum(counts) / n_tiles
        median = sorted_counts[n_tiles // 2] if n_tiles % 2 else (sorted_counts[n_tiles // 2 - 1] + sorted_counts[n_tiles // 2]) / 2
        pct_empty = 100.0 * sum(1 for c in counts if c == 0) / n_tiles
        summary[split] = {
            "n_tiles": n_tiles,
            "n_objects": sum(counts),
            "mean": mean,
            "median": median,
            "pct_empty": pct_empty,
            "max": max(counts),
        }
        print(
            f"{split:<8}{n_tiles:>10}{sum(counts):>12}{mean:>10.2f}{median:>10.1f}{pct_empty:>9.1f}%{max(counts):>8}"
        )

    labels = _bucket_labels(cap)
    x = list(range(len(labels)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("PixelOdyssey - Nombre d'objets par tuile, par split", fontsize=13, fontweight="bold")

    split_positions = {"train": (0, 0), "val": (0, 1), "test": (1, 0)}
    for split, pos in split_positions.items():
        ax = axes[pos]
        counts = raw_counts[split]
        if not counts:
            ax.set_title(f"{split} (aucune tuile)")
            ax.axis("off")
            continue
        bucketed = _bucket(counts, cap)
        n_tiles = len(counts)
        heights = [100.0 * bucketed.get(i, 0) / n_tiles for i in x]
        ax.bar(x, heights, color=SPLIT_COLORS[split])
        ax.set_title(f"{split}  (n={n_tiles} tuiles, {summary[split]['pct_empty']:.1f}% vides)")
        ax.set_xlabel("Nombre d'objets sur la tuile")
        ax.set_ylabel("% des tuiles du split")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90 if cap > 15 else 0, fontsize=7)

    ax_overlay = axes[1, 1]
    for split in SPLITS:
        counts = raw_counts[split]
        if not counts:
            continue
        bucketed = _bucket(counts, cap)
        n_tiles = len(counts)
        heights = [100.0 * bucketed.get(i, 0) / n_tiles for i in x]
        ax_overlay.plot(x, heights, marker="o", markersize=3, label=split, color=SPLIT_COLORS[split])
    ax_overlay.set_title("Comparatif superposé (% normalisé par split)")
    ax_overlay.set_xlabel("Nombre d'objets sur la tuile")
    ax_overlay.set_ylabel("% des tuiles du split")
    ax_overlay.set_xticks(x)
    ax_overlay.set_xticklabels(labels, rotation=90 if cap > 15 else 0, fontsize=7)
    ax_overlay.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"\n[SUCCÈS] Histogramme écrit : {output_png}")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "n_objets_sur_tuile", "n_tuiles", "pct_tuiles_du_split"])
            for split in SPLITS:
                counts = raw_counts[split]
                if not counts:
                    continue
                n_tiles = len(counts)
                full_counter = Counter(counts)  # non bucketé - distribution brute complète
                for n_objects in sorted(full_counter):
                    writer.writerow([split, n_objects, full_counter[n_objects], round(100.0 * full_counter[n_objects] / n_tiles, 3)])
        print(f"[SUCCÈS] Distribution brute (non bucketée) exportée : {output_csv}")

    return {"summary": summary, "output_png": str(output_png), "output_csv": str(output_csv) if output_csv else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic de densité (objets par tuile) PixelOdyssey")
    parser.add_argument(
        "--sliced-dir", default=SLICED_DIR,
        help=f"Dossier 4_sliced_dataset à diagnostiquer (défaut : {SLICED_DIR}). "
             f"Ex: .../4_sliced_dataset_mono_class pour la variante mono-classe.",
    )
    parser.add_argument(
        "--cap", type=int, default=DEFAULT_CAP,
        help=f"Nombre d'objets au-delà duquel les tuiles sont regroupées dans un dernier "
             f"bucket 'N+' pour garder l'histogramme lisible (défaut {DEFAULT_CAP}).",
    )
    parser.add_argument(
        "--out", default=OUTPUT_PNG_DEFAULT,
        help=f"Chemin du .png de sortie (défaut : {OUTPUT_PNG_DEFAULT}).",
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="Chemin optionnel d'un .csv listant la distribution brute (non bucketée) "
             "par split, pour analyse plus fine dans un tableur.",
    )
    args = parser.parse_args()
    try:
        run_diagnostic(sliced_dir=args.sliced_dir, cap=args.cap, output_png=args.out, output_csv=args.out_csv)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
