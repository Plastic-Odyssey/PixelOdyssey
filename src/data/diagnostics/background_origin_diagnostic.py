#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Diagnostic : d'où viennent les tuiles de fond (background) ?

Le pourcentage global de tuiles vides (mesuré par `tile_density_diagnostic.py`,
colonne "% vides") masque DEUX mécanismes complètement différents dans
`slicer.py`, qui n'appellent pas le même levier de correction :

1. Une image PARENTE entièrement sans déchet (aucune annotation du tout) ->
   TOUTES ses tuiles sont gardées, sans aucun sous-échantillonnage.
2. Une image parente qui contient PAR AILLEURS des objets -> une tuile vide
   sur 10 seulement est gardée (compteur roulant par image parente).

Si le taux de fond d'un split est dominé par (1), le vrai levier est de
sous-échantillonner les images parentes 100% fond (aucun garde-fou actuel
dessus - un long transect entièrement sans déchet peut à lui seul produire
des dizaines de tuiles). Si c'est (2) qui domine, le vrai levier est de
baisser le ratio 1/10. Deviner sans mesurer risque de tirer sur le mauvais
mécanisme - d'où ce diagnostic.

Note de fidélité : la détection "parent 100% fond" ci-dessous réutilise
`PlasticImageSlicer._load_yolo_labels()` (avec img_w=img_h=1, les coordonnées
YOLO étant déjà normalisées - un polygone dégénéré/invalide reste dégénéré/
invalide à n'importe quelle échelle) plutôt que de réimplémenter séparément
la même logique de validité de polygone - même risque de divergence
silencieuse entre deux copies que pour `VALID_IMG_EXTS` (partagé depuis
raw_dataset.py plutôt que redéfini ici, pour la même raison).

Entrée : `3_augmented_dataset/{images,labels}/<split>/` (niveau image parente,
pour savoir si CHAQUE parent est 100% fond) + `4_sliced_dataset/{images,labels}/<split>/`
(niveau tuile, pour compter et rattacher chaque tuile à son parent).
Sortie : tableau console par split, + option --out-csv pour le détail par parent.

Exemple :
    python -m src.data.background_origin_diagnostic
    python -m src.data.background_origin_diagnostic --out-csv origine_fond.csv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.data.utils.augment_dataset import AUGMENTED_DIR
from src.data.utils.raw_dataset import VALID_IMG_EXTS
from src.data.utils.slice_dataset import SLICED_DIR, SPLITS
from src.data.utils.slicer import PlasticImageSlicer

_WINDOW_SUFFIX_RE = re.compile(r"^\d+_\d+$")


def _parent_is_all_background(label_path: Path, slicer: PlasticImageSlicer) -> bool:
    """Réplique exactement `parent_is_all_background` de slicer.py::slice_single_pair
    (img_w=img_h=1 : coordonnées YOLO déjà normalisées, la validité/aire d'un
    polygone ne change pas de signe selon l'échelle)."""
    polygons = slicer._load_yolo_labels(label_path, img_w=1, img_h=1)
    return not polygons


def _match_parent(tile_stem: str, parent_ids_by_len: List[str]) -> Optional[str]:
    """Retrouve le parent_id d'une tuile à partir de son nom de fichier.
    `slice_dataset.py` passe `prefix=parent_id` tel quel à `slicer.slice_single_pair` ;
    `slicer.py` écrit soit `{prefix}.png` (image déjà <= tile_size), soit
    `{prefix}_{x_start}_{y_start}.png` (fenêtre glissante) - donc le nom de tuile
    est TOUJOURS soit un parent_id exact, soit `parent_id_<int>_<int>`. On teste les
    parent_id du plus long au plus court pour ne jamais se faire piéger par un
    parent_id qui serait lui-même préfixe d'un autre."""
    if tile_stem in parent_ids_by_len:
        return tile_stem
    for parent_id in parent_ids_by_len:
        if tile_stem.startswith(parent_id + "_"):
            remainder = tile_stem[len(parent_id) + 1:]
            if _WINDOW_SUFFIX_RE.match(remainder):
                return parent_id
    return None


def run_diagnostic(
    augmented_dir: str = AUGMENTED_DIR,
    sliced_dir: str = SLICED_DIR,
    out_csv: Optional[str] = None,
) -> Dict:
    augmented_dir_p = Path(augmented_dir)
    sliced_dir_p = Path(sliced_dir)
    slicer = PlasticImageSlicer()  # paramètres géométriques indifférents ici, seul _load_yolo_labels est utilisé

    csv_rows = []
    summary: Dict[str, Dict] = {}

    print(f"--- 🧭 ORIGINE DES TUILES DE FOND ({sliced_dir_p}) ---\n")
    header = f"{'Split':<8}{'Parents':>9}{'Parents bg':>12}{'Tuiles':>10}{'Tuiles(bg-parent)':>19}{'Tuiles(1/10)':>13}{'Tuiles obj.':>12}{'% fond total':>13}"
    print(header)

    for split in SPLITS:
        img_dir = augmented_dir_p / "images" / split
        lab_dir = augmented_dir_p / "labels" / split
        sliced_img_dir = sliced_dir_p / "images" / split
        sliced_lab_dir = sliced_dir_p / "labels" / split

        if not img_dir.exists() or not sliced_img_dir.exists():
            print(f"{split:<8} (dossier(s) manquant(s), split ignoré)")
            continue

        parent_ids = sorted(
            (p.stem for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS),
            key=len, reverse=True,
        )
        parent_bg: Dict[str, bool] = {
            pid: _parent_is_all_background(lab_dir / f"{pid}.txt", slicer) for pid in parent_ids
        }
        n_parents_bg = sum(parent_bg.values())

        n_tiles_bg_parent = 0       # tuiles issues d'un parent 100% fond (règle 1, jamais throttlée)
        n_tiles_partial_empty = 0   # tuiles vides issues d'un parent par ailleurs annoté (règle 1/10)
        n_tiles_partial_object = 0  # tuiles avec objet(s)
        n_unmatched = 0

        for tile_path in sliced_img_dir.glob("*.png"):
            label_path = sliced_lab_dir / f"{tile_path.stem}.txt"
            n_objects = 0
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    n_objects = sum(1 for line in f if line.strip())

            parent_id = _match_parent(tile_path.stem, parent_ids)
            if parent_id is None:
                n_unmatched += 1
                continue

            if n_objects > 0:
                n_tiles_partial_object += 1
            elif parent_bg[parent_id]:
                n_tiles_bg_parent += 1
            else:
                n_tiles_partial_empty += 1

            if out_csv:
                csv_rows.append({
                    "split": split, "parent_id": parent_id, "tile": tile_path.stem,
                    "parent_all_background": parent_bg[parent_id], "n_objects_tile": n_objects,
                })

        n_tiles_total = n_tiles_bg_parent + n_tiles_partial_empty + n_tiles_partial_object
        pct_bg_total = 100.0 * (n_tiles_bg_parent + n_tiles_partial_empty) / n_tiles_total if n_tiles_total else 0.0
        summary[split] = {
            "n_parents": len(parent_ids), "n_parents_bg": n_parents_bg,
            "n_tiles_total": n_tiles_total, "n_tiles_bg_parent": n_tiles_bg_parent,
            "n_tiles_partial_empty": n_tiles_partial_empty, "n_tiles_partial_object": n_tiles_partial_object,
            "pct_background_total": pct_bg_total, "n_unmatched": n_unmatched,
        }
        print(
            f"{split:<8}{len(parent_ids):>9}{n_parents_bg:>12}{n_tiles_total:>10}"
            f"{n_tiles_bg_parent:>19}{n_tiles_partial_empty:>13}{n_tiles_partial_object:>12}{pct_bg_total:>12.1f}%"
        )
        if n_unmatched:
            print(f"    ⚠️  {n_unmatched} tuile(s) non rattachée(s) à un parent connu - à investiguer avant de faire confiance aux chiffres ci-dessus.")

    print(
        "\nLecture : 'Tuiles(bg-parent)' = tuiles issues d'une image 100% sans déchet (règle NON "
        "throttlée - toutes ses tuiles sont gardées). 'Tuiles(1/10)' = tuiles vides issues d'une "
        "image par ailleurs annotée (règle 1 tuile vide sur 10, throttlée). C'est cette répartition, "
        "pas le seul % global, qui dit quel levier de slicer.py a du sens à ajuster."
    )

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "parent_id", "tile", "parent_all_background", "n_objects_tile"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[SUCCÈS] Détail par tuile exporté : {out_csv}")

    return {"summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic d'origine des tuiles de fond PixelOdyssey")
    parser.add_argument("--augmented-dir", default=AUGMENTED_DIR, help=f"Dossier 3_augmented_dataset (défaut : {AUGMENTED_DIR}).")
    parser.add_argument("--sliced-dir", default=SLICED_DIR, help=f"Dossier 4_sliced_dataset (défaut : {SLICED_DIR}).")
    parser.add_argument("--out-csv", default=None, help="Chemin optionnel d'un .csv listant le détail par tuile.")
    args = parser.parse_args()
    run_diagnostic(augmented_dir=args.augmented_dir, sliced_dir=args.sliced_dir, out_csv=args.out_csv)
