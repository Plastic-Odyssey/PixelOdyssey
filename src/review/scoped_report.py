#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Rapport de lecture restreint à un sous-ensemble de lots.

Isole, à l'intérieur d'un split DÉJÀ figé (typiquement `test`), les seules
imagettes appartenant à un ou plusieurs lots donnés (ex : uniquement Santa
Luzia - lots "SL ..."), et génère un rapport de lecture SUR CE SOUS-ENSEMBLE
SEUL, en réutilisant telle quelle la logique de métriques de
training_report.py (aucune duplication de calcul).

Cas d'usage : le modèle est entraîné sur TOUS les sites (SB + SL + A LEG,
pipeline standard inchangé), mais on veut savoir s'il performe différemment
sur Santa Luzia SPÉCIFIQUEMENT, sans le bruit des autres sites noyés dans le
même split de test. Contrairement à un nouveau gel de banc de test (pipeline à relancer,
donnée d'entraînement modifiée), cet outil ne touche à AUCUNE donnée : il
lit le split déjà figé tel quel et restreint seulement l'ÉVALUATION à un
sous-ensemble - zéro impact sur 2_split_dataset, 4_sliced_dataset, et sur
les runs déjà comparés via compare_runs.py sur le banc de test général (le
rapport scopé est écrit sous un nom de fichier distinct, jamais
rapport_lecture.html/rapport_metrics.json - jamais écrasé).

Important : la significativité du sous-ensemble obtenu dépend entièrement de
CE QUI EST DÉJÀ dans le split demandé pour ce filtre de lot - cet outil ne
rééquilibre rien. Un avertissement liste les classes sous
LOW_SAMPLE_WARN_THRESHOLD instances dans le sous-ensemble résultant (même
seuil que assisted_annotate.py) - à lire avant de tirer une conclusion sur
une classe rare.

Mécanique :
    1. Lit `2_split_dataset/.parent_manifest.json` (parent_id -> batch, split).
    2. Filtre les parent_id du split demandé dont le nom de lot contient
       `--batch-filter` (insensible à la casse).
    3. Retrouve les imagettes correspondantes dans
       `4_sliced_dataset/images/<split>/` (préfixe `{parent_id}_` - même
       convention que pipeline_utils.already_present).
    4. Écrit la liste de leurs chemins dans un .txt (format natif Ultralytics
       pour restreindre un split à des fichiers précis plutôt qu'à un
       dossier entier).
    5. Construit un data.yaml temporaire, copie de config/data_config.yaml
       sauf `<split>:` qui pointe vers ce .txt.
    6. Appelle training_report.generate_report() avec ce yaml temporaire et
       un `output_basename` distinct du rapport standard du run.

Entrée : --run (dossier du run), --batch-filter (sous-chaîne sur le nom de
lot, ex: "SL"), --split (défaut: test), --label (nom court utilisé dans le
titre et le nom des fichiers de sortie - déduit de --batch-filter si omis).
Sortie : `<run>/rapport_<label>_lecture.html` + `<run>/rapport_<label>_metrics.json`.

Exemple :
    python -m src.review.scoped_report --run "E:\PixelOdyssey\6. Model outputs\runs\new_ref_vanilla_yolo11n-seg_20260827_020134" \\
        --batch-filter "SL" --label SantaLuzia
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import yaml

from src.data.utils.class_config import DEFAULT_CLASS_CONFIG_PATH, load_class_config
from src.data.utils.raw_dataset import VALID_IMG_EXTS
from src.data.utils.slice_dataset import SLICED_DIR
from src.data.utils.split_dataset import PARENT_MANIFEST_FILENAME, SPLIT_DIR
from src.review.assisted_annotate import LOW_SAMPLE_WARN_THRESHOLD
from src.review.visualize_annotations import _parse_yolo_seg_label
from src.training.training_report import generate_report

DEFAULT_DATA_CONFIG_PATH = DEFAULT_CLASS_CONFIG_PATH  # même fichier, alias pour la lisibilité ici


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", label) or "scope"


def _load_parent_manifest(split_dir: str) -> Dict[str, Dict]:
    manifest_path = Path(split_dir) / PARENT_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise RuntimeError(
            f"Manifeste introuvable : {manifest_path}. Lance d'abord split_dataset.py "
            f"(ou tout le pipeline via data_pipeline.py) pour le générer."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _matching_parent_ids(parent_manifest: Dict[str, Dict], split: str, batch_filter: str) -> List[str]:
    needle = batch_filter.lower()
    return sorted(
        pid for pid, info in parent_manifest.items()
        if info["split"] == split and needle in info["batch"].lower()
    )


def _tiles_for_parents(sliced_dir: str, split: str, parent_ids: List[str]) -> Dict[str, List[Path]]:
    """Retrouve, pour chaque parent_id, ses imagettes dans
    `4_sliced_dataset/images/<split>/`. Un parent_id sans aucune imagette
    trouvée est gardé dans le résultat avec une liste vide (signalé à
    l'appelant, ne bloque pas les autres)."""
    img_dir = Path(sliced_dir) / "images" / split
    if not img_dir.exists():
        raise RuntimeError(f"Dossier introuvable : {img_dir}. Le dataset a-t-il été tuilé (data_pipeline.py) ?")
    all_files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS)

    by_parent: Dict[str, List[Path]] = {}
    for pid in parent_ids:
        prefix = f"{pid}_"
        by_parent[pid] = [p for p in all_files if p.name.startswith(prefix) or p.stem == pid]
    return by_parent


def _write_image_list(tiles: List[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in tiles:
            f.write(str(p.resolve()) + "\n")


def _write_scoped_data_yaml(base_data_config: Path, split: str, image_list_path: Path, out_yaml_path: Path) -> None:
    with open(base_data_config, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data[split] = str(image_list_path.resolve())
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _print_subset_stats(sliced_dir: str, split: str, tiles: List[Path], target_names: Dict[int, str]) -> None:
    from collections import Counter

    counts: Counter = Counter()
    lab_dir = Path(sliced_dir) / "labels" / split
    for img_path in tiles:
        label_path = lab_dir / (img_path.stem + ".txt")
        for inst in _parse_yolo_seg_label(label_path):
            counts[inst["cls"]] += 1

    print(f"\n  Composition du sous-ensemble ({len(tiles)} imagette(s)) :")
    for class_id in sorted(target_names.keys()):
        n = counts.get(class_id, 0)
        flag = "  ⚠️ échantillon faible" if 0 < n < LOW_SAMPLE_WARN_THRESHOLD else ""
        print(f"    {target_names[class_id]:<16}{n:>5}{flag}")
    total = sum(counts.values())
    print(f"    {'TOTAL':<16}{total:>5}")


def run_scoped_report(
    run_dir,
    batch_filter: str,
    split: str = "test",
    label: Optional[str] = None,
    data_config_path=None,
    sliced_dir: str = SLICED_DIR,
    split_dir: str = SPLIT_DIR,
    weights_name: str = "best.pt",
) -> Path:
    """Voir la docstring du module. Retourne le chemin du rapport HTML scopé écrit."""
    run_dir = Path(run_dir)
    label = _sanitize_label(label if label else batch_filter)
    data_config_path = Path(data_config_path) if data_config_path else DEFAULT_DATA_CONFIG_PATH

    print(f"--- 🔎 RAPPORT SCOPÉ : lot(s) contenant '{batch_filter}', split '{split}' ---")

    parent_manifest = _load_parent_manifest(split_dir)
    parent_ids = _matching_parent_ids(parent_manifest, split, batch_filter)
    if not parent_ids:
        raise RuntimeError(
            f"Aucune image parente du split '{split}' ne correspond au filtre '{batch_filter}'. "
            f"Vérifie l'orthographe du lot (ex: 'SL'), ou que ce split a bien été peuplé "
            f"(python -m src.data.data_pipeline)."
        )
    print(f"    {len(parent_ids)} image(s) parente(s) trouvée(s) : {', '.join(parent_ids)}")

    tiles_by_parent = _tiles_for_parents(sliced_dir, split, parent_ids)
    empty_parents = [pid for pid, tiles in tiles_by_parent.items() if not tiles]
    if empty_parents:
        print(f"  ⚠️  Aucune imagette trouvée pour : {', '.join(empty_parents)} (dataset pas encore tuilé pour "
              f"ce(s) parent(s) ? relance data_pipeline.py).")
    all_tiles = sorted((p for tiles in tiles_by_parent.values() for p in tiles), key=lambda p: p.name)
    if not all_tiles:
        raise RuntimeError("Aucune imagette trouvée pour les images parentes sélectionnées - rien à évaluer.")
    print(f"    {len(all_tiles)} imagette(s) au total pour l'évaluation.")

    _, target_names = load_class_config(DEFAULT_CLASS_CONFIG_PATH)
    _print_subset_stats(sliced_dir, split, all_tiles, target_names)

    scope_dir = run_dir / f".scoped_report_{label}"
    image_list_path = scope_dir / f"{split}_images.txt"
    scoped_yaml_path = scope_dir / "data_config.yaml"
    _write_image_list(all_tiles, image_list_path)
    _write_scoped_data_yaml(data_config_path, split, image_list_path, scoped_yaml_path)

    report_path = generate_report(
        run_dir,
        data_config_path=scoped_yaml_path,
        weights_name=weights_name,
        splits=[split],
        output_basename=f"rapport_{label}",
    )
    print(f"\n--- ✅ Rapport scopé généré : {report_path} ---")
    print(f"    (+ {report_path.with_name(f'rapport_{label}_metrics.json')})")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rapport de lecture PixelOdyssey restreint à un sous-ensemble de lots.")
    parser.add_argument("--run", required=True, help="Dossier du run (ex: \"E:\\PixelOdyssey\\6. Model outputs\\runs\\<run>\")")
    parser.add_argument("--batch-filter", required=True,
                         help="Sous-chaîne (insensible à la casse) sur le nom de lot, ex: 'SL' pour Santa Luzia.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                         help="Split à restreindre (défaut: test).")
    parser.add_argument("--label", default=None,
                         help="Nom court pour les fichiers de sortie (défaut : déduit de --batch-filter).")
    parser.add_argument("--data", default=None, help="Chemin vers data_config.yaml (défaut : celui du projet)")
    parser.add_argument("--weights", default="best.pt")
    args = parser.parse_args()
    try:
        run_scoped_report(
            run_dir=args.run,
            batch_filter=args.batch_filter,
            split=args.split,
            label=args.label,
            data_config_path=args.data,
            weights_name=args.weights,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
