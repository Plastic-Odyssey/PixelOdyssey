#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Audit descriptif du dataset brut (1_annotated_dataset).

Objectif : donner des chiffres concrets - nombre d'instances, nombre
d'images, dispersion de taille/forme - par classe brute, pour trancher la
question "quelles classes brutes regrouper dans quelle super-classe" à
partir de données réelles plutôt que d'une intuition sur la forme des
objets.

Délibérément indépendant de `class_taxonomy` (config/data_config.yaml) :
cet outil sert à construire cette taxonomie, il ne doit donc jamais dépendre
de son état (complet, vide, en cours de réécriture...) pour fonctionner. Les
seules correspondances d'orthographe qu'il connaît sont codées en dur
ci-dessous (ALIASES) - les mêmes divergences que class_aliases, dupliquées
ici volontairement pour ce découplage.

Pour chaque classe brute (après normalisation casse/accents/underscores +
résolution des alias d'orthographe connus), calcule :
  - nombre d'instances, nombre d'images distinctes, lots d'origine
  - aire de l'objet en % de l'aire de l'image (comparable entre lots même à
    des résolutions différentes, contrairement à l'aire en pixels bruts)
  - ratio largeur/hauteur de la boîte englobante (proxy grossier de forme)
  - coefficient de variation (écart-type / moyenne) de l'aire et du ratio
    largeur/hauteur : plus ce nombre est bas, plus les objets de cette classe
    se ressemblent en taille/forme d'une instance à l'autre - un bon indice
    (parmi d'autres, pas une preuve à lui seul) pour juger si une classe a
    une forme reconnaissable et récurrente ou si c'est un fourre-tout.

Entrée : dossier racine des lots d'annotation bruts (--raw-dir).
Sortie : rapport CSV (--output) trié par nombre d'instances décroissant, et
un résumé affiché sur stdout.

Exemple :
    python -m src.data.dataset_audit
    python -m src.data.dataset_audit --raw-dir "1_annotated_dataset" --output audit.csv
"""

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.data.class_config import load_batch_local_names, normalize_class_name
from src.data.image_io import load_image_bgr
from src.data.utils.raw_dataset import collect_parent_images

from shapely.geometry import Polygon

RAW_DIR_DEFAULT = r"E:\PixelOdyssey\3. Processed dataset\1_annotated_dataset"
OUTPUT_CSV_DEFAULT = r"E:\PixelOdyssey\3. Processed dataset\dataset_audit_report.csv"

# Mêmes divergences d'orthographe que class_aliases (config/data_config.yaml),
# dupliquées ici pour que cet outil reste utilisable même si ce fichier est
# incomplet ou en cours de réécriture - voir la docstring du module. Clé =
# nom normalisé (casse/accents/underscores déjà neutralisés), valeur = nom
# "de référence" normalisé sous lequel les compter ensemble.
ALIASES: Dict[str, str] = {
    "flipflops": "fliflops",
    "bouteilles pet": "bouteille pet",
    "bouteilles plastique rigide": "bouteille plastique rigide",
}


def _canonical_key(raw_name: str) -> str:
    key = normalize_class_name(raw_name)
    return ALIASES.get(key, key)


class _ClassStats:
    """Accumulateur pour une classe brute (canonicalisée)."""

    def __init__(self, display_name: str):
        self.display_name = display_name  # premier nom "joliment casé" rencontré
        self.n_instances = 0
        self.images: set = set()
        self.batches: Dict[str, int] = {}  # batch -> nb instances dans ce lot
        self.area_pct: List[float] = []
        self.width_pct: List[float] = []
        self.height_pct: List[float] = []
        self.aspect_ratio: List[float] = []

    def add(self, batch: str, image_key: str, area_pct: float, w_pct: float, h_pct: float, ar: float):
        self.n_instances += 1
        self.images.add(image_key)
        self.batches[batch] = self.batches.get(batch, 0) + 1
        self.area_pct.append(area_pct)
        self.width_pct.append(w_pct)
        self.height_pct.append(h_pct)
        self.aspect_ratio.append(ar)


def _cv(values: List[float]) -> float:
    """Coefficient de variation (écart-type / moyenne). None-safe : renvoie
    0.0 si trop peu de valeurs pour être calculé, plutôt que planter."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def run_audit(raw_dir: str = RAW_DIR_DEFAULT, output_csv: str = OUTPUT_CSV_DEFAULT) -> Dict[str, _ClassStats]:
    all_parents = collect_parent_images(Path(raw_dir))
    if not all_parents:
        print(f"❌ Aucune image brute trouvée dans {raw_dir}.")
        return {}

    unique_parents = list({p["parent_id"]: p for p in all_parents}.values())
    batches_seen = sorted({p["batch"] for p in unique_parents})
    print(f"--- 📊 AUDIT DESCRIPTIF ({len(unique_parents)} image(s) parente(s), {len(batches_seen)} lot(s) : "
          f"{', '.join(batches_seen)}) ---")

    local_names_by_batch: Dict[str, Dict[int, str]] = {}
    stats: Dict[str, _ClassStats] = {}
    n_background_images = 0
    n_images_with_objects = 0
    n_unreadable_images = 0
    background_by_batch: Dict[str, int] = {}
    with_objects_by_batch: Dict[str, int] = {}
    total_by_batch: Dict[str, int] = {}
    classes_by_batch: Dict[str, set] = {}
    # Résolutions natives par lot (largeur, hauteur) - sert à répondre à une question
    # concrète : les images déjà découpées d'un lot (ex: annotées directement à une
    # taille fixe plutôt que sur l'orthomosaïque complète) sont-elles exactement
    # tile_size x tile_size, ou une autre taille ? Ça détermine comment slicer.py
    # (étape 4) les traite.
    dims_by_batch: Dict[str, Dict[tuple, int]] = {}

    for item in unique_parents:
        batch_name = item["batch"]
        img_path = Path(item["img_path"])
        label_path = Path(item["label_path"])
        total_by_batch[batch_name] = total_by_batch.get(batch_name, 0) + 1

        if batch_name not in local_names_by_batch:
            local_yaml = Path(raw_dir) / batch_name / "data.yaml"
            if not local_yaml.exists():
                print(f"  ⚠️  [{batch_name}] pas de data.yaml local - lot ignoré dans l'audit.")
                local_names_by_batch[batch_name] = {}
            else:
                local_names_by_batch[batch_name] = load_batch_local_names(local_yaml)
        local_names = local_names_by_batch[batch_name]
        if not local_names:
            continue

        img = load_image_bgr(img_path)
        if img is None:
            n_unreadable_images += 1
            continue
        img_h, img_w = img.shape[:2]
        img_area = img_w * img_h
        dim_key = (img_w, img_h)
        dims_by_batch.setdefault(batch_name, {})
        dims_by_batch[batch_name][dim_key] = dims_by_batch[batch_name].get(dim_key, 0) + 1

        if not label_path.exists():
            n_background_images += 1
            background_by_batch[batch_name] = background_by_batch.get(batch_name, 0) + 1
            continue

        image_key = str(img_path)
        had_any_instance = False

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                local_id = int(parts[0])
                name = local_names.get(local_id)
                if name is None:
                    continue  # signalé séparément par raw_dataset_checker.py, pas le rôle de cet outil

                coords = [float(x) for x in parts[1:]]
                pixels = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, len(coords), 2)]
                if len(pixels) < 3:
                    continue
                geom = Polygon(pixels)
                if not geom.is_valid or geom.area <= 0:
                    continue

                minx, miny, maxx, maxy = geom.bounds
                bbox_w, bbox_h = maxx - minx, maxy - miny
                if bbox_h <= 0:
                    continue

                key = _canonical_key(name)
                if key not in stats:
                    stats[key] = _ClassStats(display_name=name)
                stats[key].add(
                    batch=batch_name,
                    image_key=image_key,
                    area_pct=100.0 * geom.area / img_area,
                    w_pct=100.0 * bbox_w / img_w,
                    h_pct=100.0 * bbox_h / img_h,
                    ar=bbox_w / bbox_h,
                )
                had_any_instance = True
                classes_by_batch.setdefault(batch_name, set()).add(name)

        if had_any_instance:
            n_images_with_objects += 1
            with_objects_by_batch[batch_name] = with_objects_by_batch.get(batch_name, 0) + 1
        else:
            n_background_images += 1
            background_by_batch[batch_name] = background_by_batch.get(batch_name, 0) + 1

    # --- Tableau par lot, AVANT le tableau par classe : rend immédiatement visible
    # que chaque lot a bien été parcouru, sans avoir à recouper les totaux à la main
    # (un lot où toutes les images ont un .txt n'apparaîtrait jamais dans une simple
    # liste de "lots avec du background" - ambigu avec "lot pas traité du tout").
    print(f"\n{'LOT':<20} {'N IMAGES':>9} {'AVEC OBJET':>11} {'BACKGROUND':>11} {'N CLASSES UTILISÉES':>21}")
    for batch in batches_seen:
        n_total = total_by_batch.get(batch, 0)
        n_obj = with_objects_by_batch.get(batch, 0)
        n_bg = background_by_batch.get(batch, 0)
        n_classes = len(classes_by_batch.get(batch, set()))
        print(f"{batch:<20} {n_total:>9} {n_obj:>11} {n_bg:>11} {n_classes:>21}")

    # --- Résolutions natives par lot : dit si un lot est à taille fixe (une seule
    # entrée) - cas typique d'imagettes déjà découpées à l'annotation - ou variable
    # (plusieurs tailles, ex: orthomosaïques brutes découpées à la main). Important
    # pour savoir comment slicer.py (étape 4) va traiter ce lot.
    print(f"\n--- 📐 RÉSOLUTIONS NATIVES PAR LOT ---")
    for batch in batches_seen:
        dims = dims_by_batch.get(batch, {})
        if not dims:
            continue
        if len(dims) == 1:
            (w, h), n = next(iter(dims.items()))
            tag = "== tile_size (640x640)" if (w, h) == (640, 640) else "taille fixe, PAS 640x640"
            print(f"  [{batch}] {n} image(s), résolution UNIQUE {w}x{h}  ({tag})")
        else:
            sorted_dims = sorted(dims.items(), key=lambda kv: -kv[1])
            top = ", ".join(f"{w}x{h}({n})" for (w, h), n in sorted_dims[:4])
            more = f" +{len(sorted_dims) - 4} autre(s) résolution(s)" if len(sorted_dims) > 4 else ""
            print(f"  [{batch}] {sum(dims.values())} image(s), résolutions VARIABLES : {top}{more}")

    # --- Rapport CSV, trié par nombre d'instances décroissant ---
    rows = []
    for key, s in sorted(stats.items(), key=lambda kv: kv[1].n_instances, reverse=True):
        rows.append({
            "classe_brute": s.display_name,
            "n_instances": s.n_instances,
            "n_images": len(s.images),
            "n_lots": len(s.batches),
            "lots": ", ".join(f"{b}({n})" for b, n in sorted(s.batches.items(), key=lambda x: -x[1])),
            "aire_mediane_pct_image": round(statistics.median(s.area_pct), 4),
            "aire_moyenne_pct_image": round(statistics.mean(s.area_pct), 4),
            "cv_aire": round(_cv(s.area_pct), 3),
            "largeur_mediane_pct_image": round(statistics.median(s.width_pct), 4),
            "hauteur_mediane_pct_image": round(statistics.median(s.height_pct), 4),
            "ratio_lxh_median": round(statistics.median(s.aspect_ratio), 3),
            "cv_ratio_lxh": round(_cv(s.aspect_ratio), 3),
        })

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # --- Résumé console ---
    print(f"\n{'CLASSE':<32} {'N INST':>7} {'N IMG':>6} {'AIRE MED %':>11} {'CV AIRE':>8} {'RATIO L/H MED':>14} {'CV RATIO':>9}")
    for r in rows:
        print(
            f"{r['classe_brute']:<32} {r['n_instances']:>7} {r['n_images']:>6} "
            f"{r['aire_mediane_pct_image']:>11} {r['cv_aire']:>8} "
            f"{r['ratio_lxh_median']:>14} {r['cv_ratio_lxh']:>9}"
        )

    print(f"\n--- 🖼️  IMAGES (détail par lot ci-dessus) ---")
    print(f"  • Images avec au moins un objet : {n_images_with_objects}")
    print(f"  • Images 'sans déchet' (background, .txt absent ou vide) : {n_background_images}")
    if n_unreadable_images:
        print(f"  ⚠️  Images illisibles (ignorées) : {n_unreadable_images}")

    print(f"\n[SUCCÈS] Rapport écrit : {output_csv}")
    print(
        "\nLecture des colonnes 'CV' (coefficient de variation = écart-type / moyenne) : plus c'est "
        "bas, plus les instances de cette classe se ressemblent en taille (cv_aire) ou en forme "
        "(cv_ratio_lxh) - un indice utile pour juger si une classe a une silhouette reconnaissable "
        "et récurrente, ou si c'est plutôt un fourre-tout. Un indice parmi d'autres : regarde aussi "
        "n_instances (une classe avec trop peu d'exemples reste inapprenable même avec un CV bas)."
    )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit descriptif du dataset brut PixelOdyssey")
    parser.add_argument("--raw-dir", default=RAW_DIR_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_CSV_DEFAULT)
    args = parser.parse_args()
    run_audit(raw_dir=args.raw_dir, output_csv=args.output)
