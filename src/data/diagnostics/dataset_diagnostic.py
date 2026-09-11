#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Diagnostic complet du dataset annoté, export .xlsx multi-feuilles.

Objectif : donner une vue EXHAUSTIVE, au niveau de chaque DÉCHET ANNOTÉ
individuel (un "item" = une instance/masque, pas une image), avec toutes les
métadonnées disponibles (lot/collecte d'origine, image d'origine, split
train/val/test si déjà connu, classe brute + super-classe résolue, taille/
forme, géolocalisation quand elle existe) - puis des feuilles agrégées
(moyennes/médianes) par super-classe et par collecte.

Complète, sans le remplacer, `dataset_audit.py` : celui-ci reste l'outil de
référence pour la conception de la taxonomie (indépendant de
class_taxonomy, classes BRUTES uniquement, CSV simple). Ce module-ci
suppose une taxonomie déjà stable, résout chaque instance vers sa
super-classe cible, et vise l'export/le partage (xlsx, plusieurs feuilles)
plutôt que l'exploration en console.

Périmètre volontaire : lit `1_annotated_dataset` (le dataset SOURCE, une
ligne par instance réellement annotée), jamais `4_sliced_dataset` (les
imagettes tuilées, qui dupliquent chaque instance sur chaque tuile qui la
recouvre - compter les items là-bas fausserait tous les totaux et toutes
les moyennes). Même choix que `dataset_audit.py`, voir sa docstring.

Géolocalisation : tentée pour CHAQUE image parente via rasterio (fonctionne
pour un GeoTIFF avec CRS embarqué - ex. les lots "SL ..." ; échoue
silencieusement et proprement pour une image sans géoréférencement - ex.
les lots "SB ..." dont les images sont déjà des fragments JPG pré-tuilés
SANS CRS, ce qui est un état de fait du dataset, pas un bug de ce script).
Quand la géoréférencement existe, un estimé de surface réelle (m²) est
calculé à partir de la résolution sol RÉELLE du GeoTIFF (GSD mesurée, pas
supposée) - comparable aux mesures manuelles de
`guide_mesure_surface_bache_qgis.md`, mais à lire comme une estimation
(empreinte 2D vue du dessus, pas un volume).

Aire en pixels + estimation cm² à GSD FIXE : en plus de `aire_m2_estimee`
(GSD réelle, seulement dispo si géoréférencé), chaque item porte aussi
`aire_px` (aire brute du masque en pixels², toujours disponible) et
`aire_cm2_estimee_gsd_fixe` (aire_px × gsd_fixe_cm_px², avec
`--gsd-fixe-cm-px` par défaut 0.6 cm/px) - calculée pour TOUS les items, y
compris les lots non géoréférencés (SB...), puisqu'elle ne dépend d'aucun
CRS. Point de vigilance à garder en tête : c'est une HYPOTHÈSE UNIQUE
appliquée à tout le dataset, alors que la GSD réelle peut varier d'un lot à
l'autre (altitude de vol, appareil) : utile pour comparer vite tous les
items sur une base commune, mais `aire_m2_estimee` (GSD mesurée) reste la
référence à privilégier pour les lots qui l'ont.

Split train/val/test : ajouté par item SI `2_split_dataset/.parent_manifest.json`
existe déjà (généré par split_dataset.py / data_pipeline.py) - sinon la
colonne l'indique explicitement plutôt que de l'omettre en silence.

Entrée : --raw-dir (dataset annoté brut, défaut 1_annotated_dataset),
--split-dir (pour le manifeste de split, optionnel), --output (.xlsx).
Sortie : un classeur .xlsx avec les feuilles :
  - Résumé            : totaux et taux de couverture (geoloc, split...)
  - Items              : une ligne par instance annotée, TOUTES les colonnes
  - Par super-classe    : agrégats/moyennes sur la taxonomie cible (7 classes)
  - Par collecte        : agrégats/moyennes par lot (dossier de 1er niveau)
  - Par classe brute     : granularité fine, pour recoupement avec dataset_audit.py
  - Par split            : totaux par split train/val/test (vérification
    d'équilibre de classe sur le dataset multi-classe - voir aussi
    "Répartition classes x split" ci-dessous, la feuille qui répond
    réellement à cette question)
  - Répartition classes x split : LA feuille pour repérer un déséquilibre de
    classe entre train/val/test - une ligne par super-classe, avec pour
    chaque split : le compte brut, "% du split" (part de cette classe PARMI
    les items de ce split - révèle une classe sur/sous-représentée dans un
    split par rapport aux autres) et "% de la classe" (part des instances de
    CETTE classe qui tombe dans ce split - révèle une classe rare presque
    absente de val/test, donc impossible à évaluer correctement). Triée par
    `ecart_max_pct_pts` décroissant (écart max de "% du split" entre les 3
    splits pour cette classe) pour faire remonter les pires déséquilibres en
    premier. Colonne `alerte` : signale explicitement une classe totalement
    absente de val ou test (le cas le plus grave - au-delà d'un déséquilibre,
    c'est une classe qu'on ne peut même pas évaluer).

EXCLUDE (classes explicitement exclues de l'entraînement, ex. "Morceaux de
bois"/"Verre") : présentes dans "Items" (dump exhaustif, colonne `est_exclue`)
et comptées dans "Résumé", mais retirées de TOUTES les feuilles agrégées par
super-classe ("Par super-classe", "Par collecte", "Par split", "Répartition
classes x split") - une classe jamais vue à l'entraînement n'a pas sa place
dans un diagnostic de composition/équilibre de ce qui EST entraîné, et sa
présence y fausserait les pourcentages (dénominateur gonflé) tout en
risquant de déclencher à tort l'alerte "absente de val/test".
NON RÉSOLU (classe brute non reconnue par la taxonomie - anomalie, pas une
exclusion voulue) reste inclus partout, volontairement : à corriger, pas à
masquer. "Par classe brute" reste lui aussi sur le jeu complet (recoupement
avec dataset_audit.py, qui n'a aucune notion d'exclusion).

Exemple :
    python -m src.data.dataset_diagnostic
    python -m src.data.dataset_diagnostic --raw-dir "1_annotated_dataset" --output diag.xlsx
    python -m src.data.dataset_diagnostic --skip-geo   # plus rapide, sans tentative de géoloc
    python -m src.data.dataset_diagnostic --gsd-fixe-cm-px 0.5   # autre hypothèse de GSD fixe
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pandas as pd
from shapely.geometry import Polygon

from src.data.class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    EXCLUDE,
    load_batch_local_names,
    load_class_config,
    normalize_class_name,
    resolve_class_name,
)
from src.data.raw_dataset import collect_parent_images
from src.paths_config import ANNOTATED_DATASET_DIR, PROCESSED_DATASET_DIR, SPLIT_DATASET_DIR

# Dérivés de paths_config.py (seule source de vérité pour la racine du
# disque de données) plutôt que redéfinis en dur ici.
RAW_DIR_DEFAULT = str(ANNOTATED_DATASET_DIR)
OUTPUT_XLSX_DEFAULT = str(PROCESSED_DATASET_DIR / "dataset_diagnostic.xlsx")
SPLIT_DIR_DEFAULT = str(SPLIT_DATASET_DIR)
PARENT_MANIFEST_FILENAME = ".parent_manifest.json"  # même nom que split_dataset.py
GSD_FIXE_CM_PAR_PX_DEFAULT = 0.6  # hypothèse unique, voir docstring du module

WEBMERCATOR = "EPSG:3857"

# Mêmes divergences d'orthographe que dataset_audit.py (voir sa docstring) -
# n'affecte QUE la feuille "Par classe brute" (granularité fine), jamais la
# résolution vers la super-classe (qui passe par class_taxonomy/class_aliases,
# seule source de vérité pour ça).
RAW_ALIASES: Dict[str, str] = {
    "flipflops": "fliflops",
    "bouteilles pet": "bouteille pet",
    "bouteilles plastique rigide": "bouteille plastique rigide",
}


def _raw_canonical_key(raw_name: str) -> str:
    key = normalize_class_name(raw_name)
    return RAW_ALIASES.get(key, key)


def _cv(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def _load_parent_manifest(split_dir: str) -> Dict[str, Dict]:
    """Retourne {} si le manifeste n'existe pas encore (pipeline pas lancé) -
    jamais bloquant, juste une colonne 'split' marquée indisponible."""
    manifest_path = Path(split_dir) / PARENT_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _geo_info_for_image(img_path: Path, skip_geo: bool = False) -> Dict:
    """Tente d'ouvrir `img_path` avec rasterio pour en tirer les dimensions ET,
    quand elles existent (et que `skip_geo` n'a pas été demandé), le
    géoréférencement (CRS + transformation affine) et la résolution sol (GSD,
    en mètres/pixel - via reprojection EPSG:3857, fonctionne quel que soit le
    CRS source, projeté ou géographique - même logique que
    `geo_density_map.generate_tile_pyramid`).

    N'ouvre JAMAIS l'image en pixels (métadonnées seules) - coût négligeable
    même sur une orthomosaïque de plusieurs centaines de Mo. `skip_geo=True`
    lit quand même les dimensions (toujours nécessaires pour dénormaliser les
    polygones) - seul le calcul de géoréférencement/GSD est court-circuité,
    plus coûteux (reprojection) que la simple ouverture des métadonnées.

    Retourne toujours : width, height (int, ou None si le fichier est
    illisible - SEUL cas où l'appelant doit ignorer ce parent). Si
    géoréférencé et `skip_geo=False` : + has_geo=True, crs (str), transform
    (objet rasterio), gsd_m (float). Sinon : has_geo=False.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform

    try:
        with rasterio.open(str(img_path)) as src:
            width, height = src.width, src.height
            if skip_geo or src.crs is None:
                return {"width": width, "height": height, "has_geo": False}
            try:
                dst_transform, _, _ = calculate_default_transform(
                    src.crs, WEBMERCATOR, width, height, *src.bounds
                )
                gsd_m = abs(dst_transform.a)
            except Exception:
                gsd_m = None
            return {
                "width": width,
                "height": height,
                "has_geo": True,
                "crs": str(src.crs),
                "transform": src.transform,
                "gsd_m": gsd_m,
            }
    except Exception:
        # Fichier illisible par rasterio (rare vu VALID_IMG_EXTS) - signalé à
        # l'appelant via width/height=None plutôt qu'un plantage du diagnostic entier.
        return {"width": None, "height": None, "has_geo": False}


def _centroids_to_lonlat(centroids_px: List[Tuple[float, float]], transform, crs) -> List[Tuple[float, float]]:
    """Même logique que geo_density_map.pixels_to_lonlat, réimportée directement
    depuis src.application.geo_density_map pour ne jamais dupliquer cette conversion -
    voir l'import plus bas dans build_items()."""
    from src.application.geo_density_map import pixels_to_lonlat

    return pixels_to_lonlat(centroids_px, transform, crs)


def build_items(
    raw_dir: str,
    split_dir: str,
    skip_geo: bool = False,
    gsd_fixe_cm_px: float = GSD_FIXE_CM_PAR_PX_DEFAULT,
) -> Tuple[List[Dict], Dict]:
    """Parcourt `raw_dir` et construit une liste de dicts, un par instance
    annotée (item). Retourne aussi un dict de compteurs de diagnostic (lignes
    ignorées, classes non résolues...) à afficher/consigner séparément.
    """
    taxonomy, target_names = load_class_config(DEFAULT_CLASS_CONFIG_PATH)
    parent_manifest = _load_parent_manifest(split_dir)

    all_parents = collect_parent_images(Path(raw_dir))
    unique_parents = list({p["parent_id"]: p for p in all_parents}.values())
    batches_seen = sorted({p["batch"] for p in unique_parents})
    print(f"--- 🔎 DIAGNOSTIC DATASET ({len(unique_parents)} image(s) parente(s), "
          f"{len(batches_seen)} collecte(s) : {', '.join(batches_seen)}) ---")
    if not parent_manifest:
        print("  ℹ️  Pas de manifeste de split trouvé (2_split_dataset/.parent_manifest.json) - "
              "colonne 'split' marquée 'indisponible' pour tous les items. Lance data_pipeline.py "
              "si tu veux aussi voir la répartition train/val/test.")

    local_names_by_batch: Dict[str, Dict[int, str]] = {}
    items: List[Dict] = []
    counters = {
        "n_parents": len(unique_parents),
        "n_parents_illisibles": 0,
        "n_parents_sans_dataYaml": 0,
        "n_lignes_ignorees_classe_locale_introuvable": 0,
        "n_instances_non_resolues": 0,
        "n_instances_exclues": 0,
        "n_parents_georeferences": 0,
        "n_instances_avec_geoloc": 0,
    }
    unresolved_names_seen: set = set()

    item_id = 0
    for item in unique_parents:
        batch_name = item["batch"]
        img_path = Path(item["img_path"])
        label_path = Path(item["label_path"])
        parent_id = item["parent_id"]

        if batch_name not in local_names_by_batch:
            local_yaml = Path(raw_dir) / batch_name / "data.yaml"
            if not local_yaml.exists():
                print(f"  ⚠️  [{batch_name}] pas de data.yaml local - lot ignoré.")
                local_names_by_batch[batch_name] = {}
                counters["n_parents_sans_dataYaml"] += 1
            else:
                local_names_by_batch[batch_name] = load_batch_local_names(local_yaml)
        local_names = local_names_by_batch[batch_name]
        if not local_names:
            continue

        geo = _geo_info_for_image(img_path, skip_geo=skip_geo)
        if geo.get("width") is None:
            counters["n_parents_illisibles"] += 1
            continue
        img_w, img_h = geo["width"], geo["height"]
        img_area = img_w * img_h
        has_geo = geo.get("has_geo", False)
        if has_geo:
            counters["n_parents_georeferences"] += 1

        split_info = parent_manifest.get(parent_id)
        split_value = split_info["split"] if split_info else "indisponible"

        if not label_path.exists():
            continue  # image "background" (aucun déchet annoté) - pas un item, rien à lister ici

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]

        # Centroïdes calculés d'abord pour TOUTES les instances valides de CE
        # parent, converti en lon/lat en un seul appel groupé (comme
        # geo_density_map) plutôt qu'un appel rasterio.warp par instance.
        parsed_instances = []
        for line_no, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            local_id = int(parts[0])
            raw_name = local_names.get(local_id)
            if raw_name is None:
                counters["n_lignes_ignorees_classe_locale_introuvable"] += 1
                continue

            coords = [float(x) for x in parts[1:]]
            pixels = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, len(coords), 2)]
            if len(pixels) < 3:
                continue
            geom = Polygon(pixels)
            if not geom.is_valid or geom.area <= 0:
                continue
            parsed_instances.append((line_no, raw_name, geom))

        centroids_px = [(g.centroid.x, g.centroid.y) for _, _, g in parsed_instances]
        lonlat = []
        if has_geo and centroids_px:
            try:
                lonlat = _centroids_to_lonlat(centroids_px, geo["transform"], geo["crs"])
            except Exception:
                lonlat = []

        for idx, (line_no, raw_name, geom) in enumerate(parsed_instances):
            minx, miny, maxx, maxy = geom.bounds
            bbox_w, bbox_h = maxx - minx, maxy - miny
            if bbox_h <= 0:
                continue

            resolved = resolve_class_name(raw_name, taxonomy)
            est_exclue = False
            if resolved is None:
                super_classe = "⚠ NON RÉSOLU"
                super_classe_id = None
                counters["n_instances_non_resolues"] += 1
                unresolved_names_seen.add(raw_name)
            elif resolved == EXCLUDE:
                super_classe = "EXCLUDE (jamais utilisée à l'entraînement)"
                super_classe_id = None
                counters["n_instances_exclues"] += 1
                est_exclue = True
            else:
                super_classe = target_names.get(resolved, f"classe_{resolved}")
                super_classe_id = resolved

            lon = lat = None
            aire_m2_estimee = None
            if has_geo and idx < len(lonlat):
                lon, lat = lonlat[idx]
                if geo.get("gsd_m"):
                    aire_m2_estimee = round(geom.area * (geo["gsd_m"] ** 2), 4)
                counters["n_instances_avec_geoloc"] += 1

            # Aire brute en pixels² - toujours disponible (pas besoin de géoréférencement),
            # sert de base à aire_cm2_estimee_gsd_fixe ci-dessous. `geom.area` est déjà en
            # pixels² à ce stade (polygone dénormalisé via img_w/img_h plus haut).
            aire_px = round(geom.area, 2)
            aire_cm2_estimee_gsd_fixe = round(aire_px * (gsd_fixe_cm_px ** 2), 4)

            item_id += 1
            items.append({
                "item_id": item_id,
                "collecte": batch_name,
                "image_origine": img_path.name,
                "chemin_image_complet": str(img_path),
                "parent_id": parent_id,
                "split": split_value,
                "label_path": str(label_path),
                "ligne_label": line_no,
                "classe_brute": raw_name,
                "super_classe": super_classe,
                "super_classe_id": super_classe_id,
                "est_exclue": est_exclue,
                "image_largeur_px": img_w,
                "image_hauteur_px": img_h,
                "aire_px": aire_px,
                "aire_pct_image": round(100.0 * geom.area / img_area, 4),
                "largeur_pct_image": round(100.0 * bbox_w / img_w, 4),
                "hauteur_pct_image": round(100.0 * bbox_h / img_h, 4),
                "ratio_largeur_hauteur": round(bbox_w / bbox_h, 3),
                "aire_cm2_estimee_gsd_fixe": aire_cm2_estimee_gsd_fixe,
                "geoloc_disponible": "Oui" if (has_geo and lon is not None) else "Non",
                "crs_source": geo.get("crs") if has_geo else None,
                "longitude": lon,
                "latitude": lat,
                "gsd_m_par_px_reelle": round(geo["gsd_m"], 4) if (has_geo and geo.get("gsd_m")) else None,
                "aire_m2_estimee": aire_m2_estimee,
            })

    if unresolved_names_seen:
        print(f"  ⚠️  {counters['n_instances_non_resolues']} instance(s) avec une classe brute NON "
              f"résolue par la taxonomie actuelle : {sorted(unresolved_names_seen)}. Normalement déjà "
              f"bloqué en amont par raw_dataset_checker.py - à vérifier si ce diagnostic tourne sur un "
              f"dataset non passé par ce garde-fou (ex: 1bis_corrected_annotation pas encore vérifié).")

    return items, counters


def _write_summary_sheet(
    writer, items: List[Dict], counters: Dict, raw_dir: str, gsd_fixe_cm_px: float
) -> None:
    from datetime import datetime

    n_geo = counters["n_instances_avec_geoloc"]
    n_total = len(items)
    pct_geo = round(100.0 * n_geo / n_total, 1) if n_total else 0.0
    n_split_ok = sum(1 for it in items if it["split"] != "indisponible")
    pct_split = round(100.0 * n_split_ok / n_total, 1) if n_total else 0.0

    rows = [
        ("Généré le", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Dataset source", raw_dir),
        ("GSD fixe utilisée pour aire_cm2_estimee_gsd_fixe (HYPOTHÈSE, pas mesurée)",
         f"{gsd_fixe_cm_px} cm/px - voir --gsd-fixe-cm-px"),
        ("Nombre d'images parentes", counters["n_parents"]),
        ("  dont géoréférencées (GeoTIFF avec CRS)", counters["n_parents_georeferences"]),
        ("  dont illisibles", counters["n_parents_illisibles"]),
        ("  dont sans data.yaml local", counters["n_parents_sans_dataYaml"]),
        ("Nombre total d'items (instances annotées)", n_total),
        ("  dont géolocalisées", f"{n_geo} ({pct_geo}%)"),
        ("  dont avec split train/val/test connu", f"{n_split_ok} ({pct_split}%)"),
        ("  dont exclues de l'entraînement (EXCLUDE)", counters["n_instances_exclues"]),
        ("  dont classe brute NON résolue (⚠ à vérifier)", counters["n_instances_non_resolues"]),
        ("Lignes de label ignorées (classe locale introuvable)",
         counters["n_lignes_ignorees_classe_locale_introuvable"]),
    ]
    df = pd.DataFrame(rows, columns=["Indicateur", "Valeur"])
    df.to_excel(writer, sheet_name="Résumé", index=False)


def _write_per_class_sheet(writer, df_items: pd.DataFrame) -> None:
    """`df_items` reçu ici est déjà filtré des instances EXCLUDE par l'appelant
    (voir run_diagnostic) - cette feuille porte sur la taxonomie CIBLE
    (super-classes réellement entraînées), une instance EXCLUDE n'y a pas sa
    place plus qu'une instance qui n'existerait pas."""
    if df_items.empty:
        return
    rows = []
    for super_classe, g in df_items.groupby("super_classe"):
        geo_g = g[g["geoloc_disponible"] == "Oui"]
        rows.append({
            "super_classe": super_classe,
            "n_instances": len(g),
            "n_images": g["chemin_image_complet"].nunique(),
            "n_collectes": g["collecte"].nunique(),
            "collectes": ", ".join(f"{b}({n})" for b, n in g["collecte"].value_counts().items()),
            "aire_px_mediane": round(g["aire_px"].median(), 2),
            "aire_px_moyenne": round(g["aire_px"].mean(), 2),
            "aire_pct_image_mediane": round(g["aire_pct_image"].median(), 4),
            "aire_pct_image_moyenne": round(g["aire_pct_image"].mean(), 4),
            "cv_aire": round(_cv(g["aire_pct_image"].tolist()), 3),
            "ratio_lxh_median": round(g["ratio_largeur_hauteur"].median(), 3),
            "cv_ratio_lxh": round(_cv(g["ratio_largeur_hauteur"].tolist()), 3),
            "aire_cm2_estimee_gsd_fixe_moyenne": round(g["aire_cm2_estimee_gsd_fixe"].mean(), 2),
            "n_instances_georeferencees": len(geo_g),
            "aire_m2_estimee_moyenne": round(geo_g["aire_m2_estimee"].mean(), 4) if len(geo_g) else None,
        })
    out = pd.DataFrame(rows).sort_values("n_instances", ascending=False)
    out.to_excel(writer, sheet_name="Par super-classe", index=False)


def _write_per_batch_sheet(writer, df_items: pd.DataFrame) -> None:
    """`df_items` reçu ici est déjà filtré des instances EXCLUDE (voir
    run_diagnostic) - sans ça, `n_super_classes_presentes`/`classes_presentes`
    afficherait "EXCLUDE (jamais utilisée à l'entraînement)" comme si c'était
    une vraie super-classe de ce lot."""
    if df_items.empty:
        return
    rows = []
    for collecte, g in df_items.groupby("collecte"):
        geo_g = g[g["geoloc_disponible"] == "Oui"]
        row = {
            "collecte": collecte,
            "n_instances": len(g),
            "n_images": g["chemin_image_complet"].nunique(),
            "n_super_classes_presentes": g["super_classe"].nunique(),
            "classes_presentes": ", ".join(f"{c}({n})" for c, n in g["super_classe"].value_counts().items()),
            "aire_pct_image_moyenne": round(g["aire_pct_image"].mean(), 4),
            "aire_cm2_estimee_gsd_fixe_moyenne": round(g["aire_cm2_estimee_gsd_fixe"].mean(), 2),
            "aire_cm2_estimee_gsd_fixe_totale": round(g["aire_cm2_estimee_gsd_fixe"].sum(), 2),
            "pct_instances_georeferencees": round(100.0 * len(geo_g) / len(g), 1) if len(g) else 0.0,
        }
        if len(geo_g):
            row["latitude_moyenne"] = round(geo_g["latitude"].mean(), 6)
            row["longitude_moyenne"] = round(geo_g["longitude"].mean(), 6)
            row["latitude_min"] = round(geo_g["latitude"].min(), 6)
            row["latitude_max"] = round(geo_g["latitude"].max(), 6)
            row["longitude_min"] = round(geo_g["longitude"].min(), 6)
            row["longitude_max"] = round(geo_g["longitude"].max(), 6)
            row["aire_m2_estimee_totale"] = round(geo_g["aire_m2_estimee"].sum(), 2)
        else:
            row.update({k: None for k in (
                "latitude_moyenne", "longitude_moyenne", "latitude_min", "latitude_max",
                "longitude_min", "longitude_max", "aire_m2_estimee_totale",
            )})
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_instances", ascending=False)
    out.to_excel(writer, sheet_name="Par collecte", index=False)


def _write_per_raw_class_sheet(writer, df_items: pd.DataFrame) -> None:
    if df_items.empty:
        return
    keys = df_items["classe_brute"].map(_raw_canonical_key)
    df = df_items.assign(_cle_brute=keys)
    rows = []
    for cle, g in df.groupby("_cle_brute"):
        display_name = g["classe_brute"].iloc[0]
        rows.append({
            "classe_brute": display_name,
            "super_classe_cible": ", ".join(sorted(g["super_classe"].unique())),
            "n_instances": len(g),
            "n_images": g["chemin_image_complet"].nunique(),
            "n_collectes": g["collecte"].nunique(),
            "aire_px_mediane": round(g["aire_px"].median(), 2),
            "aire_pct_image_mediane": round(g["aire_pct_image"].median(), 4),
            "cv_aire": round(_cv(g["aire_pct_image"].tolist()), 3),
            "ratio_lxh_median": round(g["ratio_largeur_hauteur"].median(), 3),
            "cv_ratio_lxh": round(_cv(g["ratio_largeur_hauteur"].tolist()), 3),
            "aire_cm2_estimee_gsd_fixe_moyenne": round(g["aire_cm2_estimee_gsd_fixe"].mean(), 2),
        })
    out = pd.DataFrame(rows).sort_values("n_instances", ascending=False)
    out.to_excel(writer, sheet_name="Par classe brute", index=False)


def _write_per_split_sheet(writer, df_items: pd.DataFrame) -> None:
    """Une ligne par split (train/val/test, + 'indisponible' si le manifeste
    de split n'existe pas encore) - vue d'ensemble avant le détail par classe
    de `_write_class_by_split_sheet` ci-dessous.

    `df_items` reçu ici est déjà filtré des instances EXCLUDE (voir
    run_diagnostic) : `pct_instances_dataset`/`n_super_classes_presentes` ne
    doivent porter que sur les instances réellement entraînables, sinon un
    split avec plus d'instances EXCLUDE que les autres semblerait
    artificiellement plus riche."""
    if df_items.empty:
        return
    split_order = {"train": 0, "val": 1, "test": 2, "indisponible": 3}
    n_total = len(df_items)
    rows = []
    for split, g in df_items.groupby("split"):
        rows.append({
            "split": split,
            "n_instances": len(g),
            "pct_instances_dataset": round(100.0 * len(g) / n_total, 1) if n_total else 0.0,
            "n_images": g["chemin_image_complet"].nunique(),
            "n_super_classes_presentes": g["super_classe"].nunique(),
            "classes_presentes": ", ".join(f"{c}({n})" for c, n in g["super_classe"].value_counts().items()),
        })
    out = pd.DataFrame(rows)
    out["_ordre"] = out["split"].map(split_order).fillna(99)
    out = out.sort_values("_ordre").drop(columns="_ordre")
    out.to_excel(writer, sheet_name="Par split", index=False)


def _write_class_by_split_sheet(writer, df_items: pd.DataFrame) -> List[str]:
    """LA feuille qui répond à "y a-t-il un déséquilibre de classe entre
    train/val/test ?" (dataset multi-classe - voir docstring du module).
    Une ligne par super-classe RÉELLEMENT ENTRAÎNÉE : `df_items` reçu ici
    est déjà filtré des instances EXCLUDE par l'appelant (voir
    run_diagnostic) - une classe jamais vue à l'entraînement n'a rien à
    faire dans un diagnostic de déséquilibre ENTRE splits d'entraînement.
    Sans ce filtre, `pct_du_split`/`ecart_max_pct_pts` seraient faussés par
    le poids d'EXCLUDE dans chaque split, ET EXCLUDE apparaîtrait comme une
    classe à part entière risquant de déclencher l'alerte "absente de
    val/test" alors qu'elle n'a par construction jamais vocation à y être
    évaluée.
    NON RÉSOLU reste en revanche inclus (pas concerné par cette demande) :
    contrairement à EXCLUDE (exclusion volontaire et connue), NON RÉSOLU
    signale une vraie anomalie de données à corriger, pas à masquer.

    Deux normalisations différentes, toutes deux nécessaires (l'une ne
    remplace pas l'autre) :
      - pct_du_split_<split> : part de CE split occupée par cette classe -
        révèle une classe sur/sous-représentée dans un split par rapport aux
        autres (ex: une classe qui pèse 40% du train mais 10% du test).
      - pct_de_la_classe_<split> : part des instances de CETTE classe qui
        atterrit dans ce split - révèle une classe rare presque absente de
        val/test (donc non évaluable), même si son poids relatif DANS ce
        split minuscule semble correct.

    `ecart_max_pct_pts` : écart max de pct_du_split entre les 3 splits pour
    cette classe - sert de score de tri (pires déséquilibres en premier), pas
    un jugement de gravité absolu (voir `alerte` pour le cas vraiment grave :
    classe totalement absente de val ou test).

    Sortie : liste des messages d'alerte (classe absente de val/test), pour
    affichage console immédiat par l'appelant - ne pas laisser ce signal
    enfoui dans un fichier xlsx qu'on n'ouvrira peut-être pas tout de suite.
    """
    if df_items.empty:
        return []

    all_splits = list(df_items["split"].unique())
    split_order = [s for s in ("train", "val", "test", "indisponible") if s in all_splits]
    split_order += [s for s in all_splits if s not in split_order]  # valeur inattendue -> en fin, pas perdue

    counts = df_items.groupby(["super_classe", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=split_order, fill_value=0)
    split_totals = {s: int(counts[s].sum()) for s in split_order}

    console_alerts: List[str] = []
    rows = []
    for super_classe, row in counts.iterrows():
        total_classe = int(row.sum())
        row_out = {"super_classe": super_classe, "n_total": total_classe}
        pct_du_split_values = []
        for s in split_order:
            n_s = int(row[s])
            row_out[f"n_{s}"] = n_s
            pct_du_split = round(100.0 * n_s / split_totals[s], 2) if split_totals[s] else 0.0
            row_out[f"pct_du_split_{s}"] = pct_du_split
            pct_du_split_values.append(pct_du_split)
            row_out[f"pct_de_la_classe_{s}"] = round(100.0 * n_s / total_classe, 1) if total_classe else 0.0
        row_out["ecart_max_pct_pts"] = (
            round(max(pct_du_split_values) - min(pct_du_split_values), 2) if pct_du_split_values else 0.0
        )

        alerts = []
        for s in ("val", "test"):
            if s in split_order and int(row.get(s, 0)) == 0 and total_classe > 0:
                alerts.append(f"⚠ absente de {s}")
        row_out["alerte"] = "; ".join(alerts)
        if alerts:
            console_alerts.append(f"  ⚠️  {super_classe} : {row_out['alerte']} ({total_classe} instance(s) au total)")
        rows.append(row_out)

    out = pd.DataFrame(rows).sort_values("ecart_max_pct_pts", ascending=False)
    out.to_excel(writer, sheet_name="Répartition classes x split", index=False)
    return console_alerts


def run_diagnostic(
    raw_dir: str = RAW_DIR_DEFAULT,
    split_dir: str = SPLIT_DIR_DEFAULT,
    output_xlsx: str = OUTPUT_XLSX_DEFAULT,
    skip_geo: bool = False,
    gsd_fixe_cm_px: float = GSD_FIXE_CM_PAR_PX_DEFAULT,
) -> str:
    items, counters = build_items(raw_dir, split_dir, skip_geo=skip_geo, gsd_fixe_cm_px=gsd_fixe_cm_px)
    if not items:
        print("❌ Aucun item trouvé - vérifie --raw-dir.")
        return ""

    df_items = pd.DataFrame(items)
    # Feuilles agrégées par SUPER-CLASSE (Par super-classe, Par collecte, Par
    # split, Répartition classes x split) : une instance EXCLUDE n'est par
    # construction jamais vue à l'entraînement, elle n'a donc pas sa place
    # dans un diagnostic de composition/équilibre de ce qui EST entraîné.
    # "Items" (dump exhaustif) et "Par classe brute" (recoupement avec
    # dataset_audit.py, qui n'a lui-même aucune notion d'exclusion) restent
    # sur df_items complet - voir leurs docstrings respectifs.
    df_stats = df_items[~df_items["est_exclue"]] if "est_exclue" in df_items.columns else df_items
    n_exclues_stats = len(df_items) - len(df_stats)
    if n_exclues_stats:
        print(f"  ℹ️  {n_exclues_stats} instance(s) EXCLUDE retirée(s) des feuilles agrégées par "
              f"super-classe (Par super-classe/Par collecte/Par split/Répartition classes x split) - "
              f"toujours comptées dans 'Résumé' et listées dans 'Items'.")

    n_indisponible = sum(1 for it in items if it["split"] == "indisponible")
    if n_indisponible == len(items):
        print("  ℹ️  Aucun item n'a de split connu (voir message plus haut) - les feuilles \"Par split\" "
              "et \"Répartition classes x split\" seront donc peu utiles (tout retombe dans "
              "'indisponible'). Lance data_pipeline.py (au moins l'étape split) avant de rejuger un "
              "éventuel déséquilibre.")

    output_xlsx = str(output_xlsx)
    Path(output_xlsx).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        _write_summary_sheet(writer, items, counters, raw_dir, gsd_fixe_cm_px)
        df_items.to_excel(writer, sheet_name="Items", index=False)
        _write_per_class_sheet(writer, df_stats)
        _write_per_batch_sheet(writer, df_stats)
        _write_per_raw_class_sheet(writer, df_items)
        _write_per_split_sheet(writer, df_stats)
        split_alerts = _write_class_by_split_sheet(writer, df_stats)

    print(f"\n[SUCCÈS] {len(items)} item(s) exporté(s) : {output_xlsx}")
    print(f"    Feuilles : Résumé, Items, Par super-classe, Par collecte, Par classe brute, "
          f"Par split, Répartition classes x split")
    if split_alerts:
        print(f"\n⚠️  {len(split_alerts)} classe(s) totalement absente(s) de val et/ou test "
              f"(impossible à évaluer sur ce split) :")
        for msg in split_alerts:
            print(msg)
    return output_xlsx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic complet du dataset PixelOdyssey (export .xlsx)")
    parser.add_argument("--raw-dir", default=RAW_DIR_DEFAULT)
    parser.add_argument("--split-dir", default=SPLIT_DIR_DEFAULT,
                         help="Dossier de 2_split_dataset, pour joindre la colonne 'split' si le "
                              "manifeste existe déjà (optionnel, non bloquant si absent).")
    parser.add_argument("--output", default=OUTPUT_XLSX_DEFAULT)
    parser.add_argument("--skip-geo", action="store_true",
                         help="Ne tente aucune lecture de géoréférencement (plus rapide, colonnes "
                              "géoloc/surface vides) - utile pour un premier passage rapide.")
    parser.add_argument("--gsd-fixe-cm-px", type=float, default=GSD_FIXE_CM_PAR_PX_DEFAULT,
                         help=f"Résolution sol (cm/pixel) supposée UNIFORME sur tout le dataset, "
                              f"utilisée pour aire_cm2_estimee_gsd_fixe (défaut {GSD_FIXE_CM_PAR_PX_DEFAULT}) - "
                              f"appliquée même aux lots non géoréférencés. Ne remplace pas "
                              f"aire_m2_estimee (GSD réellement mesurée), qui reste plus fiable "
                              f"quand elle est disponible.")
    args = parser.parse_args()
    run_diagnostic(
        raw_dir=args.raw_dir,
        split_dir=args.split_dir,
        output_xlsx=args.output,
        skip_geo=args.skip_geo,
        gsd_fixe_cm_px=args.gsd_fixe_cm_px,
    )
