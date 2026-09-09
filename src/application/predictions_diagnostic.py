#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Diagnostic d'un run de prédiction, export .xlsx multi-feuilles.

Équivalent de `src/data/diagnostics/dataset_diagnostic.py`, mais sur les
PRÉDICTIONS d'un modèle (un run du pipeline application, mode batch ou
orthomosaïque) plutôt que sur la vérité terrain annotée : mêmes réflexes de
lecture (une feuille "Résumé", une feuille "Items" exhaustive, des feuilles
agrégées par super-classe), adaptés à ce qui a du sens côté prédictions -
pas de split train/val/test (une session de prédiction n'en a pas), mais une
confiance par détection (qui n'existe pas côté vérité terrain) et sa propre
feuille dédiée pour aider à choisir un seuil.

Lit `detections.geojson` (écrit par `run_application.py` en mode batch et
par `geo_density_map.py` en mode orthomosaïque - même schéma de propriétés
dans les deux cas, voir leurs docstrings) : JAMAIS relancé l'inférence,
comme `training_report.py`/`compare_runs.py` relisent `rapport_metrics.json`
plutôt que de réévaluer un modèle. Périmètre des détections : exactement
celles déjà dans `detections.geojson` (après dédoublonnage inter-photos et
seuil de confiance final pour le mode batch ; après fusion des recouvrements
de tuiles pour le mode orthomosaïque) - PAS les détections brutes
intermédiaires, qui ne sont pas persistées.

Mode détecté automatiquement depuis le contenu de detections.geojson : la
présence de la propriété `source_photo` sur les features signale le mode
batch (une photo source par détection) ; sinon (membre `ortho_source` au
niveau du FeatureCollection) le mode orthomosaïque, où la notion de "photo
source" n'existe pas (une seule orthomosaïque pour tout le run) - la feuille
"Par photo source" est alors omise plutôt que vide.

Entrée : --predictions-dir (dossier d'un run, ex:
`4. Results/2_prediction/batch_20260908_143000/`, doit contenir
detections.geojson), --output (.xlsx, défaut : predictions_diagnostic.xlsx
dans ce même dossier).
Sortie : un classeur .xlsx avec les feuilles :
  - Résumé            : mode détecté, nombre de détections, poids total
    estimé, couverture d'estimation de poids, nombre de classes représentées.
  - Items              : une ligne par détection retenue, toutes les colonnes
    disponibles (classe, confiance, aire, poids, position, photo source si
    mode batch).
  - Par super-classe    : agrégats (compte, confiance moyenne, aire moyenne,
    poids total) par classe détectée.
  - Par confiance       : répartition des détections par tranche de
    confiance (0.25-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) et par classe - LA
    feuille pour juger si le seuil de confiance actuel est bien placé (une
    classe avec beaucoup de détections proches du seuil mérite un second
    regard avant de conclure sur son taux de détection réel).
  - Par photo source (mode batch uniquement) : nombre de détections par
    photo - une photo avec un nombre de détections nettement au-dessus des
    autres est un candidat à vérifier en premier dans CVAT (faux positifs
    groupés plausibles, ex: reflet, motif répété).

Appelé automatiquement par `run_inference.py` (point d'entrée unique du
pipeline application) juste après la production de la carte, dans les deux
modes - pas besoin de le relancer à la main dans l'usage normal. Cette CLI
directe reste utile pour régénérer seulement ce .xlsx (ex: après un
ajustement de `CONFIDENCE_BINS`) sans relancer toute l'inférence.

Exemple :
    python -m src.application.predictions_diagnostic --predictions-dir "E:\\PixelOdyssey\\4. Results\\2_prediction\\batch_20260908_143000"
    python -m src.application.predictions_diagnostic --predictions-dir "E:\\PixelOdyssey\\4. Results\\2_prediction\\ortho_20260908_150000" --output diagnostic_ortho.xlsx
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

CONFIDENCE_BINS = [0.0, 0.25, 0.4, 0.6, 0.8, 1.0001]
CONFIDENCE_LABELS = ["< 0.25 (sous le seuil habituel)", "0.25-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]


def _load_detections(predictions_dir: Path) -> Tuple[List[Dict], bool, Dict]:
    """Charge detections.geojson et détecte le mode (batch si `source_photo`
    présent sur au moins une feature, sinon orthomosaïque).

    Sortie : (items bruts, is_batch_mode, meta) - `meta` reprend le membre
    `ortho_source` s'il existe (vide en mode batch)."""
    geojson_path = predictions_dir / "detections.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"{geojson_path} introuvable - --predictions-dir doit pointer vers un dossier de run déjà "
            f"produit par run_application.py ou geo_density_map.py."
        )
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    is_batch_mode = any("source_photo" in feat.get("properties", {}) for feat in features)
    meta = geojson.get("ortho_source", {})

    items = []
    for i, feat in enumerate(features):
        props = feat.get("properties", {})
        items.append({
            "detection_id": i + 1,
            "class_id": props.get("class_id"),
            "class_name": props.get("class_name"),
            "confidence": props.get("confidence"),
            "area_m2": props.get("area_m2"),
            "weight_kg": props.get("weight_kg"),
            "poids_estimable": props.get("weight_kg") is not None,
            "centroid_lon": props.get("centroid_lon"),
            "centroid_lat": props.get("centroid_lat"),
            "photo_source": props.get("source_photo") if is_batch_mode else None,
            "crop_file": props.get("crop_file"),
        })
    return items, is_batch_mode, meta


def _write_summary_sheet(writer, items: List[Dict], is_batch_mode: bool, meta: Dict,
                          predictions_dir: Path) -> None:
    from datetime import datetime

    n_total = len(items)
    n_avec_poids = sum(1 for it in items if it["poids_estimable"])
    poids_total = sum(it["weight_kg"] for it in items if it["weight_kg"] is not None)
    n_classes = len({it["class_name"] for it in items if it["class_name"] is not None})
    confiances = [it["confidence"] for it in items if it["confidence"] is not None]

    rows = [
        ("Généré le", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Dossier de run", str(predictions_dir)),
        ("Mode détecté", "Batch (photos brutes)" if is_batch_mode else "Orthomosaïque"),
    ]
    if not is_batch_mode and meta:
        rows.append(("Orthomosaïque source", meta.get("tif_path", "?")))
    rows.extend([
        ("Nombre de détections", n_total),
        ("Nombre de super-classes représentées", n_classes),
        ("Confiance moyenne", round(sum(confiances) / len(confiances), 4) if confiances else None),
        ("Confiance minimale", round(min(confiances), 4) if confiances else None),
        ("Détections avec poids estimable", f"{n_avec_poids}/{n_total}"),
        ("Poids total estimé (kg, PROVISOIRE - voir weight_estimation.py)", round(poids_total, 3)),
    ])
    df = pd.DataFrame(rows, columns=["Indicateur", "Valeur"])
    df.to_excel(writer, sheet_name="Résumé", index=False)


def _write_per_class_sheet(writer, df_items: pd.DataFrame) -> None:
    if df_items.empty:
        return
    rows = []
    n_total = len(df_items)
    for class_name, g in df_items.groupby("class_name"):
        g_poids = g[g["poids_estimable"]]
        rows.append({
            "super_classe": class_name,
            "n_detections": len(g),
            "pct_du_total": round(100.0 * len(g) / n_total, 1),
            "confiance_moyenne": round(g["confidence"].mean(), 4),
            "confiance_mediane": round(g["confidence"].median(), 4),
            "aire_m2_moyenne": round(g["area_m2"].mean(), 4) if g["area_m2"].notna().any() else None,
            "n_poids_estimable": len(g_poids),
            "poids_total_kg": round(g_poids["weight_kg"].sum(), 3) if len(g_poids) else 0.0,
        })
    out = pd.DataFrame(rows).sort_values("n_detections", ascending=False)
    out.to_excel(writer, sheet_name="Par super-classe", index=False)


def _write_confidence_sheet(writer, df_items: pd.DataFrame) -> None:
    """Répartition des détections par tranche de confiance x super-classe -
    voir la docstring du module : LA feuille pour juger si le seuil de
    confiance actuel du run est bien placé. Une classe concentrée juste
    au-dessus du seuil (ex: beaucoup de 0.25-0.4) mérite un second regard
    dans CVAT avant de conclure sur son taux de détection réel."""
    if df_items.empty:
        return
    df = df_items.assign(
        tranche_confiance=pd.cut(df_items["confidence"], bins=CONFIDENCE_BINS, labels=CONFIDENCE_LABELS,
                                  right=False, include_lowest=True)
    )
    out = (
        df.groupby(["class_name", "tranche_confiance"], observed=False)
        .size()
        .reset_index(name="n_detections")
        .rename(columns={"class_name": "super_classe"})
    )
    out = out.sort_values(["super_classe", "tranche_confiance"])
    out.to_excel(writer, sheet_name="Par confiance", index=False)


def _write_per_photo_sheet(writer, df_items: pd.DataFrame) -> None:
    """Mode batch uniquement (voir run_diagnostic) - une photo avec beaucoup
    plus de détections que les autres est un candidat à vérifier en premier
    dans CVAT (faux positifs groupés plausibles)."""
    if df_items.empty:
        return
    rows = []
    for photo, g in df_items.groupby("photo_source"):
        rows.append({
            "photo_source": photo,
            "n_detections": len(g),
            "confiance_moyenne": round(g["confidence"].mean(), 4),
            "classes_presentes": ", ".join(f"{c}({n})" for c, n in g["class_name"].value_counts().items()),
        })
    out = pd.DataFrame(rows).sort_values("n_detections", ascending=False)
    out.to_excel(writer, sheet_name="Par photo source", index=False)


def run_diagnostic(predictions_dir: str, output_xlsx: str = None) -> str:
    predictions_dir = Path(predictions_dir)
    items, is_batch_mode, meta = _load_detections(predictions_dir)
    if not items:
        print("❌ Aucune détection trouvée dans detections.geojson - rien à diagnostiquer.")
        return ""

    if output_xlsx is None:
        output_xlsx = predictions_dir / "predictions_diagnostic.xlsx"
    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    df_items = pd.DataFrame(items)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        _write_summary_sheet(writer, items, is_batch_mode, meta, predictions_dir)
        df_items.drop(columns=["photo_source"] if not is_batch_mode else []).to_excel(
            writer, sheet_name="Items", index=False
        )
        _write_per_class_sheet(writer, df_items)
        _write_confidence_sheet(writer, df_items)
        if is_batch_mode:
            _write_per_photo_sheet(writer, df_items)

    sheets = "Résumé, Items, Par super-classe, Par confiance"
    if is_batch_mode:
        sheets += ", Par photo source"
    print(f"\n[SUCCÈS] {len(items)} détection(s) exportée(s) : {output_xlsx}")
    print(f"    Feuilles : {sheets}")
    return str(output_xlsx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", required=True,
                         help="Dossier d'un run de prédiction (batch_*/ortho_*, doit contenir detections.geojson).")
    parser.add_argument("--output", default=None,
                         help="Fichier .xlsx de sortie (défaut : predictions_diagnostic.xlsx dans --predictions-dir).")
    args = parser.parse_args()
    run_diagnostic(predictions_dir=args.predictions_dir, output_xlsx=args.output)
