#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Export d'un run de prédiction vers un lot au format d'import
CVAT (mêmes conventions que `assisted_annotate.py`/`split_for_cvat.py` :
`images/train/` + `labels/train/` + `data.yaml`), pour revoir visuellement
TOUTES les prédictions d'un run - et notamment repérer les ratés (faux
positifs à corriger, mais aussi les zones qui auraient dû contenir une
détection et n'en ont aucune, visibles en parcourant les images elles-mêmes
dans CVAT) - sans passer par l'interface pas-à-pas de
`assisted_annotate.py` (pensée pour VALIDER un nouveau lot d'annotation, pas
pour un simple export en lecture).

Lit `detections.geojson` (voir `run_application.py`/`geo_density_map.py`) :
JAMAIS relancé l'inférence. Périmètre des détections exporté : exactement
celles déjà dans `detections.geojson` (après dédoublonnage/seuil pour le
mode batch, après fusion des recouvrements de tuiles pour le mode
orthomosaïque) - voir `predictions_diagnostic.py` pour le même choix côté
export .xlsx.

Mode détecté automatiquement (voir `_load_predictions`) :
- BATCH : une image par photo source, copiée/réencodée telle quelle SAUF si
  une photo dépasse à elle seule `CVAT_MAX_PIXELS` (capteur à très haute
  résolution) - alors découpée en grille comme le mode orthomosaïque
  ci-dessous, via la même fonction partagée (`cvat_export.write_grid_lot`).
- ORTHOMOSAÏQUE : la totalité du GeoTIFF source est PRESQUE TOUJOURS trop
  lourde pour un import CVAT direct (voir `CVAT_MAX_PIXELS`,
  `config/`... - non, voir `split_for_cvat.py`) - découpée en grille dont
  chaque morceau reste sous ce plafond, chaque polygone de détection étant
  recadré au(x) morceau(x) qu'il recoupe (voir `cvat_export.clip_polygon_to_window`).

Chaque image (PNG, réencodée depuis la source pour les DEUX modes - jamais
une recompression JPEG-vers-JPEG à qualité dégradée, voir cv2.imwrite dans
`cvat_export.write_grid_lot`) reçoit tous ses labels EN UNE FOIS (pas de
revue interactive ici, contrairement à assisted_annotate.py) : un fichier
`.txt` par image, toutes les détections qui la concernent déjà dedans.

Entrée : --predictions-dir (dossier d'un run, doit contenir
detections.geojson), --output-dir (dossier du lot CVAT à créer, défaut :
cvat_export/ dans --predictions-dir), --n-pieces (force un nombre de
morceaux par image/orthomosaïque - sinon calculé automatiquement pour
respecter CVAT_MAX_PIXELS, voir cvat_export.resolve_n_pieces).
Sortie : lot CVAT prêt à importer (images/train + labels/train + data.yaml).

Appelé automatiquement par `run_inference.py` juste après la production de
la carte, dans les deux modes - qui compresse ensuite le lot en .zip
(racine du zip = contenu de `cvat_export/`, prêt à l'import CVAT direct) et
supprime le dossier non compressé, sauf `--no-zip-cvat-export` (voir
`run_inference.py`). CE module lui-même n'effectue jamais cette compression
- elle vit dans `run_inference.py`, pas ici, pour garder ce module (comme
`run_application.py`/`geo_density_map.py`) focalisé sur UNE seule
responsabilité. Cette CLI directe reste utile pour régénérer seulement ce
lot (ex: après un changement de `--n-pieces`) sans relancer toute
l'inférence.

Exemple :
    python -m src.application.export_predictions_to_cvat --predictions-dir "E:\\PixelOdyssey\\4. Results\\2_prediction\\batch_20260908_143000"
    python -m src.application.export_predictions_to_cvat --predictions-dir "E:\\PixelOdyssey\\4. Results\\2_prediction\\ortho_20260908_150000" --output-dir "E:\\PixelOdyssey\\4. Results\\2_prediction\\ortho_20260908_150000\\cvat_export"
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.windows
from PIL import Image

from src.review.cvat_export import write_grid_lot


def _load_predictions(predictions_dir: Path) -> Tuple[List[Dict], bool, Dict, Dict[int, str]]:
    """Charge detections.geojson - même détection de mode que
    `predictions_diagnostic._load_detections` (présence de `source_photo`
    sur les features -> batch, sinon membre `ortho_source` -> orthomosaïque).

    Sortie : (features brutes, is_batch_mode, meta_ortho, names) - `names`
    (class_id -> class_name) construit à partir des paires effectivement
    vues dans le run, pas re-résolu depuis config/data_config*.yaml (ce
    dernier a pu changer depuis - le run fait foi sur ce qu'il a VRAIMENT
    utilisé)."""
    geojson_path = predictions_dir / "detections.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"{geojson_path} introuvable - --predictions-dir doit pointer vers un dossier de run déjà "
            f"produit par run_application.py ou geo_density_map.py."
        )
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    if not features:
        return [], False, {}, {}

    is_batch_mode = any("source_photo" in feat.get("properties", {}) for feat in features)
    meta = geojson.get("ortho_source", {})

    names: Dict[int, str] = {}
    for feat in features:
        props = feat["properties"]
        names[int(props["class_id"])] = props["class_name"]

    return features, is_batch_mode, meta, names


def _export_batch(features: List[Dict], names: Dict[int, str], lot_dir: Path,
                   n_pieces: Optional[int]) -> int:
    """Un morceau de grille (ou une image entière si la photo tient sous
    CVAT_MAX_PIXELS, cas normal pour une photo drone) par PHOTO SOURCE
    distincte - regroupe d'abord toutes les détections par
    `source_photo_path` (persistée dans detections.geojson par
    run_application.py, additive - voir sa docstring)."""
    by_photo: Dict[str, List[Dict]] = defaultdict(list)
    for feat in features:
        props = feat["properties"]
        by_photo[props["source_photo_path"]].append(props)

    n_total_pieces = 0
    for photo_path, props_list in by_photo.items():
        photo_path = Path(photo_path)
        img_w = props_list[0]["photo_width_px"]
        img_h = props_list[0]["photo_height_px"]
        detections = [
            ([tuple(pt) for pt in p["pixel_polygon"]], int(p["class_id"]))
            for p in props_list
        ]

        with Image.open(photo_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            full_array = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR (convention OpenCV du reste du pipeline)

        def _read_window(x0, y0, x1, y1, _arr=full_array):
            return np.ascontiguousarray(_arr[y0:y1, x0:x1])

        n_pieces_written = write_grid_lot(
            lot_dir, photo_path.stem, img_w, img_h, detections, _read_window, names, n_pieces=n_pieces,
        )
        n_total_pieces += n_pieces_written
        print(f"    {photo_path.name} : {len(detections)} détection(s), {n_pieces_written} morceau(x) écrit(s).")

    return n_total_pieces


def _export_ortho(features: List[Dict], meta: Dict, names: Dict[int, str], lot_dir: Path,
                   n_pieces: Optional[int]) -> int:
    """La totalité du GeoTIFF est traitée comme UNE SEULE source à découper
    en grille (voir cvat_export.write_grid_lot) - `meta` (ortho_source, voir
    geo_density_map.py) donne le chemin et les dimensions, jamais recalculés
    ici (pas besoin de rouvrir le modèle ni de rescanner le fichier avant
    d'en avoir besoin pixel par pixel)."""
    tif_path = meta["tif_path"]
    img_w, img_h = meta["width_px"], meta["height_px"]
    detections = [
        ([tuple(pt) for pt in feat["properties"]["pixel_polygon"]], int(feat["properties"]["class_id"]))
        for feat in features
    ]

    with rasterio.open(tif_path) as src:
        band_count = src.count
        bands = [1, 2, 3] if band_count >= 3 else [1]

        def _read_window(x0, y0, x1, y1, _src=src, _bands=bands):
            win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
            data = _src.read(_bands, window=win)
            if len(_bands) == 1:
                data = np.repeat(data, 3, axis=0)
            arr = np.transpose(data, (1, 2, 0))  # (h, w, bands), ordre R,G,B
            return np.ascontiguousarray(arr[:, :, ::-1])  # RGB -> BGR

        n_pieces_written = write_grid_lot(
            lot_dir, Path(tif_path).stem, img_w, img_h, detections, _read_window, names, n_pieces=n_pieces,
        )

    print(f"    {Path(tif_path).name} : {len(detections)} détection(s), {n_pieces_written} morceau(x) écrit(s).")
    return n_pieces_written


def export_predictions_to_cvat(predictions_dir: str, output_dir: str = None,
                                n_pieces: Optional[int] = None) -> str:
    predictions_dir = Path(predictions_dir)
    features, is_batch_mode, meta, names = _load_predictions(predictions_dir)
    if not features:
        print("❌ Aucune détection trouvée dans detections.geojson - rien à exporter.")
        return ""

    lot_dir = Path(output_dir) if output_dir else predictions_dir / "cvat_export"
    lot_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- 📤 Export CVAT ({'batch' if is_batch_mode else 'orthomosaïque'}) : {predictions_dir} ---")
    if is_batch_mode:
        n_total = _export_batch(features, names, lot_dir, n_pieces)
    else:
        n_total = _export_ortho(features, meta, names, lot_dir, n_pieces)

    print(f"\n[SUCCÈS] {len(features)} détection(s), {n_total} morceau(x) image écrit(s) : {lot_dir}")
    print(f"    Prêt à importer dans CVAT (format \"Ultralytics YOLO Segmentation\", "
          f"images/train + labels/train + data.yaml).")
    return str(lot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", required=True,
                         help="Dossier d'un run de prédiction (batch_*/ortho_*, doit contenir detections.geojson).")
    parser.add_argument("--output-dir", default=None,
                         help="Dossier du lot CVAT à créer (défaut : cvat_export/ dans --predictions-dir).")
    parser.add_argument("--n-pieces", type=int, default=None,
                         help="Force ce nombre de morceaux par image/orthomosaïque (refuse si insuffisant pour "
                              "respecter CVAT_MAX_PIXELS) - omis par défaut, calculé automatiquement.")
    args = parser.parse_args()
    export_predictions_to_cvat(
        predictions_dir=args.predictions_dir, output_dir=args.output_dir, n_pieces=args.n_pieces,
    )
