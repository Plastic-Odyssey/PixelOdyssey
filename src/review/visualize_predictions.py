#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Visionneuse image par image des prédictions du modèle.

Fait tourner le modèle sur les images parentes une par une (même géométrie
de tuilage qu'à l'entraînement - voir tiled_inference.py), superpose GT et
prédictions en couleur selon leur statut, et génère une page HTML locale
pour naviguer image par image (boutons + flèches du clavier), avec un tri
par nombre de ratés pour aller directement aux pires cas.

Outil purement diagnostic - contrairement à label_review.py, il n'écrit
jamais rien dans 1_annotated_dataset ni dans un export destiné à CVAT. Il
sert à regarder, pas à corriger : les annotations à corriger repérées en le
parcourant se font remonter vers CVAT via label_review.py.

Réutilise l'appariement GT<->prédictions de src/review/matching.py (le même
que label_review.py) :
  - GT et prédiction appariées, IoU >= iou_mismatch_threshold -> VERT (réussi)
  - GT et prédiction appariées, IoU < iou_mismatch_threshold  -> ORANGE
    (le modèle a vu l'objet mais le masque est mal ajusté - même seuil que le
    panier C de label_review.py)
  - GT sans prédiction correspondante                          -> ROUGE
    (raté - faux négatif)
  - Prédiction sans GT correspondante, confiance >= conf_threshold
                                                                 -> BLEU
    (fausse alerte - faux positif). Une prédiction sous ce seuil n'est pas
    dessinée du tout (bruit de tuile).

Entrée : manifeste de split (parent_manifest.json), images brutes, et soit
un modèle entraîné (best.pt) soit un predict_tile_fn injecté.
Sortie : page HTML interactive sous <output_dir>/<run_id>/index.html.

Taxonomie du modèle visualisé (ajouté le 03/09/2026, suite au dataset mono-classe) :
la taxonomie n'est PAS déduite du modèle lui-même - elle vient de `--config`
(défaut : config/data_config.yaml, le référentiel 7-classes principal), qui
DOIT être le MÊME fichier que celui utilisé pour entraîner le modèle passé à
--model (voir train.py --config / data_pipeline.py --config). Le garde-fou
`assert_model_matches_taxonomy` compare `model.names` (embarqué dans les
poids) à cette taxonomie et lève une erreur claire en cas de mismatch, plutôt
que de scrambler silencieusement les noms de classe affichés. De même,
`--split-dir` doit pointer vers le manifeste DU MÊME dataset (ex:
2_split_dataset_mono_class pour un modèle mono-classe) - sinon les parent_id
du manifeste ne correspondent à rien de cohérent pour ce modèle.

Usage :
    python -m src.review.visualize_predictions
        -> sélection interactive du modèle (comme label_review.py), scope
           val par défaut, taxonomie 7-classes par défaut, ouvre ensuite
           output/../6_prediction_viewer/<run>/index.html dans un navigateur.
    python -m src.review.visualize_predictions --scope test --batch "SL"
    python -m src.review.visualize_predictions --model output/runs/<run>/weights/best.pt --limit 40

    Modèle mono-classe (taxonomie ET split différents du 7-classes par défaut) :
    python -m src.review.visualize_predictions \\
        --model output/runs/mono_class_tuned_recipe_patience0_yolo11n-seg_.../weights/best.pt \\
        --config config/data_config_mono_class.yaml \\
        --split-dir "E:\\PixelOdyssey\\3. Processed dataset\\2_split_dataset_mono_class" \\
        --scope test --run-id mono_class_tuned_recipe_test
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    assert_model_matches_taxonomy,
    load_batch_local_names,
    load_class_config,
)
from src.data.image_io import load_image_bgr
from src.data.split_dataset import PARENT_MANIFEST_FILENAME, RAW_DIR, SPLIT_DIR
from src.review.label_review import _discover_available_models, _load_gt_objects, _prompt_model_choice, RUNS_DIR
from src.review.matching import match_gt_to_predictions
from src.review.tiled_inference import make_ultralytics_predict_fn, predict_parent_image

import cv2
import numpy as np

BASE_DIR = r"E:\PixelOdyssey\3. Processed dataset"
VIEWER_DIR = os.path.join(BASE_DIR, "6_prediction_viewer")

# Couleurs BGR (convention OpenCV) - un statut = une couleur, jamais une classe
# = une couleur : ce qu'on veut voir d'un coup d'oeil ici c'est RÉUSSI/RATÉ,
# pas la taxonomie (déjà couverte par le texte du label sur chaque forme).
COLOR_TP_STRONG = (60, 180, 60)     # vert - GT et prédiction bien appariées
COLOR_TP_WEAK = (0, 140, 255)       # orange - appariées mais IoU faible (masque à ajuster)
COLOR_FN = (40, 40, 230)            # rouge - GT ratée (faux négatif)
COLOR_FP = (230, 130, 20)           # bleu - détection en trop (faux positif)

LEGEND = [
    ("Réussi (GT + prédiction bien appariées)", COLOR_TP_STRONG),
    ("Appariée mais masque mal ajusté", COLOR_TP_WEAK),
    ("Raté - annotation sans détection (faux négatif)", COLOR_FN),
    ("Fausse alerte - détection sans annotation (faux positif)", COLOR_FP),
]


def _bgr_to_css(color: Tuple[int, int, int]) -> str:
    b, g, r = color
    return f"rgb({r},{g},{b})"


def _draw_polygon(img: np.ndarray, coords, color: Tuple[int, int, int], alpha: float = 0.28) -> None:
    pts = np.array([[int(round(x)), int(round(y))] for x, y in coords], dtype=np.int32).reshape(-1, 1, 2)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)


def _put_label(img: np.ndarray, text: str, x: float, y: float, color: Tuple[int, int, int]) -> None:
    org = (max(0, int(x)), max(12, int(y) - 6))
    # Contour noir puis texte en couleur par-dessus - lisible sur n'importe
    # quel fond, pas besoin de deviner une couleur de contraste par image.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_overlay(
    img: np.ndarray,
    match_results,
    target_names: Dict[int, str],
    iou_mismatch_threshold: float,
    conf_threshold: float,
    max_display_dim: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Dessine GT/prédictions colorées selon leur statut sur une COPIE de `img`
    (jamais l'original), puis redimensionne pour un affichage rapide dans le
    navigateur (les images sources - orthomosaïques/photos drone - peuvent
    faire plusieurs milliers de pixels de large, inutile de garder ça pour un
    simple coup d'oeil visuel).
    """
    out = img.copy()
    counts = {"tp_strong": 0, "tp_weak": 0, "fn": 0, "fp": 0}

    for r in match_results:
        if r.gt is not None and r.pred is not None:
            strong = r.iou >= iou_mismatch_threshold
            color = COLOR_TP_STRONG if strong else COLOR_TP_WEAK
            counts["tp_strong" if strong else "tp_weak"] += 1
            coords = list(r.pred.geom.exterior.coords)
            _draw_polygon(out, coords, color)
            name = target_names.get(r.pred.class_id, r.pred.class_id)
            conf = r.pred.confidence if r.pred.confidence is not None else 0.0
            x, y = coords[0]
            _put_label(out, f"{name} {conf:.2f} (IoU {r.iou:.2f})", x, y, color)
        elif r.gt is not None and r.pred is None:
            counts["fn"] += 1
            coords = list(r.gt.geom.exterior.coords)
            _draw_polygon(out, coords, COLOR_FN)
            name = target_names.get(r.gt.class_id, r.gt.class_id)
            x, y = coords[0]
            _put_label(out, f"{name} (raté)", x, y, COLOR_FN)
        elif r.gt is None and r.pred is not None:
            if r.pred.confidence is None or r.pred.confidence < conf_threshold:
                continue  # bruit de tuile sous le seuil métier - jamais proposé nulle part, pas dessiné
            counts["fp"] += 1
            coords = list(r.pred.geom.exterior.coords)
            _draw_polygon(out, coords, COLOR_FP)
            name = target_names.get(r.pred.class_id, r.pred.class_id)
            x, y = coords[0]
            _put_label(out, f"{name} {r.pred.confidence:.2f} (faux positif)", x, y, COLOR_FP)

    h, w = out.shape[:2]
    scale = min(1.0, max_display_dim / max(h, w))
    if scale < 1.0:
        out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return out, counts


def _select_parents_by_scope(parent_manifest: Dict[str, Dict], scope: str, batch_filter: Optional[str]) -> List[Tuple[str, Dict]]:
    items = list(parent_manifest.items())
    if scope != "all":
        items = [(pid, info) for pid, info in items if info["split"] == scope]
    if batch_filter:
        needle = batch_filter.lower()
        items = [(pid, info) for pid, info in items if needle in info["batch"].lower()]
    items.sort(key=lambda kv: kv[0])  # ordre déterministe - important pour --limit reproductible
    return items


def _build_index_html(records: List[Dict]) -> str:
    legend_html = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{_bgr_to_css(c)}"></span>{label}</span>'
        for label, c in LEGEND
    )
    records_json = json.dumps(records, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>PixelOdyssey - Visionneuse de prédictions</title>
<style>
  body {{ margin:0; padding:0; background:#111; color:#eee; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  #topbar {{ display:flex; align-items:center; gap:16px; padding:10px 16px; background:#1a1a1a; border-bottom:1px solid #333; flex-wrap:wrap; }}
  #topbar select, #topbar input, #topbar button {{ background:#2a2a2a; color:#eee; border:1px solid #444; border-radius:4px; padding:6px 10px; font-size:13px; }}
  #topbar button {{ cursor:pointer; }}
  #topbar button:hover {{ background:#3a3a3a; }}
  #legend {{ display:flex; gap:14px; padding:8px 16px; font-size:12px; color:#bbb; flex-wrap:wrap; background:#161616; border-bottom:1px solid #2a2a2a; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; }}
  .swatch {{ width:12px; height:12px; border-radius:2px; display:inline-block; }}
  #caption {{ padding:10px 16px; font-size:14px; }}
  #caption b {{ color:#fff; }}
  #counts span {{ margin-right:14px; }}
  #imgwrap {{ display:flex; justify-content:center; align-items:center; padding:8px 16px 24px; }}
  #imgwrap img {{ max-width:100%; max-height:78vh; border:1px solid #333; border-radius:4px; }}
  #posbar {{ padding: 0 16px 14px; font-size:12px; color:#999; }}
  a {{ color:#8ab4f8; }}
</style>
</head>
<body>
<div id="topbar">
  <button id="prevBtn">&larr; Précédente</button>
  <button id="nextBtn">Suivante &rarr;</button>
  <label>Trier par :
    <select id="sortSel">
      <option value="order">Ordre du dataset</option>
      <option value="fn">Plus de ratés (FN) d'abord</option>
      <option value="fp">Plus de fausses alertes (FP) d'abord</option>
      <option value="weak">Plus de masques mal ajustés d'abord</option>
    </select>
  </label>
  <label>Filtrer (lot/parent_id) : <input id="filterInput" placeholder="ex: SL, SB 3, transect_11..."></label>
  <span id="posbar"></span>
</div>
<div id="legend">{legend_html}</div>
<div id="caption"></div>
<div id="imgwrap"><img id="mainImg" src="" alt=""></div>

<script>
const ALL_RECORDS = {records_json};
let records = ALL_RECORDS.slice();
let idx = 0;

function scoreFor(rec, key) {{
  if (key === 'fn') return rec.fn;
  if (key === 'fp') return rec.fp;
  if (key === 'weak') return rec.tp_weak;
  return 0;
}}

function applySortAndFilter() {{
  const sortKey = document.getElementById('sortSel').value;
  const filterVal = document.getElementById('filterInput').value.trim().toLowerCase();
  records = ALL_RECORDS.filter(r => !filterVal || r.parent_id.toLowerCase().includes(filterVal) || r.batch.toLowerCase().includes(filterVal));
  if (sortKey !== 'order') {{
    records = records.slice().sort((a, b) => scoreFor(b, sortKey) - scoreFor(a, sortKey));
  }}
  idx = 0;
  render();
}}

function render() {{
  if (records.length === 0) {{
    document.getElementById('caption').innerHTML = '<i>Aucune image ne correspond au filtre.</i>';
    document.getElementById('mainImg').src = '';
    document.getElementById('posbar').textContent = '';
    return;
  }}
  const r = records[idx];
  document.getElementById('mainImg').src = r.file;
  document.getElementById('caption').innerHTML =
    `<b>${{r.parent_id}}</b> - lot ${{r.batch}} - split ${{r.split}}` +
    `<div id="counts">` +
    `<span style="color:#3cb43c">Réussies : ${{r.tp_strong}}</span>` +
    `<span style="color:#ff8c00">Masques à ajuster : ${{r.tp_weak}}</span>` +
    `<span style="color:#e62828">Ratées (FN) : ${{r.fn}}</span>` +
    `<span style="color:#e68214">Fausses alertes (FP) : ${{r.fp}}</span>` +
    `</div>`;
  document.getElementById('posbar').textContent = `Image ${{idx + 1}} / ${{records.length}}`;
}}

document.getElementById('prevBtn').addEventListener('click', () => {{ idx = (idx - 1 + records.length) % records.length; render(); }});
document.getElementById('nextBtn').addEventListener('click', () => {{ idx = (idx + 1) % records.length; render(); }});
document.getElementById('sortSel').addEventListener('change', applySortAndFilter);
document.getElementById('filterInput').addEventListener('input', applySortAndFilter);
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') {{ idx = (idx - 1 + records.length) % records.length; render(); }}
  if (e.key === 'ArrowRight') {{ idx = (idx + 1) % records.length; render(); }}
}});

render();
</script>
</body>
</html>
"""


def run_visualize(
    model_path: Optional[str] = None,
    scope: str = "val",
    batch_filter: Optional[str] = None,
    limit: Optional[int] = None,
    raw_dir: str = RAW_DIR,
    split_dir: str = SPLIT_DIR,
    class_config_path=DEFAULT_CLASS_CONFIG_PATH,
    output_dir: str = VIEWER_DIR,
    tile_size: int = 640,
    overlap: int = 256,
    tile_conf_threshold: float = 0.25,
    conf_threshold: float = 0.25,
    iou_mismatch_threshold: float = 0.5,
    min_match_iou: float = 0.1,
    max_display_dim: int = 1600,
    run_id: Optional[str] = None,
    predict_tile_fn=None,
) -> Dict:
    """Voir la docstring du module. `predict_tile_fn` : même rôle qu'en
    label_review.py, pour les tests (faux modèle, pas de poids réels).

    `class_config_path` (ajouté le 03/09/2026, pour rendre l'outil utilisable
    sur un modèle mono-classe sans le forcer sur le référentiel 7-classes) :
    référentiel de classes à utiliser pour résoudre les noms affichés ET pour
    le garde-fou anti-mismatch (`assert_model_matches_taxonomy` ci-dessous) -
    DOIT être le même fichier que celui utilisé pour entraîner `model_path`.
    `split_dir` doit, de la même façon, pointer vers le manifeste DU MÊME
    dataset que ce modèle (ex: 2_split_dataset_mono_class)."""
    if predict_tile_fn is None:
        if not model_path:
            available_models = _discover_available_models(RUNS_DIR)
            if not available_models:
                raise RuntimeError(
                    f"Aucun modèle entraîné trouvé sous {RUNS_DIR} (aucun */weights/best.pt). "
                    f"Lance d'abord un entraînement (python -m src.training.train), ou fournis "
                    f"--model explicitement si le modèle se trouve ailleurs."
                )
            model_path = _prompt_model_choice(available_models, RUNS_DIR)
        predict_tile_fn = make_ultralytics_predict_fn(model_path, conf_threshold=tile_conf_threshold)

    class_taxonomy, target_names = load_class_config(class_config_path)

    # Garde-fou : voir la même vérification dans label_review.py. Absent pour un
    # predict_tile_fn injecté en test (pas d'attribut model_names).
    model_names = getattr(predict_tile_fn, "model_names", None)
    if model_names is not None:
        assert_model_matches_taxonomy(model_names, target_names, model_label=str(model_path or ""))

    manifest_path = Path(split_dir) / PARENT_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise RuntimeError(
            f"Manifeste introuvable : {manifest_path}. Lance d'abord split_dataset.py "
            f"(ou tout le pipeline via data_pipeline.py) pour le générer."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        parent_manifest = json.load(f)

    selected = _select_parents_by_scope(parent_manifest, scope, batch_filter)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        print(f"❌ Aucune image sélectionnée (scope='{scope}', batch_filter={batch_filter!r}). Rien à faire.")
        return {"images_processed": 0, "failed_images": []}

    if run_id is None:
        from datetime import datetime

        run_id = datetime.now().strftime("viz_%Y%m%d_%H%M%S")
    out_root = Path(output_dir) / run_id
    img_out_dir = out_root / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- 👁️  VISIONNEUSE DE PRÉDICTIONS ({len(selected)} image(s), scope='{scope}') ---")
    print(f"    Modèle : {model_path if model_path else '(predict_tile_fn injecté - test/appel programmatique)'}")
    print(f"    Sortie : {out_root}")

    local_names_by_batch: Dict[str, Dict[int, str]] = {}
    records: List[Dict] = []
    failed_images: List[Dict] = []
    totals = {"tp_strong": 0, "tp_weak": 0, "fn": 0, "fp": 0}

    for parent_id, info in selected:
        batch_name = info["batch"]
        split = info["split"]
        raw_img_path = Path(info["raw_img_path"])
        raw_label_path = Path(info["raw_label_path"])

        try:
            if batch_name not in local_names_by_batch:
                local_yaml = Path(raw_dir) / batch_name / "data.yaml"
                local_names_by_batch[batch_name] = load_batch_local_names(local_yaml)
            local_names = local_names_by_batch[batch_name]

            img = load_image_bgr(raw_img_path)
            if img is None:
                print(f"  ⚠️  Image illisible, ignorée : {raw_img_path}")
                continue
            img_h, img_w = img.shape[:2]

            gt_objects = _load_gt_objects(raw_label_path, local_names, class_taxonomy, img_w, img_h)
            predictions = predict_parent_image(raw_img_path, predict_tile_fn, tile_size=tile_size, overlap=overlap)
            results = match_gt_to_predictions(gt_objects, predictions, min_iou=min_match_iou)

            annotated, counts = _draw_overlay(
                img, results, target_names, iou_mismatch_threshold, conf_threshold, max_display_dim
            )
            out_file = img_out_dir / f"{parent_id}.jpg"
            cv2.imwrite(str(out_file), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

            for k in totals:
                totals[k] += counts[k]

            records.append({
                "file": f"images/{parent_id}.jpg",
                "parent_id": parent_id,
                "batch": batch_name,
                "split": split,
                **counts,
            })
            print(
                f"  ✓ {parent_id} [{batch_name}/{split}] - "
                f"réussi:{counts['tp_strong']} ajuster:{counts['tp_weak']} "
                f"raté:{counts['fn']} faux_positif:{counts['fp']}"
            )

        except Exception as e:  # noqa: BLE001 - une image en cause ne doit jamais arrêter les autres
            print(f"  ❌ Échec sur {raw_img_path} (lot '{batch_name}') - ignorée : {e}")
            failed_images.append({"batch": batch_name, "image": str(raw_img_path), "error": str(e)})

    html_doc = _build_index_html(records)
    index_path = out_root / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print("\n=== RÉSUMÉ ===")
    print(f"  • Images traitées : {len(records)}")
    print(f"  • Réussies (TP)                    : {totals['tp_strong']}")
    print(f"  • Masques à ajuster (TP, IoU faible): {totals['tp_weak']}")
    print(f"  • Ratées (FN)                      : {totals['fn']}")
    print(f"  • Fausses alertes (FP)             : {totals['fp']}")
    if failed_images:
        print(f"  • ⚠️  Images en échec (ignorées) : {len(failed_images)}")
        for fi in failed_images:
            print(f"      - [{fi['batch']}] {fi['image']}\n        {fi['error']}")
    print(f"\n  -> Ouvre {index_path} dans un navigateur pour naviguer image par image.")

    return {
        "images_processed": len(records),
        "totals": totals,
        "failed_images": failed_images,
        "index_path": str(index_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visionneuse image par image des prédictions PixelOdyssey")
    parser.add_argument(
        "--model", default=None,
        help="Chemin vers le modèle entraîné (best.pt). Optionnel : si omis, sélection interactive "
             "parmi les modèles trouvés sous output/runs/*/weights/best.pt (comme label_review.py).",
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CLASS_CONFIG_PATH),
        help="Référentiel de classes à utiliser (défaut : config/data_config.yaml, le 7-classes "
             "principal) - DOIT être le même fichier que celui utilisé pour entraîner --model. "
             "Ex: config/data_config_mono_class.yaml pour un modèle mono-classe. Un mismatch entre "
             "--model et --config est bloqué par un garde-fou explicite (assert_model_matches_taxonomy), "
             "pas silencieusement scrambé.",
    )
    parser.add_argument(
        "--split-dir", default=SPLIT_DIR,
        help=f"Dossier du split à visualiser, DOIT correspondre au dataset utilisé par --model (défaut : "
             f"{SPLIT_DIR}). Ex: .../2_split_dataset_mono_class pour un modèle mono-classe.",
    )
    parser.add_argument(
        "--raw-dir", default=RAW_DIR,
        help=f"Dossier de donnée brute source (défaut : {RAW_DIR}) - partagé entre toutes les variantes "
             f"de taxonomie/split (seul 1_annotated_dataset reste commun, voir data_pipeline.py), à "
             f"changer seulement pour une variante de donnée brute (ex: 1bis_corrected_annotation).",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Nom du sous-dossier de sortie sous 6_prediction_viewer/ (défaut : horodatage "
             "'viz_AAAAMMJJ_HHMMSS'). Utile pour retrouver facilement un run vu son nom plutôt qu'un "
             "horodatage, surtout avec plusieurs taxonomies mélangées dans le même dossier - ex: "
             "--run-id mono_class_tuned_recipe_test.",
    )
    parser.add_argument("--scope", choices=["train", "val", "test", "all"], default="val",
                        help="Quel(s) split(s) visualiser (défaut: val).")
    parser.add_argument("--batch", default=None,
                        help="Filtre optionnel (sous-chaîne, insensible à la casse) sur le nom du lot, "
                             "ex: --batch SL pour ne visualiser que les lots SL.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Nombre maximal d'images à traiter (défaut : toutes celles du scope/filtre).")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--tile-conf-threshold", type=float, default=0.25,
                        help="Seuil de confiance large appliqué par tuile avant fusion (défaut 0.25).")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="Seuil de confiance en dessous duquel une prédiction SANS GT correspondant "
                             "n'est même pas dessinée (bruit de tuile, défaut 0.25).")
    parser.add_argument("--iou-mismatch-threshold", type=float, default=0.5,
                        help="IoU en dessous de laquelle une paire appariée est dessinée en orange "
                             "(masque à ajuster) plutôt qu'en vert (défaut 0.5).")
    parser.add_argument("--min-match-iou", type=float, default=0.1,
                        help="IoU minimale pour considérer GT et prédiction comme le même objet (défaut 0.1).")
    parser.add_argument("--max-display-dim", type=int, default=1600,
                        help="Dimension max (largeur ou hauteur) des images affichées, pour un chargement "
                             "rapide dans le navigateur (défaut 1600px - l'original n'est jamais modifié).")
    args = parser.parse_args()
    try:
        run_visualize(
            model_path=args.model,
            scope=args.scope,
            batch_filter=args.batch,
            limit=args.limit,
            raw_dir=args.raw_dir,
            split_dir=args.split_dir,
            class_config_path=args.config,
            run_id=args.run_id,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_conf_threshold=args.tile_conf_threshold,
            conf_threshold=args.conf_threshold,
            iou_mismatch_threshold=args.iou_mismatch_threshold,
            min_match_iou=args.min_match_iou,
            max_display_dim=args.max_display_dim,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
