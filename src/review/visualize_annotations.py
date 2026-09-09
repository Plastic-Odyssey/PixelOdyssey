#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Visionneuse des annotations brutes (masques de segmentation).

Contrairement à visualize_predictions.py (qui fait tourner un modèle sur les
images PARENTES et compare GT/prédictions), cet outil ne fait tourner aucun
modèle et ne lit aucun poids : il lit directement les imagettes et labels
YOLO-seg déjà tuilés dans 4_sliced_dataset/{images,labels}/{train,val,test}
- exactement la donnée telle qu'elle est vue par l'entraînement - et
superpose les masques de segmentation par-dessus, colorés par classe. Sert à
inspecter visuellement le contenu du dataset (densité, qualité des masques,
tuiles de fond, classes rares...), pas à diagnostiquer un modèle.

Aucun traitement d'image côté Python (pas de cv2, pas de pré-rendu d'overlay
sur disque) : les imagettes sont référencées telles quelles par chemin
relatif vers 4_sliced_dataset (jamais copiées, jamais ouvertes), et les
polygones (déjà normalisés 0-1 dans les labels YOLO-seg) sont dessinés côté
navigateur sur un <canvas> superposé à chaque <img>, à partir d'un JSON
embarqué dans la page HTML. Génération quasi instantanée même sur plusieurs
milliers d'imagettes (aucune lecture de pixel requise, juste du texte), et
permet de garder l'opacité/l'affichage par classe interactifs sans avoir à
relancer le script.

Les noms de classes viennent de config/data_config.yaml (comme partout
ailleurs dans le repo, via class_config.load_class_config) : si la taxonomie
change, cet outil suit sans modification.

Usage :
    python -m src.review.visualize_annotations
        -> génère la visionneuse pour train+val+test (splits sélectionnables
           en direct dans la page), ouvre ensuite
           .../8_annotation_viewer/<run>/index.html dans un navigateur.
    python -m src.review.visualize_annotations --scope test
    python -m src.review.visualize_annotations --batch "SB 4"
    python -m src.review.visualize_annotations --limit 300
        -> utile pour un premier coup d'oeil rapide sur un très gros dataset
           avant de générer la vue complète.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import DEFAULT_CLASS_CONFIG_PATH, load_class_config
from src.data.raw_dataset import VALID_IMG_EXTS
from src.data.slice_dataset import SLICED_DIR

from src.paths_config import PROCESSED_DATASET_DIR as BASE_DIR  # racine centralisee (08/09/2026), voir src/paths_config.py
VIEWER_DIR = os.path.join(BASE_DIR, "8_annotation_viewer")

SPLITS = ["train", "val", "test"]

# Palette qualitative fixe (une couleur par ID de super-classe), volontairement
# plus longue que le nombre de classes actuel : si la taxonomie grandit
# (config/data_config.yaml), les nouveaux IDs ont déjà une couleur sans
# modification de ce fichier. Au-delà de la palette, repli sur un gris neutre
# plutôt qu'une erreur.
CLASS_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]
FALLBACK_COLOR = "#999999"


def _class_color(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_PALETTE):
        return CLASS_PALETTE[class_id]
    return FALLBACK_COLOR


def _guess_batch(filename: str) -> str:
    """Nom de lot dérivé du préfixe de fichier tuilé (ex :
    'SB_1_images_train_tuilage_...png' -> 'SB 1'). Purement indicatif pour
    l'affichage/le filtre de cette visionneuse - jamais utilisé pour une
    décision de pipeline (contrairement à parent_id ailleurs dans le repo,
    qui vient du manifeste de split, pas d'un parsing de nom de fichier)."""
    marker = "_images_"
    prefix = filename.split(marker, 1)[0] if marker in filename else filename.rsplit(".", 1)[0]
    return prefix.replace("_", " ")


def _parse_yolo_seg_label(label_path: Path) -> List[Dict]:
    """Lit un fichier de label YOLO-seg (une ligne = une instance :
    `class_id x1 y1 x2 y2 ... xn yn`, coordonnées déjà normalisées 0-1 par
    rapport à l'imagette - format écrit par src/data/slicer.py). Fichier
    absent ou vide -> aucune instance (tuile de fond), pas une erreur. Une
    ligne malformée est ignorée avec un avertissement plutôt que de faire
    échouer toute l'imagette (même logique défensive que visualize_predictions.py :
    un problème sur une entrée ne doit jamais bloquer les autres)."""
    if not label_path.exists():
        return []
    instances = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                print(f"  ⚠️  Ligne {line_num} illisible dans {label_path.name}, ignorée.")
                continue
            if len(coords) < 6 or len(coords) % 2 != 0:
                print(
                    f"  ⚠️  Ligne {line_num} de {label_path.name} a {len(coords)} valeur(s) de coordonnées "
                    f"(attendu un nombre pair >= 6 pour un polygone) - ignorée."
                )
                continue
            pts = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
            instances.append({"cls": class_id, "pts": pts})
    return instances


def _select_images(sliced_dir: Path, scope: str, batch_filter: Optional[str], limit: Optional[int]) -> List[Dict]:
    splits = SPLITS if scope == "all" else [scope]
    records = []
    for split in splits:
        img_dir = sliced_dir / "images" / split
        if not img_dir.exists():
            print(f"  ⚠️  Dossier introuvable, ignoré : {img_dir}")
            continue
        files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS)
        for img_path in files:
            batch = _guess_batch(img_path.name)
            if batch_filter and batch_filter.lower() not in batch.lower():
                continue
            records.append({"path": img_path, "split": split, "batch": batch})
    records.sort(key=lambda r: (r["split"], r["path"].name))  # ordre déterministe, important pour --limit reproductible
    if limit is not None:
        records = records[:limit]
    return records


def _build_index_html(records: List[Dict], target_names: Dict[int, str], images_rel_root: str) -> str:
    class_ids = sorted(target_names.keys())
    class_names_json = json.dumps({str(cid): target_names[cid] for cid in class_ids}, ensure_ascii=False)
    class_colors_json = json.dumps({str(cid): _class_color(cid) for cid in class_ids}, ensure_ascii=False)
    legend_html = "".join(
        f'<label class="legend-item"><input type="checkbox" class="classToggle" data-cls="{cid}" checked>'
        f'<span class="swatch" style="background:{_class_color(cid)}"></span>{target_names[cid]}</label>'
        for cid in class_ids
    )
    records_json = json.dumps(records, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>PixelOdyssey - Visionneuse des annotations</title>
<style>
  body {{ margin:0; padding:0; background:#111; color:#eee; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  #topbar {{ display:flex; align-items:center; gap:16px; padding:10px 16px; background:#1a1a1a; border-bottom:1px solid #333; flex-wrap:wrap; }}
  #topbar select, #topbar input, #topbar button {{ background:#2a2a2a; color:#eee; border:1px solid #444; border-radius:4px; padding:6px 10px; font-size:13px; }}
  #topbar button {{ cursor:pointer; }}
  #topbar button:hover {{ background:#3a3a3a; }}
  #topbar label {{ display:flex; align-items:center; gap:6px; font-size:13px; }}
  #legend {{ display:flex; gap:14px; padding:8px 16px; font-size:12px; color:#ddd; flex-wrap:wrap; background:#161616; border-bottom:1px solid #2a2a2a; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; cursor:pointer; user-select:none; }}
  .swatch {{ width:12px; height:12px; border-radius:2px; display:inline-block; }}
  #caption {{ padding:10px 16px; font-size:14px; }}
  #caption b {{ color:#fff; }}
  #counts {{ margin-top:4px; font-size:12px; color:#bbb; }}
  #counts span {{ margin-right:14px; }}
  #imgwrap {{ position:relative; display:inline-flex; justify-content:center; align-items:center; padding:8px 16px 24px; margin: 0 auto; width:100%; box-sizing:border-box; }}
  #stage {{ position:relative; display:inline-block; }}
  #mainImg {{ display:block; max-width:100%; max-height:76vh; border:1px solid #333; border-radius:4px; }}
  #overlay {{ position:absolute; left:0; top:0; pointer-events:none; }}
  #posbar {{ padding: 0 16px 14px; font-size:12px; color:#999; }}
  #centerwrap {{ display:flex; justify-content:center; }}
  a {{ color:#8ab4f8; }}
</style>
</head>
<body>
<div id="topbar">
  <button id="prevBtn">&larr; Précédente</button>
  <button id="nextBtn">Suivante &rarr;</button>
  <label>Split :
    <select id="splitSel">
      <option value="all">Tous</option>
      <option value="train">train</option>
      <option value="val">val</option>
      <option value="test">test</option>
    </select>
  </label>
  <label>Trier par :
    <select id="sortSel">
      <option value="order">Ordre du dataset</option>
      <option value="most">Le plus d'instances d'abord</option>
      <option value="least">Le moins d'instances d'abord (fond en premier)</option>
    </select>
  </label>
  <label>Filtrer (lot) : <input id="filterInput" placeholder="ex: SL, SB 3, transect_11..."></label>
  <label><input type="checkbox" id="hideEmptyToggle"> Masquer les tuiles de fond (0 instance)</label>
  <label><input type="checkbox" id="labelsToggle" checked> Afficher les noms de classe</label>
  <label>Opacité masques : <input type="range" id="opacitySlider" min="0" max="1" step="0.05" value="0.35"></label>
  <span id="posbar"></span>
</div>
<div id="legend">{legend_html}</div>
<div id="caption"></div>
<div id="centerwrap">
  <div id="stage">
    <img id="mainImg" src="" alt="">
    <canvas id="overlay"></canvas>
  </div>
</div>

<script>
const IMAGES_ROOT = {json.dumps(images_rel_root)};
const CLASS_NAMES = {class_names_json};
const CLASS_COLORS = {class_colors_json};
const ALL_RECORDS = {records_json};
let records = ALL_RECORDS.slice();
let idx = 0;
let hiddenClasses = new Set();

function nInstances(rec) {{ return rec.instances.length; }}

function applyFilters() {{
  const split = document.getElementById('splitSel').value;
  const sortKey = document.getElementById('sortSel').value;
  const filterVal = document.getElementById('filterInput').value.trim().toLowerCase();
  const hideEmpty = document.getElementById('hideEmptyToggle').checked;
  records = ALL_RECORDS.filter(r =>
    (split === 'all' || r.split === split) &&
    (!filterVal || r.batch.toLowerCase().includes(filterVal)) &&
    (!hideEmpty || nInstances(r) > 0)
  );
  if (sortKey === 'most') {{
    records = records.slice().sort((a, b) => nInstances(b) - nInstances(a));
  }} else if (sortKey === 'least') {{
    records = records.slice().sort((a, b) => nInstances(a) - nInstances(b));
  }}
  idx = 0;
  render();
}}

function classCounts(rec) {{
  const counts = {{}};
  for (const inst of rec.instances) {{
    counts[inst.cls] = (counts[inst.cls] || 0) + 1;
  }}
  return counts;
}}

function hexToRgba(hex, alpha) {{
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

function drawOverlay() {{
  const img = document.getElementById('mainImg');
  const canvas = document.getElementById('overlay');
  const w = img.clientWidth, h = img.clientHeight;
  if (!w || !h) return;
  canvas.width = w; canvas.height = h;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  if (records.length === 0) return;
  const rec = records[idx];
  const opacity = parseFloat(document.getElementById('opacitySlider').value);
  const showLabels = document.getElementById('labelsToggle').checked;
  for (const inst of rec.instances) {{
    if (hiddenClasses.has(inst.cls)) continue;
    const color = CLASS_COLORS[inst.cls] || '#999999';
    ctx.beginPath();
    inst.pts.forEach((pt, i) => {{
      const px = pt[0] * w, py = pt[1] * h;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }});
    ctx.closePath();
    ctx.fillStyle = hexToRgba(color, opacity);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (showLabels) {{
      const label = CLASS_NAMES[inst.cls] || ('classe ' + inst.cls);
      const lx = inst.pts[0][0] * w, ly = inst.pts[0][1] * h;
      ctx.font = '13px sans-serif';
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0,0,0,0.85)';
      ctx.strokeText(label, lx + 2, Math.max(12, ly - 4));
      ctx.fillStyle = color;
      ctx.fillText(label, lx + 2, Math.max(12, ly - 4));
    }}
  }}
}}

function render() {{
  const posbar = document.getElementById('posbar');
  if (records.length === 0) {{
    document.getElementById('caption').innerHTML = '<i>Aucune imagette ne correspond au filtre.</i>';
    document.getElementById('mainImg').src = '';
    document.getElementById('overlay').getContext('2d').clearRect(0, 0, 9999, 9999);
    posbar.textContent = '';
    return;
  }}
  const rec = records[idx];
  document.getElementById('mainImg').src = IMAGES_ROOT + '/' + rec.split + '/' + rec.file;
  const counts = classCounts(rec);
  const countsHtml = Object.keys(counts).length === 0
    ? '<i>tuile de fond (aucune annotation)</i>'
    : Object.entries(counts).map(([cls, n]) =>
        `<span style="color:${{CLASS_COLORS[cls] || '#999'}}">${{CLASS_NAMES[cls] || cls}} : ${{n}}</span>`).join('');
  document.getElementById('caption').innerHTML =
    `<b>${{rec.file}}</b> - lot ${{rec.batch}} - split ${{rec.split}} - ${{rec.instances.length}} instance(s)` +
    `<div id="counts">${{countsHtml}}</div>`;
  posbar.textContent = `Imagette ${{idx + 1}} / ${{records.length}}`;
}}

document.getElementById('mainImg').addEventListener('load', drawOverlay);
window.addEventListener('resize', drawOverlay);

document.getElementById('prevBtn').addEventListener('click', () => {{ idx = (idx - 1 + records.length) % records.length; render(); }});
document.getElementById('nextBtn').addEventListener('click', () => {{ idx = (idx + 1) % records.length; render(); }});
document.getElementById('splitSel').addEventListener('change', applyFilters);
document.getElementById('sortSel').addEventListener('change', applyFilters);
document.getElementById('filterInput').addEventListener('input', applyFilters);
document.getElementById('hideEmptyToggle').addEventListener('change', applyFilters);
document.getElementById('labelsToggle').addEventListener('change', drawOverlay);
document.getElementById('opacitySlider').addEventListener('input', drawOverlay);
document.querySelectorAll('.classToggle').forEach(cb => {{
  cb.addEventListener('change', () => {{
    const cls = parseInt(cb.dataset.cls, 10);
    if (cb.checked) hiddenClasses.delete(cls); else hiddenClasses.add(cls);
    drawOverlay();
  }});
}});
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
    scope: str = "all",
    batch_filter: Optional[str] = None,
    limit: Optional[int] = None,
    sliced_dir: str = SLICED_DIR,
    output_dir: str = VIEWER_DIR,
    run_id: Optional[str] = None,
) -> Dict:
    """Voir la docstring du module."""
    sliced_path = Path(sliced_dir)
    if not sliced_path.exists():
        raise RuntimeError(
            f"Dataset tuilé introuvable : {sliced_path}. Lance d'abord le pipeline "
            f"(python -m src.data.data_pipeline) pour le générer."
        )

    _, target_names = load_class_config(DEFAULT_CLASS_CONFIG_PATH)

    print(f"--- 🎨 VISIONNEUSE D'ANNOTATIONS (scope='{scope}') ---")
    selected = _select_images(sliced_path, scope, batch_filter, limit)
    if not selected:
        print(f"❌ Aucune imagette sélectionnée (scope='{scope}', batch_filter={batch_filter!r}). Rien à faire.")
        return {"images_processed": 0}

    records: List[Dict] = []
    total_instances = 0
    for item in selected:
        img_path: Path = item["path"]
        label_path = sliced_path / "labels" / item["split"] / (img_path.stem + ".txt")
        instances = _parse_yolo_seg_label(label_path)
        total_instances += len(instances)
        records.append({
            "file": img_path.name,
            "split": item["split"],
            "batch": item["batch"],
            "instances": instances,
        })

    if run_id is None:
        run_id = datetime.now().strftime("viz_%Y%m%d_%H%M%S")
    out_root = Path(output_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    # Les imagettes ne sont jamais copiées : la page référence directement
    # 4_sliced_dataset/images/<split>/ par chemin relatif depuis son propre
    # dossier de sortie (8_annotation_viewer/<run>/index.html).
    images_rel_root = os.path.relpath(sliced_path / "images", out_root).replace(os.sep, "/")

    html_doc = _build_index_html(records, target_names, images_rel_root)
    index_path = out_root / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    by_split: Dict[str, int] = {}
    for r in records:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1

    print("\n=== RÉSUMÉ ===")
    print(f"  • Imagettes indexées : {len(records)} ({', '.join(f'{k}={v}' for k, v in sorted(by_split.items()))})")
    print(f"  • Instances (masques) au total : {total_instances}")
    print(f"\n  -> Ouvre {index_path} dans un navigateur pour naviguer.")

    return {
        "images_processed": len(records),
        "total_instances": total_instances,
        "by_split": by_split,
        "index_path": str(index_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visionneuse des annotations brutes (masques GT) PixelOdyssey")
    parser.add_argument("--scope", choices=["train", "val", "test", "all"], default="all",
                        help="Filtre de génération initial (défaut : all - les 3 splits sont de toute façon "
                             "sélectionnables en direct dans la page une fois ouverte).")
    parser.add_argument("--batch", default=None,
                        help="Filtre optionnel (sous-chaîne, insensible à la casse) sur le lot d'origine, "
                             "ex: --batch SL. Filtrable aussi en direct dans la page.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Nombre maximal d'imagettes à indexer (défaut : toutes celles du scope/filtre). "
                             "Utile pour un premier coup d'oeil rapide avant de générer la vue complète.")
    args = parser.parse_args()
    try:
        run_visualize(scope=args.scope, batch_filter=args.batch, limit=args.limit)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
