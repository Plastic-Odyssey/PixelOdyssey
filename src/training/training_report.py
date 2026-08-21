#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Rapport de lecture d'un entraînement YOLO.

Ultralytics produit déjà `results.csv`, `confusion_matrix.png`, etc. dans le
dossier d'un run, mais ces sorties restent des artefacts bruts : pas de vue
par classe claire, pas de calcul direct des taux de faux positifs / faux
négatifs, pas de distinction entre "vu pendant l'entraînement" (val) et
"jamais vu" (test).

Ce module génère un rapport HTML autonome (`rapport_lecture.html`, à ouvrir
dans un navigateur) qui répond directement à deux questions :
  1. Pour CHAQUE classe, est-ce que je rate des déchets (faux négatifs, donc
     rappel bas) ou est-ce que je détecte des choses qui n'existent pas
     (faux positifs, donc précision basse) ?
  2. Est-ce que la performance tient sur des données jamais vues (split
     "test") ou seulement sur le split "val" déjà utilisé pendant
     l'entraînement pour ajuster les hyperparamètres (patience, etc.) ?

Point d'API Ultralytics important (vérifié empiriquement, pas deviné) :
- `model.val(...)` retourne un objet `SegmentMetrics` avec :
    - `.summary()` : déjà une table par classe (Box-P/R/F1, Mask-P/R/F1,
      mAP50, mAP50-95) — c'est la base du tableau par classe ci-dessous.
    - `.seg` / `.box` : objets avec les agrégats `.mp` (précision moyenne),
      `.mr` (rappel moyen), `.map50`, `.map`.
    - `.confusion_matrix.summary()` : la matrice de confusion déjà sous
      forme de liste de dicts {Predicted: <classe prédite>, <classe réelle
      1>: n, <classe réelle 2>: n, ..., background: n} — pas besoin de
      manipuler l'array numpy brut ni de deviner l'ordre des axes.
    - ATTENTION : la matrice de confusion est calculée sur les BOÎTES
      (`confusion_matrix.task == 'detect'`), même pour un modèle de
      segmentation. Elle sert donc à voir QUELLES classes se confondent
      entre elles, mais les taux de précision/rappel "officiels" utilisés
      partout ailleurs dans ce rapport sont les métriques MASQUE
      (`Mask-P` / `Mask-R`), plus fidèles à la vraie tâche (délimiter les
      déchets), pas les métriques boîte.
"""

import argparse
import html
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ============================================================================
# Palette (issue de la skill dataviz - ordre catégoriel validé, ne pas
# réordonner sans re-passer scripts/validate_palette.js).
# ============================================================================
COLOR_PRECISION = {"light": "#2a78d6", "dark": "#3987e5"}  # slot 1 : bleu
COLOR_RECALL = {"light": "#eb6834", "dark": "#d95926"}     # slot 2 : orange
SEQ_BLUE_LIGHT = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE_DARK = ["#1a1a19", "#10366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

RECALL_WARN_THRESHOLD = 0.5
PRECISION_WARN_THRESHOLD = 0.5

SPLIT_LABELS = {
    "val": (
        "Split VALIDATION",
        "Utilisé pendant l'entraînement pour surveiller le sur-apprentissage "
        "et déclencher l'arrêt anticipé (patience). Utile pour suivre "
        "l'entraînement, mais légèrement optimiste : les choix d'hyperparamètres "
        "ont indirectement \"vu\" ces données.",
    ),
    "test": (
        "Split TEST",
        "Jamais utilisé, ni pour l'entraînement ni pour ajuster quoi que ce "
        "soit. C'est la meilleure estimation de la performance réelle du "
        "modèle sur de nouvelles images de terrain.",
    ),
    "train": (
        "Split TRAIN",
        "Le modèle a directement appris sur ces données — sert seulement à "
        "vérifier qu'il n'y a pas de sous-apprentissage flagrant, pas à "
        "estimer une vraie performance.",
    ),
}


def _esc(x) -> str:
    return html.escape(str(x))


def _pct(x, decimals: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


# ----------------------------------------------------------------------------
# Extraction des données depuis l'objet metrics retourné par model.val()
# ----------------------------------------------------------------------------

def _per_class_rows(metrics) -> list:
    """Une ligne par classe DÉCLARÉE dans data_config.yaml (`names`), même si
    elle n'apparaît pas dans metrics.summary() (cas d'une classe à 0 instance
    dans ce split, ex. NonPlastique - cf. config/data_config.yaml) : on ne
    veut jamais qu'une classe disparaisse silencieusement du rapport."""
    names = metrics.names  # {id: nom}
    raw_nt = getattr(metrics, "nt_per_class", None)
    nt_per_class = list(raw_nt) if raw_nt is not None else []
    summary_by_name = {row["Class"]: row for row in metrics.summary()}

    rows = []
    for class_id in sorted(names.keys()):
        class_name = names[class_id]
        row = summary_by_name.get(class_name)
        instances = int(nt_per_class[class_id]) if class_id < len(nt_per_class) else 0
        if row is None:
            rows.append({
                "class_name": class_name,
                "images": 0,
                "instances": instances,
                "mask_p": None,
                "mask_r": None,
                "mask_f1": None,
                "map50": None,
                "map50_95": None,
            })
        else:
            rows.append({
                "class_name": class_name,
                "images": int(row.get("Images", 0)),
                "instances": int(row.get("Instances", instances)),
                "mask_p": float(row.get("Mask-P", 0.0)),
                "mask_r": float(row.get("Mask-R", 0.0)),
                "mask_f1": float(row.get("Mask-F1", 0.0)),
                "map50": float(row.get("mAP50", 0.0)),
                "map50_95": float(row.get("mAP50-95", 0.0)),
            })
    return rows


def _diagnostics(rows: list) -> list:
    """Constats en langage clair, générés automatiquement à partir des
    seuils ci-dessus - le coeur de la demande "plus facile à comprendre"."""
    notes = []
    for r in rows:
        name = r["class_name"]
        if r["instances"] == 0:
            notes.append(f"❌ <strong>{_esc(name)}</strong> : aucune instance dans ce split — impossible d'évaluer cette classe ici.")
            continue
        if r["mask_r"] is not None and r["mask_r"] < RECALL_WARN_THRESHOLD:
            notes.append(
                f"⚠️ <strong>{_esc(name)}</strong> : rappel {_pct(r['mask_r'])} — beaucoup de "
                f"<strong>faux négatifs</strong> (des déchets de cette classe ne sont pas détectés)."
            )
        if r["mask_p"] is not None and r["mask_p"] < PRECISION_WARN_THRESHOLD:
            notes.append(
                f"⚠️ <strong>{_esc(name)}</strong> : précision {_pct(r['mask_p'])} — beaucoup de "
                f"<strong>faux positifs</strong> (le modèle détecte cette classe là où il n'y en a pas)."
            )
    if not notes:
        notes.append("✅ Aucune classe sous les seuils d'alerte (rappel et précision ≥ 50 %) sur ce split.")
    return notes


# ----------------------------------------------------------------------------
# Rendu HTML - graphique en barres (SVG), sans dépendance externe
# ----------------------------------------------------------------------------

def _bar_path(x: float, y_top: float, w: float, baseline_y: float, radius: float = 4) -> str:
    """Rectangle qui pousse depuis la ligne de base (bas), coins arrondis en
    haut uniquement, carré à la base - spec de la skill dataviz."""
    h = baseline_y - y_top
    if h <= 0:
        return ""
    r = min(radius, w / 2, h)
    return (
        f"M{x},{baseline_y} L{x},{y_top + r} "
        f"Q{x},{y_top} {x + r},{y_top} "
        f"L{x + w - r},{y_top} Q{x + w},{y_top} {x + w},{y_top + r} "
        f"L{x + w},{baseline_y} Z"
    )


def _precision_recall_chart_svg(rows: list) -> str:
    plottable = [r for r in rows if r["instances"] > 0]
    if not plottable:
        return "<p class='muted'>Aucune classe avec des instances sur ce split — pas de graphique.</p>"

    bar_w, bar_gap, group_gap = 20, 2, 30
    margin = {"left": 46, "right": 16, "top": 16, "bottom": 64}
    chart_h = 220
    group_w = 2 * bar_w + bar_gap
    n = len(plottable)
    content_w = n * group_w + (n - 1) * group_gap
    svg_w = margin["left"] + content_w + margin["right"]
    svg_h = margin["top"] + chart_h + margin["bottom"]
    baseline_y = margin["top"] + chart_h

    # overflow:visible - sans ça, le rectangle de clipping par défaut du <svg>
    # rogne les libellés de classe pivotés (-35°) qui dépassent du viewBox
    # (noms de classe potentiellement longs, ex. "Mousse_Fragments_Souple").
    parts = [f"<svg viewBox='0 0 {svg_w} {svg_h}' width='100%' height='{svg_h}' role='img' "
              f"style='overflow: visible' aria-label='Précision et rappel par classe'>"]

    # Gridlines hairline (0%, 25%, 50%, 75%, 100%) + labels
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = baseline_y - frac * chart_h
        parts.append(
            f"<line x1='{margin['left']}' y1='{y:.1f}' x2='{svg_w - margin['right']}' y2='{y:.1f}' "
            f"class='gridline' />"
        )
        parts.append(
            f"<text x='{margin['left'] - 8}' y='{y + 4:.1f}' text-anchor='end' class='axis-label'>{int(frac * 100)}%</text>"
        )

    for i, r in enumerate(plottable):
        x0 = margin["left"] + i * (group_w + group_gap)
        p, rc = r["mask_p"] or 0.0, r["mask_r"] or 0.0
        y_p = baseline_y - p * chart_h
        y_r = baseline_y - rc * chart_h

        parts.append(
            f"<path d='{_bar_path(x0, y_p, bar_w, baseline_y)}' fill='var(--series-precision)'>"
            f"<title>{_esc(r['class_name'])} — Précision (masque) : {_pct(p)}</title></path>"
        )
        parts.append(
            f"<path d='{_bar_path(x0 + bar_w + bar_gap, y_r, bar_w, baseline_y)}' fill='var(--series-recall)'>"
            f"<title>{_esc(r['class_name'])} — Rappel (masque) : {_pct(rc)}</title></path>"
        )

        label_x = x0 + group_w / 2
        parts.append(
            f"<text x='{label_x:.1f}' y='{baseline_y + 14}' text-anchor='end' class='axis-label' "
            f"transform='rotate(-35 {label_x:.1f} {baseline_y + 14})'>{_esc(r['class_name'])}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _confusion_matrix_html(metrics) -> str:
    cm = metrics.confusion_matrix
    cm_rows = cm.summary()  # [{'Predicted': name, <true class>: n, ..., 'background': n}, ...]
    if not cm_rows:
        return "<p class='muted'>Matrice de confusion indisponible.</p>"

    true_cols = [k for k in cm_rows[0].keys() if k != "Predicted"]
    max_val = max((v for row in cm_rows for k, v in row.items() if k != "Predicted"), default=0) or 1

    head = "<th class='cm-corner'>Prédit ↓ / Réel →</th>" + "".join(f"<th>{_esc(c)}</th>" for c in true_cols)
    body_rows = []
    for row in cm_rows:
        predicted = row["Predicted"]
        cells = [f"<th class='cm-row-label'>{_esc(predicted)}</th>"]
        for true_class in true_cols:
            val = row.get(true_class, 0)
            frac = (val / max_val) if max_val else 0
            step = min(len(SEQ_BLUE_LIGHT) - 1, round(frac * (len(SEQ_BLUE_LIGHT) - 1)))
            is_diag = (predicted == true_class)
            diag_class = " cm-diag" if is_diag else ""
            cells.append(
                f"<td class='cm-cell{diag_class}' style='--cm-step:{step}' "
                f"title='Réel = {_esc(true_class)}, prédit = {_esc(predicted)} : {int(val)}'>{int(val)}</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<div class='table-scroll'><table class='cm-table'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
        "<p class='muted small'>Lecture : la diagonale (cases encadrées) = prédictions correctes. "
        "Basée sur les boîtes englobantes (et non les masques) — utile pour voir QUELLES classes se "
        "confondent entre elles, pas comme mesure officielle de précision/rappel (voir tableau par classe).</p>"
    )


def _per_class_table_html(rows: list) -> str:
    trs = []
    for r in rows:
        if r["instances"] == 0:
            trs.append(
                f"<tr class='row-empty'><td>{_esc(r['class_name'])}</td>"
                f"<td>0</td><td colspan='6' class='muted'>aucune instance dans ce split</td></tr>"
            )
            continue
        fn_rate = 1 - r["mask_r"]
        fp_rate = 1 - r["mask_p"]
        trs.append(
            "<tr>"
            f"<td>{_esc(r['class_name'])}</td>"
            f"<td>{r['instances']}</td>"
            f"<td>{_pct(r['mask_p'])}</td>"
            f"<td>{_pct(r['mask_r'])}</td>"
            f"<td class='{'flag' if fn_rate > 0.5 else ''}'>{_pct(fn_rate)}</td>"
            f"<td class='{'flag' if fp_rate > 0.5 else ''}'>{_pct(fp_rate)}</td>"
            f"<td>{_pct(r['map50'])}</td>"
            f"<td>{_pct(r['map50_95'])}</td>"
            "</tr>"
        )
    return (
        "<div class='table-scroll'><table class='data-table'><thead><tr>"
        "<th>Classe</th><th>Instances (réel)</th><th>Précision (masque)</th><th>Rappel (masque)</th>"
        "<th>Taux de faux négatifs<br><span class='muted small'>(déchets ratés)</span></th>"
        "<th>Taux de faux positifs<br><span class='muted small'>(fausses alertes)</span></th>"
        "<th>mAP50</th><th>mAP50-95</th>"
        "</tr></thead><tbody>" + "".join(trs) + "</tbody></table></div>"
    )


def _stat_tiles_html(metrics, rows: list) -> str:
    seg = metrics.seg
    total_instances = sum(r["instances"] for r in rows)
    tiles = [
        ("Instances évaluées", f"{total_instances}"),
        ("Précision moyenne (masque)", _pct(getattr(seg, "mp", 0.0))),
        ("Rappel moyen (masque)", _pct(getattr(seg, "mr", 0.0))),
        ("mAP50 (masque)", _pct(getattr(seg, "map50", 0.0))),
        ("mAP50-95 (masque)", _pct(getattr(seg, "map", 0.0))),
    ]
    return "<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='tile-label'>{_esc(label)}</div><div class='tile-value'>{value}</div></div>"
        for label, value in tiles
    ) + "</div>"


def _split_section_html(split: str, metrics) -> str:
    title, description = SPLIT_LABELS.get(split, (f"Split {split}", ""))
    rows = _per_class_rows(metrics)
    diagnostics = _diagnostics(rows)

    return f"""
    <section class="split-section">
      <h2>{_esc(title)}</h2>
      <p class="muted">{_esc(description)}</p>
      {_stat_tiles_html(metrics, rows)}
      <h3>Constats automatiques</h3>
      <ul class="diagnostics">
        {''.join(f"<li>{note}</li>" for note in diagnostics)}
      </ul>
      <h3>Précision et rappel par classe (masque)</h3>
      <div class="legend">
        <span class="legend-item"><span class="swatch swatch-precision"></span>Précision — 1 - ceci = taux de faux positifs</span>
        <span class="legend-item"><span class="swatch swatch-recall"></span>Rappel — 1 - ceci = taux de faux négatifs</span>
      </div>
      {_precision_recall_chart_svg(rows)}
      <h3>Tableau par classe</h3>
      {_per_class_table_html(rows)}
      <h3>Matrice de confusion (boîtes)</h3>
      {_confusion_matrix_html(metrics)}
    </section>
    """


_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --gridline: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-precision: %(precision_light)s;
  --series-recall: %(recall_light)s;
  --seq-blue-0: %(seq0)s; --seq-blue-1: %(seq1)s; --seq-blue-2: %(seq2)s; --seq-blue-3: %(seq3)s;
  --seq-blue-4: %(seq4)s; --seq-blue-5: %(seq5)s; --seq-blue-6: %(seq6)s; --seq-blue-7: %(seq7)s;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-precision: %(precision_dark)s;
    --series-recall: %(recall_dark)s;
    --seq-blue-0: %(seq0d)s; --seq-blue-1: %(seq1d)s; --seq-blue-2: %(seq2d)s; --seq-blue-3: %(seq3d)s;
    --seq-blue-4: %(seq4d)s; --seq-blue-5: %(seq5d)s; --seq-blue-6: %(seq6d)s; --seq-blue-7: %(seq7d)s;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px; background: var(--page);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary);
}
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 18px; margin: 28px 0 4px; }
h3 { font-size: 14px; color: var(--text-secondary); margin: 20px 0 8px; text-transform: uppercase; letter-spacing: .02em; }
.muted { color: var(--text-muted); }
.small { font-size: 12px; }
.run-meta { color: var(--text-secondary); font-size: 13px; margin-bottom: 24px; }
.split-section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; }
.tiles { display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0; }
.tile { flex: 1 1 140px; padding: 12px 14px; border: 1px solid var(--border); border-radius: 6px; }
.tile-label { font-size: 12px; color: var(--text-secondary); }
.tile-value { font-size: 22px; font-weight: 600; margin-top: 2px; }
.diagnostics { padding-left: 20px; line-height: 1.6; }
.legend { display: flex; gap: 20px; margin-bottom: 8px; font-size: 13px; color: var(--text-secondary); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.swatch-precision { background: var(--series-precision); }
.swatch-recall { background: var(--series-recall); }
.gridline { stroke: var(--gridline); stroke-width: 1; }
.axis-label { fill: var(--text-muted); font-size: 11px; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%%; font-size: 13px; margin-top: 4px; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
thead th { color: var(--text-secondary); font-weight: 500; border-bottom: 1px solid var(--baseline); }
.row-empty td { color: var(--text-muted); }
.flag { color: %(critical)s; font-weight: 600; }
.cm-table th, .cm-table td { text-align: center; }
.cm-corner { text-align: left; color: var(--text-muted); font-size: 11px; }
.cm-row-label { text-align: left; }
.cm-cell {
  background: var(--seq-blue-0);
}
.cm-cell[style*="--cm-step:1"] { background: var(--seq-blue-1); }
.cm-cell[style*="--cm-step:2"] { background: var(--seq-blue-2); }
.cm-cell[style*="--cm-step:3"] { background: var(--seq-blue-3); }
.cm-cell[style*="--cm-step:4"] { background: var(--seq-blue-4); color: white; }
.cm-cell[style*="--cm-step:5"] { background: var(--seq-blue-5); color: white; }
.cm-cell[style*="--cm-step:6"] { background: var(--seq-blue-6); color: white; }
.cm-cell[style*="--cm-step:7"] { background: var(--seq-blue-7); color: white; }
.cm-diag { box-shadow: inset 0 0 0 2px var(--text-primary); font-weight: 600; }
"""


def _build_html(run_name: str, sections_html: list) -> str:
    css = _CSS % {
        "precision_light": COLOR_PRECISION["light"], "precision_dark": COLOR_PRECISION["dark"],
        "recall_light": COLOR_RECALL["light"], "recall_dark": COLOR_RECALL["dark"],
        "seq0": SEQ_BLUE_LIGHT[0], "seq1": SEQ_BLUE_LIGHT[1], "seq2": SEQ_BLUE_LIGHT[2], "seq3": SEQ_BLUE_LIGHT[3],
        "seq4": SEQ_BLUE_LIGHT[4], "seq5": SEQ_BLUE_LIGHT[5], "seq6": SEQ_BLUE_LIGHT[6], "seq7": SEQ_BLUE_LIGHT[7],
        "seq0d": SEQ_BLUE_DARK[0], "seq1d": SEQ_BLUE_DARK[1], "seq2d": SEQ_BLUE_DARK[2], "seq3d": SEQ_BLUE_DARK[3],
        "seq4d": SEQ_BLUE_DARK[4], "seq5d": SEQ_BLUE_DARK[5], "seq6d": SEQ_BLUE_DARK[6], "seq7d": SEQ_BLUE_DARK[7],
        "critical": "#d03b3b",
    }
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport de lecture — {_esc(run_name)}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
  <h1>Rapport de lecture — {_esc(run_name)}</h1>
  <p class="run-meta">Généré automatiquement par training_report.py. Une vue par classe manque toujours dans les
  sorties Ultralytics par défaut : ce rapport calcule le taux de faux négatifs (= 1 − rappel) et le taux de faux
  positifs (= 1 − précision) pour chaque classe, sur chaque split disponible.</p>
  {''.join(sections_html)}
</div>
</body>
</html>"""


# ----------------------------------------------------------------------------
# Point d'entrée
# ----------------------------------------------------------------------------

def generate_report(run_dir, data_config_path=None, weights_name: str = "best.pt", splits=("val", "test")) -> Path:
    from ultralytics import YOLO  # import tardif : évite de charger torch si le module est juste inspecté

    run_dir = Path(run_dir)
    weights_path = run_dir / "weights" / weights_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    if data_config_path is None:
        # À défaut d'indication explicite, on retombe sur la config par défaut
        # du projet (cohérent avec train.py).
        data_config_path = Path(__file__).resolve().parent.parent.parent / "config" / "data_config.yaml"

    model = YOLO(str(weights_path))

    sections_html = []
    any_split_evaluated = False
    for split in splits:
        try:
            metrics = model.val(data=str(data_config_path), split=split, plots=False, verbose=False)
        except Exception as e:  # noqa: BLE001 - un split absent/vide ne doit pas faire échouer tout le rapport
            print(f"⚠️  Split '{split}' non évalué ({e}) — ignoré dans le rapport.")
            continue
        any_split_evaluated = True
        sections_html.append(_split_section_html(split, metrics))

    if not any_split_evaluated:
        raise RuntimeError("Aucun split n'a pu être évalué (val et test indisponibles ou vides) : rapport annulé.")

    html_doc = _build_html(run_dir.name, sections_html)
    out_path = run_dir / "rapport_lecture.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère le rapport de lecture HTML d'un run d'entraînement PixelOdyssey.")
    parser.add_argument("--run", required=True, help="Dossier du run (ex: output/runs/baseline_yolo11n-seg_20260819_143000)")
    parser.add_argument("--data", default=None, help="Chemin vers data_config.yaml (par défaut : config/data_config.yaml du projet)")
    parser.add_argument("--weights", default="best.pt", help="Nom du fichier de poids à évaluer, dans <run>/weights/ (défaut: best.pt)")
    parser.add_argument("--splits", default="val,test", help="Splits à évaluer, séparés par des virgules (défaut: val,test)")
    args = parser.parse_args()

    report_path = generate_report(
        args.run,
        data_config_path=args.data,
        weights_name=args.weights,
        splits=[s.strip() for s in args.splits.split(",") if s.strip()],
    )
    print(f"✅ Rapport généré : {report_path}")
