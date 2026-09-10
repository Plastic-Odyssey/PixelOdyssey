#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Rapport de lecture d'un entraînement YOLO.

Ultralytics produit déjà `results.csv`, `confusion_matrix.png`, etc. dans le
dossier d'un run, mais ces sorties restent des artefacts bruts : pas de vue
par classe claire, pas de calcul direct des taux de faux positifs / faux
négatifs, pas de distinction entre "vu pendant l'entraînement" (val) et
"jamais vu" (test).

Ce module génère un rapport HTML autonome (`rapport_lecture.html`) qui donne,
pour chaque split disponible (val, test) et pour chaque classe : le taux de
faux négatifs (rappel bas = déchets ratés), le taux de faux positifs
(précision basse = fausses alertes), les mAP, et la matrice de confusion.

Notes sur l'API Ultralytics utilisée :
- `model.val(...)` retourne un objet `SegmentMetrics` avec `.summary()`
  (table par classe : Box-P/R/F1, Mask-P/R/F1, mAP50, mAP50-95) et
  `.seg`/`.box` (agrégats `.mp`, `.mr`, `.map50`, `.map`).
- `.confusion_matrix.summary()` retourne la matrice de confusion sous forme
  de liste de dicts {Predicted: <classe prédite>, <classe réelle 1>: n, ...,
  background: n}.
- La matrice de confusion est calculée sur les BOÎTES (`confusion_matrix.task
  == 'detect'`), même pour un modèle de segmentation : utile pour voir
  QUELLES classes se confondent entre elles, mais les métriques officielles
  utilisées partout ailleurs dans ce rapport sont les métriques MASQUE
  (`Mask-P` / `Mask-R`), plus fidèles à la tâche réelle (délimiter les
  déchets).

Génère aussi `rapport_metrics.json`, sortie machine-lisible (mêmes chiffres
que le HTML, plus les hyperparamètres du run lus dans args.yaml et le
fingerprint de la donnée réellement utilisée à l'entraînement) - c'est ce
fichier que lit `compare_runs.py` pour comparer plusieurs runs entre eux.

Garde-fou anti-réinterprétation de taxonomie : avant d'évaluer, vérifie que
le modèle chargé (`model.names`, embarqué dans les poids au moment de
l'entraînement) correspond EXACTEMENT à la taxonomie déclarée dans
`data_config_path` - lève une RuntimeError claire sinon, plutôt que de
produire un tableau par classe scrambé (un ID de classe peut désigner une
classe différente si la taxonomie a changé depuis - voir
`class_config.assert_model_matches_taxonomy`, déjà utilisé par les 4 outils
de `src/review/`). Un run entraîné sous une taxonomie révolue doit être
réentraîné pour être comparable, pas seulement réévalué.

Entrée : dossier d'un run d'entraînement (contenant `weights/<nom>.pt`), et
config/data_config.yaml (ou un chemin explicite).
Sortie : `<run_dir>/rapport_lecture.html` + `<run_dir>/rapport_metrics.json`.

Exemple :
    python src/training/training_report.py --run "E:\PixelOdyssey\6. Model outputs\runs\mon_run"
"""

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ============================================================================
# Palette (issue de la skill dataviz - ordre catégoriel validé, ne pas
# réordonner sans re-passer scripts/validate_palette.js).
# ============================================================================
COLOR_PRECISION = {"light": "#2a78d6", "dark": "#3987e5"}  # slot 1 : bleu
COLOR_RECALL = {"light": "#eb6834", "dark": "#d95926"}     # slot 2 : orange
SEQ_BLUE_LIGHT = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE_DARK = ["#1a1a19", "#10366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

# Seuil sous lequel une classe est signalée dans les constats automatiques
# (_diagnostics) comme problématique (rappel/précision trop bas). Valeur
# ronde de repère, pas calibrée sur des données réelles - statut : question
# ouverte. Choix possibles : un seuil unique plus strict/plus lâche, ou un
# seuil différent par classe (ex. plus tolérant pour une classe rare).
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


def _global_micro_rates(rows: list) -> Dict:
    """Taux global pondéré par le nombre d'instances réelles de chaque classe -
    contrairement à mp/mr (moyenne MACRO : chaque classe pèse pareil quel que
    soit son volume), donne le taux qu'on observerait en comptant tous les
    objets réels ensemble, tous classes confondues. Reconstruit à partir des
    taux par classe : TP_c = instances_c * rappel_c, FN_c = instances_c - TP_c,
    FP_c = TP_c * (1/précision_c - 1).

    Entrée : lignes par classe (sortie de `_per_class_rows`).
    Sortie : dict {instances, global_precision, global_recall, global_fp_rate,
    global_fn_rate} - une valeur est None si aucune classe exploitable dans ce split.
    """
    total_tp, total_fp, total_instances = 0.0, 0.0, 0
    for r in rows:
        n = r["instances"]
        if n <= 0 or r["mask_r"] is None or r["mask_p"] is None:
            continue
        tp = n * r["mask_r"]
        total_tp += tp
        total_instances += n
        if r["mask_p"] > 0:
            total_fp += tp * (1.0 / r["mask_p"] - 1.0)

    global_recall = (total_tp / total_instances) if total_instances else None
    global_precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else None
    global_f1 = (
        (2 * global_precision * global_recall / (global_precision + global_recall))
        if global_precision is not None and global_recall is not None and (global_precision + global_recall) > 0
        else None
    )
    return {
        "instances": total_instances,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "global_f1": global_f1,
        "global_fp_rate": (1 - global_precision) if global_precision is not None else None,
        "global_fn_rate": (1 - global_recall) if global_recall is not None else None,
    }


def _macro_f1(rows: list) -> Optional[float]:
    """F1 macro = moyenne des F1 PAR CLASSE (déjà fournis par Ultralytics, `Mask-F1`
    dans `metrics.summary()`) - PAS F1 recalculé à partir de la précision/rappel macro
    (mp/mr) : ce sont deux quantités différentes (moyenne de F1 != F1 de la moyenne),
    et la première est la définition standard du F1 macro. Classes sans instance dans
    ce split (mask_f1=None) exclues, comme mp/mr d'Ultralytics."""
    f1_values = [r["mask_f1"] for r in rows if r["instances"] > 0 and r["mask_f1"] is not None]
    return (sum(f1_values) / len(f1_values)) if f1_values else None


def _diagnostics(rows: list) -> list:
    """Constats en langage clair, générés automatiquement à partir des
    seuils ci-dessus."""
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
                f"<td>0</td><td colspan='7' class='muted'>aucune instance dans ce split</td></tr>"
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
            f"<td><strong>{_pct(r['mask_f1'])}</strong></td>"
            f"<td class='{'flag' if fn_rate > 0.5 else ''}'>{_pct(fn_rate)}</td>"
            f"<td class='{'flag' if fp_rate > 0.5 else ''}'>{_pct(fp_rate)}</td>"
            f"<td>{_pct(r['map50'])}</td>"
            f"<td>{_pct(r['map50_95'])}</td>"
            "</tr>"
        )
    return (
        "<div class='table-scroll'><table class='data-table'><thead><tr>"
        "<th>Classe</th><th>Instances (réel)</th><th>Précision (masque)</th><th>Rappel (masque)</th>"
        "<th>F1 (masque)<br><span class='muted small'>(résumé P+R en 1 chiffre)</span></th>"
        "<th>Taux de faux négatifs<br><span class='muted small'>(déchets ratés)</span></th>"
        "<th>Taux de faux positifs<br><span class='muted small'>(fausses alertes)</span></th>"
        "<th>mAP50</th><th>mAP50-95</th>"
        "</tr></thead><tbody>" + "".join(trs) + "</tbody></table></div>"
    )


def _stat_tiles_html(metrics, rows: list) -> str:
    seg = metrics.seg
    total_instances = sum(r["instances"] for r in rows)
    global_rates = _global_micro_rates(rows)
    tiles = [
        ("Instances évaluées", f"{total_instances}"),
        ("Précision moyenne (masque, macro)", _pct(getattr(seg, "mp", 0.0))),
        ("Rappel moyen (masque, macro)", _pct(getattr(seg, "mr", 0.0))),
        ("F1 moyen (masque, macro)", _pct(_macro_f1(rows))),
        ("mAP50 (masque)", _pct(getattr(seg, "map50", 0.0))),
        ("mAP50-95 (masque)", _pct(getattr(seg, "map", 0.0))),
        ("Précision globale (pondérée instances)", _pct(global_rates["global_precision"])),
        ("Rappel global (pondéré instances)", _pct(global_rates["global_recall"])),
        ("F1 global (pondéré instances)", _pct(global_rates["global_f1"])),
    ]
    return "<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='tile-label'>{_esc(label)}</div><div class='tile-value'>{value}</div></div>"
        for label, value in tiles
    ) + "</div>"


def _split_section_html(split: str, metrics, rows: list) -> str:
    title, description = SPLIT_LABELS.get(split, (f"Split {split}", ""))
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
# Métadonnées de run (pour rapport_metrics.json - comparaison entre runs)
# ----------------------------------------------------------------------------

_HYPERPARAM_KEYS = [
    "model", "epochs", "imgsz", "batch", "patience", "degrees", "flipud",
    "fliplr", "copy_paste", "scale", "seed", "deterministic",
]


def _read_run_hyperparams(run_dir: Path) -> Dict:
    """Lit `<run_dir>/args.yaml` (écrit automatiquement par Ultralytics) pour
    retrouver les hyperparamètres RÉELS de ce run, plutôt que de supposer que
    train.py n'a pas changé depuis. Best-effort : dict vide si le fichier est
    absent ou illisible, un rapport ne doit jamais échouer pour ça."""
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return {}
    try:
        import yaml
        with open(args_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {k: data[k] for k in _HYPERPARAM_KEYS if k in data}
    except Exception:
        return {}


def _read_sliced_data_fingerprint() -> Optional[str]:
    """Lit le fingerprint de `4_sliced_dataset` (donnée réellement vue à
    l'entraînement, après la cascade split -> augmentation -> slicing) AU
    MOMENT de la génération du rapport - fiable juste après un entraînement
    (cas d'usage normal, train.py appelle generate_report() immédiatement
    après), mais reflète l'état COURANT du dossier si le rapport est
    régénéré plus tard après un nouveau run de data_pipeline.py. Best-effort :
    None si indisponible (chemin non monté, dataset jamais slicé...)."""
    try:
        from src.data.pipeline_utils import read_upstream_fingerprint
        from src.data.slice_dataset import MANIFEST_FILENAME as SLICE_MANIFEST_FILENAME
        from src.data.slice_dataset import SLICED_DIR
        return read_upstream_fingerprint(Path(SLICED_DIR) / SLICE_MANIFEST_FILENAME)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Point d'entrée
# ----------------------------------------------------------------------------

def generate_report(
    run_dir, data_config_path=None, weights_name: str = "best.pt", splits=("val", "test"),
    output_basename: str = "rapport",
) -> Path:
    """Évalue `best.pt` (ou `weights_name`) d'un run sur les splits demandés et
    écrit deux fichiers dans `run_dir` : le rapport HTML lisible, et un JSON
    machine-lisible (mêmes métriques + hyperparamètres + fingerprint de
    donnée) destiné à `compare_runs.py`.

    Entrée : dossier du run, chemin de data_config.yaml (défaut : celui du
    projet), nom du fichier de poids, splits à évaluer, `output_basename`
    (préfixe des 2 fichiers de sortie - défaut "rapport", donc
    "rapport_lecture.html"/"rapport_metrics.json" comme avant ; un préfixe
    différent permet de générer un rapport scopé - ex: un sous-ensemble de
    lots via src/review/scoped_report.py - SANS écraser le rapport standard
    du run).
    Sortie : chemin du rapport HTML écrit (le JSON est écrit à côté, même
    préfixe : `<output_basename>_metrics.json`).
    """
    import tempfile

    import yaml
    from ultralytics import YOLO  # import tardif : évite de charger torch si le module est juste inspecté

    from src.data.class_config import assert_model_matches_taxonomy

    run_dir = Path(run_dir)
    weights_path = run_dir / "weights" / weights_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    if data_config_path is None:
        # À défaut d'indication explicite, on retombe sur la config par défaut
        # du projet (cohérent avec train.py).
        data_config_path = Path(__file__).resolve().parent.parent.parent / "config" / "data_config.yaml"

    model = YOLO(str(weights_path))

    # Garde-fou anti-réinterprétation de taxonomie (même mécanisme que les 4 outils de
    # src/review/, voir class_config.assert_model_matches_taxonomy) : `model.val()`
    # construit son tableau par classe à partir de `model.names` (EMBARQUÉ dans les
    # poids au moment de
    # l'entraînement), jamais à partir de `data_config_path`. Si la taxonomie a changé
    # depuis ce run (classe retirée/renommée, ID renuméroté), les métriques par classe
    # sont silencieusement scramblées - un ID peut désigner une classe différente
    # entraînement vs config actuelle, sans qu'aucune erreur ne le signale autrement.
    # Comparé aux NOMS DÉCLARÉS DANS `data_config_path` lui-même (pas au
    # `DEFAULT_CLASS_CONFIG_PATH` du projet) : reflète exactement ce que `model.val()`
    # utilise réellement comme vérité terrain pour CET appel, y compris pour un yaml
    # temporaire scopé (src/review/scoped_report.py) qui copie data_config.yaml mais
    # pourrait en théorie diverger.
    with open(data_config_path, "r", encoding="utf-8") as f:
        data_config_names = yaml.safe_load(f).get("names", {})
    target_names = {int(k): str(v) for k, v in data_config_names.items()}
    model_names = {int(k): str(v) for k, v in model.names.items()}
    assert_model_matches_taxonomy(model_names, target_names, model_label=str(weights_path))

    sections_html = []
    splits_payload: Dict[str, Dict] = {}
    any_split_evaluated = False
    # plots=True est nécessaire pour que la matrice de confusion soit remplie : dans
    # Ultralytics (ultralytics/models/yolo/detect/val.py), `ConfusionMatrix.process_batch()`
    # - qui remplit la matrice - n'est appelé que si `self.args.plots` est vrai ; avec
    # plots=False la matrice resterait intégralement à 0 (y compris la diagonale), même si
    # les métriques P/R par classe (calculées séparément) restent correctes indépendamment.
    # Redirigé vers un dossier temporaire (project=tmp_val_dir) pour éviter que les PNG
    # qu'Ultralytics écrit dans ce cas (confusion_matrix.png, PR_curve.png...) ne polluent
    # run_dir - ce dossier temporaire est jeté à la sortie du `with`.
    with tempfile.TemporaryDirectory(prefix="pixelodyssey_val_") as tmp_val_dir:
        for split in splits:
            try:
                metrics = model.val(
                    data=str(data_config_path), split=split, plots=True, verbose=False,
                    project=tmp_val_dir, name=f"eval_{split}", exist_ok=True,
                )
            except Exception as e:  # noqa: BLE001 - un split absent/vide ne doit pas faire échouer tout le rapport
                print(f"⚠️  Split '{split}' non évalué ({e}) — ignoré dans le rapport.")
                continue
            any_split_evaluated = True
            rows = _per_class_rows(metrics)
            seg = metrics.seg
            macro_f1 = _macro_f1(rows)
            splits_payload[split] = {
                "macro": {
                    "precision": float(getattr(seg, "mp", 0.0)),
                    "recall": float(getattr(seg, "mr", 0.0)),
                    "f1": float(macro_f1) if macro_f1 is not None else None,
                    "map50": float(getattr(seg, "map50", 0.0)),
                    "map50_95": float(getattr(seg, "map", 0.0)),
                },
                "global": _global_micro_rates(rows),
                "per_class": rows,
            }
            sections_html.append(_split_section_html(split, metrics, rows))

    if not any_split_evaluated:
        raise RuntimeError("Aucun split n'a pu être évalué (val et test indisponibles ou vides) : rapport annulé.")

    html_doc = _build_html(run_dir.name, sections_html)
    out_path = run_dir / f"{output_basename}_lecture.html"
    out_path.write_text(html_doc, encoding="utf-8")

    metrics_payload = {
        "run_name": run_dir.name,
        "weights_name": weights_name,
        "hyperparams": _read_run_hyperparams(run_dir),
        "sliced_data_fingerprint": _read_sliced_data_fingerprint(),
        "splits": splits_payload,
    }
    metrics_path = run_dir / f"{output_basename}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère le rapport de lecture HTML d'un run d'entraînement PixelOdyssey.")
    parser.add_argument("--run", required=True, help="Dossier du run (ex: \"E:\\PixelOdyssey\\6. Model outputs\\runs\\baseline_yolo11n-seg_AAAAMMJJ_HHMMSS\")")
    parser.add_argument("--data", default=None, help="Chemin vers data_config.yaml (par défaut : config/data_config.yaml du projet)")
    parser.add_argument("--weights", default="best.pt", help="Nom du fichier de poids à évaluer, dans <run>/weights/ (défaut: best.pt)")
    parser.add_argument("--splits", default="val,test", help="Splits à évaluer, séparés par des virgules (défaut: val,test)")
    parser.add_argument("--output-basename", default="rapport",
                         help="Préfixe des 2 fichiers de sortie dans <run>/ (défaut: 'rapport' -> "
                              "rapport_lecture.html/rapport_metrics.json). À changer pour ne pas écraser le "
                              "rapport standard d'un run - ex: généré automatiquement par scoped_report.py.")
    args = parser.parse_args()

    try:
        report_path = generate_report(
            args.run,
            data_config_path=args.data,
            weights_name=args.weights,
            splits=[s.strip() for s in args.splits.split(",") if s.strip()],
            output_basename=args.output_basename,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
    metrics_path = report_path.with_name(f"{args.output_basename}_metrics.json")
    print(f"✅ Rapport généré : {report_path}")
    print(f"   (+ {metrics_path}, pour src.training.compare_runs)")
