#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Statistiques agrégées (surface, poids, par classe), partagées
entre les deux générateurs de carte web du projet : src/application/web_map.py
(mode batch, photos drone brutes) et src/review/geo_density_map.py (mode
orthomosaïque, GeoTIFF unique) - les mêmes statistiques doivent apparaître
sur les deux cartes, quel que soit le mode d'entrée qui les a produites.

Ce module ne construit PAS la carte entière : les deux outils gardent leur
propre fond de carte, leur propre gabarit HTML/JS et leur propre panneau
interactif (volontairement différents - voir la docstring de web_map.py :
pas de pyramide de tuiles locale en mode batch, contrairement au mode
orthomosaïque). Il fournit seulement les deux briques qui peuvent réellement
être identiques dans les deux : (a) le calcul des statistiques agrégées à
partir d'une liste de détections déjà enrichies de `area_m2`/`weight_kg`
(voir run_application.py et geo_density_map.py pour comment chacune obtient
ces deux valeurs dans son propre mode), et (b) le fragment HTML en lecture
seule qui les affiche - le filtre par classe (interactif, JS) reste écrit
séparément dans chaque gabarit, qui l'intègre à sa propre logique de
rafraîchissement de calque (boutons Points/Densité, curseur de confiance...).

Exemple :
    from src.application.stats_panel import compute_aggregate_stats, render_stats_html
    stats = compute_aggregate_stats(detection_records)
    stats_html = render_stats_html(stats)
"""

from typing import Dict, List

from src.application.weight_estimation import CALIBRATED


def compute_aggregate_stats(records: List[Dict]) -> Dict:
    """Calcule les statistiques agrégées à partir d'une liste de dicts de
    détection, CHACUN devant déjà contenir au minimum `class_name`,
    `area_m2` (peut être None/absent) et `weight_kg` (peut être None/absent,
    voir weight_estimation.estimate_weight_kg) - le calcul de ces deux
    valeurs par détection reste la responsabilité de l'appelant
    (run_application.py ou geo_density_map.py), seul à savoir comment
    obtenir une aire réelle en m² dans son propre mode (reprojection au sol
    en mètres pour le batch, GSD du GeoTIFF pour l'orthomosaïque).

    Retourne un dict :
      - n_detections, total_area_m2, total_weight_kg (somme des SEULES
        détections où weight_kg n'est pas None), n_weight_not_estimable
        (nombre de détections exclues du total pesé, affiché tel quel plutôt
        que masqué - pour que le total affiché ne soit jamais pris pour un
        vrai total sans en avoir l'air).
      - per_class : dict {class_name: {n, area_m2, weight_kg (None si AUCUNE
        détection de cette classe n'a de poids estimable), n_weight_not_estimable}},
        trié par nombre de détections décroissant (la classe la plus
        fréquente en premier - lecture la plus utile pour "qu'est-ce qui
        domine ce batch").
      - calibrated : recopie de weight_estimation.CALIBRATED, pour que
        l'appelant sache s'il doit afficher l'avertissement "estimation
        provisoire, non calibrée" à côté du total pesé.
    """
    total_area = 0.0
    total_weight = 0.0
    n_weight_not_estimable = 0
    per_class: Dict[str, Dict] = {}

    for r in records:
        class_name = r.get("class_name") or "?"
        area = r.get("area_m2")
        weight = r.get("weight_kg")

        entry = per_class.setdefault(
            class_name, {"n": 0, "area_m2": 0.0, "weight_kg": 0.0, "n_weight_not_estimable": 0}
        )
        entry["n"] += 1
        if area:
            entry["area_m2"] += area
            total_area += area
        if weight is not None:
            entry["weight_kg"] += weight
            total_weight += weight
        else:
            entry["n_weight_not_estimable"] += 1
            n_weight_not_estimable += 1

    # Une classe dont AUCUNE détection n'a de poids estimable (ex:
    # Cordage_Filet, Debris_Divers - voir weight_estimation.py) doit afficher
    # "non estimable", jamais un 0 kg trompeur qui laisserait croire à un
    # poids réellement nul.
    for entry in per_class.values():
        if entry["n_weight_not_estimable"] == entry["n"]:
            entry["weight_kg"] = None
        entry["area_m2"] = round(entry["area_m2"], 3)
        if entry["weight_kg"] is not None:
            entry["weight_kg"] = round(entry["weight_kg"], 3)

    per_class_sorted = dict(sorted(per_class.items(), key=lambda kv: kv[1]["n"], reverse=True))

    return {
        "n_detections": len(records),
        "total_area_m2": round(total_area, 3),
        "total_weight_kg": round(total_weight, 3),
        "n_weight_not_estimable": n_weight_not_estimable,
        "per_class": per_class_sorted,
        "calibrated": CALIBRATED,
    }


def render_stats_html(stats: Dict) -> str:
    """Fragment HTML (lecture seule, pas de JS) affichant les statistiques
    agrégées - identique dans web_map.py et geo_density_map.py, à insérer
    dans le panneau existant de chacun (voir leur `_build_html`)."""
    calibration_note = (
        ""
        if stats["calibrated"]
        else (
            '<div class="weight-warning">⚠️ Poids estimé par une formule PROVISOIRE '
            '(non calibrée - voir weight_estimation.py). Ordre de grandeur uniquement.</div>'
        )
    )
    not_estimable_note = (
        f'<div>{stats["n_weight_not_estimable"]} détection(s) sans poids estimable '
        f'(surface non représentative de la masse pour leur classe) - exclue(s) du total pesé.</div>'
        if stats["n_weight_not_estimable"]
        else ""
    )
    total_weight_str = (
        f'{stats["total_weight_kg"]:.2f} kg' if stats["total_weight_kg"] else "0 kg"
    )

    per_class_rows = []
    for name, entry in stats["per_class"].items():
        weight_str = f'{entry["weight_kg"]:.2f} kg' if entry["weight_kg"] is not None else "non estimable"
        per_class_rows.append(
            f'<tr><td>{name}</td><td>{entry["n"]}</td><td>{entry["area_m2"]:.2f} m²</td><td>{weight_str}</td></tr>'
        )
    per_class_table = (
        '<table class="stats-table"><thead><tr><th>Classe</th><th>N</th><th>Surface</th><th>Poids</th></tr></thead>'
        f'<tbody>{"".join(per_class_rows)}</tbody></table>'
    )

    return f"""
<div class="row stats-block">
  <div><b>{stats['n_detections']}</b> détection(s) au total</div>
  <div>Surface totale : <b>{stats['total_area_m2']:.2f} m²</b></div>
  <div>Poids total estimé : <b>{total_weight_str}</b></div>
  {calibration_note}
  {not_estimable_note}
  {per_class_table}
</div>
"""
