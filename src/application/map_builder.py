#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Gabarit HTML/JS UNIQUE de carte web d'inférence, partagé entre
les deux modes du pipeline application : `web_map.py` (batch, photos drone
brutes) et `geo_density_map.py` (orthomosaïque, GeoTIFF unique).

La SEULE différence visuelle entre les deux sorties est le fond de carte -
satellite Esri seul en mode batch, satellite + l'orthomosaïque entière
par-dessus en mode orthomosaïque (`ortho_layer`, voir plus bas). Tout le
reste (popup, légende, filtre par classe, curseur de confiance, panneau de
statistiques agrégées, bascule Points/Densité) est un SEUL gabarit, ici -
évite la duplication (et la divergence progressive) entre deux implémentations
structurellement proches.

Ce que chaque appelant garde de son côté (pas dans ce module) : le calcul des
détections elles-mêmes (inférence, reprojection, dédoublonnage...), le calcul
des bornes de la carte (`bounds`, chacun sait mieux calculer les siennes -
étendue des détections en mode batch, coins réels du GeoTIFF en mode
orthomosaïque), et tout texte réellement spécifique à son mode (`meta_extra_html`,
ex: l'avertissement sur la convention de cap caméra, pertinent seulement en
mode batch où la géolocalisation est directe photo par photo).

`ortho_layer` (None en mode batch) : dict {tiles_rel_path, min_zoom, max_zoom}
décrivant la pyramide de tuiles locale de l'orthomosaïque (voir
`geo_density_map.generate_tile_pyramid`) - affichée par-dessus le satellite à
opacité fixe (1.0, "toute l'ortho dessus" : pas de curseur de transparence,
le curseur de confiance prend cette place dans le panneau, dans les deux
modes).

Contrat de `records` (liste de dicts, un par détection retenue) - mêmes clés
dans les deux modes, sauf `source_photo` (absente en mode orthomosaïque, une
seule "photo" - le GeoTIFF - pour toute la carte) :
    lon, lat, confidence, class_id, class_name, polygon (liste [lat,lon]),
    crop (data URI ou None), area_m2 (ou None), weight_kg (ou None),
    source_photo (optionnel).

Exemple :
    from src.application.map_builder import build_map_html
    html = build_map_html(
        records, stats, bounds=(sw, ne), source_name="essai1", mode_label="Batch (photos brutes)",
    )
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def _read_vendor(filename: str) -> str:
    """Lit une librairie JS/CSS vendorisée sous src/application/vendor/
    (Leaflet + plugin leaflet.heat) - un seul exemplaire partagé par tout le
    pipeline application, y compris geo_density_map.py."""
    path = VENDOR_DIR / filename
    if not path.exists():
        raise RuntimeError(
            f"Librairie vendorisée manquante : {path}. À télécharger une fois depuis cdnjs "
            f"(leaflet 1.9.4 et leaflet.heat 0.2.0) et à conserver dans le repo."
        )
    return path.read_text(encoding="utf-8")


def render_map_page(
    records: List[Dict],
    stats: Dict,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    source_name: str,
    mode_label: str,
    map_max_zoom: int = 22,
    ortho_layer: Optional[Dict] = None,
    meta_extra_html: str = "",
    satellite_max_native_zoom: int = 19,
) -> str:
    """Construit la page HTML autonome (Leaflet + plugin leaflet.heat
    vendorisés) - à ouvrir directement dans un navigateur, volontairement PAS
    publiée comme Artifact claude.ai (CSP bloquerait le fond satellite).

    `bounds` = ((sw_lon, sw_lat), (ne_lon, ne_lat)) - calculé par l'appelant
    (étendue des détections en mode batch, coins du GeoTIFF en mode
    orthomosaïque - voir docstring du module).

    `satellite_max_native_zoom` (défaut 19) : Esri sert une tuile de
    remplacement "Map data not yet available" (code 200, invisible pour la
    logique de repli habituelle de Leaflet) dès qu'on zoome au-delà de la
    résolution RÉELLEMENT disponible pour une zone isolée - la couverture
    haute résolution d'Esri World Imagery est très inégale hors zones
    habitées. Sans risque quand le satellite n'est qu'un fond SECONDAIRE sous
    l'orthomosaïque locale (`ortho_layer` fourni, la tuile de remplacement
    reste cachée dessous), mais DOIT être abaissé (ex: 17, voir web_map.py)
    quand le satellite est le SEUL fond (`ortho_layer` absent). Statut :
    question ouverte par site - la couverture haute résolution d'Esri varie
    localement, la bonne valeur s'ajuste empiriquement (ouvrir le fond Esri
    seul sur la zone, noter le zoom auquel l'image devient floue/indisponible)."""
    from src.application.stats_panel import render_stats_html

    sw, ne = bounds
    detections_json = json.dumps(records, ensure_ascii=False)
    all_classes = sorted(stats["per_class"].keys())
    all_classes_json = json.dumps(all_classes, ensure_ascii=False)
    leaflet_css = _read_vendor("leaflet.min.css")
    leaflet_js = _read_vendor("leaflet.min.js")
    leaflet_heat_js = _read_vendor("leaflet-heat.js")
    stats_html = render_stats_html(stats)

    conf_min = stats["confidence_min"] if stats.get("confidence_min") is not None else 0.0
    conf_max = stats["confidence_max"] if stats.get("confidence_max") is not None else 1.0

    # Couche ortho (fond secondaire, PAR-DESSUS le satellite - voir docstring
    # du module) : opacité FIXE à 1.0 ("toute l'ortho dessus") - pas de
    # curseur de transparence, le curseur de confiance prend cette place dans
    # le panneau.
    if ortho_layer:
        ortho_layer_js = f"""
// Pyramide de tuiles locales de l'orthomosaïque (voir generate_tile_pyramid),
// PAR-DESSUS le satellite, à opacité fixe (1.0 - "toute l'ortho dessus" :
// pas de curseur de transparence).
const ortho = L.tileLayer('{ortho_layer["tiles_rel_path"]}/{{z}}/{{x}}/{{y}}.png', {{
  opacity: 1.0,
  minZoom: 0,
  maxZoom: {map_max_zoom},
  minNativeZoom: {ortho_layer["min_zoom"]},
  maxNativeZoom: {ortho_layer["max_zoom"]},
  bounds: BOUNDS,
  noWrap: true,
  tms: false,
}}).addTo(map);

// Rectangle discret montrant l'emprise exacte de l'orthomosaïque - repère
// utile pour situer la zone étudiée même en dézoomant au-delà de son étendue.
L.rectangle(BOUNDS, {{ color: '#3a7dff', weight: 1.5, fill: false, dashArray: '4,4' }}).addTo(map);
"""
    else:
        ortho_layer_js = ""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>PixelOdyssey — Carte d'inférence</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{leaflet_css}
</style>
<style>
  html, body {{ margin:0; padding:0; height:100%; background:#111; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  #map {{ position:absolute; top:0; bottom:0; left:0; right:0; }}
  #panel {{ position:absolute; top:12px; right:12px; z-index:1000; background:rgba(20,20,20,0.92); color:#eee;
            padding:14px 16px; border-radius:8px; width:280px; box-shadow:0 2px 10px rgba(0,0,0,0.4); font-size:13px;
            max-height:calc(100% - 24px); overflow-y:auto; }}
  #panel h1 {{ font-size:14px; margin:0 0 10px; color:#fff; }}
  #panel .row {{ margin-bottom:10px; }}
  #panel label {{ display:block; margin-bottom:4px; color:#bbb; }}
  .modebtn {{ flex:1; padding:6px 8px; border:1px solid #444; background:#2a2a2a; color:#eee; border-radius:4px;
              cursor:pointer; font-size:12px; }}
  .modebtn.active {{ background:#3a7dff; border-color:#3a7dff; color:#fff; }}
  #modebtns {{ display:flex; gap:6px; }}
  input[type=range] {{ width:100%; }}
  #legend {{ margin-top:10px; padding-top:10px; border-top:1px solid #333; }}
  #legend .swatch {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
  #legend .legend-row {{ margin-bottom:4px; display:flex; align-items:center; gap:4px; }}
  #legend .legend-row label {{ margin:0; color:#eee; cursor:pointer; flex:1; }}
  .stats-block {{ font-size:12px; line-height:1.5; color:#ddd; }}
  .weight-warning {{ color:#ffb300; margin:4px 0; font-size:11px; }}
  .stats-table {{ width:100%; border-collapse:collapse; margin-top:6px; font-size:11px; }}
  .stats-table th, .stats-table td {{ text-align:left; padding:2px 4px; border-bottom:1px solid #333; }}
  #stats {{ margin-top:10px; padding-top:10px; border-top:1px solid #333; }}
  #meta {{ margin-top:10px; padding-top:10px; border-top:1px solid #333; color:#999; font-size:11px; line-height:1.5; }}
  .leaflet-popup-content {{ font-size:12px; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>PixelOdyssey — Carte d'inférence</h1>
  <div class="row">
    <label>Affichage</label>
    <div id="modebtns">
      <button class="modebtn active" id="btnPoints">Détections</button>
      <button class="modebtn" id="btnHeat">Densité</button>
    </div>
  </div>
  <div class="row">
    <label>Confiance minimum affichée : <span id="confVal">{conf_min:.2f}</span></label>
    <input type="range" id="confSlider" min="{conf_min}" max="{conf_max}" step="0.01" value="{conf_min}">
  </div>
  <div id="legend"></div>
  <div id="stats">
    {stats_html}
  </div>
  <div id="meta">
    Mode : {mode_label}<br>
    Source : {source_name}<br>
    {meta_extra_html}
  </div>
</div>

<script>
{leaflet_js}
</script>
<script>
{leaflet_heat_js}
</script>
<script>
const DETECTIONS = {detections_json};
const ALL_CLASSES = {all_classes_json};
const SW = [{sw[1]}, {sw[0]}];
const NE = [{ne[1]}, {ne[0]}];
const BOUNDS = L.latLngBounds(SW, NE);

const map = L.map('map', {{ zoomControl: true, maxZoom: {map_max_zoom} }}).fitBounds(BOUNDS, {{ padding: [40, 40] }});

// Fond UNIQUE commun aux deux modes : satellite public Esri. En mode
// orthomosaïque, la pyramide de tuiles locale vient s'ajouter PAR-DESSUS
// (voir ortho_layer_js ci-dessous) - c'est la SEULE différence visuelle
// entre les deux sorties.
L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: 'Fond satellite : Esri, Maxar, Earthstar Geographics', maxZoom: {map_max_zoom}, maxNativeZoom: {satellite_max_native_zoom} }}
).addTo(map);
{ortho_layer_js}

// Couleurs assignées À L'AVANCE (ordre alphabétique de ALL_CLASSES) -
// nécessaire pour construire la légende/le filtre par classe avant tout
// dessin de la couche de points.
const CLASS_COLORS = {{}};
const PALETTE = ['#ff5252', '#ffb300', '#3a7dff', '#26c281', '#c77dff', '#ff8a5c', '#5cd6ff', '#ff5cbb'];
ALL_CLASSES.forEach((name, i) => {{ CLASS_COLORS[name] = PALETTE[i % PALETTE.length]; }});
function colorForClass(name) {{ return CLASS_COLORS[name] || '#999'; }}

// Filtre par classe (case décochée = classe masquée) - toutes activées par
// défaut, intégré à la légende (voir renderLegend).
let enabledClasses = new Set(ALL_CLASSES);
let currentMinConf = {conf_min};

const pointsLayer = L.layerGroup();

function buildPointsLayer() {{
  // Fermer un popup ouvert AVANT de vider la couche : sinon l'animation de
  // fermeture de Leaflet peut référencer un marqueur déjà retiré par
  // clearLayers() juste après (erreur "Cannot read properties of null
  // (reading '_animating')" observée en test si on omet cette ligne).
  map.closePopup();
  pointsLayer.clearLayers();
  DETECTIONS.forEach(d => {{
    if (d.confidence < currentMinConf) return;
    if (!enabledClasses.has(d.class_name)) return;
    const color = colorForClass(d.class_name);
    const cropHtml = d.crop
      ? `<img src="${{d.crop}}" style="display:block;margin-top:6px;max-width:260px;border-radius:4px;">`
      : '';
    const weightHtml = (d.weight_kg !== null && d.weight_kg !== undefined)
      ? `Poids estimé : ${{d.weight_kg.toFixed(2)}} kg<br>`
      : '';
    const areaHtml = (d.area_m2 !== null && d.area_m2 !== undefined)
      ? `Surface : ${{d.area_m2.toFixed(2)}} m²<br>`
      : '';
    const photoHtml = d.source_photo ? `Photo source : ${{d.source_photo}}<br>` : '';
    const popupHtml = `<b>${{d.class_name}}</b><br>Confiance : ${{(d.confidence * 100).toFixed(0)}}%<br>` +
      `${{areaHtml}}${{weightHtml}}${{photoHtml}}${{cropHtml}}`;

    // Le VRAI contour du masque (reprojeté en lon/lat) - permet de zoomer
    // sur un déchet et de juger si le masque colle à sa forme réelle.
    if (d.polygon && d.polygon.length >= 3) {{
      const mask = L.polygon(d.polygon, {{ color: color, weight: 2, fillColor: color, fillOpacity: 0.35 }});
      mask.bindPopup(popupHtml);
      pointsLayer.addLayer(mask);
    }}
    // Petit point centré en complément : à faible zoom, un masque de
    // quelques cm devient un polygone de quelques pixels écran - quasi
    // impossible à cliquer sans ce repère toujours visible.
    const marker = L.circleMarker([d.lat, d.lon], {{
      radius: 4, color: '#111', weight: 1, fillColor: color, fillOpacity: 0.9,
    }});
    marker.bindPopup(popupHtml);
    pointsLayer.addLayer(marker);
  }});
}}
buildPointsLayer();
pointsLayer.addTo(map);

function computeHeatPoints() {{
  return DETECTIONS.filter(d => d.confidence >= currentMinConf && enabledClasses.has(d.class_name))
                    .map(d => [d.lat, d.lon, 0.4 + d.confidence * 0.6]);
}}
const heatLayer = L.heatLayer(computeHeatPoints(), {{ radius: 28, blur: 22, maxZoom: {map_max_zoom} }});
// leaflet-heat.js accède à `this._map._animating` dans redraw() DÈS QU'IL Y A
// DES DONNÉES - appeler setLatLngs() avec des données non vides pendant que
// la couche n'est PAS encore ajoutée à la carte (`_map` == null) plante avec
// "Cannot read properties of null (reading '_animating')" (observé en test).
// On ne recalcule donc les points de chaleur que quand la couche est
// réellement affichée.
function refreshHeatIfVisible() {{
  if (map.hasLayer(heatLayer)) {{
    heatLayer.setLatLngs(computeHeatPoints());
  }}
}}

function renderLegend() {{
  const el = document.getElementById('legend');
  el.innerHTML = ALL_CLASSES.map(name => {{
    const checked = enabledClasses.has(name) ? 'checked' : '';
    const id = `cls_${{name.replace(/[^a-zA-Z0-9]/g, '_')}}`;
    return `<div class="legend-row">
      <input type="checkbox" id="${{id}}" ${{checked}} data-class="${{name}}">
      <span class="swatch" style="background:${{colorForClass(name)}}"></span>
      <label for="${{id}}">${{name}}</label>
    </div>`;
  }}).join('');
  el.querySelectorAll('input[type=checkbox]').forEach(cb => {{
    cb.addEventListener('change', (e) => {{
      const cls = e.target.getAttribute('data-class');
      if (e.target.checked) {{ enabledClasses.add(cls); }} else {{ enabledClasses.delete(cls); }}
      buildPointsLayer();
      refreshHeatIfVisible();
    }});
  }});
}}
renderLegend();

document.getElementById('btnPoints').addEventListener('click', () => {{
  map.removeLayer(heatLayer);
  pointsLayer.addTo(map);
  document.getElementById('btnPoints').classList.add('active');
  document.getElementById('btnHeat').classList.remove('active');
}});
document.getElementById('btnHeat').addEventListener('click', () => {{
  map.removeLayer(pointsLayer);
  heatLayer.addTo(map);
  refreshHeatIfVisible();
  document.getElementById('btnHeat').classList.add('active');
  document.getElementById('btnPoints').classList.remove('active');
}});
document.getElementById('confSlider').addEventListener('input', (e) => {{
  currentMinConf = parseFloat(e.target.value);
  document.getElementById('confVal').textContent = currentMinConf.toFixed(2);
  buildPointsLayer();
  refreshHeatIfVisible();
}});
</script>
</body>
</html>
"""
