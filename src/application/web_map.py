#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Carte web interactive pour le pipeline "application" (photos
drone brutes géoréférencées directement, PAS une orthomosaïque GeoTIFF - voir
run_application.py et la décision du 04/09/2026, journal_decisions_pipeline.md).

Distinct de `src/review/geo_density_map.py` (qui affiche les prédictions sur
une orthomosaïque WebODM déjà stitchée) : ce module consomme directement la
SORTIE de `run_application.py` (`detections.geojson` + `crops/`), déjà
dédoublonnée inter-photos et déjà en lon/lat WGS84 - aucune inférence, aucun
géoréférencement n'est refait ici, c'est une pure couche de visualisation.

Ce qui est RÉUTILISÉ tel quel de geo_density_map.py (voir sa docstring, "carte
de densité" - même stack, même choix) : Leaflet + plugin leaflet.heat
vendorisés (src/review/vendor/), le motif marqueur+polygone+popup+légende par
classe, le bascule Points/Densité. Voir `_read_vendor`.

Ce qui est DIFFÉRENT et volontairement pas repris :
- Pas de pyramide de tuiles locale (`generate_tile_pyramid`) : ce pipeline ne
  produit aucune orthomosaïque à tuiler, seulement des photos individuelles
  et leurs empreintes au sol calculées séparément. Le seul fond de carte est
  la couche satellite publique Esri (déjà présente dans geo_density_map.py en
  fond secondaire, ici c'est le fond UNIQUE).
- Les bornes de la carte sont calculées à partir de l'étendue des détections
  (pas des coins d'un GeoTIFF, qui n'existe pas ici).
- Les chips (`crops/*.jpg`, fichiers séparés sur disque - voir
  run_application.py::_build_crops) sont encodés en base64 et embarqués DANS
  le HTML au moment de la génération de cette carte, plutôt que lus comme
  fichiers relatifs à l'exécution : contrairement à `geo_density_map.py` (où
  le dossier `tiles/` peut peser plusieurs centaines de Mo, donc doit rester
  à côté du HTML), le nombre de chips d'un batch de terrain reste modeste
  (des centaines, pas des dizaines de milliers) - les embarquer donne un
  fichier HTML UNIQUE, auto-suffisant, partageable par email/Slack sans se
  soucier d'un dossier compagnon. À reconsidérer si un batch produit un
  volume de détections qui rendrait le fichier HTML ingérable.
- Curseur de confiance CÔTÉ CLIENT (filtre les détections déjà exportées,
  n'en ajoute aucune) : utile pour explorer visuellement où placer le seuil
  de confiance final, MAIS ne peut filtrer qu'À LA HAUSSE par rapport au
  `--conf-threshold` déjà appliqué par `run_application.py` avant l'export -
  une détection sous ce seuil n'existe simplement pas dans le GeoJSON. Pour
  une session d'exploration de seuil, relancer `run_application.py` avec un
  `--conf-threshold` bas (voire 0) afin que ce curseur ait vraiment quelque
  chose à explorer - voir le message de livraison de ce script.
  Le seuil `--dedup-iou-threshold`, lui, ne peut PAS s'explorer ainsi : le
  dédoublonnage est déjà figé au moment de l'export (une seule détection
  "gagnante" par cluster), il faut relancer `run_application.py` pour tester
  une autre valeur.

Entrée : --run-dir (dossier de sortie de run_application.py, contenant
detections.geojson + crops/).
Sortie : un fichier HTML autonome (carte Leaflet + chips embarqués), à ouvrir
directement dans un navigateur - volontairement PAS publié comme Artifact
claude.ai, dont la CSP bloquerait le fond satellite Esri (même raison que
geo_density_map.py, voir sa docstring).

Exemple :
    python -m src.application.web_map --run-dir output/application_runs/essai1
"""

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VENDOR_DIR = Path(__file__).resolve().parent.parent / "review" / "vendor"


def _read_vendor(filename: str) -> str:
    """Lit une librairie JS/CSS vendorisée sous src/review/vendor/ (même
    dossier que geo_density_map.py - une seule copie vendorisée pour tout le
    projet, pas de duplication)."""
    path = VENDOR_DIR / filename
    if not path.exists():
        raise RuntimeError(
            f"Librairie vendorisée manquante : {path}. Ce fichier doit déjà exister "
            f"(utilisé aussi par src/review/geo_density_map.py) - vérifier que "
            f"src/review/vendor/ n'a pas été déplacé/supprimé."
        )
    return path.read_text(encoding="utf-8")


def _crop_to_data_uri(crop_path: Path) -> Optional[str]:
    """Encode un chip JPEG en data URI base64, pour l'embarquer directement
    dans le HTML (voir docstring du module - choix différent de
    geo_density_map.py, qui embarque déjà ses crops de la même façon côté
    Python mais génère aussi un dossier tiles/ séparé pour le fond de carte,
    absent ici). Retourne None si le fichier est introuvable (signalé à
    l'appelant, jamais silencieux - un chip manquant dans un GeoJSON produit
    par run_application.py est un signe d'incohérence entre les deux, pas un
    cas normal)."""
    if not crop_path.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(crop_path.read_bytes()).decode("ascii")


def load_detection_records(run_dir: Path) -> Tuple[List[Dict], Dict]:
    """Charge `detections.geojson` produit par run_application.py et le
    convertit en la même forme de dict que geo_density_map.py utilise pour
    `_build_html` (lon/lat/confidence/class_id/class_name/polygon/crop) - même
    contrat JS des deux côtés, pour pouvoir réutiliser le même gabarit de page.

    Retourne (detection_records, stats) où `stats` résume min/max/moyenne de
    confiance et le nombre de détections par classe - affiché dans le panneau
    de la carte, utile tant que les seuils ne sont pas calibrés (voir
    docstring du module)."""
    geojson_path = run_dir / "detections.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"{geojson_path} introuvable - ce script consomme la sortie de "
            f"run_application.py (--output-dir), pas un dossier arbitraire. "
            f"Lance d'abord : python -m src.application.run_application --batch-dir ... "
            f"--output-dir {run_dir}"
        )
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    records: List[Dict] = []
    n_missing_crop = 0
    confidences: List[float] = []
    per_class: Dict[str, int] = {}

    for feature in geojson.get("features", []):
        props = feature["properties"]
        geom = feature["geometry"]
        if geom.get("type") != "Polygon":
            # Garde-fou : run_application.py n'exporte que des Polygon (voir
            # son code, mapping(Polygon(...))) - tout autre type signale un
            # GeoJSON produit par un autre outil, pas celui attendu ici.
            raise ValueError(
                f"Géométrie {geom.get('type')!r} inattendue dans {geojson_path} - "
                f"ce script attend uniquement des Polygon, comme exportés par "
                f"run_application.py."
            )
        # GeoJSON = [lon, lat] par sommet ; Leaflet (L.polygon) attend [lat, lon] -
        # même conversion que _build_html côté geo_density_map.py.
        polygon_latlon = [[lat, lon] for lon, lat in geom["coordinates"][0]]

        crop_file = props.get("crop_file")
        crop_data_uri = None
        if crop_file:
            crop_data_uri = _crop_to_data_uri(run_dir / crop_file)
            if crop_data_uri is None:
                n_missing_crop += 1

        confidence = float(props.get("confidence", 0.0))
        class_name = props.get("class_name", f"classe_{props.get('class_id')}")
        confidences.append(confidence)
        per_class[class_name] = per_class.get(class_name, 0) + 1

        records.append({
            "lon": props["centroid_lon"],
            "lat": props["centroid_lat"],
            "confidence": round(confidence, 4),
            "class_id": props.get("class_id"),
            "class_name": class_name,
            "source_photo": props.get("source_photo", ""),
            "polygon": polygon_latlon,
            "crop": crop_data_uri,
        })

    if n_missing_crop:
        print(f"⚠️  [web_map] {n_missing_crop} détection(s) référencent un chip introuvable sur disque "
              f"(crop_file absent de {run_dir}) - vérifie que le dossier crops/ a bien été copié "
              f"en entier à côté de detections.geojson.")

    stats = {
        "n_detections": len(records),
        "n_missing_crop": n_missing_crop,
        "confidence_min": round(min(confidences), 4) if confidences else None,
        "confidence_max": round(max(confidences), 4) if confidences else None,
        "confidence_mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "per_class": per_class,
    }
    return records, stats


def _compute_bounds(records: List[Dict], margin_ratio: float = 0.15) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Bornes (sw, ne) = (lon, lat) min/max des CENTROÏDES de détections, avec
    une marge - pas de coins de GeoTIFF disponibles ici (voir docstring du
    module). Si aucune détection, retombe sur un cadrage large par défaut
    plutôt que de planter (un batch sans aucune détection reste un résultat
    valide à visualiser - "aucun déchet trouvé" est une information, pas une
    erreur)."""
    if not records:
        return (-20.0, -20.0), (20.0, 20.0)

    lons = [r["lon"] for r in records]
    lats = [r["lat"] for r in records]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Marge proportionnelle à l'étendue réelle, avec un plancher (en degrés)
    # pour qu'un batch très concentré (toutes les détections à quelques
    # mètres les unes des autres) ne produise pas un cadrage ridiculement
    # serré, illisible au premier chargement.
    span_lon = max(max_lon - min_lon, 0.001)
    span_lat = max(max_lat - min_lat, 0.001)
    margin_lon = span_lon * margin_ratio
    margin_lat = span_lat * margin_ratio

    return (min_lon - margin_lon, min_lat - margin_lat), (max_lon + margin_lon, max_lat + margin_lat)


def _build_html(records: List[Dict], stats: Dict, source_name: str, satellite_max_native_zoom: int = 17) -> str:
    """Construit la page HTML autonome - même motif JS que
    `geo_density_map._build_html` (marqueur+polygone+popup+légende+bascule
    Points/Densité), adapté : pas de couche orthomosaïque locale, curseur de
    confiance côté client en plus (voir docstring du module).

    `satellite_max_native_zoom` : CORRECTIF (voir retour terrain) - la valeur
    19 copiée telle quelle de geo_density_map.py (où le satellite n'est qu'un
    fond SECONDAIRE sous l'orthomosaïque locale, donc jamais vraiment
    sollicité à ce niveau de zoom) faisait apparaître la tuile "Map data not
    yet available" qu'Esri sert (avec un code 200, pas une erreur HTTP - donc
    invisible pour la logique de repli habituelle de Leaflet) dès qu'on zoome
    au-delà de la résolution RÉELLEMENT disponible pour une zone isolée comme
    ce site - la couverture haute résolution d'Esri World Imagery est très
    inégale hors zones habitées. Ici le satellite est le SEUL fond, donc ce
    cas se déclenche dès qu'on zoome sur un déchet. Pas de moyen fiable de
    détecter cette tuile de remplacement automatiquement (c'est une vraie
    image PNG, pas une erreur) : la valeur par défaut (17, prudente) doit être
    ajustée empiriquement par site via ce paramètre - ouvre le fond Esri seul
    (ex: https://www.arcgis.com/apps/mapviewer/index.html) sur la zone du
    batch, note le zoom auquel l'image devient floue/indisponible, et passe
    cette valeur (ou un cran en dessous) en `--satellite-max-native-zoom`."""
    detections_json = json.dumps(records, ensure_ascii=False)
    sw, ne = _compute_bounds(records)
    center_lat = (sw[1] + ne[1]) / 2
    center_lon = (sw[0] + ne[0]) / 2
    leaflet_css = _read_vendor("leaflet.min.css")
    leaflet_js = _read_vendor("leaflet.min.js")
    leaflet_heat_js = _read_vendor("leaflet-heat.js")

    per_class_lines = "".join(
        f"<div>{name} : {count}</div>" for name, count in sorted(stats["per_class"].items())
    )
    conf_min = stats["confidence_min"] if stats["confidence_min"] is not None else 0.0
    conf_max = stats["confidence_max"] if stats["confidence_max"] is not None else 1.0

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>PixelOdyssey — Carte terrain</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{leaflet_css}
</style>
<style>
  html, body {{ margin:0; padding:0; height:100%; background:#111; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  #map {{ position:absolute; top:0; bottom:0; left:0; right:0; }}
  #panel {{ position:absolute; top:12px; right:12px; z-index:1000; background:rgba(20,20,20,0.92); color:#eee;
            padding:14px 16px; border-radius:8px; width:260px; box-shadow:0 2px 10px rgba(0,0,0,0.4); font-size:13px;
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
  #legend div {{ margin-bottom:4px; }}
  #stats {{ margin-top:10px; padding-top:10px; border-top:1px solid #333; color:#999; font-size:11px; line-height:1.5; }}
  #meta {{ margin-top:10px; padding-top:10px; border-top:1px solid #333; color:#999; font-size:11px; line-height:1.5; }}
  .leaflet-popup-content {{ font-size:12px; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>PixelOdyssey — Carte terrain</h1>
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
    {stats['n_detections']} détection(s) au total.<br>
    Confiance : min {conf_min:.2f} / moy {stats['confidence_mean'] or 0:.2f} / max {conf_max:.2f}<br>
    {per_class_lines}
  </div>
  <div id="meta">
    Source : {source_name}<br>
    ⚠️ Convention de cap caméra (yaw) et seuil IoU de dédoublonnage non encore
    validés empiriquement - vérifier que des photos voisines produisent bien
    des empreintes cohérentes avant de te fier aux positions affichées.
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
const SW = [{sw[1]}, {sw[0]}];
const NE = [{ne[1]}, {ne[0]}];
const BOUNDS = L.latLngBounds(SW, NE);

const map = L.map('map', {{ zoomControl: true, maxZoom: 22 }}).fitBounds(BOUNDS, {{ padding: [40, 40] }});

// Pas d'orthomosaïque locale ici (pipeline "application" = photos brutes
// géoréférencées directement, voir docstring du module) - le fond satellite
// public Esri est la SEULE couche de fond, contrairement à geo_density_map.py
// où il n'est qu'un complément à la pyramide de tuiles locale.
L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: 'Fond satellite : Esri, Maxar, Earthstar Geographics', maxZoom: 22, maxNativeZoom: {satellite_max_native_zoom} }}
).addTo(map);

const CLASS_COLORS = {{}};
const PALETTE = ['#ff5252', '#ffb300', '#3a7dff', '#26c281', '#c77dff', '#ff8a5c', '#5cd6ff', '#ff5cbb'];
let colorIdx = 0;
function colorForClass(name) {{
  if (!(name in CLASS_COLORS)) {{
    CLASS_COLORS[name] = PALETTE[colorIdx % PALETTE.length];
    colorIdx++;
  }}
  return CLASS_COLORS[name];
}}

let currentMinConf = {conf_min};
let pointLayers = [];  // {{det, marker, mask}} - reconstruit à chaque changement de seuil

function buildPointsLayer() {{
  // Fermer un popup ouvert AVANT de vider la couche : sinon l'animation de
  // fermeture de Leaflet peut référencer un marqueur déjà retiré par
  // clearLayers() juste après (erreur "Cannot read properties of null
  // (reading '_animating')" observée en test si on omet cette ligne).
  map.closePopup();
  pointsLayer.clearLayers();
  pointLayers = [];
  DETECTIONS.forEach(d => {{
    if (d.confidence < currentMinConf) return;
    const color = colorForClass(d.class_name);
    const cropHtml = d.crop
      ? `<img src="${{d.crop}}" style="display:block;margin-top:6px;max-width:260px;border-radius:4px;">`
      : '';
    const popupHtml = `<b>${{d.class_name}}</b><br>Confiance : ${{(d.confidence * 100).toFixed(0)}}%<br>` +
      `Photo source : ${{d.source_photo}}${{cropHtml}}`;

    if (d.polygon && d.polygon.length >= 3) {{
      const mask = L.polygon(d.polygon, {{ color: color, weight: 2, fillColor: color, fillOpacity: 0.35 }});
      mask.bindPopup(popupHtml);
      pointsLayer.addLayer(mask);
    }}
    const marker = L.circleMarker([d.lat, d.lon], {{
      radius: 4, color: '#111', weight: 1, fillColor: color, fillOpacity: 0.9,
    }});
    marker.bindPopup(popupHtml);
    pointsLayer.addLayer(marker);
  }});
}}

const pointsLayer = L.layerGroup();
buildPointsLayer();
pointsLayer.addTo(map);

function computeHeatPoints() {{
  return DETECTIONS.filter(d => d.confidence >= currentMinConf)
                    .map(d => [d.lat, d.lon, 0.4 + d.confidence * 0.6]);
}}
const heatLayer = L.heatLayer(computeHeatPoints(), {{ radius: 28, blur: 22, maxZoom: 22 }});
// leaflet-heat.js accède à `this._map._animating` dans redraw() DÈS QU'IL Y A
// DES DONNÉES (voir simpleheat._heat) - appeler setLatLngs() avec des
// données non vides pendant que la couche n'est PAS encore ajoutée à la
// carte (`_map` == null) plante avec "Cannot read properties of null
// (reading '_animating')" (observé en test). On ne recalcule donc les
// points de chaleur que quand la couche est réellement affichée.
function refreshHeatIfVisible() {{
  if (map.hasLayer(heatLayer)) {{
    heatLayer.setLatLngs(computeHeatPoints());
  }}
}}

function renderLegend() {{
  const el = document.getElementById('legend');
  el.innerHTML = Object.entries(CLASS_COLORS).map(([name, color]) =>
    `<div><span class="swatch" style="background:${{color}}"></span>${{name}}</div>`
  ).join('');
}}
renderLegend();

document.getElementById('btnPoints').addEventListener('click', () => {{
  map.removeLayer(heatLayer);
  pointsLayer.addTo(map);
  document.getElementById('btnPoints').classList.add('active');
  document.getElementById('btnHeat').classList.remove('active');
  document.getElementById('legend').style.display = 'block';
}});
document.getElementById('btnHeat').addEventListener('click', () => {{
  map.removeLayer(pointsLayer);
  heatLayer.addTo(map);
  refreshHeatIfVisible();  // rafraîchit avec le seuil de confiance courant, maintenant que _map est défini
  document.getElementById('btnHeat').classList.add('active');
  document.getElementById('btnPoints').classList.remove('active');
  document.getElementById('legend').style.display = 'none';
}});
document.getElementById('confSlider').addEventListener('input', (e) => {{
  currentMinConf = parseFloat(e.target.value);
  document.getElementById('confVal').textContent = currentMinConf.toFixed(2);
  buildPointsLayer();
  renderLegend();
  refreshHeatIfVisible();
}});
</script>
</body>
</html>
"""


def build_web_map(run_dir: str, output_path: Optional[str] = None, satellite_max_native_zoom: int = 17) -> Path:
    run_dir_p = Path(run_dir)
    records, stats = load_detection_records(run_dir_p)

    print(f"--- 🗺️  CARTE WEB — PIPELINE APPLICATION ---")
    print(f"    Dossier source : {run_dir_p}")
    print(f"    {stats['n_detections']} détection(s) chargée(s) ({stats['n_missing_crop']} chip(s) manquant(s)).")

    html = _build_html(records, stats, source_name=run_dir_p.name, satellite_max_native_zoom=satellite_max_native_zoom)

    out_p = Path(output_path) if output_path else run_dir_p / "carte.html"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[SUCCÈS] Carte écrite : {out_p} (fichier unique, chips embarqués - partageable tel quel).")
    return out_p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                         help="Dossier de sortie de run_application.py (contient detections.geojson + crops/).")
    parser.add_argument("--output", default=None,
                         help="Fichier HTML de sortie (défaut : <run-dir>/carte.html).")
    parser.add_argument("--satellite-max-native-zoom", type=int, default=17,
                         help="Zoom max réellement demandé au fond satellite Esri (défaut 17, prudent) - "
                              "au-delà, Leaflet agrandit la dernière tuile disponible au lieu d'en "
                              "demander une nouvelle. Nécessaire pour éviter la tuile de remplacement "
                              "'Map data not yet available' qu'Esri sert (avec un code 200 - invisible "
                              "pour Leaflet) hors des zones à imagerie haute résolution. À ajuster par "
                              "site - voir la docstring de _build_html pour la méthode.")
    args = parser.parse_args()
    build_web_map(run_dir=args.run_dir, output_path=args.output,
                  satellite_max_native_zoom=args.satellite_max_native_zoom)


if __name__ == "__main__":
    main()
