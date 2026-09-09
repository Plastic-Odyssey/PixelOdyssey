#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Carte web interactive pour le pipeline "application" (photos
drone brutes géoréférencées directement, PAS une orthomosaïque GeoTIFF - voir
run_application.py).

Distinct de `geo_density_map.py` (qui affiche les prédictions sur une
orthomosaïque WebODM déjà stitchée) : ce module consomme directement la
SORTIE de `run_application.py` (`detections.geojson` + `crops/`), déjà
dédoublonnée inter-photos et déjà en lon/lat WGS84 - aucune inférence, aucun
géoréférencement n'est refait ici, c'est une pure couche de visualisation.

Le gabarit HTML/JS de la carte (panneau, popup, légende+filtre par classe,
curseur de confiance, stats agrégées, bascule Points/Densité) est PARTAGÉ
avec `geo_density_map.py` via `map_builder.render_map_page` - la SEULE
différence visuelle entre les deux sorties est le fond de carte (satellite
seul ici, satellite + orthomosaïque entière par-dessus en mode orthomosaïque).
Ce module ne garde que ce qui lui est VRAIMENT propre : le chargement du
GeoJSON produit par `run_application.py`, l'encodage des chips en data URI,
et le calcul des bornes de la carte à partir de l'étendue des détections
(pas de coins de GeoTIFF disponibles ici, contrairement au mode orthomosaïque).

Ce qui reste DIFFÉRENT et volontairement pas repris du mode orthomosaïque :
- Pas de pyramide de tuiles locale : ce pipeline ne produit aucune
  orthomosaïque à tuiler, seulement des photos individuelles et leurs
  empreintes au sol calculées séparément.
- Les bornes de la carte sont calculées à partir de l'étendue des détections
  (pas des coins d'un GeoTIFF, qui n'existe pas ici).
- Les chips (`crops/*.jpg`, fichiers séparés sur disque - voir
  run_application.py::_save_winner_crops) sont encodés en base64 et embarqués
  DANS le HTML au moment de la génération de cette carte, plutôt que lus
  comme fichiers relatifs à l'exécution : contrairement au mode orthomosaïque
  (où le dossier `tiles/` peut peser plusieurs centaines de Mo, donc doit
  rester à côté du HTML), le nombre de chips d'un batch de terrain reste
  modeste (des centaines, pas des dizaines de milliers) - les embarquer donne
  un fichier HTML UNIQUE, auto-suffisant, partageable par email/Slack sans se
  soucier d'un dossier compagnon.
- Le curseur de confiance CÔTÉ CLIENT (filtre les détections déjà exportées,
  n'en ajoute aucune) ne peut filtrer qu'À LA HAUSSE par rapport au
  `--conf-threshold` déjà appliqué par `run_application.py` avant l'export -
  une détection sous ce seuil n'existe simplement pas dans le GeoJSON. Pour
  une session d'exploration de seuil, relancer `run_application.py` avec un
  `--conf-threshold` bas (voire 0) afin que ce curseur ait vraiment quelque
  chose à explorer.
  Le seuil `--dedup-iou-threshold`, lui, ne peut PAS s'explorer ainsi : le
  dédoublonnage est déjà figé au moment de l'export (une seule détection
  "gagnante" par cluster), il faut relancer `run_application.py` pour tester
  une autre valeur.

Entrée : --run-dir (dossier de sortie de run_application.py, contenant
detections.geojson + crops/).
Sortie : un fichier HTML autonome (carte Leaflet + chips embarqués), à ouvrir
directement dans un navigateur - volontairement PAS publié comme Artifact
claude.ai, dont la CSP bloquerait le fond satellite Esri.

Exemple :
    python -m src.application.web_map --run-dir "E:\\PixelOdyssey\\4. Results\\2_prediction\\batch_<horodatage>"
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.application.map_builder import render_map_page
from src.application.stats_panel import compute_aggregate_stats

MODE_LABEL = "Batch (photos brutes)"

META_EXTRA_HTML = (
    "⚠️ Convention de cap caméra (yaw) et seuil IoU de dédoublonnage non encore "
    "validés empiriquement - vérifier que des photos voisines produisent bien "
    "des empreintes cohérentes avant de te fier aux positions affichées."
)


def _crop_to_data_uri(crop_path: Path) -> Optional[str]:
    """Encode un chip JPEG en data URI base64, pour l'embarquer directement
    dans le HTML (voir docstring du module). Retourne None si le fichier est
    introuvable (signalé à l'appelant, jamais silencieux - un chip manquant
    dans un GeoJSON produit par run_application.py est un signe
    d'incohérence entre les deux, pas un cas normal)."""
    if not crop_path.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(crop_path.read_bytes()).decode("ascii")


def load_detection_records(run_dir: Path) -> Tuple[List[Dict], Dict]:
    """Charge `detections.geojson` produit par run_application.py et le
    convertit en la forme de dict attendue par `map_builder.render_map_page`
    (lon/lat/confidence/class_id/class_name/polygon/crop/area_m2/weight_kg/
    source_photo).

    Retourne (detection_records, stats) où `stats` fusionne les statistiques
    de confiance (min/max/moyenne, pour le curseur) et les statistiques
    agrégées de surface/poids/par-classe (voir stats_panel.compute_aggregate_stats).
    """
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
        # GeoJSON = [lon, lat] par sommet ; Leaflet (L.polygon) attend [lat, lon].
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

        records.append({
            "lon": props["centroid_lon"],
            "lat": props["centroid_lat"],
            "confidence": round(confidence, 4),
            "class_id": props.get("class_id"),
            "class_name": class_name,
            "source_photo": props.get("source_photo", ""),
            "polygon": polygon_latlon,
            "crop": crop_data_uri,
            "area_m2": props.get("area_m2"),
            "weight_kg": props.get("weight_kg"),
        })

    if n_missing_crop:
        print(f"⚠️  [web_map] {n_missing_crop} détection(s) référencent un chip introuvable sur disque "
              f"(crop_file absent de {run_dir}) - vérifie que le dossier crops/ a bien été copié "
              f"en entier à côté de detections.geojson.")

    agg_stats = compute_aggregate_stats(records)
    stats = {
        **agg_stats,
        "n_missing_crop": n_missing_crop,
        "confidence_min": round(min(confidences), 4) if confidences else None,
        "confidence_max": round(max(confidences), 4) if confidences else None,
        "confidence_mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
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

    span_lon = max(max_lon - min_lon, 0.001)
    span_lat = max(max_lat - min_lat, 0.001)
    margin_lon = span_lon * margin_ratio
    margin_lat = span_lat * margin_ratio

    return (min_lon - margin_lon, min_lat - margin_lat), (max_lon + margin_lon, max_lat + margin_lat)


def build_web_map(run_dir: str, output_path: Optional[str] = None, satellite_max_native_zoom: int = 17) -> Path:
    run_dir_p = Path(run_dir)
    records, stats = load_detection_records(run_dir_p)

    print(f"--- 🗺️  CARTE WEB — PIPELINE APPLICATION (BATCH) ---")
    print(f"    Dossier source : {run_dir_p}")
    print(f"    {stats['n_detections']} détection(s) chargée(s) ({stats['n_missing_crop']} chip(s) manquant(s)).")

    bounds = _compute_bounds(records)
    html = render_map_page(
        records, stats, bounds=bounds, source_name=run_dir_p.name, mode_label=MODE_LABEL,
        map_max_zoom=22, ortho_layer=None, meta_extra_html=META_EXTRA_HTML,
        satellite_max_native_zoom=satellite_max_native_zoom,
    )

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
                              "le satellite est ici le SEUL fond (pas d'orthomosaïque locale en mode "
                              "batch) - voir map_builder.render_map_page pour le détail.")
    args = parser.parse_args()
    build_web_map(run_dir=args.run_dir, output_path=args.output,
                  satellite_max_native_zoom=args.satellite_max_native_zoom)


if __name__ == "__main__":
    main()
