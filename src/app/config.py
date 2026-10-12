"""Chemins, registre des modèles de l'application et réglages par défaut."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MODELS_FILE = REPO_ROOT / "config" / "app_models.yaml"
SENSOR_FILE = REPO_ROOT / "config" / "sensor_specs.yaml"
IMPORT_DIR = Path.home() / "PixelOdyssey_import"
RECENT_FILE = Path.home() / ".pixelodyssey_recents.json"

PHOTO_EXTS = {".jpg", ".jpeg"}

# Réglages du traitement
FEATURE_SCALE = 4            # points homologues cherchés sur l'image réduite au quart
FEATURE_COUNT = 6000
PAIR_MIN_OVERLAP = 0.25      # recouvrement minimal (emprises GPS) pour tenter un appariement
PAIR_MAX_PER_PHOTO = 10
PAIR_MIN_INLIERS = 40
PAIR_MIN_POINTS = 30          # points homologues répartis retenus au minimum par paire
PAIR_MIN_SPREAD = 0.12        # surface couverte par ces points (fraction de l'image)
MATCH_SIGMA_PX = 4.0         # précision attendue d'un point homologue (pixels pleine résolution)
GPS_SIGMA_M = 2.0             # précision relative du GPS du drone entre deux photos
AFFINE_PRIOR = 0.10          # écart toléré au facteur d échelle commun déduit de l altitude
DEFORMATION_PRIOR = 0.03     # anisotropie / cisaillement propres à une photo (relief, perspective)
PAIR_MAX_RESIDUAL_M = 0.20     # au-delà, la paire est jugée fausse et retirée
PHOTO_SCALE_PRIOR = 0.08        # échelle propre de chaque photo (relief, altitude réelle au-dessus du sol)
PAIR_EFFECTIVE_POINTS = 4       # poids maximal d'une paire (en observations indépendantes)
YAW_PRIOR_DEG = 3.0             # cap de la nacelle (XMP DJI), une fois le biais commun retiré
ROLLING_SHUTTER_PRIOR = 0.05    # glissement pendant la lecture du capteur (fraction de la hauteur d'image)
YAW_BIAS_PRIOR_DEG = 30.0       # biais de cap commun : quasi libre
HUBER_K = 3.0                   # résidus au-delà de 3 écarts-types : poids réduit (faux appariements)
ALIGN_MAX_GPS_MEDIAN_M = 5.0    # garde-fous de l'alignement (au-delà : passe écartée)
ALIGN_MAX_GPS_SHIFT_M = 25.0
ALIGN_MAX_YAW_CORR_DEG = 45.0   # par photo (le cap XMP est parfois décalé pendant les virages)
ALIGN_MAX_YAW_MEDIAN_DEG = 12.0
ALIGN_MAX_RESIDUAL_M = 0.15     # résidu médian maximal accepté pour l alignement
OWNER_GRID_M = 0.20          # résolution de la carte de propriété
TILE_SIZE = 640
TILE_OVERLAP = 256
TILE_MARGIN_PX = 320         # marge autour de la zone propre d'une photo pour ne couper aucun objet
CONF_FLOOR = 0.05            # seuil bas conservé dans les fichiers (le curseur filtre ensuite)
DETECT_BATCH = 16
DETECTION_MODE = "fusion"     # "fusion" : toutes les vues de chaque déchet ; "proprietaire" : la photo la plus centrale seule
FUSION_SCORE = "top2"         # score d'un objet : moyenne des deux meilleures vues (une vue seule compte pour moitié)
FUSION_IOU = 0.25             # deux détections de photos différentes = même objet si elles se recouvrent au sol...
FUSION_DIST_M = 0.08          # ... ou si leurs centres sont à moins de 8 cm
CONTRAST_ENHANCE = "auto"     # False, "auto" (photos ternes seulement) ou "toujours" : égalisation locale du contraste
LOW_CONTRAST_STD = 28.0       # seuil "photo terne" (écart-type de la luminance, 0-255)
CLAHE_CLIP = 2.0
CLAHE_CELL_PX = 80
MOSAIC_MAX_PIXELS = 900e6    # au-delà, la résolution de la mosaïque est dégradée
MOSAIC_CHUNK = 2048
MAX_PITCH_DEV_DEG = 8.0      # tolérance à la verticale (au-delà : photo exclue)


def load_models() -> List[Dict]:
    with open(MODELS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = []
    for m in data.get("models", []):
        m = dict(m)
        m["chemin"] = str((REPO_ROOT / m["poids"]).resolve())
        m["disponible"] = Path(m["chemin"]).exists()
        out.append(m)
    return out


def get_model(model_id: str) -> Dict:
    for m in load_models():
        if m["id"] == model_id:
            return m
    raise KeyError(f"Modèle inconnu : {model_id}")


def load_recents() -> List[str]:
    try:
        return json.loads(RECENT_FILE.read_text(encoding="utf-8"))[:8]
    except Exception:  # noqa: BLE001
        return []


def add_recent(path: str) -> None:
    rec = [p for p in load_recents() if p != path]
    rec.insert(0, path)
    try:
        RECENT_FILE.write_text(json.dumps(rec[:8], ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def default_output_dir(photo_dir: str) -> str:
    p = Path(photo_dir)
    return str(p.parent / f"{p.name}_pixelodyssey")


def sensor_registry() -> Dict:
    try:
        with open(SENSOR_FILE, "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("sensors", {})
    except Exception:  # noqa: BLE001
        return {}


def find(model_id: Optional[str]) -> Optional[Dict]:
    try:
        return get_model(model_id) if model_id else None
    except KeyError:
        return None
