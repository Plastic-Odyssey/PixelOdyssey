"""Détection des déchets sur les photos d'origine (GPU si disponible), limitée à la zone propre
de chaque photo, puis projection au sol et mesures."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon

from src.application.weight_estimation import estimate_weight_range_kg
from src.data.utils.tiling_geometry import iter_tile_windows
from src.review.matching import LabeledPolygon
from src.review.tiled_inference import nms_merge

from . import config
from .inventory import Photo
from .ownership import Ownership
from .registration import PhotoTransform, imread

Progress = Callable[[float, str], None]

_MODEL_LOCK = threading.Lock()
_MODELS: Dict[str, object] = {}


def load_model(path: str):
    with _MODEL_LOCK:
        if path not in _MODELS:
            from ultralytics import YOLO
            _MODELS[path] = YOLO(path)
        return _MODELS[path]


def device_label() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU {torch.cuda.get_device_name(0)}"
    except Exception:  # noqa: BLE001
        pass
    return "processeur (pas de GPU détecté : traitement plus lent)"


_BATCH = [config.DETECT_BATCH]
STATS: Dict[str, int] = {}  # compteurs du dernier passage (journal)  # réduit automatiquement si la carte graphique manque de mémoire


def _free_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def needs_contrast(img: np.ndarray) -> bool:
    """Photo terne ? (écart-type de la luminance ; ~43 sur un vol normal, ~19 sur W5 voilé)"""
    mode = config.CONTRAST_ENHANCE
    if mode is True or mode == "toujours":
        return True
    if mode != "auto":
        return False
    g = cv2.cvtColor(img[::8, ::8], cv2.COLOR_BGR2GRAY)
    return float(g.std()) < config.LOW_CONTRAST_STD


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Égalisation locale du contraste (CLAHE sur la luminance) : rend aux photos ternes (ciel voilé,
    sable surexposé) le contraste des images d'entraînement, sans toucher aux couleurs."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = img.shape[:2]
    grid = (max(1, w // config.CLAHE_CELL_PX), max(1, h // config.CLAHE_CELL_PX))  # cases de ~80 px, comme sur une tuile 640 en 8 x 8
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=grid).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _detect_photo(model, img: np.ndarray, windows: List) -> List[LabeledPolygon]:
    preds: List[LabeledPolygon] = []
    i = 0
    while i < len(windows):
        bs = _BATCH[0]
        chunk = windows[i:i + bs]
        try:
            results = model.predict([img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk], conf=config.CONF_FLOOR,
                                    imgsz=config.TILE_SIZE, verbose=False)
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError hérite de RuntimeError
            if "out of memory" in str(e).lower() and bs > 1:
                _BATCH[0] = max(1, bs // 2)
                _free_gpu()
                continue
            raise
        for (x0, y0, _, _), r in zip(chunk, results):
            if r.masks is None:
                continue
            for c, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.xy):
                if len(xy) < 3:
                    continue
                g = Polygon([(float(x) + x0, float(y) + y0) for x, y in xy])
                if g.is_valid and g.area > 0:
                    preds.append(LabeledPolygon(int(c), g, float(cf)))
        i += len(chunk)
    return nms_merge(preds, iou_threshold=0.5)


def detect_all(photos: List[Photo], transforms: Dict[int, PhotoTransform], own: Ownership, model_cfg: Dict,
               cache_dir: Path, progress: Progress, cancel: Optional[threading.Event] = None) -> List[Dict]:
    fusion = config.DETECTION_MODE == "fusion"
    # cache propre à chaque réglage (photo entière ou zone utile, contraste) : changer de réglage recalcule
    cache_dir = cache_dir.with_name(cache_dir.name + ("_fusion" if fusion else "") +
                                    ("" if not config.CONTRAST_ENHANCE else f"_contraste-{config.CONTRAST_ENHANCE}"))
    STATS.clear()
    STATS.update(photos_contraste=0, detections_brutes=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_cfg["chemin"])
    names = {int(k): v for k, v in model.names.items()}
    out: List[Dict] = []
    # fenêtres utiles de chaque photo restant à traiter (les autres sont relues depuis le cache)
    plan: Dict[int, tuple] = {}
    for p in photos:
        cf = cache_dir / f"{Path(p.name).stem}_{p.idx}.json"
        if not cf.exists():
            all_w = list(dict.fromkeys(iter_tile_windows(p.width, p.height, config.TILE_SIZE,
                                                         config.TILE_SIZE - config.TILE_OVERLAP)))
            plan[p.idx] = all_w if fusion else own.useful_windows(p, all_w)
    # lecture des photos suivantes en parallèle de la détection (la carte graphique n'attend pas le disque)
    queue = [p for p in photos if plan.get(p.idx)]
    pool = ThreadPoolExecutor(max_workers=2)
    futures: Dict[int, object] = {}

    def prefetch(j: int) -> None:
        for q in queue[j:j + 3]:
            if q.idx not in futures:
                futures[q.idx] = pool.submit(imread, q.path)

    qi = 0
    try:
        for k, p in enumerate(photos):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Traitement annulé")
            dets = _photo_detections(p, plan, futures, prefetch, qi, cache_dir, transforms[p.idx], own, model, names,
                                     keep_all=fusion)
            if plan.get(p.idx):
                qi += 1
            for d in dets:
                d["photo"] = p.name
                d["photo_idx"] = p.idx
            out.extend(dets)
            msg = f"{len(out)} détections brutes" if fusion else f"{len(out)} objets retenus"
            progress((k + 1) / len(photos), f"Détection : photo {k + 1}/{len(photos)}, {msg}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    STATS["detections_brutes"] = len(out)
    if fusion:
        progress(1.0, f"Fusion des vues : {len(out)} détections brutes")
        out = fuse_views(out, photos, transforms, own)
    final: List[Dict] = []
    by_idx = {p.idx: p for p in photos}
    for d in out:
        _add_weights(by_idx[d["photo_idx"]], [d], final)
    out = final
    for i, d in enumerate(out):
        d["id"] = i
    return out


def _photo_detections(p: Photo, plan, futures, prefetch, qi, cache_dir: Path, T: PhotoTransform, own: Ownership,
                      model, names, keep_all: bool = False) -> List[Dict]:
    cf = cache_dir / f"{Path(p.name).stem}_{p.idx}.json"
    if p.idx not in plan:
        return json.loads(cf.read_text(encoding="utf-8"))
    wins = plan[p.idx]
    dets: List[Dict] = []
    if wins:
        prefetch(qi)
        img = futures.pop(p.idx).result()
        prefetch(qi + 1)
        if img is not None and needs_contrast(img):
            img = enhance_contrast(img)
            STATS["photos_contraste"] = STATS.get("photos_contraste", 0) + 1
        if img is not None:
            for lp in _detect_photo(model, img, wins):
                cen = np.array(lp.geom.centroid.coords[0])
                g_c = T.to_ground(cen[None, :])[0]
                if not keep_all and own.owner_of(g_c[None, :])[0] != p.idx:
                    continue
                geom = lp.geom.simplify(0.8, preserve_topology=True)
                if geom.geom_type != "Polygon":
                    geom = max(geom.geoms, key=lambda q: q.area)
                px = np.asarray(geom.exterior.coords)[:-1]
                ground = T.to_ground(px)
                scale = float(np.sqrt(abs(np.linalg.det(T.A))))  # m / pixel (isotrope)
                rect = cv2.minAreaRect(px.astype(np.float32))
                x0, y0, x1, y1 = geom.bounds
                dets.append({
                    "classe": names.get(lp.class_id, str(lp.class_id)), "classe_id": lp.class_id,
                    "confiance": round(lp.confidence, 4),
                    "surface_m2": round(geom.area * scale ** 2, 5),
                    "longueur_m": round(max(rect[1]) * scale, 3),
                    "px": [[round(float(a), 1), round(float(b), 1)] for a, b in px],
                    "bbox": [int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))],
                    "sol": [[round(float(a), 3), round(float(b), 3)] for a, b in ground],
                    "centre_sol": [round(float(g_c[0]), 3), round(float(g_c[1]), 3)],
                    # contour coupé par le bord de la photo : vue incomplète, moins fiable pour la forme
                    "bord": bool(x0 <= 2 or y0 <= 2 or x1 >= p.width - 3 or y1 >= p.height - 3),
                })
    cf.write_text(json.dumps(dets), encoding="utf-8")
    return dets


def _add_weights(p: Photo, dets: List[Dict], out: List[Dict]) -> None:
    for d in dets:
        d.setdefault("photo", p.name)
        d.setdefault("photo_idx", p.idx)
        rng = estimate_weight_range_kg(d["classe"], d["surface_m2"])
        d["poids_kg"] = round(rng[1], 4) if rng else None
        d["poids_bas_kg"] = round(rng[0], 4) if rng else None
        d["poids_haut_kg"] = round(rng[2], 4) if rng else None
        out.append(d)


def _available_views(points: np.ndarray, photos: List[Photo], transforms: Dict[int, PhotoTransform],
                     margin: int = 40) -> np.ndarray:
    """Nombre de photos qui voient chaque point du sol (à distance du bord de l'image)."""
    n = np.zeros(len(points), int)
    if not len(points):
        return n
    for p in photos:
        u = transforms[p.idx].to_pixel(points)
        n += (u[:, 0] >= margin) & (u[:, 0] < p.width - margin) & (u[:, 1] >= margin) & (u[:, 1] < p.height - margin)
    return n


def fuse_views(raw: List[Dict], photos: Optional[List[Photo]] = None,
               transforms: Optional[Dict[int, PhotoTransform]] = None, own: Optional[Ownership] = None) -> List[Dict]:
    """Fusion multi-vues « propriétaire + rattrapage ».

    1. Chaque zone du sol est lue sur la photo où elle est la plus centrale (« propriétaire ») : ses détections
       sont gardées telles quelles, chaque déchet n'est compté qu'une fois.
    2. Le modèle ne reconnaît pas un déchet sur toutes les photos qui le voient. Les détections faites sur les
       AUTRES photos, à un endroit où le propriétaire n'a rien trouvé, sont regroupées entre photos (recalage à
       quelques centimètres) et rattrapées si elles sont confirmées : score = moyenne des deux meilleures vues
       (une vue seule compte pour moitié, sauf si aucune autre photo ne voit cet endroit).
    Les objets denses vus par le propriétaire ne sont jamais fusionnés entre eux (pas de sous-comptage)."""
    from shapely.geometry import Polygon as _P
    from shapely.strtree import STRtree
    items = []
    for d in raw:
        try:
            g = _P(d["sol"]).buffer(0)
        except Exception:  # noqa: BLE001
            continue
        if not g.is_empty and g.area > 0:
            items.append((d, g))
    if not items:
        return []
    cen = np.array([d["centre_sol"] for d, _ in items])
    if own is not None:
        is_owner = own.owner_of(cen) == np.array([d["photo_idx"] for d, _ in items])
    else:
        is_owner = np.zeros(len(items), bool)
    kept = [dict(items[i][0]) for i in np.nonzero(is_owner)[0]]
    for o in kept:
        o["vues"] = 1
    owner_geoms = [items[i][1] for i in np.nonzero(is_owner)[0]]
    owner_tree = STRtree(owner_geoms) if owner_geoms else None
    owner_cen = cen[is_owner]
    # candidats au rattrapage : détections hors photo propriétaire, là où le propriétaire n'a rien vu
    cand = []
    for i in np.nonzero(~is_owner)[0]:
        d, g = items[i]
        dup = False
        if owner_tree is not None:
            for j in owner_tree.query(g.buffer(config.FUSION_DIST_M)):
                j = int(j)
                if owner_geoms[j].intersects(g) or np.linalg.norm(owner_cen[j] - cen[i]) < config.FUSION_DIST_M:
                    dup = True
                    break
        if not dup:
            cand.append((d, g, cen[i]))
    # regroupement des candidats entre photos
    n = len(cand)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if n:
        geoms = [g for _, g, _ in cand]
        tree = STRtree(geoms)
        for i, (d, g, c) in enumerate(cand):
            for j in tree.query(g.buffer(config.FUSION_DIST_M)):
                j = int(j)
                if j <= i or cand[j][0]["photo_idx"] == d["photo_idx"]:
                    continue
                h = geoms[j]
                inter = g.intersection(h).area
                same = np.linalg.norm(c - cand[j][2]) < config.FUSION_DIST_M
                if inter > 0:
                    same = same or inter / g.union(h).area >= config.FUSION_IOU or inter / min(g.area, h.area) >= 0.6
                if same:
                    parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    groups_l = list(groups.values())
    centers = np.array([np.mean([cand[i][2] for i in ids], 0) for ids in groups_l]) if groups_l else np.zeros((0, 2))
    avail = _available_views(centers, photos, transforms) if photos else np.full(len(groups_l), 2)
    for gi, ids in enumerate(groups_l):
        best_by_photo: Dict[int, Dict] = {}
        for i in ids:
            d = cand[i][0]
            b = best_by_photo.get(d["photo_idx"])
            if b is None or d["confiance"] > b["confiance"]:
                best_by_photo[d["photo_idx"]] = d
        views = sorted(best_by_photo.values(), key=lambda d: -d["confiance"])
        c1 = views[0]["confiance"]
        c2 = views[1]["confiance"] if len(views) > 1 else 0.0
        score = (c1 + c2) / 2 if (config.FUSION_SCORE == "top2" and avail[gi] >= 2) else c1
        rep = max(views, key=lambda d: d["confiance"] - (0.3 if d.get("bord") else 0.0))
        votes: Dict[str, float] = {}
        for d in views:
            votes[d["classe"]] = votes.get(d["classe"], 0.0) + d["confiance"]
        cls = max(votes, key=votes.get)
        obj = dict(rep)
        if cls != rep["classe"]:
            alt = next(d for d in views if d["classe"] == cls)
            obj["classe"], obj["classe_id"] = cls, alt["classe_id"]
        obj["confiance"] = round(float(score), 4)
        obj["confiance_meilleure_vue"] = round(float(c1), 4)
        obj["vues"] = len(views)
        obj["rattrape"] = True
        kept.append(obj)
    for o in kept:
        o.pop("bord", None)
    return kept
