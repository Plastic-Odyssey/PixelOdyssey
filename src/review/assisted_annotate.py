#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Production d'annotations assistées par le modèle (cold start).

Revue manuelle, une détection à la fois, des prédictions du modèle sur une
image encore vierge de toute annotation, avant de les écrire comme vérité
terrain candidate. Contrairement à `label_review.py` (qui complète/corrige
un lot déjà annoté), un nouveau lot n'a pas de taxonomie fine existante :
son data.yaml déclare directement les super-classes de
`config/data_config.yaml` comme classes locales - la classe validée ici est
la classe finale, sans passage par une sous-classe fine.

Un outil = une image = une revue = un lot - mais si l'image donnée dépasse
la limite d'import CVAT (~150M px, voir `split_for_cvat.CVAT_MAX_PIXELS`),
cet outil s'en occupe LUI-MÊME (restauré le 02/09/2026, suite à un retour
"trop lent/trop compliqué à deux commandes séparées") : découpage en grille
AUTOMATIQUE (réutilise `split_for_cvat.split_image_for_cvat`, jamais
dupliqué), PUIS chaque morceau est revu l'un après l'autre, exactement comme
si `assisted_annotate.py` était relancé à la main sur chacun. `split_for_cvat.py`
reste utilisable seul si tu veux juste les morceaux sans lancer de revue tout
de suite (ex: pour un import CVAT direct sans passer par ce module).

Point de vigilance préservé du 31/08/2026 (à ne pas réintroduire) : le
découpage se fait TOUJOURS avant toute inférence (lecture/écriture de
pixels seule, rapide, aucun modèle impliqué) - jamais après avoir fait
tourner l'inférence sur l'image ENTIÈRE puis fragmenté seulement à
l'écriture (c'est CE choix-là, pas le découpage en lui-même, qui obligeait à
attendre l'écriture de toute la grille avant même d'ouvrir l'interface).
Chaque morceau est ensuite traité en entier (inférence + revue + écriture)
avant de passer au suivant - l'interface s'ouvre donc aussi vite pour un
morceau que pour n'importe quelle image de taille normale.

Fonctionnement :
    1. Inférence tuilée sur l'image donnée (tiled_inference.predict_parent_image).
    2. Seules les prédictions >= --conf-threshold (défaut 0.5) sont proposées
       à la revue.
    3. Page web locale (serveur intégré à Python) : un déchet à la fois, chip
       à résolution native avec le contour du masque en surimpression, un
       bandeau de classe éditable (pré-rempli avec la classe prédite) et deux
       boutons - Valider (écrit le masque avec la classe actuellement
       affichée dans le bandeau) et Supprimer (rien n'est écrit).
    4. Le lot est initialisé sous RESULTS_DIR/<lot_name>/ (images/train/ +
       labels/train/ + data.yaml - même format que 1_annotated_dataset, mais
       PAS écrit directement dedans : voir RESULTS_DIR ci-dessous pour le
       raisonnement) DÈS QUE la revue démarre (image copiée et fichier de
       label VIDE créé), puis chaque clic "Valider" AJOUTE immédiatement sa
       ligne au fichier de label - rien n'est accumulé en mémoire jusqu'à la
       fin. "Enregistrer et terminer" ne fait donc plus qu'arrêter la revue
       (aucune écriture lourde à ce moment-là) : une interruption en cours de
       route (fermeture, plantage, Ctrl-C) ne perd que la revue restante,
       jamais les annotations déjà validées.
    4bis. Si `--auto-write` : saute entièrement l'étape 3 (aucune interface,
       aucune vérification humaine détection par détection). Demande
       interactivement un seuil de confiance (Entrée pour garder celui de
       --conf-threshold), puis écrit DIRECTEMENT toutes les détections
       au-dessus de ce seuil, avec leur classe PRÉDITE par le modèle telle
       quelle. À réserver à un seuil élevé et/ou à un lot qui repassera par
       une passe de contrôle complémentaire dans CVAT - un avertissement est
       affiché avant écriture.

RESULTS_DIR (E:\\PixelOdyssey\\4. Results\\1_assisted_annotation) plutôt que
1_annotated_dataset directement : ce lot est une PROPOSITION issue du
modèle + revue manuelle, pas encore une vérité terrain définitive au même
titre que les lots annotés from scratch - le format de sortie reste
strictement identique (images/train + labels/train + data.yaml) pour rester
immédiatement réimportable dans CVAT pour une passe de contrôle
complémentaire, ou copié tel quel dans 1_annotated_dataset si aucune
correction n'est jugée nécessaire.

Entrée : --image (chemin vers l'image à annoter - N'IMPORTE QUELLE taille,
le découpage sous la limite CVAT est automatique si besoin), --lot-name (nom
du nouveau lot - devient un préfixe si l'image est découpée en plusieurs
morceaux), --model (optionnel, sélection interactive sinon).
Sortie : le(s) lot(s) annoté(s) écrit(s) sous RESULTS_DIR/<lot_name>/ (un
seul lot si l'image tenait sous la limite CVAT, un lot par morceau sinon -
voir RESULTS_DIR/<lot_name>_y<y0>_x<x0>/).

Exemple :
    python -m src.review.assisted_annotate --image "chemin/vers/image.tif" --lot-name "SL_nouveau_lot"
    python -m src.review.assisted_annotate --image ... --lot-name ... --model output/runs/<run>/weights/best.pt
    python -m src.review.assisted_annotate --image ... --lot-name ... --auto-write
    python -m src.review.assisted_annotate --image "chemin/vers/grosse_ortho.tif" --lot-name "SL_nouveau_lot"
        (orthomosaïque trop grande pour CVAT : découpée automatiquement, chaque
        morceau revu l'un après l'autre sous SL_nouveau_lot_y<y0>_x<x0>/)

Puis ouvrir l'URL affichée (http://127.0.0.1:8765/ par défaut) dans un
navigateur (sauf en mode --auto-write, qui n'ouvre aucune interface) - une
URL par morceau si l'image a dû être découpée, l'une après l'autre.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import cv2
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import DEFAULT_CLASS_CONFIG_PATH, assert_model_matches_taxonomy, load_class_config
from src.data.image_io import load_image_bgr
from src.review.label_review import RUNS_DIR, _discover_available_models, _prompt_model_choice
from src.review.matching import LabeledPolygon
from src.review.split_for_cvat import CVAT_MAX_PIXELS, split_image_for_cvat
from src.review.tiled_inference import make_ultralytics_predict_fn, predict_parent_image

MARGIN_PX = 80          # marge autour du masque dans le chip de revue
MIN_CROP_DISPLAY_PX = 360  # un chip plus petit que ça est agrandi pour rester lisible
OUTLINE_COLOR_BGR = (0, 235, 255)  # jaune vif (cohérent avec geo_density_map.py, converti BGR<->RGB)

# Dossier de sortie des lots produits par cet outil - voir le raisonnement
# dans la docstring du module (staging avant CVAT/promotion vers
# 1_annotated_dataset, pas écrit directement dans le dataset d'entraînement).
RESULTS_DIR = r"E:\PixelOdyssey\4. Results\1_assisted_annotation"

# En dessous de ce nombre de candidates, une classe est signalée comme
# "échantillon trop faible" dans le résumé - repère déjà discuté avec
# l'utilisateur (2026-08-27) : sous ~10 instances, une statistique par classe
# (ici juste un compte, pas encore une métrique de qualité) est trop bruitée
# pour juger quoi que ce soit dessus, seulement pour repérer une présence.
LOW_SAMPLE_WARN_THRESHOLD = 10


def _build_crop_jpeg(img, geom, margin_px: int = MARGIN_PX) -> bytes:
    """Découpe un chip à résolution NATIVE autour du masque (jamais sous-
    échantillonné), dessine le contour en surimpression, agrandit si le chip
    est petit (LANCZOS, même choix que geo_density_map.build_detection_crops),
    encode en JPEG. `img` est le tableau BGR complet déjà chargé en mémoire -
    pas de relecture disque par détection."""
    h, w = img.shape[:2]
    minx, miny, maxx, maxy = geom.bounds
    x0 = max(0, int(minx) - margin_px)
    y0 = max(0, int(miny) - margin_px)
    x1 = min(w, int(maxx) + margin_px)
    y1 = min(h, int(maxy) + margin_px)
    crop = img[y0:y1, x0:x1].copy()

    contour = [(int(x - x0), int(y - y0)) for x, y in geom.exterior.coords]
    if len(contour) >= 3:
        import numpy as np

        pts = np.array([contour], dtype=np.int32)
        cv2.polylines(crop, pts, isClosed=True, color=OUTLINE_COLOR_BGR, thickness=3)

    ch, cw = crop.shape[:2]
    if max(ch, cw) < MIN_CROP_DISPLAY_PX and ch > 0 and cw > 0:
        scale = MIN_CROP_DISPLAY_PX / max(ch, cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_LANCZOS4)

    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Échec d'encodage JPEG du chip de revue.")
    return buf.tobytes()


def _normalize_geom_to_full_image(geom, img_w: int, img_h: int) -> List[float]:
    """Normalise les coordonnées PIXEL d'un polygone (repère image complète)
    en coordonnées YOLO-seg [0,1], dans l'ordre x1 y1 x2 y2 ... - factorisé
    entre la validation manuelle (_ReviewState.decide) et l'écriture directe
    (_run_auto_write) pour ne garder qu'une seule implémentation de cette
    conversion."""
    norm: List[float] = []
    for x, y in geom.exterior.coords:
        norm.extend([
            max(0.0, min(1.0, x / img_w)),
            max(0.0, min(1.0, y / img_h)),
        ])
    return norm


def _prompt_confidence_threshold(default: float) -> float:
    """Demande interactivement un seuil de confiance (0 exclu, 1 inclus) -
    Entrée seule garde `default`. Reboucle sur une entrée invalide plutôt que
    d'accepter silencieusement une valeur hors bornes (un seuil <= 0 ou > 1
    rendrait le filtre de confiance trivialement plein ou trivialement vide,
    sans avertissement) ou de planter sur une entrée non numérique."""
    while True:
        raw = input(
            f"    Seuil de confiance à partir duquel écrire une annotation (0-1, Entrée = {default:.2f}) : "
        ).strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("    Valeur invalide - entre un nombre entre 0 et 1 (ex: 0.6).")
            continue
        if not (0.0 < value <= 1.0):
            print("    Le seuil doit être compris entre 0 (exclu) et 1 (inclus).")
            continue
        return value


def _init_lot(
    lot_dir: Path,
    image_path: Path,
    target_names: Dict[int, str],
) -> Path:
    """Initialise le lot sous RESULTS_DIR/<lot_name>/ - images/train/ +
    labels/train/ + data.yaml - TÔT, avant la moindre décision de revue :
    copie l'image, crée un fichier de label VIDE, et écrit data.yaml. Rien
    ici ne dépend des décisions de revue, donc rien n'empêche de l'écrire
    dès que le lot est confirmé non vide (candidates non vides) plutôt que
    d'attendre la fin de la revue.

    Chaque décision validée est ensuite ajoutée au fur et à mesure au
    fichier de label déjà existant - voir `_append_validated_line`.

    Le sous-dossier `train/` est requis par le format d'import CVAT
    ("Ultralytics YOLO Segmentation" : structure `images/<subset>/` +
    `labels/<subset>/`) ; le pipeline d'ingestion standard le tolère aussi
    bien qu'un dossier `images/` à plat (mirroring récursif images->labels).
    Un seul "split" ici (`train`, nom arbitraire) : ce lot n'a pas de notion
    train/val/test avant `split_dataset.py`.

    Entrée : dossier du lot, chemin de l'image source (déjà sous la limite
    CVAT - voir split_for_cvat.py pour une orthomosaïque trop grande),
    noms de classes.
    Sortie : chemin du fichier de label (déjà créé VIDE sur disque)."""
    img_dir = lot_dir / "images" / "train"
    lab_dir = lot_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    print(f"    💾 Copie de {image_path.name} dans le nouveau lot...")
    import shutil

    shutil.copy2(image_path, img_dir / image_path.name)
    label_path = lab_dir / f"{image_path.stem}.txt"
    label_path.write_text("", encoding="utf-8")

    data_yaml_path = lot_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                # `path` + `train` : pas lus par l'ingestion pipeline (seule la
                # clé `names` l'est) mais requis par l'import CVAT pour
                # localiser les images associées aux labels.
                "path": ".",
                "train": "images/train",
                "names": {int(k): v for k, v in target_names.items()},
            },
            f, allow_unicode=True, sort_keys=False,
        )

    return label_path


def _append_validated_line(v: Dict, label_path: Path) -> None:
    """Ajoute IMMÉDIATEMENT une annotation validée (coordonnées normalisées
    PLEINE IMAGE) au fichier de label - appelée à chaque décision « Valider »
    (revue manuelle) ou pour chaque détection retenue (--auto-write), jamais
    accumulée en RAM pour une écriture groupée en fin de traitement (voir le
    raisonnement dans la docstring du module et le journal du 31/08 : c'est
    ce report de toute l'écriture à la toute fin qui a fait perdre 320
    annotations validées lors d'un plantage en cours d'écriture).

    Entrée : l'annotation validée ({class_id, coords_norm}), chemin du
    fichier de label (déjà initialisé par `_init_lot`).
    Sortie : aucune - append en mode texte (`"a"`)."""
    coords_str = " ".join(f"{c:.6f}" for c in v["coords_norm"])
    with open(label_path, "a", encoding="utf-8") as f:
        f.write(f"{v['class_id']} {coords_str}\n")


class _ReviewState:
    """État partagé du serveur de revue - un seul utilisateur, une seule
    session à la fois (pas besoin de plus pour cet outil).

    Le lot est déjà initialisé sur disque (image copiée + label VIDE +
    data.yaml) AVANT la construction de cet état, via `_init_lot` -
    `label_path` référence ce fichier déjà créé. `decide()` écrit
    immédiatement chaque validation (voir `_append_validated_line`) ;
    `finish()` ne fait donc plus AUCUNE écriture (juste marquer la revue
    comme terminée pour l'interface) - voir le raisonnement dans la
    docstring du module (journal du 31/08)."""

    def __init__(self, img, candidates: List[LabeledPolygon], target_names: Dict[int, str],
                 label_path: Path, img_w: int, img_h: int):
        self.img = img
        self.candidates = candidates
        self.target_names = target_names
        self.label_path = label_path
        self.img_w = img_w
        self.img_h = img_h
        self.cursor = 0
        self.validated_count = 0
        self.deleted_count = 0
        self.finished = False
        self.lock = threading.Lock()
        self._crop_cache: Dict[int, bytes] = {}

    def crop_bytes(self, idx: int) -> bytes:
        if idx not in self._crop_cache:
            self._crop_cache[idx] = _build_crop_jpeg(self.img, self.candidates[idx].geom)
        return self._crop_cache[idx]

    def state_json(self) -> dict:
        total = len(self.candidates)
        if self.finished:
            return {
                "finished": True,
                "total": total,
                "validated": self.validated_count,
                "deleted": self.deleted_count,
                "output": [str(self.label_path)],
            }
        if self.cursor >= total:
            return {"finished": False, "done_reviewing": True, "total": total,
                    "validated": self.validated_count, "deleted": self.deleted_count}
        pred = self.candidates[self.cursor]
        classes = [{"id": cid, "name": name} for cid, name in sorted(self.target_names.items())]
        return {
            "finished": False,
            "done_reviewing": False,
            "index": self.cursor,
            "total": total,
            "reviewed": self.cursor,
            "validated": self.validated_count,
            "deleted": self.deleted_count,
            "predicted_class_id": pred.class_id,
            "predicted_class_name": self.target_names.get(pred.class_id, str(pred.class_id)),
            "confidence": round(pred.confidence, 3) if pred.confidence is not None else None,
            "classes": classes,
        }

    def decide(self, idx: int, action: str, class_id: int) -> None:
        with self.lock:
            if idx != self.cursor or self.finished:
                return  # décision périmée (double-clic, page rechargée) - ignorée
            if action == "valider":
                geom = self.candidates[idx].geom
                norm = _normalize_geom_to_full_image(geom, self.img_w, self.img_h)
                v = {"class_id": int(class_id), "coords_norm": norm}
                _append_validated_line(v, self.label_path)
                self.validated_count += 1
            else:
                self.deleted_count += 1
            self.cursor += 1

    def finish(self) -> dict:
        with self.lock:
            # Rien à écrire ici : chaque validation a déjà été persistée
            # immédiatement par decide(). finish() ne fait plus que marquer
            # la revue comme terminée pour l'interface web.
            self.finished = True
            return self.state_json()


def _run_auto_write(
    predictions: List[LabeledPolygon],
    conf_threshold: float,
    target_names: Dict[int, str],
    lot_dir: Path,
    image_path: Path,
    img_w: int,
    img_h: int,
) -> List[Path]:
    """Chemin ALTERNATIF à la revue manuelle (`--auto-write`) : demande
    interactivement un seuil de confiance, puis écrit directement toutes les
    détections au-dessus de ce seuil comme annotations validées - la classe
    écrite est TOUJOURS celle prédite par le modèle, jamais vérifiée une à
    une. Aucune interface web n'est ouverte dans ce chemin.

    Entrée : détections brutes (non filtrées par --conf-threshold - le seuil
    réellement appliqué est celui choisi interactivement ici, --conf-threshold
    ne sert que de valeur par défaut suggérée dans le prompt).
    Sortie : liste des chemins de labels écrits (un seul élément - même
    contrat de retour que le chemin manuel, pour ne pas avoir à distinguer
    les deux côté appelant).

    Le lot est initialisé (`_init_lot`) puis chaque détection retenue est
    ajoutée au fur et à mesure (`_append_validated_line`) plutôt que
    d'attendre d'avoir toute la liste pour écrire en un bloc - même
    raisonnement qu'en revue manuelle, même si le risque de perte humaine
    est ici moindre (aucune décision manuelle en jeu)."""
    print("\n--- ⚡ MODE ÉCRITURE DIRECTE (--auto-write, sans revue manuelle) ---")
    print("    Chaque détection au-dessus du seuil choisi sera écrite avec sa classe PRÉDITE, sans")
    print("    aucune vérification humaine - à réserver à un seuil élevé, ou à un lot qui repassera")
    print("    par CVAT pour une passe de contrôle avant intégration à 1_annotated_dataset.")
    write_threshold = _prompt_confidence_threshold(conf_threshold)

    to_write = sorted(
        [p for p in predictions if p.confidence is not None and p.confidence >= write_threshold],
        key=lambda p: p.confidence, reverse=True,
    )
    print(f"    {len(to_write)} détection(s) >= {write_threshold:.0%} seront écrites directement.")
    if not to_write:
        raise RuntimeError(
            f"Aucune détection >= {write_threshold:.0%} sur cette image à ce seuil - aucun lot créé. "
            f"Relance avec un seuil plus bas si besoin."
        )

    label_path = _init_lot(lot_dir, image_path, target_names)
    for p in to_write:
        v = {"class_id": p.class_id, "coords_norm": _normalize_geom_to_full_image(p.geom, img_w, img_h)}
        _append_validated_line(v, label_path)

    print(f"\n--- ✅ Lot écrit : {lot_dir} ---")
    print(f"    {len(to_write)} annotation(s) écrite(s) SANS validation manuelle (seuil {write_threshold:.0%}).")
    print(f"    Ce lot est au format standard (images/train + labels/train + data.yaml) mais N'EST PAS")
    print(f"    dans 1_annotated_dataset : repasse-le par CVAT si besoin d'une passe de contrôle, ou")
    print(f"    copie-le tel quel dans 1_annotated_dataset/{lot_dir.name}/ puis lance data_pipeline.py --force.")
    return [label_path]


def _make_handler(state: _ReviewState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence le log par requête - trop verbeux pour cet usage
            pass

        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - nom imposé par BaseHTTPRequestHandler
            path = urlparse(self.path).path
            if path == "/":
                body = _PAGE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self._send_json(state.state_json())
            elif path.startswith("/api/crop/"):
                try:
                    idx = int(path.rsplit("/", 1)[-1])
                    body = state.crop_bytes(idx)
                except (ValueError, IndexError):
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                payload = {}

            if path == "/api/decide":
                state.decide(
                    idx=int(payload.get("idx", -1)),
                    action=str(payload.get("action", "")),
                    class_id=int(payload.get("class_id", -1)),
                )
                self._send_json(state.state_json())
            elif path == "/api/finish":
                self._send_json(state.finish())
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


_PAGE_HTML = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>PixelOdyssey — annotation assistée</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 640px; margin: 2.5rem auto; padding: 0 1.2rem; }
  h1 { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.2rem; }
  #progress { color: #888; font-size: 0.85rem; margin-bottom: 1.2rem; }
  #crop-wrap { text-align: center; background: #1118; border-radius: 8px; padding: 0.8rem; }
  #crop { max-width: 100%; max-height: 60vh; border-radius: 4px; }
  #band { display: flex; align-items: center; gap: 0.6rem; margin: 1rem 0; }
  #band label { font-size: 0.8rem; color: #888; }
  #class-select { flex: 1; font-size: 1.05rem; padding: 0.5rem 0.6rem; border-radius: 6px; }
  #conf { font-size: 0.8rem; color: #888; }
  #buttons { display: flex; gap: 0.8rem; margin-top: 0.6rem; }
  button { flex: 1; padding: 0.9rem; font-size: 1rem; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
  #btn-valider { background: #2f7f78; color: white; }
  #btn-supprimer { background: #a8532c; color: white; }
  #btn-finish { background: transparent; border: 1px solid #888; color: inherit; margin-top: 1.5rem; width: 100%; padding: 0.6rem; font-weight: 400; }
  #done { text-align: center; padding: 3rem 0; }
  .hidden { display: none; }
</style>
</head>
<body>
  <h1>Production d'annotations assistées</h1>
  <div id="progress"></div>

  <div id="review">
    <div id="crop-wrap"><img id="crop" src="" alt="détection à valider"></div>
    <div id="band">
      <label for="class-select">Classe</label>
      <select id="class-select"></select>
      <span id="conf"></span>
    </div>
    <div id="buttons">
      <button id="btn-valider">✓ Valider</button>
      <button id="btn-supprimer">✕ Supprimer</button>
    </div>
    <button id="btn-finish">Enregistrer et terminer maintenant</button>
  </div>

  <div id="done" class="hidden"></div>

<script>
let current = null;

async function refresh() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();
    if (s.finished) {
      document.getElementById('review').classList.add('hidden');
      const d = document.getElementById('done');
      d.classList.remove('hidden');
      d.innerHTML = `<h2>Terminé</h2>
        <p>${s.validated} annotation(s) validée(s), ${s.deleted} supprimée(s).</p>
        <p style="color:#888;font-size:0.85rem;">Écrit dans : ${s.output.join(', ')}</p>
        <p style="color:#888;font-size:0.85rem;">Tu peux fermer cette page.</p>`;
      return;
    }
    if (s.done_reviewing) {
      document.getElementById('progress').textContent =
        `${s.total} / ${s.total} passées en revue — clique « Enregistrer et terminer » pour écrire le lot.`;
      document.getElementById('review').querySelector('#crop-wrap').classList.add('hidden');
      document.getElementById('band').classList.add('hidden');
      document.getElementById('buttons').classList.add('hidden');
      current = null;
      return;
    }
    current = s;
    document.getElementById('progress').textContent =
      `${s.reviewed} / ${s.total} revues — ${s.validated} validée(s), ${s.deleted} supprimée(s)`;
    document.getElementById('crop').src = `/api/crop/${s.index}?t=${Date.now()}`;
    const sel = document.getElementById('class-select');
    sel.innerHTML = '';
    for (const c of s.classes) {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.name;
      if (c.id === s.predicted_class_id) opt.selected = true;
      sel.appendChild(opt);
    }
    document.getElementById('conf').textContent =
      s.confidence !== null ? `confiance modèle : ${(s.confidence * 100).toFixed(0)}%` : '';
  } catch (err) {
    document.getElementById('progress').textContent =
      'Erreur de communication avec le serveur local (' + err + ') — recharge la page. ' +
      'Rassure-toi : toute annotation déjà validée avant cette erreur est déjà sur disque.';
  }
}

async function decide(action) {
  if (!current) return;
  const class_id = parseInt(document.getElementById('class-select').value, 10);
  await fetch('/api/decide', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx: current.index, action, class_id}),
  });
  refresh();
}

document.getElementById('btn-valider').addEventListener('click', () => decide('valider'));
document.getElementById('btn-supprimer').addEventListener('click', () => decide('supprimer'));
document.getElementById('btn-finish').addEventListener('click', async () => {
  const btn = document.getElementById('btn-finish');
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = 'Enregistrement…';
  try {
    await fetch('/api/finish', {method: 'POST'});
    await refresh();
  } catch (err) {
    btn.disabled = false;
    btn.textContent = original;
    alert(
      "Échec de la requête de fin (connexion perdue ?) : " + err +
      "\\nRassure-toi : chaque annotation validée a déjà été écrite sur disque au moment du clic sur " +
      "Valider, rien n'est perdu - réessaie simplement ce bouton."
    );
  }
});
document.addEventListener('keydown', (e) => {
  if (!current) return;
  if (e.key === 'Enter') decide('valider');
  if (e.key === 'Backspace' || e.key === 'Delete') decide('supprimer');
});

refresh();
</script>
</body>
</html>
"""


def run_assisted_annotate(
    image_path: Union[str, Path],
    lot_name: str,
    model_path: Optional[str] = None,
    results_dir: Union[str, Path] = RESULTS_DIR,
    conf_threshold: float = 0.5,
    tile_conf_threshold: float = 0.25,
    tile_size: int = 640,
    overlap: int = 256,
    auto_write: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    predict_tile_fn=None,
    max_pixels: int = CVAT_MAX_PIXELS,
) -> List[Path]:
    """`predict_tile_fn` : comme dans label_review.py/visualize_predictions.py,
    permet d'injecter un faux prédicteur pour les tests sans modèle réel.

    `image_path` peut être de N'IMPORTE QUELLE taille : si elle dépasse
    `max_pixels` (la limite d'import CVAT par défaut), cette fonction découpe
    D'ABORD automatiquement via `split_for_cvat.split_image_for_cvat` (pixels
    seulement, aucune inférence à ce stade - voir la docstring du module),
    puis s'appelle elle-même récursivement UNE FOIS PAR MORCEAU (chaque
    appel se comporte alors exactement comme sur une image de taille normale
    - inférence, revue, écriture, l'un après l'autre, jamais en parallèle).
    `lot_name` devient le préfixe de chaque lot de morceau
    (`<lot_name>_y<y0>_x<x0>`). Retourne alors la liste concaténée des
    chemins de labels écrits (un par morceau).

    `auto_write` : saute entièrement l'interface de revue manuelle - demande
    interactivement (input()) un seuil de confiance, puis écrit directement
    toutes les détections au-dessus comme annotations validées, classe
    prédite telle quelle. Voir `_run_auto_write` pour le détail et la mise en
    garde. `host`/`port`/`open_browser` sont ignorés dans ce mode (aucun
    serveur n'est démarré).

    Retourne la liste des chemins de labels écrits (un seul élément). En
    mode revue manuelle (`auto_write=False`, comportement par défaut
    inchangé), bloque jusqu'à la fin de la revue (cette fonction ne rend la
    main qu'après que le serveur a reçu la décision de terminer, via
    /api/finish déclenché par le bouton). En mode `auto_write=True`, rend la
    main dès l'écriture terminée (après le seul prompt de seuil)."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise RuntimeError(f"Image introuvable : {image_path}")

    # Garde-fou taille, vérifié TÔT (avant tout chargement d'image coûteux) : si
    # l'image dépasse la limite d'import CVAT, on découpe D'ABORD (pixels seuls,
    # rapide, voir split_for_cvat.py) puis on traite chaque morceau comme une
    # image normale - un par un, jamais en parallèle (voir docstring de la
    # fonction et le point de vigilance du 31/08/2026 dans la docstring du
    # module : ne JAMAIS faire tourner l'inférence sur l'image entière avant de
    # découper, c'est cet ordre-là qui retardait l'ouverture de l'interface).
    probe_img = load_image_bgr(image_path)
    if probe_img is None:
        raise RuntimeError(f"Image illisible : {image_path}")
    probe_h, probe_w = probe_img.shape[:2]
    if probe_w * probe_h > max_pixels:
        print(
            f"⚠️  {image_path.name} ({probe_w}x{probe_h} = {probe_w * probe_h:,} px) dépasse la limite "
            f"d'import CVAT ({max_pixels:,} px) - découpage automatique en morceaux (voir "
            f"split_for_cvat.split_image_for_cvat), chacun ensuite revu l'un après l'autre sous "
            f"'{lot_name}_y<y0>_x<x0>'."
        )
        pieces_source_dir = Path(results_dir) / "_pieces_source" / lot_name
        pieces = split_image_for_cvat(image_path, pieces_source_dir, max_pixels=max_pixels)
        print(
            f"    {len(pieces)} morceau(x) source écrit(s) sous {pieces_source_dir} (pas des lots "
            f"d'annotation - juste l'image découpée, revue à suivre ci-dessous)."
        )
        all_label_paths: List[Path] = []
        stem_prefix = f"{image_path.stem}_"
        for i, piece_path in enumerate(pieces, start=1):
            suffix = piece_path.stem[len(stem_prefix):] if piece_path.stem.startswith(stem_prefix) else str(i)
            piece_lot_name = f"{lot_name}_{suffix}"
            print(f"\n=== Morceau {i}/{len(pieces)} : {piece_path.name} → lot '{piece_lot_name}' ===")
            piece_labels = run_assisted_annotate(
                piece_path, piece_lot_name,
                model_path=model_path, results_dir=results_dir,
                conf_threshold=conf_threshold, tile_conf_threshold=tile_conf_threshold,
                tile_size=tile_size, overlap=overlap, auto_write=auto_write,
                host=host, port=port, open_browser=open_browser,
                predict_tile_fn=predict_tile_fn, max_pixels=max_pixels,
            )
            all_label_paths.extend(piece_labels)
        print(f"\n--- ✅ {len(pieces)} morceau(x) traité(s), {len(all_label_paths)} lot(s) écrit(s) au total ---")
        return all_label_paths

    lot_dir = Path(results_dir) / lot_name
    if lot_dir.exists() and any(lot_dir.iterdir()):
        raise RuntimeError(
            f"Le dossier de lot {lot_dir} existe déjà et n'est pas vide - choisis un autre "
            f"--lot-name pour ne jamais écraser un lot existant sans le vouloir."
        )

    if predict_tile_fn is None:
        if not model_path:
            available_models = _discover_available_models(RUNS_DIR)
            if not available_models:
                raise RuntimeError(
                    f"Aucun modèle entraîné trouvé sous {RUNS_DIR}. Lance d'abord un entraînement "
                    f"(python -m src.training.train), ou fournis --model explicitement."
                )
            model_path = _prompt_model_choice(available_models, RUNS_DIR)
        predict_tile_fn = make_ultralytics_predict_fn(model_path, conf_threshold=tile_conf_threshold)

    _, target_names = load_class_config(DEFAULT_CLASS_CONFIG_PATH)

    # Garde-fou : ce lot cold-start écrit directement la classe prédite comme classe
    # finale (pas de placeholder à corriger, voir la docstring du module) - un modèle
    # entraîné sous une autre taxonomie que la config actuelle produirait donc des ID
    # silencieusement réinterprétés comme une AUTRE classe, sans aucun garde-fou visible
    # dans l'interface. Voir assert_model_matches_taxonomy. Absent pour un
    # predict_tile_fn injecté en test (pas d'attribut model_names).
    model_names = getattr(predict_tile_fn, "model_names", None)
    if model_names is not None:
        assert_model_matches_taxonomy(model_names, target_names, model_label=str(model_path or ""))

    print(f"--- 🔎 Inférence sur {image_path.name} ---")
    # `probe_img`/`probe_w`/`probe_h` déjà chargés plus haut pour le contrôle de
    # taille CVAT - réutilisés ici tels quels plutôt que de relire l'image une
    # 2e fois (coûteux sur une grosse orthomosaïque).
    img, img_w, img_h = probe_img, probe_w, probe_h

    predictions = predict_parent_image(image_path, predict_tile_fn, tile_size=tile_size, overlap=overlap)
    candidates = sorted(
        [p for p in predictions if p.confidence is not None and p.confidence >= conf_threshold],
        key=lambda p: p.confidence, reverse=True,
    )
    print(f"    {len(predictions)} détection(s) brute(s), {len(candidates)} au-dessus de "
          f"{conf_threshold:.0%} de confiance - proposée(s) à la revue.")

    stats = _prediction_stats(predictions, candidates, target_names, img_w, img_h)
    _print_prediction_stats(stats, conf_threshold)

    if not predictions:
        print("    Rien à revoir - aucun lot créé.")
        raise RuntimeError(
            f"Aucune détection sur cette image. Vérifie que le bon modèle est utilisé, ou baisse "
            f"--tile-conf-threshold."
        )

    if auto_write:
        return _run_auto_write(predictions, conf_threshold, target_names, lot_dir, image_path, img_w, img_h)

    if not candidates:
        print("    Rien à revoir à ce seuil de confiance - aucun lot créé.")
        raise RuntimeError(
            f"Aucune détection >= {conf_threshold:.0%} sur cette image. Essaie un --conf-threshold "
            f"plus bas, ou vérifie que le bon modèle est utilisé."
        )

    # Le lot est initialisé (image + label VIDE + data.yaml) ICI, avant même
    # d'ouvrir l'interface de revue - voir _init_lot. Chaque clic "Valider"
    # ajoutera ensuite immédiatement sa ligne au fichier de label
    # (_ReviewState.decide -> _append_validated_line) : plus aucune écriture
    # lourde n'est différée jusqu'à "Enregistrer et terminer" (voir journal
    # du 31/08 - c'est ce report qui a fait perdre 320 annotations validées
    # lors d'un plantage en cours d'écriture). Une seule image ici (déjà sous
    # la limite CVAT si besoin, voir split_for_cvat.py) : cette écriture est
    # rapide, l'interface s'ouvre sans délai perceptible.
    label_path = _init_lot(lot_dir, image_path, target_names)
    state = _ReviewState(img, candidates, target_names, label_path, img_w, img_h)
    server = ThreadingHTTPServer((host, port), _make_handler(state))

    url = f"http://{host}:{port}/"
    print(f"\n--- 🖥️  Interface de validation prête : {url} ---")
    print("    (Entrée = Valider, Retour/Suppr = Supprimer, au clavier)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        while not state.finished:
            server_thread.join(timeout=0.5)
            if not server_thread.is_alive():
                break
    except KeyboardInterrupt:
        print(
            f"\nInterrompu - {state.validated_count} annotation(s) déjà validée(s) sont bien sur disque sous "
            f"{lot_dir} (écriture immédiate à chaque « Valider ») : seule la revue restante est perdue, "
            f"aucune annotation déjà validée ne l'est."
        )
        server.shutdown()
        raise
    finally:
        server.shutdown()

    print(f"\n--- ✅ Lot écrit : {lot_dir} ---")
    print(f"    {state.validated_count} annotation(s) validée(s), {state.deleted_count} supprimée(s).")
    print(f"    Ce lot est au format standard (images/train + labels/train + data.yaml) mais N'EST PAS")
    print(f"    dans 1_annotated_dataset : repasse-le par CVAT si besoin d'une passe de contrôle, ou")
    print(f"    copie-le tel quel dans 1_annotated_dataset/{lot_name}/ puis lance data_pipeline.py --force.")
    return [label_path]


def _class_group_stats(
    items: List[LabeledPolygon], target_names: Dict[int, str], img_area: Optional[float]
) -> List[Dict]:
    """Statistiques descriptives PAR CLASSE sur un groupe de détections
    (prédictions brutes, ou candidates au-dessus du seuil de confiance) :
    effectif, distribution de confiance (min/médiane/moyenne/max), et aire
    relative médiane du masque (aire du polygone / aire de l'image parente -
    même logique que l'aire relative déjà utilisée pour diagnostiquer
    `Bouchon`/`Bouee` dans l'audit du dataset, voir journal du 26/08).

    Entrée : liste de détections, noms de classes cibles, aire de l'image
    parente en pixels² (None si indisponible - l'aire relative est alors
    omise plutôt que de risquer une division par zéro).
    Sortie : une ligne par classe REPRÉSENTÉE dans `items` (pas une ligne par
    classe déclarée - contrairement à training_report.py, ici on résume ce
    qui a été détecté sur UNE image, une classe totalement absente n'apporte
    rien à afficher), triées par effectif décroissant.
    """
    by_class: Dict[int, List[LabeledPolygon]] = defaultdict(list)
    for p in items:
        by_class[p.class_id].append(p)

    rows = []
    for class_id, group in by_class.items():
        confs = sorted(p.confidence for p in group if p.confidence is not None)
        rel_areas = sorted(p.geom.area / img_area for p in group) if img_area else []
        rows.append({
            "class_id": class_id,
            "class_name": target_names.get(class_id, str(class_id)),
            "n": len(group),
            "conf_min": confs[0] if confs else None,
            "conf_median": statistics.median(confs) if confs else None,
            "conf_mean": statistics.fmean(confs) if confs else None,
            "conf_max": confs[-1] if confs else None,
            "rel_area_median": statistics.median(rel_areas) if rel_areas else None,
        })
    rows.sort(key=lambda r: r["n"], reverse=True)
    return rows


def _prediction_stats(
    predictions: List[LabeledPolygon],
    candidates: List[LabeledPolygon],
    target_names: Dict[int, str],
    img_w: int,
    img_h: int,
) -> Dict:
    """Statistiques descriptives complètes sur les détections d'une image de
    revue assistée : à la fois sur TOUTES les détections brutes (avant seuil
    de confiance métier) et sur les seules `candidates` réellement proposées
    à la revue - comparer les deux permet de voir si une classe attendue
    existe mais reste sous le seuil, sans avoir à rebaisser --conf-threshold
    à l'aveugle.

    Entrée : détections brutes, candidates filtrées (>= conf_threshold),
    noms de classes cibles, dimensions de l'image parente.
    Sortie : dict {raw: {n, per_class}, candidates: {n, per_class}} -
    `per_class` est la sortie de `_class_group_stats`.
    """
    img_area = float(img_w * img_h) if img_w and img_h else None
    return {
        "raw": {"n": len(predictions), "per_class": _class_group_stats(predictions, target_names, img_area)},
        "candidates": {"n": len(candidates), "per_class": _class_group_stats(candidates, target_names, img_area)},
    }


def _fmt_pct(x: Optional[float], decimals: int = 0) -> str:
    return f"{x * 100:.{decimals}f}%" if x is not None else "—"


def _print_prediction_stats(stats: Dict, conf_threshold: float) -> None:
    """Affiche `stats` (sortie de `_prediction_stats`) sur la console, dans le
    même style que le reste du module. Purement informatif - n'écrit rien sur
    disque (contrairement à `_init_lot`/`_append_validated_line`), sert à juger AVANT d'ouvrir
    l'interface de revue si l'image vaut la peine d'être revue en l'état."""

    def _print_group(label: str, group: Dict) -> None:
        rows = group["per_class"]
        print(f"\n  {label} ({group['n']} détection(s)) :")
        if not rows:
            print("    (aucune)")
            return
        header = f"    {'Classe':<16}{'n':>5}{'conf. min':>11}{'conf. médiane':>15}{'conf. max':>11}{'aire médiane':>14}"
        print(header)
        for r in rows:
            print(
                f"    {r['class_name']:<16}{r['n']:>5}"
                f"{_fmt_pct(r['conf_min']):>11}{_fmt_pct(r['conf_median']):>15}{_fmt_pct(r['conf_max']):>11}"
                f"{_fmt_pct(r['rel_area_median'], 2):>14}"
            )

    print("\n--- 📊 STATISTIQUES DESCRIPTIVES DES PRÉDICTIONS ---")
    _print_group("Détections brutes (avant seuil de confiance)", stats["raw"])
    _print_group(f"Candidates proposées à la revue (confiance ≥ {conf_threshold:.0%})", stats["candidates"])

    low_sample = [r for r in stats["candidates"]["per_class"] if r["n"] < LOW_SAMPLE_WARN_THRESHOLD]
    if low_sample:
        detail = ", ".join(f"{r['class_name']} ({r['n']})" for r in low_sample)
        print(
            f"\n  ⚠️  Classe(s) à échantillon faible (< {LOW_SAMPLE_WARN_THRESHOLD} candidates) sur "
            f"cette image : {detail}. Rappel : en dessous de ce seuil, le nombre repéré ne permet "
            f"de juger que la PRÉSENCE de la classe sur cette image, pas la fiabilité du modèle "
            f"dessus - voir échange du 27/08 sur la significativité statistique."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production d'annotations assistées par le modèle (cold start) - PixelOdyssey")
    parser.add_argument("--image", required=True,
                        help="Chemin vers l'image à annoter - doit déjà être sous la limite d'import CVAT "
                             "(voir split_for_cvat.py pour découper une orthomosaïque trop grande AVANT).")
    parser.add_argument("--lot-name", required=True,
                        help="Nom du nouveau dossier sous RESULTS_DIR/ (doit ne pas déjà exister).")
    parser.add_argument("--model", default=None,
                        help="Chemin vers le modèle entraîné (best.pt). Optionnel : sélection interactive sinon.")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Seuil de confiance métier après fusion inter-tuiles (défaut 0.5).")
    parser.add_argument("--tile-conf-threshold", type=float, default=0.25,
                        help="Seuil de confiance large appliqué par tuile avant fusion (défaut 0.25).")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                         help=f"Dossier de sortie des lots (défaut : {RESULTS_DIR}).")
    parser.add_argument("--auto-write", action="store_true",
                         help="Saute l'interface de revue manuelle : demande interactivement un seuil de "
                              "confiance (Entrée pour garder --conf-threshold), puis écrit directement toutes "
                              "les détections au-dessus comme annotations validées, classe PRÉDITE telle "
                              "quelle - AUCUNE détection n'est vérifiée une à une. À réserver à un seuil élevé "
                              "et/ou à un lot destiné à repasser par un contrôle qualité complémentaire dans "
                              "CVAT avant intégration à 1_annotated_dataset.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="N'ouvre pas le navigateur automatiquement.")
    parser.add_argument("--max-pixels", type=int, default=CVAT_MAX_PIXELS,
                         help=f"Plafond de pixels au-delà duquel l'image est découpée automatiquement "
                              f"avant revue (défaut : {CVAT_MAX_PIXELS:,}, la limite d'import CVAT "
                              f"constatée - même valeur que split_for_cvat.py).")
    args = parser.parse_args()
    try:
        run_assisted_annotate(
            image_path=args.image,
            lot_name=args.lot_name,
            model_path=args.model,
            results_dir=args.results_dir,
            conf_threshold=args.conf_threshold,
            tile_conf_threshold=args.tile_conf_threshold,
            tile_size=args.tile_size,
            overlap=args.overlap,
            auto_write=args.auto_write,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            max_pixels=args.max_pixels,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
