#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Validation assistée pour démarrer un NOUVEAU lot (cold start).

Contexte (voir journal de décisions, "En rade" -> "Annotation assistée CVAT -
cold-start pour un nouveau lot jamais entraîné dessus") : `label_review.py`
suppose toujours un lot DÉJÀ annoté (il complète/corrige une vérité terrain
existante). Ce script répond au cas inverse - une image ou orthomosaïque
encore vierge de toute annotation - en te laissant valider toi-même, une à
une, les prédictions du modèle avant de les figer en vérité terrain.

Différence structurelle avec label_review.py (pas juste une question
d'interface) : un nouveau lot n'a PAR CONSTRUCTION aucune taxonomie fine
existante (contrairement à SB/SL/A LEG, ~20 sous-classes chacun) - inventer
une sous-classe qui n'existerait que pour ce lot n'aurait aucun sens. Son
data.yaml déclare donc directement les 8 SUPER-classes de
`config/data_config.yaml` comme classes locales (voir les entrées
d'auto-mapping ajoutées le 25/08/2026 dans class_taxonomy, "Bouteille": 0,
etc.) - pas de placeholder à corriger plus tard dans CVAT, la classe validée
ICI est la classe finale. Contrepartie assumée : ce lot n'aura jamais de
sous-classe fine tant que quelqu'un ne le réannote pas plus précisément à la
main - vitesse de démarrage contre granularité, pas un oubli.

Fonctionnement :
    1. Inférence tuilée sur l'image/mosaïque donnée (même géométrie que
       l'entraînement - réutilise tiled_inference.predict_parent_image, comme
       label_review.py et visualize_predictions.py).
    2. Seules les prédictions >= --conf-threshold (défaut 0.5) sont proposées
       à la revue - en dessous, jamais montrées (bruit de tuile, voir le
       double seuil déjà en place dans label_review.py : filtre large par
       tuile avant fusion, seuil métier après).
    3. Petite page web locale (serveur intégré à Python, aucune dépendance
       ajoutée) : un déchet à la fois, chip à résolution native avec le
       contour du masque en surimpression (même esprit que les chips de
       geo_density_map.py), un bandeau de classe éditable (menu déroulant,
       pré-rempli avec la classe prédite) et DEUX boutons - Valider (écrit le
       masque avec la classe ACTUELLEMENT affichée dans le bandeau, que ce
       soit celle prédite ou celle que tu as changée) et Supprimer (rien
       n'est écrit). Décision utilisateur du 25/08/2026 : pas de 3e bouton
       "classe à revoir" séparé - le bandeau éditable rend ce cas identique à
       "Valider" une fois la bonne classe sélectionnée, une seule action
       suffit.
    4. À la fin de la revue (ou sur "Enregistrer et terminer" pour s'arrêter
       en cours de route), écrit directement dans
       1_annotated_dataset/<lot_name>/ (images/ + labels/ + data.yaml) - le
       lot obtenu est immédiatement ingérable par le pipeline standard
       (split_dataset.py etc.), aucun format de sortie ni glue supplémentaire.

Usage :
    python -m src.review.bootstrap_annotate --image "chemin/vers/image.tif" --lot-name "SL_nouveau_lot"
        -> aucun --model fourni : sélection interactive du modèle entraîné
           (même mécanisme que label_review.py).
    python -m src.review.bootstrap_annotate --image ... --lot-name ... --model output/runs/<run>/weights/best.pt
        -> saute la sélection interactive.

Puis ouvrir l'URL affichée (http://127.0.0.1:8765/ par défaut) dans un
navigateur.
"""

import argparse
import json
import os
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import cv2
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import DEFAULT_CLASS_CONFIG_PATH, load_class_config
from src.data.image_io import load_image_bgr
from src.data.split_dataset import RAW_DIR
from src.review.label_review import RUNS_DIR, _discover_available_models, _prompt_model_choice
from src.review.matching import LabeledPolygon
from src.review.tiled_inference import make_ultralytics_predict_fn, predict_parent_image

MARGIN_PX = 80          # marge autour du masque dans le chip de revue
MIN_CROP_DISPLAY_PX = 360  # un chip plus petit que ça est agrandi pour rester lisible
OUTLINE_COLOR_BGR = (0, 235, 255)  # jaune vif (cohérent avec geo_density_map.py, converti BGR<->RGB)


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


def _write_new_lot(
    lot_dir: Path,
    image_path: Path,
    validated: List[Dict],
    target_names: Dict[int, str],
    img_w: int,
    img_h: int,
) -> Path:
    """Écrit le nouveau lot dans 1_annotated_dataset/<lot_name>/ - images/ +
    labels/ + data.yaml, structure identique à celle attendue par
    src.data.raw_dataset.collect_parent_images (mêmes noms de dossiers,
    label retrouvé via le mirroring images/ -> labels/)."""
    img_dir = lot_dir / "images"
    lab_dir = lot_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, img_dir / image_path.name)

    lines = []
    for v in validated:
        coords = v["coords_norm"]
        coords_str = " ".join(f"{c:.6f}" for c in coords)
        lines.append(f"{v['class_id']} {coords_str}\n")

    label_path = lab_dir / f"{image_path.stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    data_yaml_path = lot_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                # `path` + `train` : pas nécessaires pour l'ingestion pipeline
                # (raw_dataset.collect_parent_images / class_config.load_batch_local_names
                # ne lisent QUE la clé `names`, voir class_config.py) - mais
                # indispensables pour l'IMPORT CVAT (format "Ultralytics YOLO
                # Segmentation"), qui doit savoir où se trouvent les images à
                # associer aux labels. Bug corrigé le 25/08/2026 : sans ces 2
                # clés, l'import CVAT ne plante pas mais n'importe RIEN (0
                # annotation, y compris dans le panneau Objects) - même classe
                # de bug déjà rencontrée et corrigée le 23/08/2026 dans
                # label_review._write_review_data_yaml, pas réutilisée ici à
                # l'origine alors qu'elle aurait dû l'être. `train: "images"`
                # pointe vers le dossier plat déjà écrit ci-dessus (pas de
                # sous-dossier images/train/ - CVAT retrouve les labels en
                # substituant "images" -> "labels" dans ce même chemin, la
                # convention standard Ultralytics).
                "path": ".",
                "train": "images",
                "names": {int(k): v for k, v in target_names.items()},
            },
            f, allow_unicode=True, sort_keys=False,
        )

    return label_path


class _ReviewState:
    """État partagé du serveur de revue - un seul utilisateur, une seule
    session à la fois (pas besoin de plus pour cet outil)."""

    def __init__(self, img, candidates: List[LabeledPolygon], target_names: Dict[int, str],
                 lot_dir: Path, image_path: Path, img_w: int, img_h: int):
        self.img = img
        self.candidates = candidates
        self.target_names = target_names
        self.lot_dir = lot_dir
        self.image_path = image_path
        self.img_w = img_w
        self.img_h = img_h
        self.cursor = 0
        self.validated: List[Dict] = []
        self.deleted_count = 0
        self.finished = False
        self.result_path: Optional[Path] = None
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
                "validated": len(self.validated),
                "deleted": self.deleted_count,
                "output": str(self.result_path) if self.result_path else None,
            }
        if self.cursor >= total:
            return {"finished": False, "done_reviewing": True, "total": total,
                    "validated": len(self.validated), "deleted": self.deleted_count}
        pred = self.candidates[self.cursor]
        classes = [{"id": cid, "name": name} for cid, name in sorted(self.target_names.items())]
        return {
            "finished": False,
            "done_reviewing": False,
            "index": self.cursor,
            "total": total,
            "reviewed": self.cursor,
            "validated": len(self.validated),
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
                norm = []
                for x, y in geom.exterior.coords:
                    norm.extend([
                        max(0.0, min(1.0, x / self.img_w)),
                        max(0.0, min(1.0, y / self.img_h)),
                    ])
                self.validated.append({"class_id": int(class_id), "coords_norm": norm})
            else:
                self.deleted_count += 1
            self.cursor += 1

    def finish(self) -> dict:
        with self.lock:
            if not self.finished:
                self.result_path = _write_new_lot(
                    self.lot_dir, self.image_path, self.validated,
                    self.target_names, self.img_w, self.img_h,
                )
                self.finished = True
            return self.state_json()


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
<title>PixelOdyssey — validation de nouveau lot</title>
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
  <h1>Validation de nouveau lot</h1>
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
  const r = await fetch('/api/state');
  const s = await r.json();
  if (s.finished) {
    document.getElementById('review').classList.add('hidden');
    const d = document.getElementById('done');
    d.classList.remove('hidden');
    d.innerHTML = `<h2>Terminé</h2>
      <p>${s.validated} annotation(s) validée(s), ${s.deleted} supprimée(s).</p>
      <p style="color:#888;font-size:0.85rem;">Écrit dans : ${s.output}</p>
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
  await fetch('/api/finish', {method: 'POST'});
  refresh();
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


def run_bootstrap_annotate(
    image_path: Union[str, Path],
    lot_name: str,
    model_path: Optional[str] = None,
    annotated_dir: Union[str, Path] = RAW_DIR,
    conf_threshold: float = 0.5,
    tile_conf_threshold: float = 0.25,
    tile_size: int = 640,
    overlap: int = 256,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    predict_tile_fn=None,
) -> Path:
    """`predict_tile_fn` : comme dans label_review.py/visualize_predictions.py,
    permet d'injecter un faux prédicteur pour les tests sans modèle réel.

    Retourne le chemin du fichier de labels écrit une fois la revue terminée
    (bloque jusque-là - cette fonction ne rend la main qu'après que le
    serveur a reçu la décision de terminer, via /api/finish déclenché par le
    bouton ou automatiquement quand la file est épuisée)."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise RuntimeError(f"Image introuvable : {image_path}")

    lot_dir = Path(annotated_dir) / lot_name
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

    print(f"--- 🔎 Inférence sur {image_path.name} ---")
    img = load_image_bgr(image_path)
    if img is None:
        raise RuntimeError(f"Image illisible : {image_path}")
    img_h, img_w = img.shape[:2]

    predictions = predict_parent_image(image_path, predict_tile_fn, tile_size=tile_size, overlap=overlap)
    candidates = sorted(
        [p for p in predictions if p.confidence is not None and p.confidence >= conf_threshold],
        key=lambda p: p.confidence, reverse=True,
    )
    print(f"    {len(predictions)} détection(s) brute(s), {len(candidates)} au-dessus de "
          f"{conf_threshold:.0%} de confiance - proposée(s) à la revue.")

    if not candidates:
        print("    Rien à revoir à ce seuil de confiance - aucun lot créé.")
        raise RuntimeError(
            f"Aucune détection >= {conf_threshold:.0%} sur cette image. Essaie un --conf-threshold "
            f"plus bas, ou vérifie que le bon modèle est utilisé."
        )

    _, target_names = load_class_config(DEFAULT_CLASS_CONFIG_PATH)

    state = _ReviewState(img, candidates, target_names, lot_dir, image_path, img_w, img_h)
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
        print("\nInterrompu - rien n'est écrit (utilise le bouton « Enregistrer et terminer » avant de fermer).")
        server.shutdown()
        raise
    finally:
        server.shutdown()

    print(f"\n--- ✅ Lot écrit : {lot_dir} ---")
    print(f"    {len(state.validated)} annotation(s) validée(s), {state.deleted_count} supprimée(s).")
    print(f"    Prêt pour le pipeline standard : python -m src.data.data_pipeline --force")
    return state.result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation assistée pour démarrer un nouveau lot PixelOdyssey")
    parser.add_argument("--image", required=True, help="Chemin vers l'image ou l'orthomosaïque à annoter.")
    parser.add_argument("--lot-name", required=True,
                        help="Nom du nouveau dossier sous 1_annotated_dataset/ (doit ne pas déjà exister).")
    parser.add_argument("--model", default=None,
                        help="Chemin vers le modèle entraîné (best.pt). Optionnel : sélection interactive sinon.")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Seuil de confiance métier après fusion inter-tuiles (défaut 0.5).")
    parser.add_argument("--tile-conf-threshold", type=float, default=0.25,
                        help="Seuil de confiance large appliqué par tuile avant fusion (défaut 0.25).")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="N'ouvre pas le navigateur automatiquement.")
    args = parser.parse_args()
    try:
        run_bootstrap_annotate(
            image_path=args.image,
            lot_name=args.lot_name,
            model_path=args.model,
            conf_threshold=args.conf_threshold,
            tile_conf_threshold=args.tile_conf_threshold,
            tile_size=args.tile_size,
            overlap=args.overlap,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
