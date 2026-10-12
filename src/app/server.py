"""Serveur web local (bibliothèque standard uniquement) : interface + API du traitement."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from shapely.geometry import Polygon

from . import __version__, config, inventory, pipeline, tiles

STATE: Dict = {"appareil": "détection du GPU en cours…"}
_IMG_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_IMG_LOCK = threading.Lock()
_INDEX_CACHE: Dict[str, list] = {}


def _warmup() -> None:
    try:
        from .detection import device_label
        STATE["appareil"] = device_label()
    except Exception as e:  # noqa: BLE001
        STATE["appareil"] = f"indisponible ({e.__class__.__name__})"


def pick_folder() -> str:
    code = ("import tkinter as tk, tkinter.filedialog as fd\n"
            "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True); r.update()\n"
            "p = fd.askdirectory(title='Choisir le dossier des photos du vol', mustexist=True)\n"
            "import sys; sys.stdout.buffer.write((p or '').encode('utf-8'))\n")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=900, env=env)
    return r.stdout.decode("utf-8", errors="ignore").strip()


def _photo(path: str) -> Optional[np.ndarray]:
    with _IMG_LOCK:
        if path in _IMG_CACHE:
            _IMG_CACHE.move_to_end(path)
            return _IMG_CACHE[path]
    from .registration import imread
    img = imread(path)
    with _IMG_LOCK:
        _IMG_CACHE[path] = img
        while len(_IMG_CACHE) > 4:
            _IMG_CACHE.popitem(last=False)
    return img


def imagette(job: pipeline.Job, det_id: int) -> Optional[bytes]:
    key = f"{job.out}|{job.model_id}"
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        f = job.out / "cache" / f"imagettes_{job.model_id}.json"
        if not f.exists():
            return None
        idx = json.loads(f.read_text(encoding="utf-8"))
        _INDEX_CACHE[key] = idx
    if det_id < 0 or det_id >= len(idx):
        return None
    d = idx[det_id]
    img = _photo(d["chemin"])
    if img is None:
        return None
    x0, y0, x1, y1 = d["bbox"]
    side = max(x1 - x0, y1 - y0)
    m = int(side * 0.6) + 40
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = side // 2 + m
    X0, Y0 = max(0, cx - half), max(0, cy - half)
    X1, Y1 = min(img.shape[1], cx + half), min(img.shape[0], cy + half)
    crop = img[Y0:Y1, X0:X1].copy()
    pts = (np.array(d["px"]) - [X0, Y0]).astype(np.int32)
    s = 360 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    cv2.polylines(crop, [np.round(pts * s).astype(np.int32)], True, (238, 211, 34), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


_LIGHT: Dict[str, tuple] = {}
_LIGHT_LOCK = threading.Lock()
LIGHT_KEYS = ("id", "classe", "confiance", "surface_m2", "longueur_m", "poids_kg", "poids_bas_kg", "poids_haut_kg",
              "photo", "vues", "lon", "lat")


def light_detections(path: Path) -> Optional[bytes]:
    """Version allégée des détections pour la carte : contours simplifiés (~1,5 cm), champs utiles seulement.
    Le fichier complet reste celui des téléchargements (QGIS)."""
    if not path.exists():
        return None
    key, mtime = str(path), path.stat().st_mtime
    with _LIGHT_LOCK:
        hit = _LIGHT.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    gj = json.loads(path.read_text(encoding="utf-8"))
    feats = []
    for f in gj.get("features", []):
        ring = f["geometry"]["coordinates"][0]
        try:
            g = Polygon(ring).simplify(1.5e-7, preserve_topology=True)
            if g.is_empty or g.geom_type != "Polygon":
                raise ValueError
            ring = [[round(x, 7), round(y, 7)] for x, y in g.exterior.coords]
        except Exception:  # noqa: BLE001
            pass
        props = {k: f["properties"].get(k) for k in LIGHT_KEYS}
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    data = json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")).encode("utf-8")
    with _LIGHT_LOCK:
        _LIGHT[key] = (mtime, data)
    return data


def results(job: pipeline.Job) -> Dict:
    out = job.out
    files = []
    for name, label in [(f"detections_{job.model_id}.csv", "Tableau des détections (CSV, Excel)"),
                        (f"detections_{job.model_id}.geojson", "Détections (GeoJSON, QGIS)"),
                        ("mosaique.tif", "Mosaïque géoréférencée (GeoTIFF, QGIS)"),
                        ("mosaique_apercu.jpg", "Aperçu de la mosaïque (JPEG)"),
                        ("emprises.geojson", "Emprises des photos (GeoJSON)"),
                        (f"synthese_{job.model_id}.json", "Synthèse chiffrée (JSON)"),
                        ("journal.txt", "Journal du traitement")]:
        f = out / name
        if f.exists():
            files.append({"nom": name, "libelle": label, "taille": f.stat().st_size, "url": f"/fichier/{job.id}/{name}"})
    res = {"job": job.id, "sortie": str(out), "telechargements": files, "synthese": job.result.get("synthese"),
           "alignement": job.result.get("alignement"), "modele": config.find(job.model_id),
           "detections_url": f"/api/detections/{job.id}", "emprises_url": f"/api/emprises/{job.id}"}
    if job.mosaic_path and Path(job.mosaic_path).exists():
        b = tiles.bounds_lonlat(job.mosaic_path)
        res["mosaique"] = {"bounds": [[b[1], b[0]], [b[3], b[2]]], "maxzoom": tiles.max_native_zoom(job.mosaic_path),
                           "url": f"/tuiles/{job.id}/{{z}}/{{x}}/{{y}}.png"}
    return res


class Handler(BaseHTTPRequestHandler):
    server_version = f"PixelOdyssey/{__version__}"

    def log_message(self, fmt, *args):  # silence
        pass

    # ------------------------------------------------------------ utilitaires
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8", cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _err(self, msg: str, code: int = 400):
        self._json({"erreur": msg}, code)

    def _body(self) -> Dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def _file(self, path: Path, download: bool = False):
        if not path.exists() or not path.is_file():
            return self._err("Fichier introuvable", 404)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return

    def _job(self, jid: str) -> Optional[pipeline.Job]:
        return pipeline.JOBS.get(jid)

    # ------------------------------------------------------------ routes
    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self._file(config.STATIC_DIR / "index.html")
            if path.startswith("/static/"):
                f = (config.STATIC_DIR / path[len("/static/"):]).resolve()
                if config.STATIC_DIR.resolve() not in f.parents:
                    return self._err("Accès refusé", 403)
                return self._file(f)
            if path == "/api/modeles":
                return self._json({"modeles": config.load_models(), "appareil": STATE["appareil"], "version": __version__})
            if path == "/api/recents":
                return self._json({"recents": [p for p in config.load_recents() if Path(p).exists()]})
            m = re.fullmatch(r"/api/etat/(\w+)", path)
            if m:
                job = self._job(m.group(1))
                return self._json(job.status()) if job else self._err("Traitement inconnu", 404)
            m = re.fullmatch(r"/api/resultats/(\w+)", path)
            if m:
                job = self._job(m.group(1))
                return self._json(results(job)) if job else self._err("Traitement inconnu", 404)
            m = re.fullmatch(r"/api/(detections|emprises)/(\w+)", path)
            if m:
                job = self._job(m.group(2))
                if not job:
                    return self._err("Traitement inconnu", 404)
                if m.group(1) == "emprises":
                    return self._file(job.out / "emprises.geojson")
                data = light_detections(job.out / f"detections_{job.model_id}.geojson")
                return self._send(200, data, "application/json") if data else self._err("Détections absentes", 404)
            m = re.fullmatch(r"/tuiles/(\w+)/(\d+)/(\d+)/(\d+)\.png", path)
            if m:
                job = self._job(m.group(1))
                if not job or not job.mosaic_path:
                    return self._send(200, tiles.empty_png(), "image/png")
                png = tiles.render_tile(job.mosaic_path, int(m.group(2)), int(m.group(3)), int(m.group(4)))
                return self._send(200, png, "image/png", cache=job.state == "termine")
            m = re.fullmatch(r"/imagette/(\w+)/(\d+)\.jpg", path)
            if m:
                job = self._job(m.group(1))
                data = imagette(job, int(m.group(2))) if job else None
                return self._send(200, data, "image/jpeg", cache=True) if data else self._err("Imagette indisponible", 404)
            m = re.fullmatch(r"/fichier/(\w+)/([\w.\-]+)", path)
            if m:
                job = self._job(m.group(1))
                if not job:
                    return self._err("Traitement inconnu", 404)
                return self._file(job.out / m.group(2), download=True)
            return self._err("Page introuvable", 404)
        except Exception as e:  # noqa: BLE001
            return self._err(f"{e.__class__.__name__} : {e}", 500)

    def do_PUT(self):
        path = unquote(urlparse(self.path).path)
        m = re.fullmatch(r"/api/import/(\w+)/(.+)", path)
        if not m:
            return self._err("Page introuvable", 404)
        name = Path(m.group(2)).name
        if Path(name).suffix.lower() not in config.PHOTO_EXTS:
            return self._json({"ignore": name})
        dest = config.IMPORT_DIR / m.group(1)
        dest.mkdir(parents=True, exist_ok=True)
        n = int(self.headers.get("Content-Length") or 0)
        with open(dest / name, "wb") as f:
            remaining = n
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        return self._json({"ok": name})

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            body = self._body()
            if path == "/api/choisir-dossier":
                return self._json({"chemin": pick_folder()})
            if path == "/api/import/debut":
                sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
                return self._json({"session": sid, "dossier": str(config.IMPORT_DIR / sid)})
            if path == "/api/inventaire":
                folder = body.get("dossier", "").strip().strip('"')
                inv = inventory.scan(folder)
                data = inventory.to_json(inv)
                data["sortie_defaut"] = config.default_output_dir(folder)
                data["deja_traite"] = [p.stem.replace("synthese_", "") for p in Path(data["sortie_defaut"]).glob("synthese_*.json")]
                config.add_recent(str(Path(folder)))
                return self._json(data)
            if path == "/api/lancer":
                folder = body["dossier"]
                job = pipeline.Job(folder, body["modele"], body.get("sortie") or None)
                job.start()
                return self._json({"job": job.id})
            if path == "/api/annuler":
                job = self._job(body.get("job", ""))
                if job:
                    job.cancel.set()
                return self._json({"ok": True})
            if path == "/api/ouvrir":
                job = pipeline.open_existing(body["sortie"], body.get("modele"))
                return self._json({"job": job.id})
            if path == "/api/ouvrir-dossier":
                p = body.get("chemin", "")
                if p and Path(p).exists():
                    if sys.platform.startswith("win"):
                        os.startfile(p)  # type: ignore[attr-defined]
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", p])
                    else:
                        subprocess.Popen(["xdg-open", p])
                return self._json({"ok": True})
            return self._err("Page introuvable", 404)
        except (FileNotFoundError, ValueError, KeyError) as e:
            return self._err(str(e) if not isinstance(e, KeyError) else f"Paramètre manquant : {e}", 400)
        except Exception as e:  # noqa: BLE001
            return self._err(f"{e.__class__.__name__} : {e}", 500)


def serve(port: int = 8765, open_browser: bool = True) -> None:
    threading.Thread(target=_warmup, daemon=True).start()
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        raise RuntimeError("Aucun port libre entre 8765 et 8785")
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  PixelOdyssey {__version__} est prêt : {url}")
    print("  (laisser cette fenêtre ouverte pendant l'utilisation ; Ctrl+C pour arrêter)\n")
    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Arrêt.")
