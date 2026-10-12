"""python -m src.app [verifier | traiter --dossier D --modele M [--sortie S]] [--port N] [--sans-navigateur]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def verifier() -> int:
    ok_all = True

    def line(ok, label, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  [{'OK' if ok else '!!'}] {label}" + (f" : {detail}" if detail else ""))

    print("\nVérification de l'installation PixelOdyssey\n")
    print(f"  Python {sys.version.split()[0]} ({sys.executable})")
    for mod in ("numpy", "cv2", "shapely", "scipy", "rasterio", "yaml", "PIL", "torch", "ultralytics"):
        try:
            m = __import__(mod)
            line(True, mod, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            line(False, mod, f"absent ({e.__class__.__name__}) - pip install {mod if mod != 'cv2' else 'opencv-python'}")
    try:
        import cv2
        cv2.SIFT_create()
        line(True, "SIFT (OpenCV)")
    except Exception as e:  # noqa: BLE001
        line(False, "SIFT (OpenCV)", str(e))
    try:
        import torch
        line(torch.cuda.is_available(), "GPU CUDA", torch.cuda.get_device_name(0) if torch.cuda.is_available()
             else "non disponible : la détection tournera sur le processeur (beaucoup plus lent)")
    except Exception:  # noqa: BLE001
        pass
    try:
        import tkinter  # noqa: F401
        line(True, "tkinter (fenêtre de choix de dossier)")
    except Exception:  # noqa: BLE001
        line(False, "tkinter", "absent : saisir le chemin du dossier à la main")
    from src.app import config
    for m in config.load_models():
        line(m["disponible"], f"modèle {m['id']}", m["chemin"])
    print("\n  Tout est prêt." if ok_all else "\n  Des éléments manquent (voir les lignes !!).")
    return 0 if ok_all else 1


def traiter(args) -> int:
    from src.app import pipeline
    job = pipeline.Job(args.dossier, args.modele, args.sortie)
    job.start()
    last = ""
    while job.state in ("en_attente", "en_cours"):
        time.sleep(1)
        s = job.status()
        msg = f"{s['progression'] * 100:5.1f} %  {s['message']}"
        if msg != last:
            print(msg, flush=True)
            last = msg
    print(f"\nÉtat : {job.state}" + (f" - {job.error}" if job.error else ""))
    if job.state == "termine":
        syn = job.result.get("synthese", {})
        print(f"{syn.get('n_objets')} déchets (seuil {syn.get('seuil')}), surface couverte {syn.get('surface_couverte_m2')} m², "
              f"poids ≈ {syn.get('poids_kg')} kg [{syn.get('poids_bas_kg')} - {syn.get('poids_haut_kg')}]")
        print(f"Résultats : {job.out}")
    return 0 if job.state == "termine" else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m src.app", description="PixelOdyssey - application locale")
    ap.add_argument("commande", nargs="?", default="interface", choices=["interface", "verifier", "traiter"])
    ap.add_argument("--dossier")
    ap.add_argument("--modele", default="mono_20261008")
    ap.add_argument("--sortie")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--sans-navigateur", action="store_true")
    args = ap.parse_args()
    if args.commande == "verifier":
        return verifier()
    if args.commande == "traiter":
        if not args.dossier:
            ap.error("--dossier est obligatoire pour traiter")
        return traiter(args)
    from src.app.server import serve
    serve(args.port, open_browser=not args.sans_navigateur)
    return 0


if __name__ == "__main__":
    sys.exit(main())
