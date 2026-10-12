"""Orchestration d'un traitement complet, avec progression, reprise et sorties.

Dossier de sortie (un par vol) :
  cache/                     points homologues et détections par photo (reprise sans tout refaire)
  alignement.json            transformation pixel -> sol de chaque photo, diagnostic
  mosaique.tif               mosaïque GeoTIFF (UTM), ouvrable dans QGIS
  mosaique_apercu.jpg
  emprises.geojson           emprise de chaque photo et zone qui lui est attribuée
  detections_<modele>.geojson / .csv, synthese_<modele>.json
  parametres.json, journal.txt
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rasterio.crs import CRS

from . import __version__, config, detection, inventory, mosaic, ownership, registration, tiles

ALIGN_VERSION = 4
DETECTION_VERSION = 2  # 2 : fusion multi-vues  # à incrémenter quand le modèle d'alignement change (les anciens résultats sont recalculés)

STEPS = [
    ("inventaire", "Lecture des photos", 0.02),
    ("points", "Points homologues", 0.14),
    ("paires", "Appariement des photos voisines", 0.12),
    ("alignement", "Alignement global", 0.03),
    ("attribution", "Attribution des zones", 0.01),
    ("mosaique", "Mosaïque", 0.25),
    ("detection", "Détection des déchets", 0.38),
    ("synthese", "Synthèse et exports", 0.05),
]
JOBS: Dict[str, "Job"] = {}


def _signature(inv: inventory.Inventory) -> str:
    h = hashlib.sha1()
    for p in inv.usable:
        st = Path(p.path).stat()
        h.update(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


class Job:
    def __init__(self, folder: str, model_id: str, out_dir: Optional[str] = None, existing: bool = False):
        self.id = uuid.uuid4().hex[:10]
        self.folder = folder
        self.model_id = model_id
        self.out = Path(out_dir or config.default_output_dir(folder))
        self.state = "en_attente"
        self.step = None
        self.step_progress = 0.0
        self.progress = 0.0
        self.message = ""
        self.log: List[str] = []
        self.error: Optional[str] = None
        self.started = time.time()
        self.finished: Optional[float] = None
        self.cancel = threading.Event()
        self.result: Dict = {}
        self.steps_done: List[str] = []
        self.mosaic_path: Optional[str] = None
        self.existing = existing
        JOBS[self.id] = self

    # ---------------------------------------------------------------- suivi
    def _log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.log.append(line)
        try:
            with open(self.out / "journal.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _set_step(self, key: str) -> None:
        if self.step and self.step not in self.steps_done:
            self.steps_done.append(self.step)
        self.step = key
        self.step_progress = 0.0
        label = dict((k, l) for k, l, _ in STEPS)[key]
        self._log(label + "…")
        self._update(0.0, label)

    def _update(self, frac: float, msg: str) -> None:
        self.step_progress = max(0.0, min(1.0, frac))
        done = sum(w for k, _, w in STEPS if k in self.steps_done)
        cur = dict((k, w) for k, _, w in STEPS).get(self.step, 0.0)
        self.progress = min(0.999, done + cur * self.step_progress)
        self.message = msg

    def status(self) -> Dict:
        el = (self.finished or time.time()) - self.started
        eta = None
        if self.state == "en_cours" and self.progress > 0.03:
            eta = el / self.progress * (1 - self.progress)
        return {"id": self.id, "etat": self.state, "etape": self.step, "etapes_faites": self.steps_done,
                "progression": round(self.progress, 4), "progression_etape": round(self.step_progress, 3),
                "message": self.message, "journal": self.log[-12:], "erreur": self.error,
                "ecoule_s": int(el), "restant_s": int(eta) if eta is not None else None,
                "mosaique_prete": bool(self.mosaic_path), "sortie": str(self.out),
                "etapes": [{"cle": k, "libelle": l} for k, l, _ in STEPS]}

    # ---------------------------------------------------------------- traitement
    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        self.state = "en_cours"
        try:
            self.out.mkdir(parents=True, exist_ok=True)
            self._log(f"PixelOdyssey {__version__} - dossier {self.folder} - modèle {self.model_id}")
            self._log(f"Calcul : {detection.device_label()}")
            self.run_pipeline()
            self.state = "termine"
            self.progress = 1.0
            self.message = "Traitement terminé"
            self._log(f"Terminé en {int(time.time() - self.started)} s")
        except Exception as e:  # noqa: BLE001
            self.state = "erreur"
            self.error = f"{e.__class__.__name__} : {e}"
            self._log("ERREUR : " + self.error)
            self._log(traceback.format_exc()[-1500:])
        finally:
            self.finished = time.time()

    def run_pipeline(self) -> None:
        model_cfg = config.get_model(self.model_id)
        if not model_cfg["disponible"]:
            raise FileNotFoundError(f"Poids du modèle introuvables : {model_cfg['chemin']}")
        cache = self.out / "cache"
        # 1. inventaire
        self._set_step("inventaire")
        inv = inventory.scan(self.folder)
        photos = inv.usable
        sig = _signature(inv)
        excl = len(inv.photos) - len(photos)
        self._log(f"{len(photos)} photos utilisables, {excl} exclues")
        self._update(1.0, f"{len(photos)} photos utilisables")
        # alignement déjà calculé pour exactement ces photos ?
        al_path = self.out / "alignement.json"
        al = None
        if al_path.exists():
            try:
                old = json.loads(al_path.read_text(encoding="utf-8"))
                if old.get("signature") == sig and old.get("version") == ALIGN_VERSION:
                    al = old
            except Exception:  # noqa: BLE001
                al = None
        if al is None:
            # 2-3. points et paires
            self._set_step("points")
            feats = registration.compute_features(inv, cache / "points", self._step_progress())
            self._set_step("paires")
            matches = registration.match_pairs(inv, feats, self._step_progress())
            self._log(f"{len(matches)} paires de photos recalées")
            del feats
            # 4. alignement
            self._set_step("alignement")
            res = registration.global_alignment(inv, matches, self._step_progress())
            diag = res["diagnostic"]
            self._log("Alignement : résidu médian {residu_median_cm} cm, {photos_recalees}/{photos_total} photos "
                      "recalées, écart au GPS médian {ecart_gps_median_m} m, échelle x{facteur_echelle}, "
                      "biais de cap {biais_cap_deg}°".format(**diag))
            for note in diag.get("remarques", []):
                self._log(f"Alignement : {note}")
            if diag.get("mode") == "gps":
                self._log("ATTENTION : photos placées sur leur seul GPS (précision 2 à 5 m) : la mosaïque aura des "
                          "décalages et des déchets situés dans les recouvrements peuvent être comptés deux fois")
            elif diag.get("mode") == "rigide":
                self._log("Alignement simplifié (sans déformation des photos) : coutures de la mosaïque moins nettes")
            al = {"version": ALIGN_VERSION, "signature": sig, "epsg": inv.crs.to_epsg(), "origine": inv.origin.tolist(),
                  "photos": {str(p.idx): p.name for p in photos}, "transforms": {str(k): v for k, v in res["transforms"].items()},
                  "diagnostic": diag}
            al_path.write_text(json.dumps(al), encoding="utf-8")
            # les détections et la mosaïque dépendent de l'alignement
            for d in cache.glob("detections_*"):
                shutil.rmtree(d, ignore_errors=True)
            tiles.release(self.out / "mosaique.tif")
            for f in (self.out / "mosaique.tif", self.out / "mosaique_apercu.jpg"):
                if f.exists():
                    f.unlink()
        else:
            for k in ("points", "paires", "alignement"):
                self._set_step(k)
                self._update(1.0, "Déjà calculé pour ces photos (repris)")
            self._log("Alignement repris du traitement précédent")
        transforms = {int(k): registration.PhotoTransform.from_json(v) for k, v in al["transforms"].items()}
        self.result["alignement"] = al["diagnostic"]
        # 5. attribution
        self._set_step("attribution")
        own = ownership.Ownership(photos, transforms)
        self._update(1.0, f"Surface couverte : {own.covered_area_m2():.0f} m²")
        self._write_footprints(inv, photos, transforms, own)
        # 6. mosaïque
        self._set_step("mosaique")
        mpath = self.out / "mosaique.tif"
        if mpath.exists():
            self._update(1.0, "Mosaïque déjà calculée (reprise)")
            minfo = {"chemin": str(mpath), "apercu": str(self.out / "mosaique_apercu.jpg")}
        else:
            minfo = mosaic.render(photos, transforms, own, inv.crs, inv.origin, mpath, self._step_progress())
            self._log(f"Mosaïque : {minfo['largeur']} x {minfo['hauteur']} pixels, {minfo['resolution_m'] * 100:.1f} cm/pixel")
        self.mosaic_path = str(mpath)
        # 7. détection
        self._set_step("detection")
        dets = detection.detect_all(photos, transforms, own, model_cfg, cache / f"detections_{self.model_id}",
                                    self._step_progress(), self.cancel)
        st = detection.STATS
        if config.DETECTION_MODE == "fusion":
            n_resc = sum(1 for d in dets if d.get("rattrape"))
            self._log(f"Fusion des vues : {st.get('detections_brutes', 0)} détections sur l'ensemble des photos, "
                      f"{len(dets)} objets distincts dont {n_resc} rattrapés sur une autre photo que la plus centrale")
        else:
            self._log(f"{len(dets)} objets détectés (avant seuil de confiance)")
        if st.get("photos_contraste"):
            self._log(f"Contraste renforcé avant détection sur {st['photos_contraste']} photo(s) ternes "
                      f"(écart-type de luminance < {config.LOW_CONTRAST_STD:g})")
        # 8. synthèse
        self._set_step("synthese")
        self.write_outputs(inv, dets, own, model_cfg, al)
        self._update(1.0, "Exports écrits")
        self.steps_done.append("synthese")

    def _step_progress(self):
        return lambda f, m: self._update(f, m)

    # ---------------------------------------------------------------- sorties
    def _write_footprints(self, inv, photos, transforms, own) -> None:
        feats = []
        crs = inv.crs
        for p in photos:
            g = own.fp[p.idx]
            ll = inv.to_lonlat(np.vstack([g, g[:1]]))
            feats.append({"type": "Feature", "properties": {"photo": p.name, "alertes": p.alerts,
                                                            "surface_attribuee_m2": round(own.owned_area_m2(p.idx), 1)},
                          "geometry": {"type": "Polygon", "coordinates": [[[round(a, 7), round(b, 7)] for a, b in ll]]}})
        (self.out / "emprises.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                                                    encoding="utf-8")

    def write_outputs(self, inv, dets: List[Dict], own, model_cfg: Dict, al: Dict) -> None:
        crs = inv.crs
        all_pts = [d["sol"] for d in dets]
        feats = []
        if dets:
            flat = np.vstack([np.asarray(s) for s in all_pts])
            ll = inv.to_lonlat(flat)
            k = 0
            for d in dets:
                n = len(d["sol"])
                ring = ll[k:k + n]
                k += n
                ring = np.vstack([ring, ring[:1]])
                cen = inv.to_lonlat(np.asarray([d["centre_sol"]]))[0]
                props = {key: d[key] for key in ("id", "classe", "confiance", "surface_m2", "longueur_m", "poids_kg",
                                                 "poids_bas_kg", "poids_haut_kg", "photo")}
                props["vues"] = d.get("vues", 1)
                props["lon"], props["lat"] = round(float(cen[0]), 7), round(float(cen[1]), 7)
                feats.append({"type": "Feature", "properties": props,
                              "geometry": {"type": "Polygon", "coordinates": [[[round(float(a), 7), round(float(b), 7)] for a, b in ring]]}})
        gj = {"type": "FeatureCollection", "features": feats}
        mid = model_cfg["id"]
        (self.out / f"detections_{mid}.geojson").write_text(json.dumps(gj), encoding="utf-8")
        # index interne pour les imagettes (pixels dans la photo d'origine)
        index = [{"id": d["id"], "photo": d["photo"], "chemin": next(p.path for p in inv.photos if p.name == d["photo"]),
                  "bbox": d["bbox"], "px": d["px"]} for d in dets]
        (self.out / "cache" / f"imagettes_{mid}.json").write_text(json.dumps(index), encoding="utf-8")
        thr = float(model_cfg["seuil"])
        with open(self.out / f"detections_{mid}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["id", "classe", "confiance", "longitude", "latitude", "longueur_m", "surface_m2", "poids_kg",
                        "poids_bas_kg", "poids_haut_kg", "photo", "nb_photos_detectant", f"au_dessus_du_seuil_{thr}"])
            for ft in feats:
                p = ft["properties"]
                w.writerow([p["id"], p["classe"], p["confiance"], p["lon"], p["lat"], p["longueur_m"], p["surface_m2"],
                            p["poids_kg"], p["poids_bas_kg"], p["poids_haut_kg"], p["photo"], p["vues"], int(p["confiance"] >= thr)])
        syn = summarize(dets, thr, own.covered_area_m2())
        syn.update({"modele": {k: model_cfg[k] for k in ("id", "nom", "seuil", "source")},
                    "photos": len(inv.usable), "photos_exclues": len(inv.photos) - len(inv.usable),
                    "alignement": al["diagnostic"], "date_traitement": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "version": __version__})
        (self.out / f"synthese_{mid}.json").write_text(json.dumps(syn, ensure_ascii=False, indent=1), encoding="utf-8")
        params = {"version_application": __version__, "dossier_photos": self.folder, "modele": model_cfg,
                  "epsg": crs.to_epsg(), "origine_utm": inv.origin.tolist(),
                  "reglages": {k: getattr(config, k) for k in dir(config) if k.isupper() and isinstance(getattr(config, k), (int, float))}}
        (self.out / "parametres.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
        self.result["synthese"] = syn


def summarize(dets: List[Dict], thr: float, area_m2: float) -> Dict:
    sel = [d for d in dets if d["confiance"] >= thr]
    per: Dict[str, Dict] = {}
    for d in sel:
        e = per.setdefault(d["classe"], {"n": 0, "surface_m2": 0.0, "poids_kg": 0.0, "poids_bas_kg": 0.0, "poids_haut_kg": 0.0})
        e["n"] += 1
        e["surface_m2"] += d["surface_m2"]
        for k in ("poids_kg", "poids_bas_kg", "poids_haut_kg"):
            e[k] += d[k] or 0.0
    for e in per.values():
        for k in list(e):
            if k != "n":
                e[k] = round(e[k], 3)
    return {"seuil": thr, "n_objets": len(sel), "surface_couverte_m2": round(area_m2, 1),
            "densite_par_m2": round(len(sel) / area_m2, 4) if area_m2 else None,
            "surface_dechets_m2": round(sum(d["surface_m2"] for d in sel), 3),
            "poids_kg": round(sum(d["poids_kg"] or 0 for d in sel), 2),
            "poids_bas_kg": round(sum(d["poids_bas_kg"] or 0 for d in sel), 2),
            "poids_haut_kg": round(sum(d["poids_haut_kg"] or 0 for d in sel), 2),
            "par_classe": dict(sorted(per.items(), key=lambda kv: -kv[1]["n"]))}


def open_existing(out_dir: str, model_id: Optional[str] = None) -> Job:
    """Réouvre un traitement terminé (sans recalcul) pour afficher ses résultats."""
    out = Path(out_dir)
    if not out.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {out_dir}")
    syns = sorted(out.glob("synthese_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if model_id:
        syns = [s for s in syns if s.stem == f"synthese_{model_id}"] or syns
    if not syns:
        raise FileNotFoundError("Aucun traitement terminé dans ce dossier (fichier synthese_*.json absent).")
    syn = json.loads(syns[0].read_text(encoding="utf-8"))
    params = json.loads((out / "parametres.json").read_text(encoding="utf-8")) if (out / "parametres.json").exists() else {}
    job = Job(params.get("dossier_photos", ""), syn["modele"]["id"], str(out), existing=True)
    job.state = "termine"
    job.progress = 1.0
    job.message = "Traitement repris depuis le disque"
    job.result["synthese"] = syn
    job.result["alignement"] = syn.get("alignement", {})
    job.steps_done = [k for k, _, _ in STEPS]
    if (out / "mosaique.tif").exists():
        job.mosaic_path = str(out / "mosaique.tif")
    job.finished = time.time()
    return job
