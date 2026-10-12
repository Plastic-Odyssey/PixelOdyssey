#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Contrôle d'alignement annotations / images (1_annotated_dataset).

Pourquoi : le 12/10/2026, deux lots se sont révélés faux sans qu'aucun outil ne le signale :
  - `SL 6-10_26-27` : fichiers d'annotation des transects 6 à 10 inversés (6<->10, 7<->9) et tracés
    sur un export légèrement différent de l'orthophoto (zoom ~1,05, rotation ~-1,5°, ~80 px) ;
  - `SL W6 n` : annotations faites sur les photos du vol W4 qui portent le même nom (DJI_07xx).
Les contours tombaient sur du sable vide (ou dans la mer) : le modèle apprenait le contraire de la réalité.

Principe : un contour bien placé entoure un objet qui se distingue du fond. On calcule une carte
d'« objectivité » (écart de couleur entre l'image et sa version très floutée), puis le score moyen de
cette carte à l'intérieur des contours annotés :
  - tel quel (score_identite) ;
  - et après le meilleur petit zoom (0,94 à 1,08) + décalage libre (score_meilleur, corrélation FFT).
Une image bien annotée a un score élevé tel quel (0,8 à 2,5 sur nos lots) et ne gagne rien à être décalée.
Un score tel quel < 0,4, ou un gain net après décalage, mérite un coup d'œil dans CVAT.
Faux positifs connus : sols très texturés (roche d'Aldabra) donnent des scores bas mais sans décalage.

Option --vols-bruts : pour les lots de photos DJI, teste aussi les photos de même nom dans les autres
vols du dossier de photos brutes (détecte le cas « photo du mauvais vol »).

Exemples :
    python -m src.data.diagnostics.label_alignment_check
    python -m src.data.diagnostics.label_alignment_check --lot "SL W6 n" --vols-bruts "E:\\PixelOdyssey\\2. Raw data\\3. Santa Luzia\\Raw pictures"
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def objectness(img: np.ndarray, f: int) -> np.ndarray:
    alpha = img[:, :, 3] > 0 if img.ndim == 3 and img.shape[2] == 4 else None
    bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (w // f, h // f), interpolation=cv2.INTER_AREA).astype(np.float32)
    o = np.linalg.norm(small - cv2.GaussianBlur(small, (0, 0), 8), axis=2)
    valid = np.ones(o.shape, bool) if alpha is None else cv2.resize(alpha.astype(np.uint8), (w // f, h // f)) > 0
    o = (o - o[valid].mean()) / (o[valid].std() + 1e-6)
    o[~valid] = 0
    return o


def read_polygons(label_path: str):
    polys = []
    for line in open(label_path, encoding="utf-8"):
        v = line.split()
        if len(v) >= 7:
            polys.append(np.array(v[1:], float).reshape(-1, 2))
    return polys


def scores(o: np.ndarray, polys, w: int, h: int, f: int):
    F = np.fft.rfft2(o)

    def one(s):
        m = np.zeros(o.shape, np.float32)
        for xy in polys:
            cv2.fillPoly(m, [(xy * [w / f * s, h / f * s]).astype(np.int32)], 1.0)
        cc = np.fft.irfft2(F * np.conj(np.fft.rfft2(m, s=o.shape)), s=o.shape) / max(m.sum(), 1.0)
        k = np.unravel_index(np.argmax(cc), cc.shape)
        dy = k[0] if k[0] < o.shape[0] // 2 else k[0] - o.shape[0]
        dx = k[1] if k[1] < o.shape[1] // 2 else k[1] - o.shape[1]
        return float(cc[0, 0]), float(cc.max()), dx * f, dy * f
    ident = one(1.0)[0]
    best = max(((one(s), s) for s in np.arange(0.94, 1.081, 0.02)), key=lambda t: t[0][1])
    (_, b, dx, dy), s = best
    return ident, b, float(s), int(dx), int(dy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=None, help="dossier 1_annotated_dataset (défaut : celui du disque de données)")
    ap.add_argument("--lot", action="append", help="limiter à ce(s) lot(s)")
    ap.add_argument("--vols-bruts", default=None, help="dossier des vols bruts (photos de même nom dans les autres vols)")
    ap.add_argument("--output", default="controle_alignement_annotations.csv")
    ap.add_argument("--min-objets", type=int, default=5, help="ignorer les images avec moins d'objets (score trop bruité)")
    args = ap.parse_args()
    if args.raw_dir:
        root = Path(args.raw_dir)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from src.paths_config import DATA_ROOT
        root = Path(DATA_ROOT) / "3. Processed dataset" / "1_annotated_dataset"
    lots = sorted(d for d in os.listdir(root) if (root / d / "images").is_dir() and not d.startswith("."))
    if args.lot:
        lots = [l for l in lots if l in args.lot]
    rows = []
    for lot in lots:
        img_root = root / lot / "images"
        for ip in sorted(glob.glob(str(img_root / "**" / "*"), recursive=True)):
            if not ip.lower().endswith(IMG_EXT):
                continue
            rel = Path(ip).relative_to(img_root).with_suffix(".txt")  # sous-dossiers éventuels (lots SB)
            lp = root / lot / "labels" / rel
            if not lp.exists():
                continue
            polys = read_polygons(str(lp))
            if len(polys) < args.min_objets:
                continue
            img = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            f = max(1, int(round(max(w, h) / 1400)))
            ident, best, s, dx, dy = scores(objectness(img, f), polys, w, h, f)
            suspect = ident < 0.4 or (best > 1.3 * max(ident, 0.05) and (abs(dx) + abs(dy) > 12 or abs(s - 1) > 0.01))
            other = ""
            if args.vols_bruts and suspect:
                cands = []
                for fl in sorted(os.listdir(args.vols_bruts)):
                    p2 = Path(args.vols_bruts) / fl / Path(ip).name
                    if p2.exists():
                        im2 = cv2.imread(str(p2))
                        if im2 is not None and im2.shape[:2] == (h, w):
                            o2 = objectness(im2, f)
                            m = np.zeros(o2.shape, np.float32)
                            for xy in polys:
                                cv2.fillPoly(m, [(xy * [w / f, h / f]).astype(np.int32)], 1.0)
                            cands.append((float((o2 * m).sum() / m.sum()), fl))
                if cands:
                    sc, fl = max(cands)
                    if sc > max(1.0, 2 * ident):
                        other = f"{fl} ({sc:.2f})"
            rows.append(dict(lot=lot, image=Path(ip).name, n_objets=len(polys), score_identite=round(ident, 2),
                             score_meilleur=round(best, 2), zoom=round(s, 2), dx_px=dx, dy_px=dy,
                             suspect="OUI" if suspect else "", meilleure_photo_autre_vol=other))
            r = rows[-1]
            print(f"{lot[:30]:30s} {r['image'][:24]:24s} n={len(polys):4d} tel quel {ident:5.2f} | "
                  f"meilleur {best:5.2f} (zoom {s:.2f}, {dx:+d},{dy:+d} px)"
                  + ("  <== à vérifier" if suspect else "") + (f"  -> annotations de {other}" if other else ""),
                  flush=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["lot"], delimiter=";")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n{sum(1 for r in rows if r['suspect'])} image(s) à vérifier sur {len(rows)} ; détail : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
