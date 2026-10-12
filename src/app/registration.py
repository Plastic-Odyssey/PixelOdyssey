"""Recalage des photos entre elles et alignement global.

1. Points SIFT sur chaque photo réduite au quart (mis en cache sur disque).
2. Appariement des photos voisines (emprises GPS qui se recouvrent) : homographie RANSAC,
   avec contrôles de cohérence (nombre de points, échelle, accord avec le GPS).
3. Ajustement global : une transformation affine par photo (pixel -> mètres), estimée par
   moindres carrés à partir des points homologues (précision ~cm), ancrée sur le GPS
   (précision ~m) et sur l'échelle/cap déduits de l'altitude (contrainte faible).
   Les paires dont le résidu reste élevé sont retirées, puis le système est résolu à nouveau.
Hypothèse : sol à peu près plan (vrai sur une plage vue de ~17 m).
"""
from __future__ import annotations

import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from shapely.strtree import STRtree

from . import config
from .inventory import Inventory, Photo

Progress = Callable[[float, str], None]


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """cv2.imread qui accepte les chemins Windows avec accents."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    except Exception:  # noqa: BLE001
        return None


def _features_one(ph: Photo, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    f = cache_dir / (Path(ph.name).stem + f"_{ph.idx}.npz")
    if f.exists():
        d = np.load(f)
        return d["pts"], d["desc"]
    img = imread(ph.path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    sift = cv2.SIFT_create(nfeatures=config.FEATURE_COUNT)
    kps, desc = sift.detectAndCompute(img, None)
    sx = ph.width / img.shape[1]
    sy = ph.height / img.shape[0]
    pts = np.array([[k.pt[0] * sx, k.pt[1] * sy] for k in kps], dtype=np.float32).reshape(-1, 2)
    desc = (np.clip(desc, 0, 255).astype(np.uint8) if desc is not None else np.zeros((0, 128), np.uint8))
    np.savez(f, pts=pts, desc=desc)
    return pts, desc


def compute_features(inv: Inventory, cache_dir: Path, progress: Progress) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    photos = inv.usable
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 4))) as ex:
        futs = {ex.submit(_features_one, p, cache_dir): p for p in photos}
        for fut in as_completed(futs):
            p = futs[fut]
            out[p.idx] = fut.result()
            done += 1
            progress(done / len(photos), f"Points homologues : {done}/{len(photos)} photos")
    return out


def candidate_pairs(inv: Inventory) -> List[Tuple[int, int, float]]:
    photos = inv.usable
    polys = [inv.footprint_local(p) for p in photos]
    tree = STRtree(polys)
    per: Dict[int, List[Tuple[float, int]]] = {p.idx: [] for p in photos}
    for i, g in enumerate(polys):
        for j in tree.query(g):
            j = int(j)
            if j <= i:
                continue
            inter = g.intersection(polys[j]).area
            ov = inter / min(g.area, polys[j].area)
            if ov >= config.PAIR_MIN_OVERLAP:
                per[photos[i].idx].append((ov, photos[j].idx))
                per[photos[j].idx].append((ov, photos[i].idx))
    keep = set()
    for a, lst in per.items():
        for ov, b in sorted(lst, reverse=True)[:config.PAIR_MAX_PER_PHOTO]:
            keep.add((min(a, b), max(a, b), round(ov, 3)))
    return sorted(keep)


def _match_pair(pa: Photo, pb: Photo, fa, fb) -> Optional[Dict]:
    cv2.setRNGSeed(pa.idx * 100003 + pb.idx)  # RANSAC reproductible d'un traitement à l'autre
    (ka, da), (kb, db) = fa, fb
    if len(da) < 30 or len(db) < 30:
        return None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    m = flann.knnMatch(db.astype(np.float32), da.astype(np.float32), k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < config.PAIR_MIN_INLIERS:
        return None
    src = np.float32([kb[g.queryIdx] for g in good])
    dst = np.float32([ka[g.trainIdx] for g in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 8.0, maxIters=4000, confidence=0.999)
    if H is None:
        return None
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    if n < config.PAIR_MIN_INLIERS or n < 0.12 * len(good):
        return None
    # échelle plausible
    J = H[:2, :2] / H[2, 2]
    det = abs(np.linalg.det(J))
    if not (0.5 < det < 2.0):
        return None
    # accord avec le GPS : le centre de b doit tomber près de l'endroit prévu dans a
    cb = np.array([pb.width / 2.0, pb.height / 2.0])
    pred = cv2.perspectiveTransform(cb.reshape(1, 1, 2).astype(np.float64), H).reshape(2)
    ca = np.array([pa.width / 2.0, pa.height / 2.0])
    gps_px = np.linalg.solve(pa.A0, pb.t0 - pa.t0) + ca
    if np.linalg.norm(pred - gps_px) * pa.gsd > 25.0:
        return None
    s, d = src[inl], dst[inl]
    proj = cv2.perspectiveTransform(s.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - d) ** 2, axis=1))))
    # sous-échantillon réparti : grille 8x8 sur l'image a, au plus 3 points par case
    idx = list(range(len(s)))
    random.Random(pa.idx * 100003 + pb.idx).shuffle(idx)
    cells: Dict[Tuple[int, int], int] = {}
    pick = []
    for k in idx:
        key = (int(d[k, 0] / pa.width * 8), int(d[k, 1] / pa.height * 8))
        if cells.get(key, 0) < 3:
            cells[key] = cells.get(key, 0) + 1
            pick.append(k)
    # une paire dont les points tiennent dans une bande étroite contraint mal la géométrie
    if len(pick) < config.PAIR_MIN_POINTS:
        return None
    hull = cv2.convexHull(d[pick].astype(np.float32))
    if cv2.contourArea(hull) < config.PAIR_MIN_SPREAD * pa.width * pa.height:
        return None
    return {"a": pa.idx, "b": pb.idx, "pa": d[pick].astype(np.float64), "pb": s[pick].astype(np.float64),
            "inliers": n, "rms_px": rms}


def match_pairs(inv: Inventory, feats, progress: Progress) -> List[Dict]:
    by_idx = {p.idx: p for p in inv.usable}
    pairs = candidate_pairs(inv)
    out: List[Dict] = []
    if not pairs:
        return out
    done = 0
    with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 4))) as ex:
        futs = [ex.submit(_match_pair, by_idx[a], by_idx[b], feats[a], feats[b]) for a, b, _ in pairs]
        for fut in futs:
            r = fut.result()
            if r is not None:
                out.append(r)
            done += 1
            if done % 5 == 0 or done == len(pairs):
                progress(done / len(pairs), f"Appariement des photos voisines : {done}/{len(pairs)} paires")
    return out


def _Q(th: np.ndarray):
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([c, -s], -1), np.stack([-s, -c], -1)], -2)


def _dQ(th: np.ndarray):
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([-s, -c], -1), np.stack([-c, s], -1)], -2)


class PhotoTransform:
    """Pixel -> sol (mètres, repère local) : g = A (u (1 + k1 r²)) + t, avec u = pixel - centre,
    r = |u| / focale en pixels et A = S * gsd * Q(cap) * B (B : petite déformation affine propre
    à la photo, qui absorbe notamment l'obturateur roulant). Inverse par itérations."""

    def __init__(self, A, t, k1: float, fpx: float, width: int, height: int):
        self.A = np.asarray(A, dtype=np.float64)
        self.Ainv = np.linalg.inv(self.A)
        self.t = np.asarray(t, dtype=np.float64)
        self.k1 = float(k1)
        self.fpx = float(fpx)
        self.c = np.array([width / 2.0, height / 2.0])

    def to_ground(self, px: np.ndarray) -> np.ndarray:
        u = np.asarray(px, dtype=np.float64) - self.c
        r2 = np.sum(u * u, axis=-1, keepdims=True) / self.fpx ** 2
        return (u * (1.0 + self.k1 * r2)) @ self.A.T + self.t

    def to_pixel(self, g: np.ndarray) -> np.ndarray:
        up = (np.asarray(g, dtype=np.float64) - self.t) @ self.Ainv.T
        u = up.copy()
        for _ in range(4):
            r2 = np.sum(u * u, axis=-1, keepdims=True) / self.fpx ** 2
            u = up / (1.0 + self.k1 * r2)
        return u + self.c

    def to_json(self) -> Dict:
        return {"A": self.A.tolist(), "t": self.t.tolist(), "k1": self.k1, "fpx": self.fpx,
                "w": int(self.c[0] * 2), "h": int(self.c[1] * 2)}

    @classmethod
    def from_json(cls, d: Dict) -> "PhotoTransform":
        return cls(d["A"], d["t"], d["k1"], d["fpx"], d["w"], d["h"])


def _sym(e):
    return np.array([[e[0], e[1]], [e[1], e[2]]])


class _Problem:
    """Moindres carrés de l'alignement global (voir global_alignment)."""
    # inconnues par photo : cap, x, y, puis 5 termes de déformation B = I + somme(p_j G_j) :
    # échelle propre (hauteur réelle au-dessus du sol), anisotropie, cisaillement, et les deux composantes du
    # glissement « obturateur roulant » (déplacement du drone pendant la lecture du capteur, proportionnel à la ligne)
    NP = 8
    GEN = np.array([[[1, 0], [0, 1]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 1]]], dtype=np.float64)

    def __init__(self, inv: Inventory, matches: List[Dict]):
        photos = inv.usable
        self.photos = photos
        self.n = n = len(photos)
        col = {p.idx: k for k, p in enumerate(photos)}
        self.gsd = np.array([p.gsd for p in photos])
        self.th0 = np.radians(np.array([p.yaw for p in photos]))
        self.gps = np.array([p.t0 for p in photos])
        self.fpx = np.array([p.focal_mm / p.sensor_width_mm * p.width for p in photos])
        cen = np.array([[p.width / 2.0, p.height / 2.0] for p in photos])
        self.nv = self.NP * n + 3
        self.i_sig, self.ik, self.i_yaw = self.NP * n, self.NP * n + 1, self.NP * n + 2
        self.P = []
        for m in matches:
            ka, kb = col[m["a"]], col[m["b"]]
            ua, ub = m["pa"] - cen[ka], m["pb"] - cen[kb]
            self.P.append((ka, kb, ua, ub, np.sum(ua * ua, 1) / self.fpx[ka] ** 2, np.sum(ub * ub, 1) / self.fpx[kb] ** 2))
        # les écarts d'une même paire ne sont pas indépendants (relief, parallaxe, obturateur) : chaque paire
        # pèse au plus comme PAIR_EFFECTIVE_POINTS observations indépendantes, sinon le GPS ne retient plus rien
        self.pw = [min(1.0, config.PAIR_EFFECTIVE_POINTS / len(P[2])) for P in self.P]

    def x0(self) -> np.ndarray:
        x = np.zeros(self.nv)
        x[1:self.NP * self.n:self.NP] = self.gps[:, 0]
        x[2:self.NP * self.n:self.NP] = self.gps[:, 1]
        x[self.i_sig] = 1.0
        return x

    def unpack(self, x):
        NP, n = self.NP, self.n
        delta = x[0:NP * n:NP]
        t = np.stack([x[1:NP * n:NP], x[2:NP * n:NP]], 1)
        P = np.stack([x[3 + j:NP * n:NP] for j in range(5)], 1)  # termes de déformation
        return delta, t, P, x[self.i_sig], x[self.ik]

    def Bmat(self, P_k) -> np.ndarray:
        return np.eye(2) + np.tensordot(P_k, self.GEN, axes=1)

    def pair(self, x, pi, jac=True):
        """Résidu (pixels de a) et dérivées d'une paire."""
        delta, t, E, sig, k1 = self.unpack(x)
        ka, kb, ua, ub, ra2, rb2 = self.P[pi]
        gsd = self.gsd
        yg = x[self.i_yaw]  # biais de cap commun à tout le vol (compas / nacelle)
        tha, thb = self.th0[ka] + delta[ka] + yg, self.th0[kb] + delta[kb] + yg
        Qa, Qb = _Q(tha), _Q(thb)
        uda = ua * (1 + k1 * ra2)[:, None]
        udb = ub * (1 + k1 * rb2)[:, None]
        Ba, Bb = self.Bmat(E[ka]), self.Bmat(E[kb])
        Bai = np.linalg.inv(Ba)
        gr = gsd[kb] / gsd[ka]
        ta = sig / gsd[ka]
        dt = t[kb] - t[ka]
        ubB = udb @ Bb.T
        Y = gr * (ubB @ (Qa @ Qb).T) + ta * (Qa @ dt)
        Z = Y @ Bai.T
        res = Z - uda
        if not jac:
            return res
        dQa, dQb = _dQ(tha), _dQ(thb)
        nq = len(ua)
        D = {}
        D[("a", 0)] = (gr * (ubB @ (dQa @ Qb).T) + ta * (dQa @ dt)) @ Bai.T
        D[("b", 0)] = gr * (ubB @ (Bai @ Qa @ dQb).T)
        Jt = ta * (Bai @ Qa)
        D[("b", 1)] = np.broadcast_to(Jt[:, 0], (nq, 2))
        D[("b", 2)] = np.broadcast_to(Jt[:, 1], (nq, 2))
        D[("a", 1)] = -D[("b", 1)]
        D[("a", 2)] = -D[("b", 2)]
        MQ = gr * (Bai @ Qa @ Qb)
        for j, G in enumerate(self.GEN):
            D[("b", 3 + j)] = (udb @ G.T) @ MQ.T
            D[("a", 3 + j)] = -((Z @ G.T) @ Bai.T)
        D[("g", self.i_yaw)] = D[("a", 0)] + D[("b", 0)]
        D[("g", self.i_sig)] = np.broadcast_to(Bai @ ((1.0 / gsd[ka]) * (Qa @ dt)), (nq, 2))
        D[("g", self.ik)] = gr * ((ub * rb2[:, None]) @ (Bai @ Qa @ Qb @ Bb).T) - ua * ra2[:, None]
        return res, D

    def _priors(self, x):
        NP = self.NP
        delta, t, E, sig, k1 = self.unpack(x)
        pri_c, pri_w, pri_r = [], [], []
        wg = 1.0 / config.GPS_SIGMA_M
        for k in range(self.n):
            for d in range(2):
                pri_c.append(NP * k + 1 + d); pri_w.append(wg); pri_r.append(wg * (t[k, d] - self.gps[k, d]))
            pri_c.append(NP * k); pri_w.append(1.0 / math.radians(config.YAW_PRIOR_DEG)); pri_r.append(delta[k] / math.radians(config.YAW_PRIOR_DEG))
            xs = x[NP * k + 3:NP * k + 8]
            sigmas = (config.PHOTO_SCALE_PRIOR, config.DEFORMATION_PRIOR, config.DEFORMATION_PRIOR,
                      config.ROLLING_SHUTTER_PRIOR, config.ROLLING_SHUTTER_PRIOR)
            for e, sigma in enumerate(sigmas):
                pri_c.append(NP * k + 3 + e); pri_w.append(1.0 / sigma); pri_r.append(xs[e] / sigma)
        pri_c.append(self.i_sig); pri_w.append(1.0 / config.AFFINE_PRIOR); pri_r.append((sig - 1.0) / config.AFFINE_PRIOR)
        pri_c.append(self.ik); pri_w.append(1.0 / 0.3); pri_r.append(k1 / 0.3)
        wy = 1.0 / math.radians(config.YAW_BIAS_PRIOR_DEG)
        pri_c.append(self.i_yaw); pri_w.append(wy); pri_r.append(wy * x[self.i_yaw])
        return np.array(pri_c), np.array(pri_w), np.array(pri_r)

    @staticmethod
    def _huber(rn):
        """Poids (IRLS) et coût de Huber pour des résidus normalisés (en écarts-types)."""
        k = config.HUBER_K
        w = np.ones_like(rn)
        m = rn > k
        w[m] = k / rn[m]
        cost = np.where(m, 2 * k * rn - k * k, rn * rn)
        return w, cost

    def cost(self, x, active) -> float:
        wm = 1.0 / config.MATCH_SIGMA_PX
        c = 0.0
        for pi in active:
            res = self.pair(x, pi, jac=False)
            if not np.all(np.isfinite(res)):
                return float("inf")
            c += self.pw[pi] * float(np.sum(self._huber(wm * np.linalg.norm(res, axis=1))[1]))
        return c + float(np.sum(self._priors(x)[2] ** 2))

    def build(self, x, active):
        """Jacobienne creuse et résidus pondérés (Huber par point), et coût robuste correspondant."""
        NP = self.NP
        wm = 1.0 / config.MATCH_SIGMA_PX
        rows, cols, vals, rr = [], [], [], []
        r0 = 0
        cost = 0.0
        for pi in active:
            ka, kb = self.P[pi][0], self.P[pi][1]
            res, D = self.pair(x, pi)
            nq = res.shape[0]
            w, c = self._huber(wm * np.linalg.norm(res, axis=1))
            cost += self.pw[pi] * float(np.sum(c))
            sw = np.repeat(np.sqrt(w * self.pw[pi]), 2) * wm
            ridx = r0 + np.arange(2 * nq)
            for (who, k), dv in D.items():
                col = (NP * ka + k) if who == "a" else (NP * kb + k) if who == "b" else k
                rows.append(ridx)
                cols.append(np.full(2 * nq, col))
                vals.append(sw * np.ascontiguousarray(dv).reshape(-1))
            rr.append(sw * res.reshape(-1))
            r0 += 2 * nq
        pri_c, pri_w, pri_r = self._priors(x)
        cost += float(np.sum(pri_r ** 2))
        npri = len(pri_c)
        rows.append(r0 + np.arange(npri)); cols.append(pri_c); vals.append(pri_w)
        rr.append(pri_r)
        R = np.concatenate(rr)
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0 + npri, self.nv))
        return J, R, cost

    def pair_residual_m(self, x, pi) -> float:
        res = self.pair(x, pi, jac=False)
        delta, t, E, sig, k1 = self.unpack(x)
        return float(np.median(np.linalg.norm(res, axis=1))) * self.gsd[self.P[pi][0]] / sig


def _levenberg_marquardt(pb: "_Problem", x: np.ndarray, active: List[int], free: np.ndarray,
                         max_iter: int = 40) -> Tuple[np.ndarray, float]:
    """Levenberg-Marquardt : un pas n'est accepté que s'il fait baisser le coût réel, ce qui empêche
    les divergences (bloc de photos peu relié qui « tourne » ou se déforme sans limite)."""
    J, R, cost = pb.build(x, active)
    lam = 1e-4
    fixed = (~free).astype(np.float64) * 1e12
    for _ in range(max_iter):
        H = (J.T @ J).tocsc()
        g = J.T @ R
        dH = H.diagonal()
        improved = False
        for _try in range(10):
            A = H + sp.diags(lam * np.maximum(dH, 1e-9) + fixed)
            try:
                dx = spsolve(A.tocsc(), -g)
            except Exception:  # noqa: BLE001
                dx = np.full_like(x, np.nan)
            if np.all(np.isfinite(dx)):
                dx[~free] = 0.0
                xn = x + dx
                cn = pb.cost(xn, active)
                if cn < cost:
                    improved = True
                    break
            lam *= 8.0
        if not improved:
            break
        rel = (cost - cn) / max(cost, 1e-12)
        x = xn
        J, R, cost = pb.build(x, active)
        lam = max(lam / 4.0, 1e-7)
        if rel < 1e-6:
            break
    return x, cost


def _free_mask(pb: "_Problem", full: bool) -> np.ndarray:
    free = np.zeros(pb.nv, dtype=bool)
    NP, n = pb.NP, pb.n
    for k in range(4 if not full else NP):
        free[k:NP * n:NP] = True  # cap, x, y, échelle propre (+ anisotropie et cisaillement si complet)
    if full:
        # l'échelle et le biais de cap communs, fixés par la passe rigide, restent figés : sinon ils s'échangent
        # avec la déformation commune de toutes les photos (ambiguïté que seuls les a priori départagent)
        free[pb.ik] = True
    else:
        free[pb.i_sig] = True   # échelle commune (l'altitude relative ne la donne qu'à ~10 % près)
        free[pb.i_yaw] = True   # biais de cap commun (souvent plusieurs degrés sur les drones grand public)
    return free


def _check(pb: "_Problem", x: np.ndarray, active: List[int], full: bool) -> Tuple[bool, str]:
    """Garde-fous : un alignement physiquement aberrant n'est jamais utilisé."""
    delta, t, E, sig, k1 = pb.unpack(x)
    if not np.all(np.isfinite(x)) or sig <= 0:
        return False, "calcul instable"
    shift = np.linalg.norm(t - pb.gps, axis=1)
    if np.median(shift) > config.ALIGN_MAX_GPS_MEDIAN_M or np.max(shift) > config.ALIGN_MAX_GPS_SHIFT_M:
        return False, f"écart au GPS trop grand ({np.median(shift):.1f} m médian, {np.max(shift):.1f} m max)"
    if np.max(np.abs(delta)) > math.radians(config.ALIGN_MAX_YAW_CORR_DEG) or \
            np.median(np.abs(delta)) > math.radians(config.ALIGN_MAX_YAW_MEDIAN_DEG):
        return False, (f"corrections de cap trop grandes ({math.degrees(np.median(np.abs(delta))):.0f}° en médiane, "
                       f"{math.degrees(np.max(np.abs(delta))):.0f}° max)")
    eff = (1.0 / sig) * (1.0 + float(np.median(E[:, 0])))
    if not (1 / 1.25 < eff < 1.25):
        return False, f"facteur d'échelle aberrant (x{eff:.2f})"
    if full:
        if abs(k1) > 0.25 or np.max(np.abs(E[:, :3])) > 0.30 or np.max(np.abs(E[:, 3:])) > 0.8:
            return False, "déformation des photos aberrante"
    if active:
        res = np.median([pb.pair_residual_m(x, pi) for pi in active])
        if not np.isfinite(res) or res > config.ALIGN_MAX_RESIDUAL_M:
            return False, f"résidu trop élevé ({res * 100:.0f} cm)"
    return True, ""


def _initial_yaw_bias(pb: "_Problem", matches: List[Dict]) -> float:
    """Biais de cap commun estimé directement : angle entre le déplacement GPS de deux photos voisines et
    le déplacement mesuré sur les images (médiane sur toutes les paires assez éloignées)."""
    col = {p.idx: k for k, p in enumerate(pb.photos)}
    angles = []
    for m in matches:
        ka, kb = col[m["a"]], col[m["b"]]
        pa, pbb = pb.photos[ka], pb.photos[kb]
        gps = pb.gps[kb] - pb.gps[ka]
        if np.linalg.norm(gps) < 3.0:
            continue
        T, _ = cv2.estimateAffinePartial2D(m["pb"].astype(np.float32), m["pa"].astype(np.float32), method=cv2.LMEDS)
        if T is None:
            continue
        cb = T[:, :2] @ np.array([pbb.width / 2.0, pbb.height / 2.0]) + T[:, 2]
        img = pa.A0 @ (cb - np.array([pa.width / 2.0, pa.height / 2.0]))
        angles.append(math.atan2(gps[1], gps[0]) - math.atan2(img[1], img[0]))
    if len(angles) < 10:
        return 0.0
    a = (np.array(angles) + math.pi) % (2 * math.pi) - math.pi
    # Q(cap + b) tourne les vecteurs au sol de -b : on corrige donc de l'opposé de l'angle mesuré
    return -float(np.median(a))


def global_alignment(inv: Inventory, matches: List[Dict], progress: Progress) -> Dict:
    """Ajustement global (moindres carrés robustes, Levenberg-Marquardt) :
    - par photo : correction de cap, position, petite déformation affine (3 termes) ;
    - pour tout le vol : facteur d'échelle commun et distorsion radiale de l'objectif.
    Les écarts entre points homologues sont mesurés en PIXELS de la photo de référence de
    chaque paire : contracter le bloc ne réduit donc pas les résidus, et l'échelle d'ensemble
    reste fixée par le GPS. Deux passes : rigide (cap + position) puis complète ; chacune est
    contrôlée par des garde-fous, et à défaut on garde la meilleure passe valide (ou le GPS seul).
    Les paires dont le résidu reste élevé sont retirées."""
    pb = _Problem(inv, matches)
    x_gps = pb.x0()
    x_gps[pb.i_yaw] = _initial_yaw_bias(pb, matches)
    active = list(range(len(matches)))
    notes: List[str] = []
    # passe 1 : rigide (cap + position), échelle et objectif fixes
    progress(0.05, "Alignement global : passe rigide")
    x_rigid, _ = _levenberg_marquardt(pb, x_gps.copy(), active, _free_mask(pb, False))
    ok_rigid, why = _check(pb, x_rigid, active, False)
    if not ok_rigid:
        notes.append(f"passe rigide écartée : {why}")
    # passe 2 : complète, avec rejet progressif des paires incohérentes
    x = (x_rigid if ok_rigid else x_gps).copy()
    free = _free_mask(pb, True)
    for outer in range(5):
        x, _ = _levenberg_marquardt(pb, x, active, free)
        resid = {pi: pb.pair_residual_m(x, pi) for pi in active}
        bad = [pi for pi in active if not np.isfinite(resid[pi]) or resid[pi] > config.PAIR_MAX_RESIDUAL_M]
        progress(min(1.0, 0.2 + 0.16 * (outer + 1)), f"Alignement global : {len(active)} paires, {len(bad)} rejetées")
        if not bad:
            break
        worst = set(sorted(bad, key=lambda pi: -resid[pi] if np.isfinite(resid[pi]) else -1e9)[:max(1, len(bad) // 2 + 1)])
        active = [pi for pi in active if pi not in worst]
    ok_full, why = _check(pb, x, active, True)
    if ok_full:
        mode = "complet"
    else:
        notes.append(f"passe complète écartée : {why}")
        active = list(range(len(matches)))
        if ok_rigid:
            mode, x = "rigide", x_rigid
        else:
            mode, x = "gps", x_gps
            active = []
    resid = {pi: pb.pair_residual_m(x, pi) for pi in active}
    delta, t, E, sig, k1 = pb.unpack(x)
    S = 1.0 / sig
    linked = set()
    for pi in active:
        linked.add(pb.photos[pb.P[pi][0]].idx)
        linked.add(pb.photos[pb.P[pi][1]].idx)
    transforms = {}
    for k, p in enumerate(pb.photos):
        A = S * pb.gsd[k] * (_Q(pb.th0[k] + delta[k] + x[pb.i_yaw]) @ pb.Bmat(E[k]))
        T = PhotoTransform(A, t[k], k1, pb.fpx[k], p.width, p.height)
        d = T.to_json()
        d["recalee"] = p.idx in linked
        d["echelle"] = float(S * pb.gsd[k])
        transforms[p.idx] = d
    res = np.array([resid[pi] for pi in active]) if active else np.array([0.0])
    gps_shift = np.linalg.norm(t - pb.gps, axis=1)
    return {
        "transforms": transforms,
        "diagnostic": {
            "paires_candidates": len(matches), "paires_retenues": len(active),
            "photos_recalees": len(linked), "photos_total": pb.n,
            "residu_median_cm": round(float(np.median(res)) * 100, 1),
            "residu_p90_cm": round(float(np.percentile(res, 90)) * 100, 1),
            "ecart_gps_median_m": round(float(np.median(gps_shift)), 2),
            "ecart_gps_max_m": round(float(np.max(gps_shift)), 2),
            "facteur_echelle": round(float(S), 4), "distorsion_k1": round(float(k1), 4),
            "biais_cap_deg": round(float(np.degrees(x[pb.i_yaw])), 2),
            "correction_cap_mediane_deg": round(float(np.degrees(np.median(np.abs(delta)))), 2),
            "echelle_photos_pct": [round(float(np.min(E[:, 0])) * 100, 1), round(float(np.max(E[:, 0])) * 100, 1)],
            "deformation_max_pct": round(float(np.max(np.abs(E[:, 1:3]))) * 100, 2),
            "obturateur_median_pct": round(float(np.median(np.linalg.norm(E[:, 3:], axis=1))) * 100, 1),
            "mode": mode, "remarques": notes,
        },
    }
