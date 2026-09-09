#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Production d'annotations assistées par le modèle (cold start).

Revue manuelle, une détection à la fois, des prédictions du modèle sur une
image ou orthomosaïque encore vierge de toute annotation, avant de les
écrire comme vérité terrain candidate. Contrairement à `label_review.py` (qui
complète/corrige un lot déjà annoté), un nouveau lot n'a pas de taxonomie
fine existante : son data.yaml déclare directement les super-classes de
`config/data_config.yaml` comme classes locales - la classe validée ici est
la classe finale, sans passage par une sous-classe fine.

Fonctionnement :
    1. Inférence tuilée sur l'image/mosaïque donnée
       (tiled_inference.predict_parent_image).
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
       raisonnement) DÈS QUE la revue démarre (image(s) copiée(s)/découpée(s)
       et fichiers de label VIDES créés), puis chaque clic "Valider" AJOUTE
       immédiatement sa ligne au(x) fichier(s) de label concerné(s) - rien
       n'est accumulé en mémoire jusqu'à la fin. "Enregistrer et terminer" ne
       fait donc plus qu'arrêter la revue (aucune écriture lourde à ce
       moment-là) : une interruption en cours de route (fermeture, plantage,
       Ctrl-C) ne perd que la revue restante, jamais les annotations déjà
       validées.
    4bis. Si `--auto-write` : saute entièrement l'étape 3 (aucune interface,
       aucune vérification humaine détection par détection). Demande
       interactivement un seuil de confiance (Entrée pour garder celui de
       --conf-threshold), puis écrit DIRECTEMENT toutes les détections
       au-dessus de ce seuil, avec leur classe PRÉDITE par le modèle telle
       quelle. À réserver à un seuil élevé et/ou à un lot qui repassera par
       une passe de contrôle complémentaire dans CVAT - un avertissement est
       affiché avant écriture. Compatible avec `--n-pieces`.
    5. `--n-pieces` : nombre de morceaux dans lesquels fragmenter l'image
       source à l'écriture, en grille SANS chevauchement, la plus carrée
       possible pour ce nombre (tiling_geometry.most_square_grid) - chaque
       morceau devient sa propre image + son propre label, les annotations
       validées étant recadrées à la géométrie de chaque morceau (même
       logique de recadrage que le slicer d'entraînement, src/data/slicer.py).
       Par défaut (omis) : calculé AUTOMATIQUEMENT pour que chaque morceau
       reste sous CVAT_MAX_PIXELS (50 000 000 px - voir sa définition dans
       split_for_cvat.py, seule source de vérité pour cette limite,
       importée ici plutôt que dupliquée)
       (1 seul morceau si l'image tient déjà dedans). Si précisé
       explicitement et insuffisant pour respecter ce plafond, l'outil refuse
       plutôt que d'écrire un lot inutilisable (voir `_resolve_n_pieces`) -
       les deux contraintes ("le plus carré possible" et "sous le plafond
       CVAT") ne s'opposent jamais : la surface d'un morceau ne dépend que de
       leur nombre total, pas de la forme de grille choisie pour ce nombre
       (voir tiling_geometry.min_pieces_for_pixel_cap).

RESULTS_DIR (E:\\PixelOdyssey\\4. Results\\1_assisted_annotation) plutôt que
1_annotated_dataset directement : ce lot est une PROPOSITION issue du
modèle + revue manuelle, pas encore une vérité terrain définitive au même
titre que les lots annotés from scratch - le format de sortie reste
strictement identique (images/train + labels/train + data.yaml) pour rester
immédiatement réimportable dans CVAT pour une passe de contrôle
complémentaire, ou copié tel quel dans 1_annotated_dataset si aucune
correction n'est jugée nécessaire.

Entrée : --image (chemin vers l'image/orthomosaïque à annoter), --lot-name
(nom du nouveau lot), --model (optionnel, sélection interactive sinon).
Sortie : le lot annoté écrit sous RESULTS_DIR/<lot_name>/.

Exemple :
    python -m src.review.assisted_annotate --image "chemin/vers/image.tif" --lot-name "SL_nouveau_lot"
    python -m src.review.assisted_annotate --image ... --lot-name ... --model output/runs/<run>/weights/best.pt
    python -m src.review.assisted_annotate --image "chemin/vers/grosse_ortho.tif" --lot-name "SL_nouveau_lot" --n-pieces 6
    python -m src.review.assisted_annotate --image ... --lot-name ... --auto-write

Puis ouvrir l'URL affichée (http://127.0.0.1:8765/ par défaut) dans un
navigateur (sauf en mode --auto-write, qui n'ouvre aucune interface).
"""

import argparse
import json
import math
import os
import shutil
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
from shapely.geometry import MultiPolygon, Polygon, box

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import DEFAULT_CLASS_CONFIG_PATH, assert_model_matches_taxonomy, load_class_config
from src.data.image_io import load_image_bgr
from src.data.tiling_geometry import iter_grid_windows, min_pieces_for_pixel_cap, most_square_grid
from src.review.label_review import RUNS_DIR, _discover_available_models, _prompt_model_choice
from src.review.matching import LabeledPolygon
from src.review.split_for_cvat import CVAT_MAX_PIXELS
from src.review.tiled_inference import make_ultralytics_predict_fn, predict_parent_image

MARGIN_PX = 80          # marge autour du masque dans le chip de revue
MIN_CROP_DISPLAY_PX = 360  # un chip plus petit que ça est agrandi pour rester lisible
OUTLINE_COLOR_BGR = (0, 235, 255)  # jaune vif (cohérent avec geo_density_map.py, converti BGR<->RGB)

# Dossier de sortie des lots produits par cet outil - voir le raisonnement
# dans la docstring du module (staging avant CVAT/promotion vers
# 1_annotated_dataset, pas écrit directement dans le dataset d'entraînement).
RESULTS_DIR = r"E:\PixelOdyssey\4. Results\1_assisted_annotation"

# Ratio d'aire minimal conservé pour un fragment d'annotation recadré au bord
# d'un morceau de grille (--n-pieces > 1) - même valeur par défaut et même
# raisonnement que PlasticImageSlicer.min_area_ratio (src/data/slicer.py) :
# écarte un fragment résiduel négligeable plutôt que de polluer le label
# d'un morceau voisin avec un confetti sans valeur.
GRID_MIN_AREA_RATIO = 0.05

# CVAT_MAX_PIXELS (limite d'import CVAT) est importé depuis split_for_cvat.py
# plutôt que redéfini ici, pour n'avoir qu'une seule source de vérité pour
# cette valeur - voir sa définition et son raisonnement là-bas. Sert de
# plafond par défaut pour la résolution automatique de --n-pieces (voir
# _resolve_n_pieces) - une orthomosaïque plus grande que ça DOIT être
# fragmentée avant d'être proposée à CVAT, ce n'est plus une option.

# En dessous de ce nombre de candidates, une classe est signalée comme
# "échantillon trop faible" dans le résumé : sous ce seuil, une statistique
# par classe (ici juste un compte, pas encore une métrique de qualité) est
# trop bruitée pour juger quoi que ce soit dessus, seulement pour repérer une
# présence. Question ouverte : la valeur (10) est un repère raisonnable mais
# arbitraire, pas dérivé d'une analyse formelle de puissance statistique.
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


def _clip_one_to_window(
    v: Dict, img_w: int, img_h: int, x0: int, y0: int, x1: int, y1: int,
    min_area_ratio: float = GRID_MIN_AREA_RATIO,
) -> List[str]:
    """Recadre UNE annotation validée (coordonnées normalisées PLEINE IMAGE)
    sur une fenêtre de la grille, retourne des lignes de label YOLO-seg en
    coordonnées normalisées LOCALES à cette fenêtre.

    Même logique de recadrage que PlasticImageSlicer.slice_single_pair
    (src/data/slicer.py) : intersection shapely avec la fenêtre, filtre
    `min_area_ratio` pour écarter un fragment négligeable en bord de
    découpe (une grille SANS chevauchement, contrairement au tuilage
    d'entraînement, ne rattrape pas ailleurs l'objet coupé - le fragment
    gardé est la seule trace de cet objet dans ce morceau).

    Factorisé à l'unité pour être appelable dès qu'UNE annotation est validée
    (`_append_validated_to_pieces`, revue manuelle) et pas seulement en bloc
    à la fin (`_clip_validated_to_window`, conservé pour --auto-write qui
    construit sa liste en un coup)."""
    window = box(x0, y0, x1, y1)
    piece_w, piece_h = x1 - x0, y1 - y0
    lines: List[str] = []

    coords = v["coords_norm"]
    pixel_pts = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, len(coords), 2)]
    if len(pixel_pts) < 3:
        return lines
    geom = Polygon(pixel_pts)
    if not geom.is_valid or geom.area <= 0 or not window.intersects(geom):
        return lines

    intersection = window.intersection(geom)
    if intersection.is_empty or intersection.area <= 0:
        return lines
    if (intersection.area / geom.area) < min_area_ratio:
        return lines

    if isinstance(intersection, Polygon):
        parts = [intersection]
    elif isinstance(intersection, MultiPolygon):
        parts = list(intersection.geoms)
    else:
        return lines

    for part in parts:
        if part.area <= 0:
            continue
        local: List[float] = []
        for x, y in part.exterior.coords:
            local.extend([
                max(0.0, min(1.0, (x - x0) / piece_w)),
                max(0.0, min(1.0, (y - y0) / piece_h)),
            ])
        if len(local) >= 6:
            coords_str = " ".join(f"{c:.6f}" for c in local)
            lines.append(f"{v['class_id']} {coords_str}\n")

    return lines


def _clip_validated_to_window(
    validated: List[Dict], img_w: int, img_h: int, x0: int, y0: int, x1: int, y1: int,
    min_area_ratio: float = GRID_MIN_AREA_RATIO,
) -> List[str]:
    """Version en lot de `_clip_one_to_window`, conservée pour --auto-write
    (qui construit toute sa liste `validated` avant écriture, contrairement
    à la revue manuelle qui valide un élément à la fois)."""
    lines: List[str] = []
    for v in validated:
        lines.extend(_clip_one_to_window(v, img_w, img_h, x0, y0, x1, y1, min_area_ratio))
    return lines


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

    
def _resolve_n_pieces(n_pieces: Optional[int], img_w: int, img_h: int, max_pixels: int) -> int:
    """Résout le nombre de morceaux final à partir de la valeur demandée
    (`None` = automatique) et du plafond de pixels CVAT, en combinant les
    deux contraintes plutôt que de les traiter comme indépendantes :

    - `n_pieces=None` (défaut CLI) : calcule le minimum requis pour rester
      sous `max_pixels` par morceau (`min_pieces_for_pixel_cap`) - 1 si
      l'image entière tient déjà dedans, comportement identique à avant
      l'ajout de cette contrainte.
    - `n_pieces` fourni explicitement : REFUSE plutôt que d'écrire
      silencieusement un lot que CVAT rejettera, si la valeur donnée est
      insuffisante pour respecter le plafond - indique le minimum requis
      pour cette image précise. Ne refuse jamais une valeur PLUS GRANDE que
      le minimum (l'utilisateur peut vouloir des morceaux plus petits que le
      strict nécessaire, ex: pour paralléliser une revue manuelle).

    Entrée : valeur demandée (ou None), dimensions de l'image, plafond de
    pixels par morceau.
    Sortie : nombre de morceaux à utiliser (toujours >= 1)."""
    min_required = min_pieces_for_pixel_cap(img_w, img_h, max_pixels)
    total_px = img_w * img_h

    if n_pieces is None:
        if min_required > 1:
            print(
                f"    --n-pieces non précisé : image {img_w}x{img_h} = {total_px:,} px > limite CVAT "
                f"({max_pixels:,} px/image) - découpage automatique en {min_required} morceau(x) minimum "
                f"pour rester importable."
            )
        return min_required

    if n_pieces < 1:
        raise RuntimeError(f"--n-pieces doit être >= 1 (reçu {n_pieces}).")
    if n_pieces < min_required:
        raise RuntimeError(
            f"--n-pieces {n_pieces} insuffisant pour cette image ({img_w}x{img_h} = {total_px:,} px) : "
            f"chaque morceau ferait encore ~{total_px // n_pieces:,} px, au-dessus de la limite CVAT "
            f"({max_pixels:,} px/image) - l'import serait refusé. Utilise au moins --n-pieces {min_required} "
            f"(ou omets --n-pieces pour laisser l'outil calculer ce minimum automatiquement)."
        )
    return n_pieces


def _warn_if_elongated(n_pieces: int, rows: int, cols: int, img_w: int, img_h: int) -> None:
    """Avertit si la grille trouvée pour `n_pieces` reste très allongée
    (diviseurs déséquilibrés, typiquement n_pieces premier) - n'empêche pas
    l'exécution, juste un signal pour reconsidérer --n-pieces si le résultat
    visuel ne convient pas."""
    piece_w, piece_h = img_w / cols, img_h / rows
    ratio = max(piece_w, piece_h) / min(piece_w, piece_h)
    if ratio > 1.5:
        suggestions = [n for n in range(max(2, n_pieces - 2), n_pieces + 3) if n != n_pieces]
        print(
            f"  ⚠️  Avec {n_pieces} morceau(x), la grille la plus carrée possible est {rows}x{cols} "
            f"(morceaux {piece_w:.0f}x{piece_h:.0f}px, ratio {ratio:.1f}:1 - assez allongé). "
            f"{n_pieces} a peu de diviseurs adaptés à ce ratio d'image. Essaie éventuellement "
            f"{', '.join(str(n) for n in suggestions)} pour un résultat plus carré."
        )


class Piece:
    """Un morceau de la grille de sortie (ou l'unique morceau si
    n_pieces <= 1, fenêtre = l'image entière) - son image et son fichier de
    label VIDE existent déjà sur disque dès la construction (voir
    `_init_lot`) ; `label_path` est ensuite complété au fur et à mesure par
    `_append_validated_to_pieces`, une ligne à la fois, jamais en un seul
    bloc final (voir le raisonnement dans la docstring du module) : une
    interruption en cours de route ne perd alors que la revue restante,
    jamais les annotations déjà validées."""

    __slots__ = ("label_path", "x0", "y0", "x1", "y1")

    def __init__(self, label_path: Path, x0: int, y0: int, x1: int, y1: int):
        self.label_path = label_path
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


def _init_lot(
    lot_dir: Path,
    image_path: Path,
    img,
    img_w: int,
    img_h: int,
    target_names: Dict[int, str],
    n_pieces: int = 1,
) -> List[Piece]:
    """Initialise le lot sous RESULTS_DIR/<lot_name>/ - images/train/ +
    labels/train/ + data.yaml - TÔT, avant la moindre décision de revue :
    copie l'image (ou découpe + écrit chaque morceau de la grille), crée un
    fichier de label VIDE par morceau, et écrit data.yaml. Rien ici ne
    dépend des décisions de revue (seulement de l'image source et de
    --n-pieces), donc rien n'empêche de l'écrire dès que le lot est confirmé
    non vide (candidates non vides) plutôt que d'attendre la fin de la revue.

    Chaque décision validée est ensuite ajoutée au fur et à mesure à son(ses)
    fichier(s) de label déjà existant(s) - voir `_append_validated_to_pieces`.

    Le sous-dossier `train/` est requis par le format d'import CVAT
    ("Ultralytics YOLO Segmentation" : structure `images/<subset>/` +
    `labels/<subset>/`) ; le pipeline d'ingestion standard le tolère aussi
    bien qu'un dossier `images/` à plat (mirroring récursif images->labels).
    Un seul "split" ici (`train`, nom arbitraire) : ce lot n'a pas de notion
    train/val/test avant `split_dataset.py`.

    `n_pieces == 1` (défaut) : comportement inchangé - l'image source est
    copiée telle quelle, un seul fichier de label (Piece dont la fenêtre =
    l'image entière).
    `n_pieces > 1` : l'image est fragmentée en grille (voir
    tiling_geometry.most_square_grid/iter_grid_windows), chaque morceau
    devenant sa propre image (.png, écrite depuis les pixels déjà chargés en
    mémoire - jamais une recompression du fichier source) + son propre
    label VIDE. Nommage `{stem}_y{y0}_x{x0}` - même convention que les
    imagettes St Brandon déjà présentes dans 1_annotated_dataset
    (`StBrandonTransect 1_y10164_x3696.jpg`).

    Entrée : dossier du lot, chemin de l'image source, image déjà chargée en
    mémoire (BGR - rechargée depuis `image_path` si absente et n_pieces > 1),
    dimensions de l'image, noms de classes, nombre de morceaux.
    Sortie : liste des `Piece` (un par morceau, un seul si n_pieces == 1),
    chacun déjà pourvu de son image et de son fichier de label VIDE sur
    disque."""
    img_dir = lot_dir / "images" / "train"
    lab_dir = lot_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    pieces: List[Piece] = []

    if n_pieces <= 1:
        print(f"    💾 Copie de {image_path.name} dans le nouveau lot...")
        shutil.copy2(image_path, img_dir / image_path.name)
        label_path = lab_dir / f"{image_path.stem}.txt"
        label_path.write_text("", encoding="utf-8")
        pieces.append(Piece(label_path, 0, 0, img_w, img_h))
    else:
        rows, cols = most_square_grid(n_pieces, img_w, img_h)
        _warn_if_elongated(n_pieces, rows, cols, img_w, img_h)
        print(f"    💾 Découpage de {image_path.name} en grille {rows}x{cols} ({rows * cols} morceau(x))...")

        if img is None:
            img = load_image_bgr(image_path)
            if img is None:
                raise RuntimeError(f"Image illisible pour le découpage en grille : {image_path}")

        stem = image_path.stem
        n_written = 0
        for x0, y0, x1, y1 in iter_grid_windows(img_w, img_h, rows, cols):
            piece_img = img[y0:y1, x0:x1]
            piece_name = f"{stem}_y{y0}_x{x0}"
            cv2.imwrite(str(img_dir / f"{piece_name}.png"), piece_img)
            label_path = lab_dir / f"{piece_name}.txt"
            label_path.write_text("", encoding="utf-8")
            pieces.append(Piece(label_path, x0, y0, x1, y1))
            n_written += 1
            print(f"      ... morceau {n_written}/{rows * cols} écrit ({piece_name}.png).")
        print(f"    {n_written} morceau(x) écrit(s) - labels vides prêts, remplis au fur et à mesure de la revue.")

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

    return pieces


def _append_validated_to_pieces(v: Dict, img_w: int, img_h: int, pieces: List[Piece]) -> None:
    """Ajoute IMMÉDIATEMENT une annotation validée (coordonnées normalisées
    PLEINE IMAGE) à chaque morceau qu'elle recoupe - appelée à chaque
    décision « Valider » (revue manuelle) ou pour chaque détection retenue
    (--auto-write), jamais accumulée en RAM pour une écriture groupée en fin
    de traitement (voir Piece/_init_lot pour le raisonnement).

    Cas `n_pieces == 1` (un seul Piece, fenêtre = l'image entière) : la
    ligne pleine image est déjà dans le bon référentiel, écrite telle
    quelle, aucun recadrage nécessaire. Sinon, recadre sur chaque morceau via
    `_clip_one_to_window` et n'ajoute une ligne qu'aux morceaux réellement
    concernés.

    Entrée : l'annotation validée ({class_id, coords_norm}), dimensions de
    l'image, morceaux du lot (déjà initialisés par `_init_lot`).
    Sortie : aucune - append en mode texte (`"a"`) sur le(s) fichier(s) de
    label concerné(s)."""
    if len(pieces) == 1 and (pieces[0].x0, pieces[0].y0, pieces[0].x1, pieces[0].y1) == (0, 0, img_w, img_h):
        coords_str = " ".join(f"{c:.6f}" for c in v["coords_norm"])
        with open(pieces[0].label_path, "a", encoding="utf-8") as f:
            f.write(f"{v['class_id']} {coords_str}\n")
        return

    for piece in pieces:
        lines = _clip_one_to_window(v, img_w, img_h, piece.x0, piece.y0, piece.x1, piece.y1)
        if lines:
            with open(piece.label_path, "a", encoding="utf-8") as f:
                f.writelines(lines)


class _ReviewState:
    """État partagé du serveur de revue - un seul utilisateur, une seule
    session à la fois (pas besoin de plus pour cet outil).

    Le lot est déjà initialisé sur disque (image(s) copiée(s)/découpée(s) +
    labels VIDES + data.yaml) AVANT la construction de cet état, via
    `_init_lot` - `pieces` référence ces fichiers de label déjà créés.
    `decide()` écrit immédiatement chaque validation (voir
    `_append_validated_to_pieces`) ; `finish()` ne fait donc plus AUCUNE
    écriture (juste marquer la revue comme terminée pour l'interface) - voir
    le raisonnement dans la docstring du module."""

    def __init__(self, img, candidates: List[LabeledPolygon], target_names: Dict[int, str],
                 pieces: List[Piece], img_w: int, img_h: int):
        self.img = img
        self.candidates = candidates
        self.target_names = target_names
        self.pieces = pieces
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
                "output": [str(p.label_path) for p in self.pieces],
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
                _append_validated_to_pieces(v, self.img_w, self.img_h, self.pieces)
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
    img,
    target_names: Dict[int, str],
    lot_dir: Path,
    image_path: Path,
    img_w: int,
    img_h: int,
    n_pieces: int,
) -> List[Path]:
    """Chemin ALTERNATIF à la revue manuelle (`--auto-write`) : demande
    interactivement un seuil de confiance, puis écrit directement toutes les
    détections au-dessus de ce seuil comme annotations validées - la classe
    écrite est TOUJOURS celle prédite par le modèle, jamais vérifiée une à
    une. Aucune interface web n'est ouverte dans ce chemin.

    Entrée : détections brutes (non filtrées par --conf-threshold - le seuil
    réellement appliqué est celui choisi interactivement ici, --conf-threshold
    ne sert que de valeur par défaut suggérée dans le prompt), image en
    mémoire, reste identique à run_assisted_annotate.
    Sortie : liste des chemins de labels écrits (même contrat que le retour
    normal de run_assisted_annotate - un chemin par morceau si n_pieces > 1).

    Le lot est initialisé (`_init_lot`) puis chaque détection retenue est
    ajoutée au fur et à mesure (`_append_validated_to_pieces`) plutôt que
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

    pieces = _init_lot(lot_dir, image_path, img, img_w, img_h, target_names, n_pieces=n_pieces)
    for p in to_write:
        v = {"class_id": p.class_id, "coords_norm": _normalize_geom_to_full_image(p.geom, img_w, img_h)}
        _append_validated_to_pieces(v, img_w, img_h, pieces)

    written = [piece.label_path for piece in pieces]
    print(f"\n--- ✅ Lot écrit : {lot_dir} ({len(written)} fichier(s) de label) ---")
    print(f"    {len(to_write)} annotation(s) écrite(s) SANS validation manuelle (seuil {write_threshold:.0%}).")
    print(f"    Ce lot est au format standard (images/train + labels/train + data.yaml) mais N'EST PAS")
    print(f"    dans 1_annotated_dataset : repasse-le par CVAT si besoin d'une passe de contrôle, ou")
    print(f"    copie-le tel quel dans 1_annotated_dataset/{lot_dir.name}/ puis lance data_pipeline.py --force.")
    return written


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
      "\nRassure-toi : chaque annotation validée a déjà été écrite sur disque au moment du clic sur " +
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
    n_pieces: Optional[int] = None,
    auto_write: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    predict_tile_fn=None,
) -> List[Path]:
    """`predict_tile_fn` : comme dans label_review.py/visualize_predictions.py,
    permet d'injecter un faux prédicteur pour les tests sans modèle réel.

    `n_pieces` : `None` (défaut) = calculé automatiquement pour rester sous
    la limite d'import CVAT (`CVAT_MAX_PIXELS` px/morceau) - 1 seul morceau
    si l'image tient déjà dedans. Si fourni explicitement et insuffisant
    pour respecter cette limite, échoue avec le minimum requis plutôt que
    d'écrire un lot que CVAT refusera à l'import (voir `_resolve_n_pieces`).

    `auto_write` : saute entièrement l'interface de revue manuelle - demande
    interactivement (input()) un seuil de confiance, puis écrit directement
    toutes les détections au-dessus comme annotations validées, classe
    prédite telle quelle. Voir `_run_auto_write` pour le détail et la mise en
    garde. `host`/`port`/`open_browser` sont ignorés dans ce mode (aucun
    serveur n'est démarré).

    Retourne la liste des chemins de labels écrits. En mode revue manuelle
    (`auto_write=False`, comportement par défaut inchangé), bloque jusqu'à la
    fin de la revue (cette fonction ne rend la main qu'après que le serveur a
    reçu la décision de terminer, via /api/finish déclenché par le bouton ou
    automatiquement quand la file est épuisée). En mode `auto_write=True`,
    rend la main dès l'écriture terminée (après le seul prompt de seuil)."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise RuntimeError(f"Image introuvable : {image_path}")

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
    img = load_image_bgr(image_path)
    if img is None:
        raise RuntimeError(f"Image illisible : {image_path}")
    img_h, img_w = img.shape[:2]

    n_pieces = _resolve_n_pieces(n_pieces, img_w, img_h, CVAT_MAX_PIXELS)
    if n_pieces > 1:
        rows, cols = most_square_grid(n_pieces, img_w, img_h)
        _warn_if_elongated(n_pieces, rows, cols, img_w, img_h)
        print(f"    Sortie prévue en grille {rows}x{cols} ({n_pieces} morceau(x)) - "
              f"l'inférence/la revue restent sur l'image entière, le découpage n'intervient qu'à l'écriture.")

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
        return _run_auto_write(
            predictions, conf_threshold, img, target_names, lot_dir, image_path, img_w, img_h, n_pieces,
        )

    if not candidates:
        print("    Rien à revoir à ce seuil de confiance - aucun lot créé.")
        raise RuntimeError(
            f"Aucune détection >= {conf_threshold:.0%} sur cette image. Essaie un --conf-threshold "
            f"plus bas, ou vérifie que le bon modèle est utilisé."
        )

    # Le lot est initialisé (image(s)/grille + labels VIDES + data.yaml) ICI,
    # avant même d'ouvrir l'interface de revue - voir _init_lot. Chaque clic
    # "Valider" ajoutera ensuite immédiatement sa ligne au fichier de label
    # concerné (_ReviewState.decide -> _append_validated_to_pieces) : plus
    # aucune écriture lourde n'est différée jusqu'à "Enregistrer et
    # terminer", pour qu'une interruption en cours de revue ne perde jamais
    # les annotations déjà validées.
    pieces = _init_lot(lot_dir, image_path, img, img_w, img_h, target_names, n_pieces=n_pieces)
    state = _ReviewState(img, candidates, target_names, pieces, img_w, img_h)
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

    print(f"\n--- ✅ Lot écrit : {lot_dir} ({len(pieces)} fichier(s) de label) ---")
    print(f"    {state.validated_count} annotation(s) validée(s), {state.deleted_count} supprimée(s).")
    print(f"    Ce lot est au format standard (images/train + labels/train + data.yaml) mais N'EST PAS")
    print(f"    dans 1_annotated_dataset : repasse-le par CVAT si besoin d'une passe de contrôle, ou")
    print(f"    copie-le tel quel dans 1_annotated_dataset/{lot_name}/ puis lance data_pipeline.py --force.")
    return [piece.label_path for piece in pieces]


def _class_group_stats(
    items: List[LabeledPolygon], target_names: Dict[int, str], img_area: Optional[float]
) -> List[Dict]:
    """Statistiques descriptives PAR CLASSE sur un groupe de détections
    (prédictions brutes, ou candidates au-dessus du seuil de confiance) :
    effectif, distribution de confiance (min/médiane/moyenne/max), et aire
    relative médiane du masque (aire du polygone / aire de l'image parente) -
    utile pour repérer une classe dont les masques prédits sont anormalement
    petits ou grands par rapport à ce qu'on attend (ex: `Bouchon`/`Bouee`).

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
    disque (contrairement à `_init_lot`/`_append_validated_to_pieces`), sert à juger AVANT d'ouvrir
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
            f"dessus."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production d'annotations assistées par le modèle (cold start) - PixelOdyssey")
    parser.add_argument("--image", required=True, help="Chemin vers l'image ou l'orthomosaïque à annoter.")
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
    parser.add_argument("--n-pieces", type=int, default=None,
                         help="Nombre de morceaux dans lesquels fragmenter l'image/l'orthomosaïque à "
                              "l'ÉCRITURE du résultat (grille sans chevauchement, la plus carrée possible - "
                              "voir tiling_geometry.most_square_grid). Par défaut (omis) : calculé "
                              f"automatiquement pour rester sous la limite d'import CVAT ({CVAT_MAX_PIXELS:,} "
                              "px/morceau) - 1 seul morceau si l'image tient déjà dedans. Si précisé "
                              "explicitement et insuffisant pour respecter cette limite, l'outil refuse "
                              "avec le nombre minimal requis plutôt que d'écrire un lot que CVAT rejettera.")
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
            n_pieces=args.n_pieces,
            auto_write=args.auto_write,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
