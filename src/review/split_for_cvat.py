#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Découpage d'une orthomosaïque en grille sous la limite CVAT.

Étape AUTONOME et préalable à la revue (assisted_annotate.py) - pas
d'inférence, pas d'annotation, pas de label ici : uniquement un découpage
géométrique en grille SANS chevauchement, la plus carrée possible pour le
nombre de morceaux retenu (tiling_geometry.most_square_grid), chaque
morceau restant sous CVAT_MAX_PIXELS (50 000 000 px - constaté par
l'utilisateur, au-delà CVAT refuse l'import).

Pourquoi une étape séparée plutôt qu'un découpage intégré à la revue :
faire l'inférence + la revue sur l'orthomosaïque ENTIÈRE, puis ne
fragmenter qu'à l'écriture, forcerait à attendre l'inférence complète
(potentiellement longue sur une grosse mosaïque) ET, avec l'écriture
incrémentale, l'écriture de TOUS les morceaux de la grille AVANT même
d'ouvrir l'interface de revue - un délai qui donnerait l'impression que
l'interface n'affiche rien. Découper D'ABORD (rapide - uniquement de la
lecture/écriture de pixels, aucune inférence) puis revoir CHAQUE morceau
indépendamment via `assisted_annotate.py` (qui reste ainsi un outil "une
image, une revue, un lot" - voir sa docstring) répartit le travail en
unités plus petites et plus rapides à démarrer, et fait de la contrainte
CVAT le découpage naturel du travail plutôt qu'un détail d'écriture
différé.

Entrée : --image (chemin vers l'image/orthomosaïque source), --output-dir
(dossier où écrire les morceaux - créé si besoin), --n-pieces (optionnel :
calculé automatiquement pour respecter CVAT_MAX_PIXELS si omis, refuse si
fourni et insuffisant - voir `_resolve_n_pieces`).
Sortie : N fichiers image dans --output-dir, nommés {stem}_y{y0}_x{x0}.png
(même convention que les imagettes St Brandon déjà présentes dans
1_annotated_dataset) - AUCUN label, AUCUN data.yaml : chaque morceau est
ensuite revu indépendamment via assisted_annotate.py, qui crée son propre
lot pour chacun.

Exemple :
    python -m src.review.split_for_cvat --image "chemin/vers/grosse_ortho.tif" --output-dir "chemin/vers/dossier_pieces"
    python -m src.review.split_for_cvat --image ... --output-dir ... --n-pieces 6

Puis, pour CHAQUE morceau produit (une revue indépendante par morceau) :
    python -m src.review.assisted_annotate --image "chemin/vers/dossier_pieces/grosse_ortho_y0_x0.png" --lot-name "SL_nouveau_lot_y0_x0"
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.image_io import load_image_bgr
from src.data.tiling_geometry import iter_grid_windows, min_pieces_for_pixel_cap, most_square_grid

# Limite d'import CVAT (constatée empiriquement, pas documentée officiellement) :
# une image de plus de ~50 millions de pixels est refusée à l'import.
CVAT_MAX_PIXELS = 50_000_000


def _resolve_n_pieces(n_pieces: Optional[int], img_w: int, img_h: int, max_pixels: int) -> int:
    """Résout le nombre de morceaux final à partir de la valeur demandée
    (`None` = automatique) et du plafond de pixels CVAT :

    - `n_pieces=None` (défaut CLI) : calcule le minimum requis pour rester
      sous `max_pixels` par morceau (`min_pieces_for_pixel_cap`) - 1 si
      l'image entière tient déjà dedans.
    - `n_pieces` fourni explicitement : REFUSE plutôt que d'écrire
      silencieusement des morceaux que CVAT rejettera, si la valeur donnée
      est insuffisante pour respecter le plafond - indique le minimum
      requis pour cette image précise. Ne refuse jamais une valeur PLUS
      GRANDE que le minimum.

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


def split_image_for_cvat(
    image_path, output_dir, n_pieces: Optional[int] = None, max_pixels: int = CVAT_MAX_PIXELS,
) -> List[Path]:
    """Découpe `image_path` en grille sous `max_pixels` par morceau, écrit
    chaque morceau comme fichier image indépendant dans `output_dir` (créé si
    besoin) - AUCUNE inférence, AUCUN label : cette étape ne fait QUE de la
    géométrie et de l'I/O image, rapide même sur une grosse orthomosaïque.

    `n_pieces` résolu à 1 : l'image tient déjà sous `max_pixels` - copiée
    telle quelle dans `output_dir` (même nom de fichier), aucun découpage.

    Entrée : chemin de l'image source, dossier de sortie, nombre de morceaux
    demandé (None = automatique), plafond de pixels par morceau.
    Sortie : liste des chemins des fichiers écrits dans `output_dir` (un par
    morceau, ou un seul si aucun découpage nécessaire)."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if not image_path.exists():
        raise RuntimeError(f"Image introuvable : {image_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    img = load_image_bgr(image_path)
    if img is None:
        raise RuntimeError(f"Image illisible : {image_path}")
    img_h, img_w = img.shape[:2]

    n_pieces = _resolve_n_pieces(n_pieces, img_w, img_h, max_pixels)

    if n_pieces <= 1:
        dest = output_dir / image_path.name
        print(f"    L'image tient déjà sous la limite CVAT ({max_pixels:,} px) - copie sans découpage vers {dest}.")
        shutil.copy2(image_path, dest)
        return [dest]

    rows, cols = most_square_grid(n_pieces, img_w, img_h)
    _warn_if_elongated(n_pieces, rows, cols, img_w, img_h)
    print(f"    Découpage de {image_path.name} en grille {rows}x{cols} ({rows * cols} morceau(x))...")

    stem = image_path.stem
    written: List[Path] = []
    for x0, y0, x1, y1 in iter_grid_windows(img_w, img_h, rows, cols):
        piece_img = img[y0:y1, x0:x1]
        piece_path = output_dir / f"{stem}_y{y0}_x{x0}.png"
        cv2.imwrite(str(piece_path), piece_img)
        written.append(piece_path)
        print(f"      ... morceau {len(written)}/{rows * cols} écrit ({piece_path.name}).")

    print(
        f"    {len(written)} morceau(x) écrit(s) dans {output_dir} - à revoir indépendamment, un par un, "
        f"via assisted_annotate.py."
    )
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Découpage d'une orthomosaïque en grille sous la limite d'import CVAT - PixelOdyssey"
    )
    parser.add_argument("--image", required=True, help="Chemin vers l'image ou l'orthomosaïque à découper.")
    parser.add_argument("--output-dir", required=True, help="Dossier où écrire les morceaux (créé si besoin).")
    parser.add_argument(
        "--n-pieces", type=int, default=None,
        help="Nombre de morceaux (grille sans chevauchement, la plus carrée possible). Par défaut (omis) : "
             f"calculé automatiquement pour rester sous la limite d'import CVAT ({CVAT_MAX_PIXELS:,} px/morceau) "
             "- 1 seul morceau (copie simple) si l'image tient déjà dedans. Si précisé explicitement et "
             "insuffisant, l'outil refuse avec le minimum requis plutôt que d'écrire des morceaux que CVAT "
             "rejettera.",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=CVAT_MAX_PIXELS,
        help=f"Plafond de pixels par morceau (défaut : {CVAT_MAX_PIXELS:,}, la limite d'import CVAT constatée).",
    )
    args = parser.parse_args()
    try:
        written = split_image_for_cvat(
            args.image, args.output_dir, n_pieces=args.n_pieces, max_pixels=args.max_pixels
        )
        print(f"\n--- ✅ {len(written)} fichier(s) écrit(s) dans {args.output_dir} ---")
        print("    Revue indépendante à lancer pour chaque morceau, par exemple :")
        for p in written:
            print(f'    python -m src.review.assisted_annotate --image "{p}" --lot-name "<nom_du_lot>"')
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
