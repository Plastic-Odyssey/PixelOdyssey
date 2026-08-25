#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Découverte des images parentes dans le dataset brut annoté.

"""

from pathlib import Path
from typing import Dict, List

VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_corresponding_label(img_path: Path) -> Path:
    """Cherche l'annotation .txt soit à côté de l'image, soit dans le dossier miroir 'labels'."""
    direct_label = img_path.with_suffix(".txt")
    if direct_label.exists():
        return direct_label

    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        alt_label = Path(*parts).with_suffix(".txt")
        if alt_label.exists():
            return alt_label

    return direct_label


def collect_parent_images(raw_dir: Path) -> List[Dict]:
    """Parcourt récursivement `raw_dir` (ex: 1_annotated_dataset) et recense les
    paires images/labels valides, avec le nom du lot d'origine de chacune.

    Chaque entrée retournée contient :
      - parent_id : identifiant unique dérivé de l'arborescence (utilisé comme
        préfixe de nom de fichier dans les étapes suivantes)
      - img_path / label_path : chemins absolus vers l'image et son label brut
      - batch : nom du sous-dossier de premier niveau sous raw_dir (ex: "SB 1"),
        utilisé pour retrouver le data.yaml LOCAL de ce lot (local_id -> nom),
        seule façon de traduire ses classes vers le référentiel - voir
        class_config.load_batch_local_names et split_dataset.py.
    """
    raw_dir = Path(raw_dir)
    parent_images: List[Dict] = []

    if not raw_dir.exists():
        print(f"[ERREUR] Le dossier source n'existe pas : {raw_dir}")
        return parent_images

    for img_path in raw_dir.rglob("*"):
        # Exclut les fichiers/dossiers cachés (commençant par '.')
        if img_path.name.startswith(".") or any(p.startswith(".") for p in img_path.parts):
            continue

        if img_path.is_file() and img_path.suffix.lower() in VALID_IMG_EXTS:
            label_path = find_corresponding_label(img_path)

            rel_path = img_path.relative_to(raw_dir)
            parent_id = str(rel_path.with_suffix("")).replace("\\", "_").replace("/", "_").replace(" ", "_")
            batch_name = rel_path.parts[0]

            parent_images.append({
                "parent_id": parent_id,
                "img_path": str(img_path),
                "label_path": str(label_path),
                "batch": batch_name,
            })

    return parent_images
