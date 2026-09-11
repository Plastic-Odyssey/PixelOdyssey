#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Vérificateur universel d'intégrité d'un dataset train/val/test.

Réutilisable à PLUSIEURS étapes du pipeline (pas seulement la sortie finale) :
- après l'étape 1 (split_dataset.py) : `--base-path ".../2_split_dataset"`
- après l'étape 2 (augment_dataset.py) : `--base-path ".../3_augmented_dataset"`
- après l'étape 3 (slice_dataset.py) : sans argument, lit `path` dans
  config/data_config.yaml (qui pointe vers 4_sliced_dataset).
Les trois dossiers ont la même structure images/{split} + labels/{split}, donc
le même checker s'applique à tous sans modification. Appelé automatiquement
après chaque étape par l'orchestrateur (data_pipeline.py).

Entrée : chemin d'un dossier `images/{split}` + `labels/{split}` (via
--base-path ou config/data_config.yaml).
Sortie : dict {"ok", "missing_splits", "orphan_issues"} ; code de sortie 0/1
en usage CLI.

Exemple :
    python -m src.data.dataset_sanity_check --base-path data/2_split_dataset
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Source partagée (voir raw_dataset.py) - à ne pas redéfinir localement : ce
# checker tourne aussi sur `2_split_dataset`, qui contient encore les images
# parentes brutes (dont des .tif), donc la liste d'extensions doit couvrir
# .tif/.tiff en plus de .jpg/.jpeg/.png pour ne pas générer de faux orphelins.
from src.data.raw_dataset import VALID_IMG_EXTS
# Idem pour le chemin par défaut du référentiel de classes : dérivé de
# class_config.py (seule source de vérité), jamais réécrit en dur ici.
from src.data.class_config import DEFAULT_CLASS_CONFIG_PATH


class PlasticDatasetChecker:
    def __init__(self, config_path: str = str(DEFAULT_CLASS_CONFIG_PATH), base_path: Optional[str] = None):
        """`base_path` prend le pas sur `config_path` s'il est fourni - pratique pour
        pointer ponctuellement vers une étape intermédiaire (ex: 2_split_dataset)
        sans avoir à maintenir un fichier de config séparé pour chaque étape."""
        if base_path is not None:
            self.base_path = Path(base_path)
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.base_path = Path(config["path"])

        self.splits = ["train", "val", "test"]

    def verify_all_splits(self) -> dict:
        """Retourne {"ok", "missing_splits", "orphan_issues"} plutôt qu'un simple bool :

        un split manquant/vide (`missing_splits`) et un vrai désalignement
        images/labels (`orphan_issues`) n'ont pas la même gravité. Un split vide
        est souvent normal (petit échantillon, split pas encore peuplé à ce
        stade du pipeline) ; un orphelin trahit un vrai bug (une étape a copié
        une image sans son label, ou l'inverse). Les confondre sous un seul
        booléen ferait crier au bug un simple split de test avec 0 image.
        `ok` reste True tant qu'aucun orphelin n'est détecté, même si des
        splits sont manquants.
        """
        print(f"--- 🔍 VÉRIFICATION DU DATASET SUR SSD : {self.base_path} ---")
        result = {"ok": True, "missing_splits": [], "orphan_issues": []}
        if not self.base_path.exists():
            print(f"❌ Erreur : Le dossier cible n'existe pas sur le disque : {self.base_path}")
            result["ok"] = False
            result["missing_splits"] = list(self.splits)
            return result

        for split in self.splits:
            img_dir = self.base_path / "images" / split
            lab_dir = self.base_path / "labels" / split

            if not img_dir.exists() or not lab_dir.exists():
                print(f"⚠️ Split [{split}] : Dossier images ou labels non encore créé.")
                result["missing_splits"].append(split)
                continue

            images = sorted(f.stem for f in img_dir.glob("*") if f.suffix.lower() in VALID_IMG_EXTS)
            labels = sorted(f.stem for f in lab_dir.glob("*.txt"))

            orphan_images = set(images) - set(labels)
            orphan_labels = set(labels) - set(images)

            if orphan_images or orphan_labels:
                result["ok"] = False
                if orphan_images:
                    msg = f"Split [{split}] : {len(orphan_images)} image(s) orpheline(s) sans label .txt !"
                    print(f"⚠️ {msg}")
                    result["orphan_issues"].append(msg)
                if orphan_labels:
                    msg = f"Split [{split}] : {len(orphan_labels)} label(s) .txt sans image correspondante !"
                    print(f"⚠️ {msg}")
                    result["orphan_issues"].append(msg)
            else:
                print(f"✅ Split [{split}] : {len(images)} tuiles alignées avec {len(labels)} labels.")

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérificateur d'intégrité images/labels PixelOdyssey")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CLASS_CONFIG_PATH),
        help="Fichier de config à lire pour 'path' (ignoré si --base-path est fourni).",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Pointe directement vers un dossier images/{split}+labels/{split} "
             "(ex: '.../2_split_dataset' ou '.../4_sliced_dataset'), sans passer par un fichier de config.",
    )
    args = parser.parse_args()

    checker = PlasticDatasetChecker(config_path=args.config, base_path=args.base_path)
    result = checker.verify_all_splits()
    sys.exit(0 if result["ok"] else 1)
