#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Revue web légère des faux positifs (panier A), sans passer par CVAT.

Contexte (voir journal du 28/08/2026, "F1 ajouté + décision de priorité") : une partie
des "faux positifs" mesurés dans les rapports d'entraînement sont probablement des
déchets réels que l'annotation a simplement oubliés - une détection correcte du modèle
sur un objet non annoté compte comme une erreur alors que c'est un trou dans la vérité
terrain. `label_review.py` détecte déjà ce cas (panier A : prédiction confiante SANS
annotation correspondante) mais son seul chemin de correction est un export vers CVAT,
jugé trop lent à l'usage pour ce genre de décision "oui/non" simple.

CE MODULE NE FAIT QUE PANIER A (annotation probablement oubliée) - PAS le panier C de
label_review.py (masque existant probablement mal ajusté, à retoucher). Une revue
oui/non ne peut pas corriger un contour de masque : ce cas continue de nécessiter CVAT
(ou un outil dédié à écrire plus tard). `label_review.py --scope val_test` reste la
bonne commande pour un diagnostic complet des DEUX paniers.

Fonctionnement :
    1. Réutilise le moteur de détection de `label_review.py` (inférence tuilée +
       appariement GT/prédictions par classe+IoU, voir `matching.py`) pour repérer
       tous les candidats panier A du périmètre choisi (--scope), SANS rien écrire -
       contrairement à `run_review()`, qui exporte inconditionnellement chaque
       candidat vers 5_review_dataset/.
    2. Page web locale (même moteur que `assisted_annotate.py` : serveur HTTP intégré
       à Python, un candidat à la fois, chip à résolution native avec le contour du
       masque en surimpression) : bandeau de sous-classe éditable, menu déroulant
       proposant TOUT le référentiel PROJET (~20 sous-classes, class_taxonomy de
       config/data_config.yaml - voir class_config.load_global_class_options), pas
       seulement les sous-classes que CE lot déclare localement (CORRECTIF 28/08/2026,
       voir journal - l'ancienne limite au référentiel local du lot empêchait de choisir
       une sous-classe pertinente que ce lot n'avait simplement jamais eu l'occasion de
       déclarer, ex: un lot cold-start bootstrap_annotate qui ne connaît que ses 7
       super-classes). Le modèle ne prédit qu'en espace super-classe (label_review.py a
       la même limite) : les sous-classes qui résolvent vers la super-classe PRÉDITE
       restent listées en premier (choix par défaut du menu), le reste du référentiel
       projet suit. Deux boutons Valider/Supprimer.
    3. À la fin de la revue (ou sur "Enregistrer et terminer" pour s'arrêter en cours
       de route) : copie INTÉGRALE de --raw-dir (1_annotated_dataset par défaut) vers
       --output-dir (1bis_corrected_annotation par défaut, À CÔTÉ de 1_annotated_dataset
       - jamais écrit dedans), puis ajoute les lignes validées aux fichiers de label
       correspondants dans cette copie. Comme le menu est désormais PROJET (pas limité
       au référentiel local du lot), une sous-classe choisie peut ne pas exister encore
       dans le data.yaml local du lot concerné : dans ce cas un nouvel ID local est
       alloué (max existant + 1) et le data.yaml COPIÉ de ce lot (jamais l'original) est
       mis à jour en conséquence - voir `_ReviewState._write_corrected_dataset`. Le
       résultat est un dataset COMPLET, structure identique à 1_annotated_dataset,
       immédiatement utilisable comme --raw-dir alternatif de data_pipeline.py (voir sa
       docstring, "Variante de donnée brute") :

           python -m src.data.data_pipeline --raw-dir <output-dir> --suffix _corrected

       Un review_manifest.csv est aussi écrit à la racine de --output-dir, traçant
       CHAQUE décision (validé/supprimé, classe choisie, confiance) - jamais silencieux,
       même principe que review_manifest.csv de label_review.py.

Coût à anticiper : l'étape 3 copie tout le dataset brut sur disque (toutes les images,
même celles sans aucun candidat) - le temps et l'espace disque dépendent donc de la
taille totale de 1_annotated_dataset, pas seulement du nombre de candidats revus.

Entrée : manifeste de split (parent_manifest.json, déjà produit par split_dataset.py),
images/labels bruts, et soit un modèle entraîné (best.pt) soit un predict_tile_fn injecté.
Sortie : --output-dir (dataset corrigé complet) + review_manifest.csv à sa racine.

Usage :
    python -m src.review.review_false_positives
        -> sélection interactive du modèle (le plus récent par défaut), scope=all.
    python -m src.review.review_false_positives --model output/runs/<run>/weights/best.pt --scope val_test
        -> modèle explicite, périmètre restreint à val/test (revue plus rapide).
Puis ouvrir l'URL affichée (http://127.0.0.1:8766/ par défaut - port différent
d'assisted_annotate.py pour pouvoir lancer les deux outils en parallèle si besoin).
"""

import argparse
import csv
import json
import os
import shutil
import sys
import threading
import webbrowser
from collections import OrderedDict, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union
from urllib.parse import urlparse

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    assert_model_matches_taxonomy,
    load_batch_local_names,
    load_class_config,
    load_global_class_options,
    normalize_class_name,
)
from src.data.image_io import load_image_bgr
from src.data.split_dataset import PARENT_MANIFEST_FILENAME, RAW_DIR, SPLIT_DIR
from src.review.assisted_annotate import _build_crop_jpeg, _normalize_geom_to_full_image
from src.review.label_review import (
    BASE_DIR,
    RUNS_DIR,
    _discover_available_models,
    _load_gt_objects,
    _prompt_model_choice,
    _select_parents,
)
from src.review.matching import match_gt_to_predictions
from src.review.tiled_inference import make_ultralytics_predict_fn, predict_parent_image

from shapely.geometry import Polygon

# À CÔTÉ de 1_annotated_dataset (jamais dedans) - voir docstring du module.
CORRECTED_DIR = os.path.join(BASE_DIR, "1bis_corrected_annotation")

# Taille de cache d'images sources gardées en mémoire pendant la revue web - les
# images de ce projet peuvent être de grandes orthomosaïques, on ne veut JAMAIS
# toutes les garder en mémoire à la fois. 2 suffit dans le cas courant (candidats
# groupés par image consécutive dans la liste, voir collect_bucket_a_candidates) :
# l'image courante + la précédente le temps qu'une requête tardive du navigateur
# (rechargement de page) arrive encore à retrouver son chip.
IMAGE_CACHE_SIZE = 2


class ReviewCandidate(NamedTuple):
    """Un candidat panier A (prédiction confiante sans annotation correspondante),
    prêt pour la revue web - tout ce qu'il faut pour afficher le chip et écrire la
    ligne de label si validé, SANS avoir besoin de relire le manifeste de split."""
    batch: str
    split: str
    raw_img_path: Path
    raw_label_path: Path
    img_w: int
    img_h: int
    geom: Polygon
    target_class_id: int
    target_class_name: str
    confidence: float
    # (nom_canonique, libellé affiché, ID_super_classe_cible), voir
    # _ordered_global_class_options. `nom_canonique` est ce qui circule avec le navigateur
    # (identifiant du <option>) et ce qui est stocké dans pending_additions - PAS un ID
    # local, qui n'a de sens que dans le data.yaml d'UN lot (voir _write_corrected_dataset
    # pour la résolution/allocation de l'ID local au moment de l'écriture).
    class_options: List[tuple]


def _ordered_global_class_options(
    global_options: List[tuple], predicted_target_id: int
) -> List[tuple]:
    """Réordonne la liste globale des sous-classes PROJET (déjà chargée une seule fois pour
    tout le run - voir `class_config.load_global_class_options` et son appel dans
    `collect_bucket_a_candidates`) pour un candidat donné.

    CORRECTIF (28/08/2026) : le menu propose désormais TOUT le référentiel du PROJET
    (~20 sous-classes), plus seulement les sous-classes que CE lot déclare localement dans
    son propre data.yaml (ancien comportement, voir `_matching_local_ids`/`_review_class_options`
    dans le journal de décisions) - une revue humaine doit pouvoir choisir la sous-classe
    correcte même si ce lot particulier ne l'a encore jamais rencontrée (elle sera alors
    ajoutée au data.yaml LOCAL COPIÉ de ce lot, voir `_write_corrected_dataset`), et pas
    seulement corriger la super-classe mal prédite à l'intérieur du sous-ensemble déjà connu
    de ce lot.

    Les sous-classes qui résolvent vers `predicted_target_id` (la suggestion du modèle) restent
    listées EN PREMIER, dans leur ordre d'apparition dans class_taxonomy - c'est toujours le
    choix par défaut du menu (voir _PAGE_HTML : aucune présélection explicite, le navigateur
    retient le premier <option> du DOM) : le bon réflexe pour le cas courant (super-classe
    correcte, sous-classe à préciser) reste aussi rapide qu'avant. Le reste du référentiel
    projet suit, pour les cas où la super-classe elle-même doit être corrigée."""
    priority = [o for o in global_options if o[2] == predicted_target_id]
    rest = [o for o in global_options if o[2] != predicted_target_id]
    return priority + rest


def collect_bucket_a_candidates(
    model_path: Optional[str] = None,
    scope: str = "all",
    raw_dir: str = RAW_DIR,
    split_dir: str = SPLIT_DIR,
    tile_size: int = 640,
    overlap: int = 256,
    tile_conf_threshold: float = 0.25,
    conf_threshold: float = 0.6,
    min_match_iou: float = 0.1,
    predict_tile_fn=None,
) -> List[ReviewCandidate]:
    """Même moteur de détection que `label_review.run_review()` (voir sa docstring pour le
    détail de l'appariement GT/prédictions), réduit au panier A et SANS AUCUNE ÉCRITURE sur
    disque - retourne juste la liste des candidats, à charge de l'appelant (la revue web,
    ci-dessous) de décider quoi en faire. `predict_tile_fn` : comme dans label_review.py, pour
    injecter un faux prédicteur en test."""
    if predict_tile_fn is None:
        if not model_path:
            available_models = _discover_available_models(RUNS_DIR)
            if not available_models:
                raise RuntimeError(
                    f"Aucun modèle entraîné trouvé sous {RUNS_DIR} (aucun */weights/best.pt). "
                    f"Lance d'abord un entraînement (python -m src.training.train), ou fournis "
                    f"--model explicitement si le modèle se trouve ailleurs."
                )
            model_path = _prompt_model_choice(available_models, RUNS_DIR)
        predict_tile_fn = make_ultralytics_predict_fn(model_path, conf_threshold=tile_conf_threshold)

    class_taxonomy, target_names = load_class_config(DEFAULT_CLASS_CONFIG_PATH)

    # Menu déroulant PROJET (pas par lot) - voir docstring du module et de
    # _ordered_global_class_options. Calculé UNE FOIS pour tout le run (indépendant du lot
    # ou du candidat, seul l'ORDRE varie ensuite par candidat) : (nom_canonique, libellé
    # affiché, ID_super_classe). Le libellé inclut la super-classe entre parenthèses -
    # nécessaire maintenant que le menu n'est plus limité à un seul lot dont le regroupement
    # par super-classe allait de soi.
    global_class_options = [
        (name, f"{name} ({target_names.get(target_id, str(target_id))})", target_id)
        for name, target_id in load_global_class_options(DEFAULT_CLASS_CONFIG_PATH)
    ]

    model_names = getattr(predict_tile_fn, "model_names", None)
    if model_names is not None:
        assert_model_matches_taxonomy(model_names, target_names, model_label=str(model_path or ""))

    manifest_path = Path(split_dir) / PARENT_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise RuntimeError(
            f"Manifeste introuvable : {manifest_path}. Lance d'abord split_dataset.py "
            f"(ou tout le pipeline via data_pipeline.py) pour le générer."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        parent_manifest = json.load(f)

    selected = _select_parents(parent_manifest, scope)
    if not selected:
        print(f"❌ Aucune image sélectionnée pour scope='{scope}'. Rien à faire.")
        return []

    print(f"--- 🔎 DÉTECTION DES CANDIDATS PANIER A ({len(selected)} image(s), scope='{scope}') ---")
    print(f"    Modèle : {model_path if model_path else '(predict_tile_fn injecté - test/appel programmatique)'}")

    local_names_by_batch: Dict[str, Dict[int, str]] = {}
    candidates: List[ReviewCandidate] = []
    images_with_candidates = 0
    skipped_no_placeholder = 0
    failed_images: List[Dict] = []

    for parent_id, info in selected.items():
        batch_name = info["batch"]
        split = info["split"]
        raw_img_path = Path(info["raw_img_path"])
        raw_label_path = Path(info["raw_label_path"])

        try:
            if batch_name not in local_names_by_batch:
                local_yaml = Path(raw_dir) / batch_name / "data.yaml"
                local_names_by_batch[batch_name] = load_batch_local_names(local_yaml)
            local_names = local_names_by_batch[batch_name]

            img = load_image_bgr(raw_img_path)
            if img is None:
                print(f"  ⚠️  Image illisible, ignorée : {raw_img_path}")
                continue
            img_h, img_w = img.shape[:2]
            del img  # dims seulement ici - le chip sera relu à la demande pendant la revue web (cache borné)

            gt_objects = _load_gt_objects(raw_label_path, local_names, class_taxonomy, img_w, img_h)
            predictions = predict_parent_image(raw_img_path, predict_tile_fn, tile_size=tile_size, overlap=overlap)
            results = match_gt_to_predictions(gt_objects, predictions, min_iou=min_match_iou)

            image_candidates: List[ReviewCandidate] = []
            for r in results:
                if r.gt is None and r.pred is not None and r.pred.confidence is not None \
                        and r.pred.confidence >= conf_threshold:
                    options = _ordered_global_class_options(global_class_options, r.pred.class_id)
                    if not options:
                        # Ne peut arriver que si le référentiel PROJET lui-même ne contient aucune
                        # sous-classe résolvant vers une vraie super-classe (config/data_config.yaml
                        # vide ou mal formé) - garde de sécurité, plus un cas "par lot" comme avant
                        # le correctif du 28/08 (le menu est désormais global, pas limité à ce que
                        # CE lot déclare localement).
                        skipped_no_placeholder += 1
                        print(
                            f"  ⚠️  Référentiel projet sans aucune sous-classe résolvant vers une "
                            f"super-classe réelle - candidat ignoré pour {raw_img_path.name}."
                        )
                        continue
                    image_candidates.append(ReviewCandidate(
                        batch=batch_name, split=split,
                        raw_img_path=raw_img_path, raw_label_path=raw_label_path,
                        img_w=img_w, img_h=img_h, geom=r.pred.geom,
                        target_class_id=r.pred.class_id,
                        target_class_name=target_names.get(r.pred.class_id, str(r.pred.class_id)),
                        confidence=float(r.pred.confidence),
                        class_options=options,
                    ))

            if image_candidates:
                images_with_candidates += 1
                candidates.extend(image_candidates)

        except Exception as e:  # noqa: BLE001 - une image en cause ne doit jamais arrêter les autres
            print(f"  ❌ Échec sur {raw_img_path} (lot '{batch_name}') - ignorée, détection continue : {e}")
            failed_images.append({"batch": batch_name, "image": str(raw_img_path), "error": str(e)})
            continue

    print("\n=== RÉSUMÉ DE LA DÉTECTION ===")
    print(f"  • Images avec au moins un candidat : {images_with_candidates} / {len(selected)}")
    print(f"  • Candidats panier A à revoir : {len(candidates)}")
    if skipped_no_placeholder:
        print(f"  • ⚠️  Candidats ignorés (référentiel projet sans sous-classe utilisable) : {skipped_no_placeholder}")
    if failed_images:
        print(f"  • ⚠️  Images en échec (ignorées) : {len(failed_images)}")
        for fi in failed_images:
            print(f"      - [{fi['batch']}] {fi['image']}\n        {fi['error']}")
    print("  (rappel : panier C - masques à retoucher - n'est pas couvert par cet outil, voir "
          "label_review.py pour un diagnostic complet des deux paniers)")

    return candidates


class _ReviewState:
    """État partagé du serveur de revue - un seul utilisateur, une seule session à la fois
    (même principe que assisted_annotate._ReviewState), mais les candidats couvrent
    potentiellement PLUSIEURS images/lots plutôt qu'une seule image."""

    def __init__(self, candidates: List[ReviewCandidate], raw_dir: Path, output_dir: Path):
        self.candidates = candidates
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.cursor = 0
        self.validated_count = 0
        self.deleted_count = 0
        self.finished = False
        self.decisions: List[Dict] = []
        # Ajouts en attente, regroupés par fichier de label SOURCE (pas encore la destination
        # copiée - résolu au moment de l'écriture, voir _write_corrected_dataset). PAS encore
        # de ligne de label prête (contrairement à avant le correctif du 28/08) : le menu étant
        # désormais PROJET, l'ID LOCAL (propre à chaque lot) qui ira dans le .txt n'est connu
        # qu'au moment de l'écriture, une fois le data.yaml de CE lot rechargé (réutilisation
        # d'un ID existant si ce lot connaît déjà cette sous-classe, sinon allocation d'un
        # nouveau) - chaque entrée garde donc (lot, nom_canonique, coordonnées) en attendant.
        self.pending_additions: Dict[str, List[Dict]] = defaultdict(list)
        self.lock = threading.Lock()
        self._img_cache: "OrderedDict[str, object]" = OrderedDict()

    def _load_image_cached(self, path: Path):
        key = str(path)
        if key in self._img_cache:
            self._img_cache.move_to_end(key)
            return self._img_cache[key]
        img = load_image_bgr(path)
        if img is None:
            raise RuntimeError(f"Image illisible : {path}")
        self._img_cache[key] = img
        while len(self._img_cache) > IMAGE_CACHE_SIZE:
            self._img_cache.popitem(last=False)
        return img

    def crop_bytes(self, idx: int) -> bytes:
        c = self.candidates[idx]
        img = self._load_image_cached(c.raw_img_path)
        return _build_crop_jpeg(img, c.geom)

    def state_json(self) -> dict:
        total = len(self.candidates)
        if self.finished:
            return {
                "finished": True, "total": total,
                "validated": self.validated_count, "deleted": self.deleted_count,
                "output": str(self.output_dir),
            }
        if self.cursor >= total:
            return {
                "finished": False, "done_reviewing": True, "total": total,
                "validated": self.validated_count, "deleted": self.deleted_count,
            }
        c = self.candidates[self.cursor]
        return {
            "finished": False, "done_reviewing": False,
            "index": self.cursor, "total": total, "reviewed": self.cursor,
            "validated": self.validated_count, "deleted": self.deleted_count,
            "batch": c.batch, "split": c.split, "image": c.raw_img_path.name,
            "predicted_class_name": c.target_class_name,
            "confidence": round(c.confidence, 3),
            "classes": [{"id": name, "name": label} for name, label, _target_id in c.class_options],
        }

    def decide(self, idx: int, action: str, class_name: str) -> None:
        with self.lock:
            if idx != self.cursor or self.finished:
                return  # décision périmée (double-clic, page rechargée) - ignorée, même garde qu'assisted_annotate
            c = self.candidates[idx]
            if action == "valider":
                valid_names = {name for name, _label, _target_id in c.class_options}
                if class_name not in valid_names:
                    # Choix hors du menu envoyé au navigateur (ne devrait pas arriver en usage
                    # normal - le <select> ne propose que c.class_options) - ignoré plutôt que
                    # d'injecter une sous-classe non vérifiée dans le dataset corrigé.
                    return
                norm = _normalize_geom_to_full_image(c.geom, c.img_w, c.img_h)
                coords_str = " ".join(f"{v:.6f}" for v in norm)
                self.pending_additions[str(c.raw_label_path)].append({
                    "batch": c.batch, "canonical_name": class_name, "coords_str": coords_str,
                })
                self.validated_count += 1
                self.decisions.append({
                    "bucket": "A_missing_annotation", "batch": c.batch, "split": c.split,
                    "image": c.raw_img_path.name, "target_class": c.target_class_name,
                    "action": "valide", "chosen_local_class": class_name,
                    "confidence": round(c.confidence, 3),
                })
            else:
                self.deleted_count += 1
                self.decisions.append({
                    "bucket": "A_missing_annotation", "batch": c.batch, "split": c.split,
                    "image": c.raw_img_path.name, "target_class": c.target_class_name,
                    "action": "supprime", "chosen_local_class": "",
                    "confidence": round(c.confidence, 3),
                })
            self.cursor += 1

    def finish(self) -> dict:
        with self.lock:
            if not self.finished:
                self._write_corrected_dataset()
                self.finished = True
            return self.state_json()

    def _write_corrected_dataset(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise RuntimeError(
                f"{self.output_dir} existe déjà et n'est pas vide - choisis un autre --output-dir "
                f"pour ne jamais écraser une passe de correction précédente sans le vouloir."
            )
        print(f"\n--- 💾 Copie intégrale de {self.raw_dir} vers {self.output_dir} (peut prendre un moment) ---")
        shutil.copytree(self.raw_dir, self.output_dir)

        # Résolution des IDs locaux, LOT PAR LOT (voir docstring du module) : le menu de revue
        # est PROJET, pas par lot - une sous-classe choisie peut donc être nouvelle pour le lot
        # concerné. On recharge le data.yaml COPIÉ de chaque lot une seule fois (cache), on
        # réutilise un ID local déjà déclaré si son nom normalisé correspond, sinon on en
        # alloue un nouveau (max existant + 1) et on le mémorise pour le réécrire à la fin -
        # UNE seule écriture par lot modifié, pas une par ligne ajoutée.
        local_names_by_batch: Dict[str, Dict[int, str]] = {}
        dirty_batches: set = set()

        def _resolve_local_id(batch: str, canonical_name: str) -> int:
            if batch not in local_names_by_batch:
                local_names_by_batch[batch] = load_batch_local_names(self.output_dir / batch / "data.yaml")
            names = local_names_by_batch[batch]
            norm_target = normalize_class_name(canonical_name)
            for lid, existing_name in names.items():
                if normalize_class_name(existing_name) == norm_target:
                    return lid
            new_id = (max(names) + 1) if names else 0
            names[new_id] = canonical_name
            dirty_batches.add(batch)
            print(f"    ➕ [{batch}] nouvelle sous-classe locale ajoutée à data.yaml : {new_id} = '{canonical_name}'")
            return new_id

        raw_dir_resolved = self.raw_dir.resolve()
        for raw_label_path_str, items in self.pending_additions.items():
            raw_label_path = Path(raw_label_path_str).resolve()
            try:
                rel = raw_label_path.relative_to(raw_dir_resolved)
            except ValueError:
                print(f"  ⚠️  {raw_label_path} n'est pas sous {raw_dir_resolved} - ajout ignoré (ne devrait pas arriver).")
                continue
            label_dst = self.output_dir / rel
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            new_lines = [
                f"{_resolve_local_id(item['batch'], item['canonical_name'])} {item['coords_str']}\n"
                for item in items
            ]
            # Mode 'a' : crée le fichier s'il n'existait pas côté source (image jusque-là sans
            # aucune annotation - shutil.copytree n'a alors rien copié à ce chemin) - pas besoin
            # de distinguer les deux cas, 'a' gère les deux uniformément.
            with open(label_dst, "a", encoding="utf-8") as f:
                f.writelines(new_lines)

        # Réécrit le data.yaml (copié) de chaque lot où une nouvelle sous-classe a été allouée -
        # ne touche QUE la clé "names" (et "nc" si présente), le reste du fichier (path, splits,
        # etc.) est préservé tel quel.
        for batch in dirty_batches:
            yaml_path = self.output_dir / batch / "data.yaml"
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            yaml_data["names"] = {lid: name for lid, name in sorted(local_names_by_batch[batch].items())}
            if "nc" in yaml_data:
                yaml_data["nc"] = len(local_names_by_batch[batch])
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)

        if self.decisions:
            manifest_path = self.output_dir / "review_manifest.csv"
            fieldnames = ["bucket", "batch", "split", "image", "target_class", "action", "chosen_local_class", "confidence"]
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.decisions)
            print(f"    Manifeste de revue : {manifest_path} ({len(self.decisions)} décision(s))")

        print(f"--- ✅ Dataset corrigé écrit : {self.output_dir} ---")


def _make_handler(state: _ReviewState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence le log par requête
            pass

        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
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
                except (ValueError, IndexError, RuntimeError):
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
                    class_name=str(payload.get("class_name", "")),
                )
                self._send_json(state.state_json())
            elif path == "/api/finish":
                try:
                    self._send_json(state.finish())
                except RuntimeError as e:
                    self._send_json({"error": str(e)}, status=409)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


_PAGE_HTML = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>PixelOdyssey — revue des faux positifs</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 640px; margin: 2.5rem auto; padding: 0 1.2rem; }
  h1 { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.2rem; }
  #progress { color: #888; font-size: 0.85rem; margin-bottom: 0.3rem; }
  #context { color: #888; font-size: 0.78rem; margin-bottom: 1rem; }
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
  <h1>Revue des faux positifs (panier A)</h1>
  <div id="progress"></div>
  <div id="context"></div>

  <div id="review">
    <div id="crop-wrap"><img id="crop" src="" alt="candidat à valider"></div>
    <div id="band">
      <label for="class-select">Sous-classe</label>
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
      <p style="color:#888;font-size:0.85rem;">Dataset corrigé écrit dans : ${s.output}</p>
      <p style="color:#888;font-size:0.85rem;">Tu peux fermer cette page.</p>`;
    return;
  }
  if (s.done_reviewing) {
    document.getElementById('progress').textContent =
      `${s.total} / ${s.total} passés en revue — clique « Enregistrer et terminer » pour écrire le dataset corrigé.`;
    document.getElementById('context').textContent = '';
    document.getElementById('review').querySelector('#crop-wrap').classList.add('hidden');
    document.getElementById('band').classList.add('hidden');
    document.getElementById('buttons').classList.add('hidden');
    current = null;
    return;
  }
  current = s;
  document.getElementById('progress').textContent =
    `${s.reviewed} / ${s.total} revus — ${s.validated} validé(s), ${s.deleted} supprimé(s)`;
  document.getElementById('context').textContent =
    `${s.batch} · ${s.split} · ${s.image}`;
  document.getElementById('crop').src = `/api/crop/${s.index}?t=${Date.now()}`;
  const sel = document.getElementById('class-select');
  sel.innerHTML = '';
  for (const c of s.classes) {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = c.name;
    sel.appendChild(opt);
  }
  document.getElementById('conf').textContent =
    `classe cible modèle : ${s.predicted_class_name} · confiance ${(s.confidence * 100).toFixed(0)}%`;
}

async function decide(action) {
  if (!current) return;
  const class_name = document.getElementById('class-select').value;
  await fetch('/api/decide', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx: current.index, action, class_name}),
  });
  refresh();
}

document.getElementById('btn-valider').addEventListener('click', () => decide('valider'));
document.getElementById('btn-supprimer').addEventListener('click', () => decide('supprimer'));
document.getElementById('btn-finish').addEventListener('click', async () => {
  const r = await fetch('/api/finish', {method: 'POST'});
  const s = await r.json();
  if (s.error) { alert(s.error); return; }
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


def run_review_false_positives(
    model_path: Optional[str] = None,
    scope: str = "all",
    raw_dir: str = RAW_DIR,
    split_dir: str = SPLIT_DIR,
    output_dir: str = CORRECTED_DIR,
    tile_size: int = 640,
    overlap: int = 256,
    tile_conf_threshold: float = 0.25,
    conf_threshold: float = 0.6,
    min_match_iou: float = 0.1,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = True,
    predict_tile_fn=None,
) -> Dict:
    """Bloque jusqu'à la fin de la revue (le serveur ne rend la main qu'après réception de la
    décision de terminer, bouton ou file épuisée puis "Enregistrer et terminer" cliqué) -
    même contrat que `assisted_annotate.run_assisted_annotate`. Retourne un résumé
    {candidates_found, validated, deleted, output_dir}."""
    candidates = collect_bucket_a_candidates(
        model_path=model_path, scope=scope, raw_dir=raw_dir, split_dir=split_dir,
        tile_size=tile_size, overlap=overlap, tile_conf_threshold=tile_conf_threshold,
        conf_threshold=conf_threshold, min_match_iou=min_match_iou, predict_tile_fn=predict_tile_fn,
    )
    if not candidates:
        print("\n  -> Rien à revoir sur ce périmètre - aucune interface ouverte, aucun dataset écrit.")
        return {"candidates_found": 0, "validated": 0, "deleted": 0, "output_dir": None}

    state = _ReviewState(candidates, raw_dir=Path(raw_dir), output_dir=Path(output_dir))
    server = ThreadingHTTPServer((host, port), _make_handler(state))

    url = f"http://{host}:{port}/"
    print(f"\n--- 🖥️  Interface de revue prête : {url} ---")
    print(f"    {len(candidates)} candidat(s) à revoir. (Entrée = Valider, Retour/Suppr = Supprimer, au clavier)")
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

    print(f"\n--- ✅ {state.validated_count} annotation(s) validée(s), {state.deleted_count} supprimée(s). ---")
    if state.finished:
        print(f"    Dataset corrigé : {output_dir}")
        print(f"    Pour l'utiliser : python -m src.data.data_pipeline --raw-dir \"{output_dir}\" --suffix _corrected")
    return {
        "candidates_found": len(candidates),
        "validated": state.validated_count,
        "deleted": state.deleted_count,
        "output_dir": str(output_dir) if state.finished else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Revue web légère des faux positifs (panier A, sans CVAT) - PixelOdyssey"
    )
    parser.add_argument(
        "--model", default=None,
        help="Chemin vers le modèle entraîné (best.pt). Optionnel : sélection interactive sinon.",
    )
    parser.add_argument("--scope", choices=["val_test", "all"], default="all",
                        help="all (défaut) : tout le dataset, train + val + test. val_test : "
                             "seulement val/test, pour une revue plus rapide/ciblée.")
    parser.add_argument("--raw-dir", default=RAW_DIR, help=f"Dataset brut source (défaut : {RAW_DIR}).")
    parser.add_argument("--split-dir", default=SPLIT_DIR, help=f"Dossier de split (défaut : {SPLIT_DIR}).")
    parser.add_argument("--output-dir", default=CORRECTED_DIR,
                        help=f"Dossier de sortie du dataset corrigé (défaut : {CORRECTED_DIR}) - doit ne "
                             f"pas déjà exister ou être vide.")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--tile-conf-threshold", type=float, default=0.25,
                        help="Seuil de confiance large appliqué par tuile avant fusion (défaut 0.25).")
    parser.add_argument("--conf-threshold", type=float, default=0.6,
                        help="Seuil de confiance métier pour proposer un candidat à la revue (défaut 0.6).")
    parser.add_argument("--min-match-iou", type=float, default=0.1,
                        help="IoU minimale pour considérer GT et prédiction comme le même objet (défaut 0.1).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true", help="N'ouvre pas le navigateur automatiquement.")
    args = parser.parse_args()
    try:
        run_review_false_positives(
            model_path=args.model,
            scope=args.scope,
            raw_dir=args.raw_dir,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_conf_threshold=args.tile_conf_threshold,
            conf_threshold=args.conf_threshold,
            min_match_iou=args.min_match_iou,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
