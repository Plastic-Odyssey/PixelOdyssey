#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Relecture assistée par modèle (human-in-the-loop).

Utilise le modèle entraîné (best.pt) comme assistant d'annotation pour
repérer les oublis dans 1_annotated_dataset, plutôt que de ré-inspecter les
images à la main. Une annotation manquante fausse les métriques de val/test :
une détection correcte du modèle sur un déchet non annoté compte comme un
faux positif, alors que c'est un trou dans la vérité terrain, pas une erreur
du modèle.

Ce script ne modifie jamais 1_annotated_dataset directement (règle non
négociable du projet - voir data_pipeline.py). Il produit un export non
destructif, prêt à importer dans une tâche de relecture CVAT, dans un
sous-dossier horodaté sous 5_review_dataset/. Une fois validé/corrigé dans
CVAT, c'est à l'utilisateur de réexporter vers le lot correspondant dans
1_annotated_dataset - ce script ne fait que proposer.

Trois paniers de triage (voir src/review/matching.py pour l'appariement
GT <-> prédictions par classe + IoU) :
  A. Prédiction confiante SANS annotation correspondante -> objet
     probablement oublié à l'annotation. Injectée comme nouvelle ligne dans
     le label exporté (voir la limite de granularité ci-dessous).
  C. GT et prédiction du modèle APPARIÉES (même classe, se recoupent) mais
     avec une IoU faible -> masque existant probablement mal ajusté. Signalé
     dans le manifeste de relecture pour retouche manuelle dans CVAT (pas de
     masque de remplacement automatique proposé).
  (GT sans prédiction correspondante : purement informationnel, jamais
  injecté - ça peut vouloir dire "le modèle a un angle mort", pas
  nécessairement "l'annotation est fausse". Compté dans le résumé, pas
  exporté comme changement.)

LIMITE DE GRANULARITÉ (importante, lire avant d'utiliser) : le modèle prédit
en espace SUPER-CLASSE (les classes de `names` dans config/data_config.yaml),
mais l'annotation brute d'un lot se fait en espace FIN (sous-classes selon
le lot - "bouteille PET", "cagette", etc.). Le modèle ne peut donc jamais
dire quelle sous-classe précise il a vue - seulement sa famille. Toute
injection du panier A porte donc une sous-classe "placeholder" (la première
sous-classe DE CE LOT qui pointe vers la super-classe prédite - voir
class_config.pick_placeholder_local_id), à corriger manuellement pendant la
relecture CVAT. C'est une conséquence inévitable d'entraîner sur des
super-classes en annotant sur des sous-classes plus fines : la relecture
humaine est ce qui referme cet écart.

Périmètre par défaut : tout le dataset (train + val + test), voir --scope
ci-dessous. Passe --scope val_test pour te restreindre à val/test (utile
pour un diagnostic rapide de bruit d'annotation sans lancer une relecture
complète).

Entrée : manifeste de split (parent_manifest.json), images/labels bruts du
lot, et soit un modèle entraîné (best.pt) soit un predict_tile_fn injecté.
Sortie : export non destructif sous 5_review_dataset/<run_id>/ (images,
labels, data.yaml par lot) + review_manifest.csv listant les paniers A et C.

Usage :
    python -m src.review.label_review
        -> aucun --model fourni : liste interactivement les modèles entraînés
           trouvés sous output/runs/*/weights/best.pt (le plus récent en
           premier) et demande lequel utiliser - plus besoin d'aller chercher
           le chemin à la main. Entrée seule = le plus récent.
    python -m src.review.label_review --model output/runs/<run>/weights/best.pt
        -> saute la sélection interactive, utilise directement ce modèle
           (utile pour scripter/relancer sans interaction).
    python -m src.review.label_review --scope val_test
        -> revient à l'ancien périmètre restreint (diagnostic rapide de bruit
           d'annotation sur val/test seuls, sans relire tout le dataset).
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    EXCLUDE,
    assert_model_matches_taxonomy,
    load_batch_local_names,
    load_class_config,
    pick_placeholder_local_id,
    resolve_class_name,
)
from src.data.image_io import load_image_bgr
from src.data.split_dataset import PARENT_MANIFEST_FILENAME, RAW_DIR, SPLIT_DIR
from src.review.matching import LabeledPolygon, match_gt_to_predictions
from src.review.tiled_inference import make_ultralytics_predict_fn, predict_parent_image

from shapely.geometry import Polygon

from src.paths_config import PROCESSED_DATASET_DIR as BASE_DIR  # racine centralisee (08/09/2026), voir src/paths_config.py
REVIEW_DIR = os.path.join(BASE_DIR, "5_review_dataset")

# src/review/label_review.py -> parents[2] = racine du repo. Même calcul que
# PROJECT_ROOT dans src/training/train.py - c'est là que train.py écrit
# chaque run (output/runs/<tag>_<modèle>_<horodatage>/weights/best.pt), donc
# c'est là qu'on va chercher les modèles disponibles pour la sélection
# interactive ci-dessous (voir _discover_available_models).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "output" / "runs"


def _load_gt_objects(label_path: Path, local_names: Dict[int, str], class_taxonomy, img_w: int, img_h: int):
    """Charge les objets GT d'une image (liste vide si le label n'existe pas -
    image confirmée "sans déchet", ou simplement pas encore annotée : les
    deux cas sont traités IDENTIQUEMENT ici, voir la docstring du module -
    dans les deux cas, toute prédiction confiante sur cette image devient un
    candidat panier A puisqu'il n'y a rien à quoi la comparer)."""
    gt_objects: List[LabeledPolygon] = []
    if not label_path.exists():
        return gt_objects

    with open(label_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            local_id = int(parts[0])
            name = local_names.get(local_id)
            if name is None:
                print(f"  ⚠️  {label_path} (L{lineno}) : ID {local_id} absent du data.yaml du lot - ignoré.")
                continue
            resolved = resolve_class_name(name, class_taxonomy)
            if resolved is None or resolved == EXCLUDE:
                continue

            coords = [float(x) for x in parts[1:]]
            pixels = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, len(coords), 2)]
            if len(pixels) < 3:
                continue
            geom = Polygon(pixels)
            if geom.is_valid and geom.area > 0:
                gt_objects.append(LabeledPolygon(class_id=resolved, geom=geom, confidence=None))

    return gt_objects


def _select_parents(parent_manifest: Dict[str, Dict], scope: str) -> Dict[str, Dict]:
    if scope == "all":
        return parent_manifest
    return {pid: info for pid, info in parent_manifest.items() if info["split"] in ("val", "test")}


def _write_review_data_yaml(dst_path: Path, local_names: Dict[int, str], splits_present: set) -> None:
    """Écrit le data.yaml du lot exporté sous 5_review_dataset - jamais une
    copie du data.yaml original du lot dans 1_annotated_dataset.

    Le data.yaml original d'un lot décrit la mise en page de l'export CVAT
    d'origine (souvent `train: train.txt`, une liste de chemins - pas
    nécessairement de clé val/test). Le copier tel quel serait faux : cet
    export utilise toujours `images/<split>/` + `labels/<split>/` (format
    Ultralytics YOLO Segmentation attendu par CVAT), et peut couvrir train
    et/ou val et/ou test pour un même lot selon --scope.

    Ce data.yaml généré ici décrit fidèlement la structure réellement
    exportée : mêmes `names` (mêmes ID locaux que les .txt de labels,
    INCHANGÉS - à ne pas confondre avec le référentiel de super-classes
    cibles), et une clé par split réellement présent dans cet export.

    Entrée : chemin de destination, table des noms locaux du lot, ensemble
    des splits présents dans l'export.
    Sortie : fichier data.yaml écrit sur disque.
    """
    data = {
        "path": ".",
        "names": {int(k): v for k, v in local_names.items()},
    }
    for split in ("train", "val", "test"):
        if split in splits_present:
            data[split] = f"images/{split}"

    with open(dst_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _discover_available_models(runs_dir: Union[str, Path]) -> List[Dict]:
    """Scanne <runs_dir>/*/weights/best.pt (un dossier par run lancé via
    src/training/train.py - voir _build_run_name là-bas) et retourne une
    fiche par modèle trouvé, du plus RÉCENT au plus ancien (mtime de best.pt -
    c'est le modèle qu'on veut proposer en premier, celui sur lequel on
    travaille le plus probablement).

    Les métadonnées descriptives (architecture, epochs réellement effectués,
    mAP50-95 masque sur val) sont lues en best-effort depuis args.yaml et
    results.csv DU RUN (écrits par Ultralytics, pas par ce projet) - si l'un
    ou l'autre manque ou est illisible, le modèle est quand même listé, juste
    avec moins d'info : on ne veut jamais qu'un run entraîné disparaisse de la
    liste à cause d'un fichier annexe absent.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []

    found: List[Dict] = []
    for weights_path in sorted(runs_dir.glob("*/weights/best.pt")):
        run_dir = weights_path.parent.parent
        entry: Dict = {
            "run_name": run_dir.name,
            "path": str(weights_path),
            "mtime": weights_path.stat().st_mtime,
            "architecture": None,
            "epochs": None,
            "map50_95_mask_val": None,
        }

        args_path = run_dir / "args.yaml"
        if args_path.exists():
            try:
                import yaml

                with open(args_path, "r", encoding="utf-8") as f:
                    args_data = yaml.safe_load(f) or {}
                entry["architecture"] = args_data.get("model")
            except Exception:  # noqa: BLE001 - annexe optionnelle, ne doit jamais faire disparaître le modèle
                pass

        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            try:
                with open(results_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    last = rows[-1]
                    if last.get("epoch"):
                        entry["epochs"] = int(float(last["epoch"]))
                    map_key = "metrics/mAP50-95(M)"  # mAP50-95 MASQUE, val (voir training_report.py)
                    if last.get(map_key):
                        entry["map50_95_mask_val"] = float(last[map_key])
            except Exception:  # noqa: BLE001 - idem, purement informatif
                pass

        found.append(entry)

    found.sort(key=lambda e: e["mtime"], reverse=True)
    return found


def _format_model_choice_line(idx: int, entry: Dict) -> str:
    marker = " (le plus récent)" if idx == 0 else ""
    details = []
    if entry["architecture"]:
        details.append(str(entry["architecture"]))
    if entry["epochs"] is not None:
        details.append(f"{entry['epochs']} epochs")
    if entry["map50_95_mask_val"] is not None:
        details.append(f"mAP50-95 masque (val) = {entry['map50_95_mask_val']:.3f}")
    detail_str = f"\n        {' • '.join(details)}" if details else ""
    return f"  [{idx}] {entry['run_name']}{marker}{detail_str}"


def _prompt_model_choice(models: List[Dict], runs_dir: Path) -> str:
    """Liste les modèles trouvés et demande lequel utiliser. Entrée seule (ou
    '0') -> le plus récent, pour que le cas le plus fréquent ("je veux
    relire avec mon dernier entraînement") ne demande qu'une touche."""
    print(f"\n--- 🗂️  MODÈLES ENTRAÎNÉS DISPONIBLES ({len(models)} trouvé(s) sous {runs_dir}) ---")
    for idx, entry in enumerate(models):
        print(_format_model_choice_line(idx, entry))

    raw = input(
        f"\nNuméro du modèle à utiliser pour la relecture [0-{len(models) - 1}, "
        f"Entrée = 0 = le plus récent] : "
    ).strip()
    choice = 0 if raw == "" else None
    if choice is None:
        try:
            choice = int(raw)
        except ValueError:
            raise RuntimeError(f"Choix invalide : '{raw}' n'est pas un numéro de la liste ci-dessus.")
    if not (0 <= choice < len(models)):
        raise RuntimeError(f"Choix hors limites : {choice}. Entre un numéro entre 0 et {len(models) - 1}.")

    chosen = models[choice]
    print(f"    -> Modèle sélectionné : {chosen['path']}\n")
    return chosen["path"]


def run_review(
    model_path: Optional[str] = None,
    scope: str = "all",
    raw_dir: str = RAW_DIR,
    split_dir: str = SPLIT_DIR,
    output_dir: str = REVIEW_DIR,
    tile_size: int = 640,
    overlap: int = 256,
    tile_conf_threshold: float = 0.25,
    conf_threshold: float = 0.6,
    iou_mismatch_threshold: float = 0.5,
    min_match_iou: float = 0.1,
    run_id: Optional[str] = None,
    predict_tile_fn=None,
) -> Dict:
    """
    `predict_tile_fn` : permet d'injecter une fonction de prédiction par tuile
    déjà construite (ex: pour les tests, avec un faux modèle - voir
    tiled_inference.py) au lieu de charger un modèle Ultralytics réel depuis
    `model_path`. En usage normal, laisser `predict_tile_fn=None`.

    `model_path` : si fourni, ce modèle est utilisé directement (aucune
    interaction). Si `None` ET `predict_tile_fn` est aussi `None`, une liste
    des modèles entraînés trouvés sous `RUNS_DIR` (output/runs/*/weights/
    best.pt) est proposée de façon interactive - voir _discover_available_models
    / _prompt_model_choice - pour ne plus avoir à chercher le chemin à la main.
    """
    # Résolution du modèle EN PREMIER, avant tout travail sur le manifeste -
    # échouer vite (ou proposer la sélection interactive) sans avoir déjà
    # relu/imprimé quoi que ce soit sur les images sélectionnées.
    if predict_tile_fn is None:
        if not model_path:
            # Pas de --model fourni : on propose la liste des modèles entraînés
            # trouvés sous output/runs/ plutôt que d'exiger que l'utilisateur
            # aille chercher le chemin de best.pt à la main à chaque fois.
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

    # Garde-fou : un modèle entraîné sous une taxonomie différente de la config
    # actuelle prédirait des ID de classe qui ne veulent plus dire la même chose -
    # voir assert_model_matches_taxonomy. Absent pour un predict_tile_fn injecté en
    # test (pas d'attribut model_names, pas de vrai modèle chargé).
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
        return {"images_processed": 0, "bucket_a": 0, "bucket_c": 0, "gt_unmatched": 0}

    if run_id is None:
        from datetime import datetime

        run_id = datetime.now().strftime("review_%Y%m%d_%H%M%S")
    out_root = Path(output_dir) / run_id
    print(f"--- 🔎 RELECTURE ASSISTÉE PAR MODÈLE ({len(selected)} image(s), scope='{scope}') ---")
    print(f"    Modèle : {model_path if model_path else '(predict_tile_fn injecté - test/appel programmatique)'}")
    print(f"    Sortie : {out_root}")

    local_names_by_batch: Dict[str, Dict[int, str]] = {}
    manifest_rows: List[Dict] = []
    # Splits RÉELLEMENT exportés pour chaque lot (peut être un sous-ensemble de
    # {train, val, test} - ex: un lot qui n'a des images flaguées qu'en train,
    # ou qui en a dans les 3) - nécessaire pour écrire un data.yaml d'export
    # fidèle à ce qui existe vraiment sur disque, voir _write_review_data_yaml.
    batches_splits_written: Dict[str, set] = {}
    counts = {"bucket_a": 0, "bucket_c": 0, "gt_unmatched": 0, "images_processed": 0}
    # Une image qui plante (fichier corrompu, format inattendu que même
    # image_io.load_image_bgr ne rattrape pas, etc.) ne doit jamais faire
    # perdre le travail déjà fait sur les autres images - voir le try/except
    # ci-dessous. Chaque échec est listé ici avec son chemin exact pour
    # pouvoir aller inspecter le fichier en cause.
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

            gt_objects = _load_gt_objects(raw_label_path, local_names, class_taxonomy, img_w, img_h)
            predictions = predict_parent_image(
                raw_img_path, predict_tile_fn, tile_size=tile_size, overlap=overlap
            )
            results = match_gt_to_predictions(gt_objects, predictions, min_iou=min_match_iou)

            new_lines: List[str] = []
            image_has_flag = False
            for r in results:
                if r.gt is None and r.pred is not None:
                    if r.pred.confidence is not None and r.pred.confidence >= conf_threshold:
                        placeholder_id = pick_placeholder_local_id(local_names, class_taxonomy, r.pred.class_id)
                        if placeholder_id is None:
                            print(
                                f"  ⚠️  [{batch_name}] aucune sous-classe locale ne pointe vers la "
                                f"super-classe {r.pred.class_id} ({target_names.get(r.pred.class_id)}) - "
                                f"détection ignorée pour {raw_img_path.name}."
                            )
                            continue
                        coords = list(r.pred.geom.exterior.coords)
                        norm = []
                        for x, y in coords:
                            norm.extend([max(0.0, min(1.0, x / img_w)), max(0.0, min(1.0, y / img_h))])
                        coords_str = " ".join(f"{c:.6f}" for c in norm)
                        new_lines.append(f"{placeholder_id} {coords_str}\n")
                        counts["bucket_a"] += 1
                        image_has_flag = True
                        manifest_rows.append(
                            {
                                "bucket": "A_missing_annotation",
                                "batch": batch_name,
                                "split": split,
                                "image": raw_img_path.name,
                                "target_class": target_names.get(r.pred.class_id, r.pred.class_id),
                                "placeholder_local_class": local_names.get(placeholder_id, placeholder_id),
                                "confidence": round(r.pred.confidence, 3),
                                "iou": "",
                                "bbox_xyxy": ",".join(f"{v:.1f}" for v in r.pred.geom.bounds),
                            }
                        )
                elif r.gt is not None and r.pred is not None and r.iou < iou_mismatch_threshold:
                    counts["bucket_c"] += 1
                    image_has_flag = True
                    manifest_rows.append(
                        {
                            "bucket": "C_mask_to_refine",
                            "batch": batch_name,
                            "split": split,
                            "image": raw_img_path.name,
                            "target_class": target_names.get(r.gt.class_id, r.gt.class_id),
                            "placeholder_local_class": "",
                            "confidence": round(r.pred.confidence, 3) if r.pred.confidence is not None else "",
                            "iou": round(r.iou, 3),
                            "bbox_xyxy": ",".join(f"{v:.1f}" for v in r.gt.geom.bounds),
                        }
                    )
                elif r.gt is not None and r.pred is None:
                    counts["gt_unmatched"] += 1

            if not image_has_flag:
                # Rien à signaler sur cette image (tout est déjà bien annoté, ou seulement
                # des GT non retrouvés par le modèle - informationnel, pas exporté) : pas
                # besoin de l'inclure dans l'export de relecture.
                counts["images_processed"] += 1
                continue

            # Export non destructif : lot recréé sous out_root, image + label copiés.
            # Le label brut ORIGINAL est conservé intégralement (y compris les objets
            # flagués panier C, dont le masque n'est PAS modifié automatiquement en v1 -
            # seulement signalé dans review_manifest.csv, à corriger toi-même dans CVAT)
            # + les nouvelles lignes du panier A sont ajoutées à la fin.
            img_dst_dir = out_root / batch_name / "images" / split
            lab_dst_dir = out_root / batch_name / "labels" / split
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            lab_dst_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(raw_img_path, img_dst_dir / raw_img_path.name)
            existing_lines = (
                raw_label_path.read_text(encoding="utf-8").splitlines(keepends=True)
                if raw_label_path.exists() else []
            )
            # Le nom du label exporté suit toujours celui de l'image (comme le fait
            # find_corresponding_label côté source) - pas de dépendance à un éventuel
            # suffixe déjà présent sur raw_label_path.
            with open(lab_dst_dir / f"{raw_img_path.stem}.txt", "w", encoding="utf-8") as f:
                f.writelines(existing_lines)
                f.writelines(new_lines)

            # data.yaml régénéré (PAS copié depuis 1_annotated_dataset - voir
            # _write_review_data_yaml) à chaque image traitée pour ce lot :
            # coût négligeable, et garantit qu'il reflète TOUJOURS les splits
            # réellement présents à date dans cet export, y compris si ce lot
            # a des images flaguées dans plusieurs splits à la fois.
            batches_splits_written.setdefault(batch_name, set()).add(split)
            _write_review_data_yaml(
                out_root / batch_name / "data.yaml", local_names, batches_splits_written[batch_name]
            )

            counts["images_processed"] += 1

        except Exception as e:  # noqa: BLE001 - une image en cause ne doit jamais arrêter les autres
            print(f"  ❌ Échec sur {raw_img_path} (lot '{batch_name}') - ignorée, relecture continue : {e}")
            failed_images.append({"batch": batch_name, "image": str(raw_img_path), "error": str(e)})
            continue

    if manifest_rows:
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_csv = out_root / "review_manifest.csv"
        fieldnames = [
            "bucket", "batch", "split", "image", "target_class",
            "placeholder_local_class", "confidence", "iou", "bbox_xyxy",
        ]
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"    Manifeste de relecture : {manifest_csv} ({len(manifest_rows)} lignes)")

    print("\n=== RÉSUMÉ ===")
    print(f"  • Images traitées : {counts['images_processed']}")
    print(f"  • Panier A (annotation probablement oubliée) : {counts['bucket_a']}")
    print(f"  • Panier C (masque probablement à retoucher)  : {counts['bucket_c']}")
    print(f"  • GT sans prédiction correspondante (info seulement) : {counts['gt_unmatched']}")
    if failed_images:
        print(f"  • ⚠️  Images en échec (ignorées, PAS incluses dans l'export) : {len(failed_images)}")
        for fi in failed_images:
            print(f"      - [{fi['batch']}] {fi['image']}\n        {fi['error']}")
    if counts["bucket_a"] or counts["bucket_c"]:
        print(f"\n  -> Importe {out_root} dans une tâche de relecture CVAT pour valider/corriger.")
    else:
        print("\n  -> Rien à signaler sur ce périmètre.")

    counts["failed_images"] = failed_images
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relecture assistée par modèle PixelOdyssey")
    parser.add_argument(
        "--model", default=None,
        help="Chemin vers le modèle entraîné (best.pt). Optionnel : si omis, la liste des "
             "modèles trouvés sous output/runs/*/weights/best.pt est proposée interactivement "
             "(le plus récent en premier, Entrée seule = le sélectionner directement).",
    )
    parser.add_argument("--scope", choices=["val_test", "all"], default="all",
                        help="all (défaut) : tout le dataset, train + val + test - pour faire "
                             "grandir le dataset annoté dans son ensemble. val_test : seulement "
                             "val/test, pour un diagnostic rapide de bruit d'annotation sans "
                             "relire tout le dataset.")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--tile-conf-threshold", type=float, default=0.25,
                        help="Seuil de confiance large appliqué par tuile avant fusion (défaut 0.25).")
    parser.add_argument("--conf-threshold", type=float, default=0.6,
                        help="Seuil de confiance métier pour proposer une annotation manquante (défaut 0.6).")
    parser.add_argument("--iou-mismatch-threshold", type=float, default=0.5,
                        help="IoU en dessous de laquelle une paire GT/prédiction appariée est "
                             "signalée comme masque à retoucher (défaut 0.5).")
    parser.add_argument("--min-match-iou", type=float, default=0.1,
                        help="IoU minimale pour considérer GT et prédiction comme le même objet (défaut 0.1).")
    args = parser.parse_args()
    try:
        run_review(
            model_path=args.model,
            scope=args.scope,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_conf_threshold=args.tile_conf_threshold,
            conf_threshold=args.conf_threshold,
            iou_mismatch_threshold=args.iou_mismatch_threshold,
            min_match_iou=args.min_match_iou,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
