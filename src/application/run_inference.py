#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Point d'entrée UNIQUE du pipeline "application" : dispatch
automatique entre inférence orthomosaïque (un seul GeoTIFF géoréférencé) et
inférence batch (dossier de photos drone brutes), selon `--input`.

Permet de choisir, à partir du fichier/dossier sélectionné, entre les deux
modes du pipeline plutôt que de les lancer séparément à la main :
  - `src/application/geo_density_map.py` (orthomosaïque unique WebODM déjà
    stitchée - voir sa docstring).
  - `src/application/run_application.py` + `src/application/web_map.py`
    (batch de photos brutes, dédoublonnage inter-photos inclus - voir la
    docstring de run_application.py pour l'état du dédoublonnage `iou`).

Tout le pipeline application (les deux modes, plus tous ses modules de
support) vit sous `src/application/`. Les deux modes produisent leur carte
web via un gabarit HTML/JS UNIQUE et PARTAGÉ (`map_builder.py`) : la SEULE
différence visuelle entre les deux sorties est le fond de carte (satellite
seul en mode batch, satellite + l'orthomosaïque entière par-dessus en mode
orthomosaïque). Ce module (`run_inference.py`) ne réimplémente rien des deux
pipelines : il se contente de détecter le type d'entrée, résoudre le modèle
via le registre commun (`model_registry.py` - même registre pour les deux
modes, voir plus bas) et appeler la fonction Python correspondante.

DÉTECTION DU MODE (voir `detect_input_mode`) :
  - `--input` est un DOSSIER -> mode batch (photos brutes).
  - `--input` est un FICHIER -> tentative d'ouverture avec `rasterio` ; s'il
    a un CRS (géoréférencement) -> mode orthomosaïque. Un fichier image SANS
    géoréférencement (une simple photo isolée, ou un GeoTIFF corrompu/sans
    CRS) n'est PAS une entrée valide pour ce pipeline - erreur explicite
    plutôt qu'une tentative de traitement dégradée : ce cas n'a jamais été
    supporté par aucun des deux pipelines existants (une photo isolée sans
    métadonnées de vol n'est pas exploitable par le mode batch non plus, qui
    a besoin d'un DOSSIER pour reprojeter/dédoublonner entre plusieurs vues).

RÉSOLUTION DU MODÈLE : les deux modes utilisent le MÊME registre
`config/models_registry.yaml` (voir model_registry.py). Unifier sur le
registre évite de devoir retenir un chemin de fichier différent selon le
mode, et propage automatiquement le bon `taxonomy_config` du modèle choisi
au mode orthomosaïque (voir `class_config_path` dans geo_density_map.py) -
sans quoi un modèle mono-classe choisi pour l'orthomosaïque serait comparé,
à tort, à la taxonomie 7-classes par défaut. C'est pour la MÊME raison que
`geo_density_map.py` appelé DIRECTEMENT (pas via ce module) reste exposé à
ce piège : sa propre CLI attend un chemin de poids brut et un `--class-config`
optionnel (défaut 7-classes silencieux si omis) - passer par CE module est
ce qui garantit la résolution automatique depuis le registre, dans les deux
modes.

POST-TRAITEMENT AUTOMATIQUE (identique dans les deux modes, car les deux
modes écrivent `detections.geojson` au même format - voir
`run_application.py`/`geo_density_map.py`) : une fois la carte produite, ce
module enchaîne automatiquement, sans étape manuelle supplémentaire :
  - l'export .xlsx de diagnostic des prédictions (`predictions_diagnostic.py`).
  - l'export du lot d'import CVAT (`export_predictions_to_cvat.py`), puis sa
    compression en .zip (le dossier `cvat_export/` non compressé est
    supprimé après coup, sauf `--no-zip-cvat-export`) - zip dont la racine
    contient directement `images/`, `labels/`, `data.yaml` (pas de dossier
    intermédiaire), prêt à être importé tel quel dans CVAT.
Chacune de ces deux étapes est individuellement désactivable (`--skip-stats`,
`--skip-cvat-export`) et n'interrompt jamais la production de la carte si
elle échoue (avertissement affiché, carte déjà produite conservée) - la carte
reste le livrable principal de ce module, ces deux exports sont additifs.

Exemple :
    # Mode orthomosaïque (détecté automatiquement : --input est un fichier GeoTIFF)
    python -m src.application.run_inference --input chemin/vers/orthomosaique.tif --model mono_class_v1

    # Mode batch (détecté automatiquement : --input est un dossier de photos)
    python -m src.application.run_inference --input chemin/vers/photos/ --model mono_class_v1

    # Sans les exports additifs (carte seule, comportement d'avant leur ajout)
    python -m src.application.run_inference --input chemin/vers/photos/ --model mono_class_v1 \\
        --skip-stats --skip-cvat-export
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional

from src.application.model_registry import load_operational_models, resolve_model_choice

MODE_ORTHOMOSAIC = "orthomosaic"
MODE_BATCH = "batch"


def detect_input_mode(input_path: Path) -> str:
    """Détecte le mode d'inférence à partir de `--input` (voir docstring du
    module). Lève une erreur explicite plutôt que de deviner si l'entrée
    n'est ni un dossier ni un GeoTIFF géoréférencé reconnaissable."""
    if input_path.is_dir():
        return MODE_BATCH

    if not input_path.is_file():
        raise FileNotFoundError(f"Entrée introuvable : {input_path} (ni fichier, ni dossier).")

    try:
        import rasterio
        with rasterio.open(input_path) as src:
            has_crs = src.crs is not None
    except Exception as e:
        raise ValueError(
            f"{input_path} est un fichier mais n'a pas pu être ouvert comme GeoTIFF géoréférencé "
            f"({e}). Ce pipeline ne prend en entrée que (a) un dossier de photos drone brutes "
            f"(mode batch) ou (b) un unique GeoTIFF géoréférencé issu d'un stitching WebODM (mode "
            f"orthomosaïque) - une photo isolée sans métadonnées de vol n'est exploitable par aucun "
            f"des deux modes (voir docstring du module)."
        ) from e

    if not has_crs:
        raise ValueError(
            f"{input_path} s'ouvre comme image mais n'a AUCUN géoréférencement (CRS absent) - un "
            f"GeoTIFF sans CRS ne peut pas être placé sur une carte. Vérifie qu'il s'agit bien d'un "
            f"export WebODM (orthomosaïque stitchée), pas d'une photo brute individuelle - pour des "
            f"photos brutes, passe le DOSSIER qui les contient en --input (mode batch)."
        )
    return MODE_ORTHOMOSAIC


def _zip_and_remove_folder(folder: Path) -> str:
    """Compresse `folder` en `<folder>.zip` À CÔTÉ (racine du zip = contenu
    de `folder`, jamais un dossier `folder/` imbriqué - c'est ce que CVAT
    attend d'un zip d'import direct : `images/`, `labels/`, `data.yaml` à la
    racine), puis supprime le dossier non compressé. Retourne le chemin du
    .zip."""
    archive_base = str(folder)  # shutil.make_archive ajoute lui-même ".zip"
    zip_path = shutil.make_archive(base_name=archive_base, format="zip", root_dir=str(folder))
    shutil.rmtree(folder)
    return zip_path


def _run_post_processing(
    predictions_dir: Path, skip_stats: bool, skip_cvat_export: bool,
    cvat_n_pieces: Optional[int], zip_cvat_export: bool,
) -> Dict[str, Optional[str]]:
    """Enchaîne les deux exports additifs (voir docstring du module) sur le
    dossier de run déjà produit par l'un ou l'autre mode - identique pour les
    deux, puisque `predictions_dir` désigne dans les deux cas le dossier
    contenant `detections.geojson` (voir `run_inference`, qui calcule ce
    chemin différemment selon le mode mais appelle CETTE fonction une seule
    fois, sans dupliquer la logique de post-traitement par mode).

    Chaque étape est indépendante et n'interrompt jamais l'autre ni la carte
    déjà produite si elle échoue - un avertissement est affiché, pas une
    exception propagée (voir docstring du module)."""
    results: Dict[str, Optional[str]] = {"stats_xlsx": None, "cvat_export": None}

    if not skip_stats:
        print("--- 📊 Statistiques des prédictions (predictions_diagnostic.py) ---")
        try:
            from src.application.predictions_diagnostic import run_diagnostic
            xlsx_path = run_diagnostic(str(predictions_dir))
            results["stats_xlsx"] = xlsx_path or None
        except Exception as e:
            print(f"⚠️  Export .xlsx de diagnostic échoué (carte déjà produite conservée) : {e}")

    if not skip_cvat_export:
        print("--- 📤 Export CVAT (export_predictions_to_cvat.py) ---")
        try:
            from src.application.export_predictions_to_cvat import export_predictions_to_cvat
            lot_dir_str = export_predictions_to_cvat(str(predictions_dir), n_pieces=cvat_n_pieces)
            if lot_dir_str:
                lot_dir = Path(lot_dir_str)
                if zip_cvat_export:
                    zip_path = _zip_and_remove_folder(lot_dir)
                    print(f"    Lot CVAT compressé : {zip_path}")
                    results["cvat_export"] = zip_path
                else:
                    results["cvat_export"] = str(lot_dir)
        except Exception as e:
            print(f"⚠️  Export CVAT échoué (carte déjà produite conservée) : {e}")

    return results


def run_inference(
    input_path: str,
    model_name: Optional[str] = None,
    conf_threshold: float = 0.25,
    output_dir: Optional[str] = None,
    # Options spécifiques au mode batch (voir run_application.py) - ignorées en mode orthomosaïque.
    dedup_method: str = "iou",
    dedup_iou_threshold: float = 0.3,
    # Options spécifiques au mode orthomosaïque (voir geo_density_map.py) - ignorées en mode batch.
    tile_size: int = 640,
    overlap: int = 256,
    ortho_max_zoom_cap: int = 21,
    satellite_max_native_zoom: int = 17,
    # Post-traitement additif (voir docstring du module) - identique dans les deux modes.
    skip_stats: bool = False,
    skip_cvat_export: bool = False,
    cvat_n_pieces: Optional[int] = None,
    zip_cvat_export: bool = True,
) -> Dict[str, Optional[str]]:
    """Point d'entrée unique : détecte le mode à partir de `input_path`,
    résout le modèle via le registre commun, délègue entièrement au pipeline
    approprié (voir docstring du module) pour produire la carte, PUIS
    enchaîne automatiquement les deux exports additifs (stats .xlsx, lot
    CVAT) sur le résultat - voir `_run_post_processing`.

    `conf_threshold` : mappé sur le seuil qui a le même rôle métier dans
    chaque mode - le seuil final APRÈS dédoublonnage en mode batch
    (`run_application.py`), le seuil de la 1ère (et unique) passe d'inférence
    tuilée en mode orthomosaïque (`geo_density_map.py`, pas de dédoublonnage
    inter-photos dans ce mode - une seule vue par zone, voir sa docstring).

    Retourne un dict {"map_path", "stats_xlsx", "cvat_export"} - les deux
    derniers sont `None` si l'étape correspondante a été sautée (`--skip-*`)
    ou a échoué (avertissement affiché, jamais une exception propagée depuis
    cette partie additive)."""
    input_p = Path(input_path)
    mode = detect_input_mode(input_p)

    models = load_operational_models()
    model = resolve_model_choice(models, requested_name=model_name)
    print(f"[run_inference] Mode détecté : {mode} — modèle : {model.name} ({model.weights_path})")

    if mode == MODE_BATCH:
        from src.application.run_application import run_application
        from src.application.web_map import build_web_map

        run_dir = run_application(
            batch_dir=str(input_p),
            model_name=model.name,
            conf_threshold=conf_threshold,
            dedup_method=dedup_method,
            dedup_iou_threshold=dedup_iou_threshold,
            output_dir=output_dir,
        )
        map_path = build_web_map(run_dir=str(run_dir), satellite_max_native_zoom=satellite_max_native_zoom)
        # Mode batch : detections.geojson vit directement dans run_dir (voir
        # run_application.py) - c'est aussi le dossier de la carte (carte.html).
        predictions_dir = Path(run_dir)
    else:
        # mode == MODE_ORTHOMOSAIC
        from src.application.geo_density_map import run_geo_density_map

        result = run_geo_density_map(
            tif_path=str(input_p),
            model_path=model.weights_path,
            tile_size=tile_size,
            overlap=overlap,
            tile_conf_threshold=conf_threshold,
            ortho_max_zoom_cap=ortho_max_zoom_cap,
            output_dir=output_dir,
            class_config_path=model.taxonomy_config,
        )
        map_path = Path(result["output_path"])
        # Mode orthomosaïque : detections.geojson vit à côté d'index.html
        # (voir geo_density_map.py, out_p.parent) - jamais dans un
        # sous-dossier séparé.
        predictions_dir = map_path.parent

    print(f"\n[SUCCÈS] Carte générée : {map_path}")

    post = _run_post_processing(
        predictions_dir, skip_stats=skip_stats, skip_cvat_export=skip_cvat_export,
        cvat_n_pieces=cvat_n_pieces, zip_cvat_export=zip_cvat_export,
    )

    if post["stats_xlsx"]:
        print(f"[SUCCÈS] Statistiques : {post['stats_xlsx']}")
    if post["cvat_export"]:
        print(f"[SUCCÈS] Export CVAT : {post['cvat_export']}")

    return {"map_path": str(map_path), **post}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                         help="Un DOSSIER de photos drone brutes (mode batch) OU un unique fichier "
                              "GeoTIFF géoréférencé (mode orthomosaïque) - détecté automatiquement, "
                              "voir docstring du module.")
    parser.add_argument("--model", default=None,
                         help="Nom du modèle dans config/models_registry.yaml (invite interactive si "
                              "omis et plusieurs modèles disponibles - voir model_registry.py).")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                         help="Seuil de confiance (rôle métier différent selon le mode détecté, voir "
                              "docstring de run_inference()).")
    parser.add_argument("--output-dir", default=None,
                         help="Dossier de sortie (défaut : 4. Results/2_prediction/<préfixe>_<horodatage>, "
                              "préfixe batch_ ou ortho_ selon le mode détecté - voir paths.py).")
    parser.add_argument("--dedup-method", choices=["visual", "iou"], default="iou",
                         help="Mode batch uniquement - voir run_application.py.")
    parser.add_argument("--dedup-iou-threshold", type=float, default=0.3,
                         help="Mode batch uniquement - voir run_application.py.")
    parser.add_argument("--tile-size", type=int, default=640, help="Mode orthomosaïque uniquement.")
    parser.add_argument("--overlap", type=int, default=256, help="Mode orthomosaïque uniquement.")
    parser.add_argument("--ortho-max-zoom-cap", type=int, default=21, help="Mode orthomosaïque uniquement.")
    parser.add_argument("--satellite-max-native-zoom", type=int, default=17,
                         help="Mode batch uniquement - voir web_map.py.")
    parser.add_argument("--skip-stats", action="store_true",
                         help="Ne pas générer l'export .xlsx de diagnostic des prédictions après la carte.")
    parser.add_argument("--skip-cvat-export", action="store_true",
                         help="Ne pas générer le lot d'import CVAT après la carte.")
    parser.add_argument("--cvat-n-pieces", type=int, default=None,
                         help="Force ce nombre de morceaux par image/orthomosaïque pour l'export CVAT "
                              "(voir export_predictions_to_cvat.py --n-pieces) - défaut : calculé "
                              "automatiquement.")
    parser.add_argument("--no-zip-cvat-export", action="store_true",
                         help="Garde le lot CVAT en dossier non compressé (images/labels/data.yaml) "
                              "plutôt que de le compresser en .zip puis supprimer le dossier (défaut).")
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        model_name=args.model,
        conf_threshold=args.conf_threshold,
        output_dir=args.output_dir,
        dedup_method=args.dedup_method,
        dedup_iou_threshold=args.dedup_iou_threshold,
        tile_size=args.tile_size,
        overlap=args.overlap,
        ortho_max_zoom_cap=args.ortho_max_zoom_cap,
        satellite_max_native_zoom=args.satellite_max_native_zoom,
        skip_stats=args.skip_stats,
        skip_cvat_export=args.skip_cvat_export,
        cvat_n_pieces=args.cvat_n_pieces,
        zip_cvat_export=not args.no_zip_cvat_export,
    )


if __name__ == "__main__":
    main()
