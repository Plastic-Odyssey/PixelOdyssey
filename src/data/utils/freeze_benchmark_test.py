#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Gel du banc de test pour les comparaisons entre runs.

Prend une photo du split test ACTUEL (lu dans `2_split_dataset/.parent_manifest.json`,
donc `split_dataset.py` doit déjà avoir tourné au moins une fois) et l'écrit dans
`config/frozen_test_parents.json`. À partir de là, `split_dataset.py` épingle ces
parent_id en test de façon PERMANENTE, quel que soit le lot ou les données ajoutées
plus tard - voir sa docstring.

Effet définitif à comprendre avant de lancer : une fois gelé, plus aucune image du
lot dont provient un parent_id figé ne peut "reprendre sa place" en test - le reste
de ce lot (et tout nouveau lot futur) n'alimente plus que train/val. Toute
comparaison de runs faite APRÈS ce gel porte donc sur exactement les mêmes images
test, mais le test lui-même cesse de refléter les nouvelles sources ajoutées au
fil du temps - à rafraîchir explicitement (une nouvelle exécution de ce script,
décision consciente) si le test devient trop daté par rapport aux nouvelles
sources, pas automatiquement.

N'écrase jamais un banc déjà gelé sans --force explicite (un gel accidentel
répété casserait justement la garantie "toujours les mêmes images").

Entrée : `2_split_dataset/.parent_manifest.json` (déjà généré par split_dataset.py).
Sortie : `config/frozen_test_parents.json` (liste de parent_id).

Deux modes de gel sont disponibles :

1. Gel du split ALÉATOIRE actuel (comportement d'origine) : prend le split test
   produit par le dernier `split_dataset.py`, tel quel.
2. Gel d'une sélection MANUELLE (curée à la main) : lit une liste d'images et/ou
   de lots entiers depuis un fichier JSON (défaut : config/manual_test_selection.json)
   et convertit chaque entrée en parent_id via EXACTEMENT la même dérivation que
   raw_dataset.collect_parent_images (nom de lot + chemin relatif, séparateurs et
   espaces remplacés par "_"). Utile quand le split aléatoire par lot laisse des
   petits lots sans aucune image de test, ou tire un transect atypique en densité
   d'objets.

Exemple :
    python -m src.data.utils.freeze_benchmark_test
    python -m src.data.utils.freeze_benchmark_test --force   # regèle un nouveau banc (écrase l'ancien)
    python -m src.data.utils.freeze_benchmark_test --manual
    python -m src.data.utils.freeze_benchmark_test --manual --manual-file config/manual_test_selection.json --force
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.utils.split_dataset import FROZEN_TEST_PATH, PARENT_MANIFEST_FILENAME, RAW_DIR, SPLIT_DIR
from src.data.utils.raw_dataset import collect_parent_images

# Fichier de sélection manuelle par défaut (voir freeze_manual_test()). Format :
# {"images": ["<lot>/images/train/<fichier>", ...], "full_batches": ["<lot>", ...]}
# - "images" : chemins relatifs à RAW_DIR (1_annotated_dataset), séparateurs / ou \,
#   ou chemin Windows absolu complet (le nom du dossier RAW_DIR sert alors d'ancre).
# - "full_batches" : noms de lots (dossiers de premier niveau sous RAW_DIR, ex: "SB 2")
#   dont TOUTES les images parentes doivent être gelées en test.
DEFAULT_MANUAL_SELECTION_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "manual_test_selection.json"


def freeze_current_test(force: bool = False) -> Path:
    manifest_path = Path(SPLIT_DIR) / PARENT_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} introuvable - lance d'abord `python -m src.data.utils.split_dataset` "
            f"(ou le pipeline complet) au moins une fois avant de geler le banc de test."
        )

    if FROZEN_TEST_PATH.exists() and not force:
        raise RuntimeError(
            f"{FROZEN_TEST_PATH} existe déjà - un banc de test est déjà gelé. Relance avec --force "
            f"UNIQUEMENT si tu veux délibérément le remplacer par le split actuel (ça casse la "
            f"comparabilité avec tout run évalué sur l'ancien banc)."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        parent_manifest = json.load(f)

    test_entries = {pid: info for pid, info in parent_manifest.items() if info.get("split") == "test"}
    if not test_entries:
        raise RuntimeError(f"Aucune image en split 'test' dans {manifest_path} - rien à geler.")

    batch_counts = Counter(info["batch"] for info in test_entries.values())

    payload = {
        "note": (
            "Banc de test figé - ces parent_id restent en test pour tous les runs futurs, "
            "voir freeze_benchmark_test.py et le journal de décisions (26/08/2026). "
            "Ne pas éditer à la main ; régénérer avec --force pour un gel délibéré."
        ),
        "n_parent_ids": len(test_entries),
        "parent_ids": sorted(test_entries.keys()),
    }
    FROZEN_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Banc de test figé : {len(test_entries)} image(s) parente(s) épinglées en test, "
          f"écrit dans {FROZEN_TEST_PATH}.")
    print("   Répartition par lot :")
    for batch, n in sorted(batch_counts.items()):
        print(f"     • {batch} : {n}")
    print(
        "\n⚠️  Prochaine étape : relance le pipeline avec --force "
        "(`python -m src.data.utils.data_pipeline --force`) pour que le split existant applique "
        "immédiatement ce gel - sans ça, 2_split_dataset garde son contenu actuel jusqu'au "
        "prochain changement de config qui déclencherait de toute façon une régénération."
    )
    return FROZEN_TEST_PATH


def _path_to_parent_id(img_path: str, raw_dir: Path) -> str:
    """Convertit un chemin d'image en parent_id, selon EXACTEMENT la même dérivation
    que raw_dataset.collect_parent_images (indispensable pour que le parent_id calculé
    ici corresponde à celui réellement utilisé par split_dataset.py).

    Accepte soit un chemin relatif à raw_dir (ex: "A LEG3_1/images/train/DJI_0018.jpg",
    séparateurs / ou \\), soit un chemin absolu qui contient le nom de raw_dir quelque
    part dans son arborescence (pratique pour coller un chemin Windows complet tel
    qu'affiché dans l'explorateur de fichiers, ex: "E:\\PixelOdyssey\\...\\1_annotated_dataset\\...").
    """
    raw_dir = Path(raw_dir)
    p = PureWindowsPath(img_path) if "\\" in img_path else Path(img_path)
    parts = p.parts
    raw_dir_name = raw_dir.name
    if raw_dir_name in parts:
        idx = parts.index(raw_dir_name)
        rel_parts = parts[idx + 1:]
    else:
        # Pas de raw_dir_name dans le chemin -> on suppose qu'il est déjà relatif à raw_dir.
        rel_parts = parts
    if not rel_parts:
        raise ValueError(f"Chemin '{img_path}' ne pointe vers aucun fichier sous '{raw_dir_name}'.")
    rel = Path(*rel_parts).with_suffix("")
    return str(rel).replace("\\", "_").replace("/", "_").replace(" ", "_")


def freeze_manual_test(
    manual_file: Path = DEFAULT_MANUAL_SELECTION_PATH,
    raw_dir: str = RAW_DIR,
    force: bool = False,
) -> Path:
    if FROZEN_TEST_PATH.exists() and not force:
        raise RuntimeError(
            f"{FROZEN_TEST_PATH} existe déjà - un banc de test est déjà gelé. Relance avec --force "
            f"UNIQUEMENT si tu veux délibérément le remplacer par cette sélection manuelle (ça casse "
            f"la comparabilité avec tout run évalué sur l'ancien banc)."
        )

    manual_file = Path(manual_file)
    if not manual_file.exists():
        raise RuntimeError(
            f"{manual_file} introuvable - crée ce fichier (voir la docstring de ce module pour le "
            f"format attendu : clés 'images' et/ou 'full_batches')."
        )
    with open(manual_file, "r", encoding="utf-8") as f:
        selection = json.load(f)

    image_entries: List[str] = selection.get("images", [])
    full_batches: List[str] = selection.get("full_batches", [])
    if not image_entries and not full_batches:
        raise RuntimeError(f"{manual_file} ne liste ni 'images' ni 'full_batches' - rien à geler.")

    raw_dir = Path(raw_dir)
    all_parents = collect_parent_images(raw_dir)
    if not all_parents:
        raise RuntimeError(f"Aucune image parente trouvée sous {raw_dir}.")
    all_by_id = {p["parent_id"]: p for p in all_parents}
    # Repli insensible à la casse : le dataset brut a des sous-dossiers "train"/"Train"
    # incohérents SELON LE LOT (SL/A LEG utilisent "train" en minuscule, SB "Train" en
    # majuscule) - un chemin tapé à la main dans
    # manual_test_selection.json avec la mauvaise casse ne doit pas silencieusement
    # échouer à résoudre tout un lot. En cas de collision entre deux parent_id qui ne
    # diffèrent QUE par la casse (jamais observé, mais possible en théorie), le premier
    # rencontré gagne - suffisant ici car cette table sert seulement de repli best-effort.
    all_by_id_ci = {pid.lower(): pid for pid in all_by_id}
    by_batch: Dict[str, List[str]] = defaultdict(list)
    for p in all_parents:
        by_batch[p["batch"]].append(p["parent_id"])

    selected: set = set()
    errors: List[str] = []
    case_fallbacks: List[str] = []

    for batch in full_batches:
        if batch not in by_batch:
            errors.append(f"lot entier '{batch}' introuvable sous {raw_dir} (lots disponibles : {sorted(by_batch)})")
            continue
        selected.update(by_batch[batch])

    for img in image_entries:
        try:
            pid = _path_to_parent_id(img, raw_dir)
        except ValueError as e:
            errors.append(str(e))
            continue
        if pid not in all_by_id:
            pid_ci = all_by_id_ci.get(pid.lower())
            if pid_ci is None:
                errors.append(f"image '{img}' -> parent_id '{pid}' introuvable sous {raw_dir}")
                continue
            case_fallbacks.append(f"'{img}' : casse corrigée automatiquement ('{pid}' -> '{pid_ci}')")
            pid = pid_ci
        selected.add(pid)

    if case_fallbacks:
        print(f"ℹ️  {len(case_fallbacks)} chemin(s) résolu(s) via un repli insensible à la casse "
              f"(dossier train/Train incohérent entre lots) :")
        for line in case_fallbacks:
            print(f"     • {line}")

    if errors:
        raise RuntimeError(
            f"{len(errors)} entrée(s) de {manual_file} n'ont pas pu être résolues, rien n'a été écrit "
            f"(corrige {manual_file} et relance) :\n  - " + "\n  - ".join(errors)
        )
    if not selected:
        raise RuntimeError("Aucun parent_id résolu depuis la sélection manuelle - rien à geler.")

    batch_counts = Counter(all_by_id[pid]["batch"] for pid in selected)

    payload = {
        "note": (
            "Banc de test figé MANUELLEMENT (curation par transect représentatif, pas un split "
            f"aléatoire) - voir {manual_file.name} et le journal de décisions (27/08/2026). "
            "Ne pas éditer à la main ; régénérer avec --manual --force pour un gel délibéré."
        ),
        "source": "manual",
        "manual_file": str(manual_file),
        "n_parent_ids": len(selected),
        "parent_ids": sorted(selected),
    }
    FROZEN_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Banc de test figé MANUELLEMENT : {len(selected)} image(s) parente(s) épinglées en test, "
          f"écrit dans {FROZEN_TEST_PATH}.")
    print("   Répartition par lot :")
    for batch, n in sorted(batch_counts.items()):
        print(f"     • {batch} : {n}")
    print(
        "\n⚠️  Prochaine étape : relance le pipeline avec --force "
        "(`python -m src.data.utils.data_pipeline --force`) pour que ce gel manuel s'applique."
    )
    return FROZEN_TEST_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gèle le split test actuel de PixelOdyssey pour les comparaisons entre runs."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remplace un banc de test déjà gelé (geste délibéré, casse la comparabilité avec les "
             "runs déjà évalués sur l'ancien banc).",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Gèle une sélection MANUELLE (config/manual_test_selection.json par défaut) au lieu du "
             "split aléatoire actuel produit par split_dataset.py - voir freeze_manual_test().",
    )
    parser.add_argument(
        "--manual-file",
        default=str(DEFAULT_MANUAL_SELECTION_PATH),
        help=f"Chemin du fichier de sélection manuelle à utiliser avec --manual "
             f"(défaut : {DEFAULT_MANUAL_SELECTION_PATH}).",
    )
    args = parser.parse_args()
    try:
        if args.manual:
            freeze_manual_test(manual_file=Path(args.manual_file), force=args.force)
        else:
            freeze_current_test(force=args.force)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
