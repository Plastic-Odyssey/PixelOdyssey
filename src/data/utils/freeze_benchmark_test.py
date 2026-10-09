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

3. Gel STRATIFIÉ (24/09/2026, `--stratified`, le banc v2) : construit un banc de test
   neuf, lot par lot, qui vise TEST_RATIO (15 %) des objets retenus de chaque lot, 15 %
   de ses images (pour garder des images sans déchet et mesurer les fausses alertes) et
   une part équilibrée de chaque classe, en ne prenant que des GROUPES SPATIAUX entiers
   (voir config/split_groups.json). À score égal, préfère les images qui étaient déjà en
   test, puis en val, dans le split courant : les modèles déjà entraînés restent ainsi
   évaluables sans fuite sur la plus grande partie possible du nouveau banc
   (object_level_eval.py --unseen-by). L'ancien banc est archivé dans
   frozen_test_parents_v1.json.

Exemple :
    python -m src.data.utils.freeze_benchmark_test --stratified --force
    python -m src.data.utils.freeze_benchmark_test
    python -m src.data.utils.freeze_benchmark_test --force   # regèle un nouveau banc (écrase l'ancien)
    python -m src.data.utils.freeze_benchmark_test --manual
    python -m src.data.utils.freeze_benchmark_test --manual --manual-file config/manual_test_selection.json --force
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.utils.split_dataset import (
    FROZEN_TEST_PATH, PARENT_MANIFEST_FILENAME, RAW_DIR, SPLIT_DIR, TEST_RATIO, _load_split_groups,
    _stable_unit_interval,
)
from src.data.utils.class_config import (
    DEFAULT_CLASS_CONFIG_PATH, EXCLUDE, load_batch_local_names, load_class_config, resolve_class_name,
)
from src.data.utils.raw_dataset import collect_parent_images, site_of_batch

# Fichier de sélection manuelle par défaut (voir freeze_manual_test()). Format :
# {"images": ["<lot>/images/train/<fichier>", ...], "full_batches": ["<lot>", ...]}
# - "images" : chemins relatifs à RAW_DIR (1_annotated_dataset), séparateurs / ou \,
#   ou chemin Windows absolu complet (le nom du dossier RAW_DIR sert alors d'ancre).
# - "full_batches" : noms de lots (dossiers de premier niveau sous RAW_DIR, ex: "SB 2")
#   dont TOUTES les images parentes doivent être gelées en test.
# parents[3] = racine du dépôt (utils -> data -> src -> racine). Corrigé le 23/09/2026 :
# parent.parent.parent pointait vers src/config/ depuis le déplacement de ce fichier dans
# src/data/utils/ (11/09/2026) - même bug que FROZEN_TEST_PATH dans split_dataset.py.
DEFAULT_MANUAL_SELECTION_PATH = Path(__file__).resolve().parents[3] / "config" / "manual_test_selection.json"


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
        "\n⚠️  Prochaine étape : relance le pipeline (`python -m src.data.data_pipeline`) - le changement "
        "de banc modifie le fingerprint du split, qui sera reconstruit (confirmation demandée)."
    )
    print(
        "\n⚠️  Rappel (23/09/2026) : ce banc n'est appliqué que si FROZEN_TEST_BENCH_ENABLED = True dans "
        "src/data/utils/split_dataset.py (False par défaut). Le passer à True change la répartition "
        "train/val/test de TOUS les datasets : réentraîner ensuite tous les modèles à comparer."
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
        "\n⚠️  Prochaine étape : relance le pipeline (`python -m src.data.data_pipeline`) pour que ce "
        "gel manuel s'applique (reconstruction du split, confirmation demandée)."
    )
    print(
        "\n⚠️  Rappel (23/09/2026) : ce banc n'est appliqué que si FROZEN_TEST_BENCH_ENABLED = True dans "
        "src/data/utils/split_dataset.py (False par défaut). Le passer à True change la répartition "
        "train/val/test de TOUS les datasets : réentraîner ensuite tous les modèles à comparer."
    )
    return FROZEN_TEST_PATH


STRATIFIED_MIN_UNITS = 3  # un lot de moins de 3 groupes n'envoie rien en test (trop petit pour être coupé)
PREFERENCE_PENALTY = {"test": 0.0, "val": 0.02, "train": 0.04}


def _kept_counts_by_class(label_path: Path, local_names: Dict[int, str], taxonomy) -> Counter:
    out: Counter = Counter()
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        name = local_names.get(int(parts[0]))
        target = resolve_class_name(name, taxonomy) if name is not None else None
        if target is not None and target != EXCLUDE:
            out[int(target)] += 1
    return out


def freeze_stratified_test(force: bool = False, target_ratio: float = TEST_RATIO,
                           class_config_path: Path = DEFAULT_CLASS_CONFIG_PATH, dry_run: bool = False) -> Path:
    """Banc de test v2 (voir le point 3 de la docstring du module).

    Par lot, sélection gloutonne de groupes spatiaux entiers qui minimise l'écart
    normalisé à trois cibles : objets retenus (poids 1), images (poids 1), objets par
    classe (poids 0,5, moyenne des écarts relatifs des classes présentes dans le lot).
    S'arrête dès qu'aucun groupe supplémentaire ne réduit l'écart. Déterministe."""
    if FROZEN_TEST_PATH.exists() and not force and not dry_run:
        raise RuntimeError(f"{FROZEN_TEST_PATH} existe déjà - relance avec --force pour le remplacer "
                           f"(l'ancien banc sera archivé).")
    taxonomy, target_names = load_class_config(class_config_path)
    parents = list({p["parent_id"]: p for p in collect_parent_images(Path(RAW_DIR))}.values())
    groups = _load_split_groups()
    current_split: Dict[str, str] = {}
    manifest_path = Path(SPLIT_DIR) / PARENT_MANIFEST_FILENAME
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            current_split = {pid: v["split"] for pid, v in json.load(f).items()}

    names_by_batch: Dict[str, Dict[int, str]] = {}
    by_batch: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[str, Counter] = {}
    for p in parents:
        b = p["batch"]
        if b not in names_by_batch:
            names_by_batch[b] = load_batch_local_names(Path(RAW_DIR) / b / "data.yaml")
        counts[p["parent_id"]] = _kept_counts_by_class(Path(p["label_path"]), names_by_batch[b], taxonomy)
        by_batch[b][groups.get(p["parent_id"], p["parent_id"])].append(p)

    selected: List[str] = []
    report = []
    for batch in sorted(by_batch):
        units = by_batch[batch]
        n_img = sum(len(m) for m in units.values())
        tot = Counter()
        for m in units.values():
            for p in m:
                tot.update(counts[p["parent_id"]])
        t_obj, t_img = target_ratio * sum(tot.values()), target_ratio * n_img
        t_cls = {c: target_ratio * n for c, n in tot.items()}
        if len(units) < STRATIFIED_MIN_UNITS:
            report.append((batch, 0, n_img, 0, sum(tot.values()), "lot trop petit : aucun groupe en test"))
            continue

        def unit_stats(uid):
            c = Counter()
            for p in units[uid]:
                c.update(counts[p["parent_id"]])
            return c, len(units[uid])

        stats = {uid: unit_stats(uid) for uid in units}

        def gap(obj_c: Counter, n: int) -> float:
            g = abs(t_obj - sum(obj_c.values())) / max(t_obj, 1.0) + abs(t_img - n) / max(t_img, 1.0)
            if t_cls:
                g += 0.5 * sum(abs(t_cls[c] - obj_c.get(c, 0)) / max(t_cls[c], 1.0) for c in t_cls) / len(t_cls)
            return g

        def penalty(uid):
            splits = [current_split.get(p["parent_id"], "train") for p in units[uid]]
            return max(PREFERENCE_PENALTY.get(s, 0.04) for s in splits)

        chosen: List[str] = []
        cur_c, cur_n = Counter(), 0
        cur_gap = gap(cur_c, cur_n)
        while True:
            best = None
            for uid in sorted(units, key=lambda u: _stable_unit_interval(f"bench_v2:{batch}:{u}")):
                if uid in chosen:
                    continue
                c, n = stats[uid]
                g = gap(cur_c + c, cur_n + n) + penalty(uid)
                if best is None or g < best[0]:
                    best = (g, uid)
            if best is None or best[0] >= cur_gap:
                break
            chosen.append(best[1])
            cur_c += stats[best[1]][0]
            cur_n += stats[best[1]][1]
            cur_gap = gap(cur_c, cur_n)
        for uid in chosen:
            selected.extend(p["parent_id"] for p in units[uid])
        report.append((batch, cur_n, n_img, sum(cur_c.values()), sum(tot.values()), ""))

    print(f"--- Banc de test v2 (stratifié, cible {target_ratio:.0%}) ---")
    for batch, n_sel, n_img, o_sel, o_tot, note in report:
        pct = f"{100 * o_sel / o_tot:.0f}%" if o_tot else "-"
        print(f"  • {batch:<24} images {n_sel:>3}/{n_img:<3} objets {o_sel:>4}/{o_tot:<5} ({pct}) {note}")
    sel = set(selected)
    cls_tot, cls_sel, site_tot, site_sel = Counter(), Counter(), Counter(), Counter()
    for p in parents:
        c = counts[p["parent_id"]]
        cls_tot.update(c)
        site_tot[site_of_batch(p["batch"])] += sum(c.values())
        if p["parent_id"] in sel:
            cls_sel.update(c)
            site_sel[site_of_batch(p["batch"])] += sum(c.values())
    print("  Par site : " + " | ".join(f"{s} {site_sel[s]}/{site_tot[s]} ({100 * site_sel[s] / max(1, site_tot[s]):.0f}%)"
                                     for s in sorted(site_tot)))
    print("  Par classe : " + " | ".join(f"{target_names.get(c, c)} {cls_sel[c]}/{cls_tot[c]}" for c in sorted(cls_tot)))
    already = Counter(current_split.get(pid, "absent") for pid in sel)
    print(f"  Provenance dans le split actuel : {dict(already)} (seules les images venues de 'train' "
          f"sont des fuites pour les modèles déjà entraînés)")
    if dry_run:
        print("  (dry-run : rien n'est écrit)")
        return FROZEN_TEST_PATH

    if FROZEN_TEST_PATH.exists():
        archive = FROZEN_TEST_PATH.with_name("frozen_test_parents_v1.json")
        if not archive.exists():
            shutil.copyfile(FROZEN_TEST_PATH, archive)  # copie octet pour octet (fins de ligne comprises)
            print(f"  Ancien banc archivé : {archive.name}")
    payload = {
        "note": ("Banc de test v2 (24/09/2026), stratifié par lot sur les objets, les images et les classes, "
                 "par groupes spatiaux entiers - voir freeze_benchmark_test.py --stratified. Figé pour tous "
                 "les runs à venir : ne pas éditer à la main."),
        "source": "stratified_v2",
        "target_ratio": target_ratio,
        "class_config": Path(class_config_path).name,
        "n_parent_ids": len(sel),
        "parent_ids": sorted(sel),
    }
    with open(FROZEN_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✅ {FROZEN_TEST_PATH} écrit ({len(sel)} images parentes).")
    return FROZEN_TEST_PATH


def remap_renamed_frozen(dry_run: bool = False) -> Path:
    """Suit un RENOMMAGE de lot sans changer le contenu du banc (08/10/2026).

    L'identifiant d'une image (parent_id) dérive du nom de son lot : renommer un dossier de
    lot change donc l'identifiant de ses images, et une image du banc figé devient
    "introuvable" alors qu'elle est toujours là. Pour chaque identifiant figé introuvable,
    cherche une image actuelle du MÊME site avec le même nom de fichier ET la même taille
    (même photo, octet pour octet en pratique) et remplace l'ancien identifiant par le
    nouveau. Le banc garde exactement les mêmes images. Un identifiant sans correspondance
    unique est laissé tel quel et signalé (image réellement disparue). L'ancien fichier est
    archivé dans config/frozen_test_history/."""
    if not FROZEN_TEST_PATH.exists():
        raise RuntimeError(f"{FROZEN_TEST_PATH} introuvable - rien à remapper.")
    with open(FROZEN_TEST_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ids = list(payload.get("parent_ids", []))
    parents = list({p["parent_id"]: p for p in collect_parent_images(Path(RAW_DIR))}.values())
    current = {p["parent_id"] for p in parents}
    old_manifest: Dict[str, Dict] = {}
    manifest_path = Path(SPLIT_DIR) / PARENT_MANIFEST_FILENAME
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            old_manifest = json.load(f)
    remap: Dict[str, str] = {}
    lost: List[str] = []
    for pid in ids:
        if pid in current:
            continue
        info = old_manifest.get(pid)
        old_img = Path(str(info["raw_img_path"]).replace("\\", "/")) if info else None
        candidates = []
        for p in parents:
            if p["parent_id"] in ids:
                continue
            img = Path(p["img_path"])
            if info is not None:
                same_name = old_img is not None and img.name == old_img.name
                same_site = site_of_batch(p["batch"]) == site_of_batch(info["batch"])
            else:
                # Sans l'ancien manifeste (2_split_dataset déjà reconstruit ou absent) : même fin
                # d'identifiant ("..._<nom de fichier sans extension>") et même code de site.
                same_name = pid.endswith("_" + img.stem)
                same_site = site_of_batch(p["batch"]) == pid.split("_", 1)[0]
            if same_name and same_site:
                copy = (Path(SPLIT_DIR) / "images" / info["split"] / f"{pid}{img.suffix.lower()}") if info else None
                size_ref = copy.stat().st_size if copy is not None and copy.exists() else None
                if size_ref is None or img.stat().st_size == size_ref:
                    candidates.append(p["parent_id"])
        if len(candidates) == 1:
            remap[pid] = candidates[0]
        else:
            lost.append(pid)
    for old, new in remap.items():
        print(f"  ↪ {old}  ->  {new}")
    for pid in lost:
        print(f"  ⚠️  {pid} : introuvable, aucune correspondance unique - laissé tel quel (image disparue ?)")
    if not remap:
        print("✅ Aucun identifiant du banc à remapper.")
        return FROZEN_TEST_PATH
    if dry_run:
        print("  (dry-run : rien n'est écrit)")
        return FROZEN_TEST_PATH
    hist = FROZEN_TEST_PATH.parent / "frozen_test_history"
    hist.mkdir(exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(FROZEN_TEST_PATH, hist / f"frozen_test_parents_{stamp}.json")
    payload["parent_ids"] = sorted(remap.get(pid, pid) for pid in ids)
    payload["n_parent_ids"] = len(payload["parent_ids"])
    payload.setdefault("remapped", []).append({"date": stamp, "renamed": remap})
    with open(FROZEN_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✅ {len(remap)} identifiant(s) remappé(s) dans {FROZEN_TEST_PATH} (même contenu de banc ; "
          f"ancienne version archivée dans {hist.name}/).")
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
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Construit le banc v2 stratifié par lot/objets/classes et par groupes spatiaux "
             "(24/09/2026) - voir freeze_stratified_test().",
    )
    parser.add_argument(
        "--remap-renamed",
        action="store_true",
        help="Après le renommage d'un dossier de lot : remplace dans le banc les identifiants devenus "
             "introuvables par ceux des MÊMES images sous leur nouveau nom (08/10/2026).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Avec --stratified/--remap-renamed : affiche sans écrire.")
    args = parser.parse_args()
    try:
        if args.remap_renamed:
            remap_renamed_frozen(dry_run=args.dry_run)
        elif args.stratified:
            freeze_stratified_test(force=args.force, dry_run=args.dry_run)
        elif args.manual:
            freeze_manual_test(manual_file=Path(args.manual_file), force=args.force)
        else:
            freeze_current_test(force=args.force)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
