#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - [HISTORIQUE] Tableau croisé Entraînement x Évaluation, métrique PAR TUILE.

Depuis le 08/10/2026, le grand tableau se produit avec `python -m src.training.campaign` (métrique
niveau objet sur le banc figé, entraînement des 9 modèles compris). Ce script reste pour relire
les anciennes campagnes ; sa liste MODELS ci-dessous date du 23/09/2026.

Génère le tableau croisé Entraînement x Évaluation (F1 macro, split test)
pour les 8 modèles "un site/taxonomie" déjà entraînés (32 cellules = 8 modèles x 4 colonnes
d'évaluation : Full, Santa Luzia, Saint Brandon, Aldabra).

Logique : les 4 colonnes (Full, Santa Luzia, Saint Brandon, Aldabra) sont TOUTES
calculées via src.review.scoped_report, avec un filtre de lot vide ("") pour la
colonne "Full" - `"" in nom_de_lot` est toujours vrai, donc AUCUN lot n'est exclu :
c'est très exactement "pas de restriction", en réutilisant le même mécanisme que les
3 autres colonnes plutôt qu'un chemin de code séparé. Ce choix n'est PAS cosmétique :
pour une ligne "modèle dédié à un site" (Santa Luzia/Saint Brandon/Aldabra), le
rapport_metrics.json déjà présent dans le dossier du run a été généré À
L'ENTRAÎNEMENT avec la config DE CE SITE (ex: data_config_SB.yaml -> 4_sliced_dataset_SB,
qui ne contient QUE des images Saint Brandon) - le relire directement donnerait en
réalité la case DIAGONALE (Saint Brandon entraîné x Saint Brandon évalué), pas la
colonne Full (repéré le 11/09/2026 : 39 instances dans ce json pour SB_dedie, contre
911 pour le modèle Full - clairement pas le même test set). Toutes les colonnes sont
donc recalculées ici sur le dataset "Full" de la bonne taxonomie
(4_sliced_dataset(+_mono_class)/2_split_dataset(+_mono_class)) et écrites dans des
fichiers dédiés (rapport_Full_metrics.json, etc.) qui ne touchent jamais le
rapport_metrics.json d'origine du run.

scoped_report réutilise le split test de `2_split_dataset(+_mono_class)` - les mêmes
images test, quel que soit le modèle évalué, ce qui rend les 8 lignes comparables
entre elles AU SEIN D'UN MÊME PASSAGE de ce script. Attention (constat du 23/09/2026) :
ce test n'est PAS le banc figé de config/frozen_test_parents.json (désactivé de fait
depuis le 10-11/09/2026, voir FROZEN_TEST_BENCH_ENABLED dans split_dataset.py) mais le
tirage stable par hash 70/20/10 : une image déjà en test y reste, mais chaque NOUVELLE
image annotée a ~10 % de chances d'y entrer - le test grossit avec le dataset. Deux
passages de ce script à des dates différentes ne sont donc comparables que si
`test_instances`/`n_test_parents` sont identiques (voir le CSV versionné). Chaque ligne
est en outre contrôlée automatiquement contre toute fuite train/val -> test. Et JAMAIS le 4_sliced_dataset_SL/_SB/_A d'un modèle dédié à
UN site pour les colonnes de site : ce dataset-là ne contient QUE les images de ce
site (aucune image Saint Brandon/Aldabra dedans) et ne permettrait donc pas de
mesurer la généralisation croisée (SL_dedie -> évalué sur Saint Brandon, par exemple).

Sortie : tableau imprimé dans le terminal (2 grilles, multi-classe et mono-classe) +
CSV `evaluation_matrix.csv` à côté de ce script, une ligne par cellule.

À lancer depuis la racine du repo (celle qui contient src/) :
    python src\\training\\evaluation_matrix.py
ou, si placé dans le package src/training/ :
    python -m src.training.evaluation_matrix
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.paths_config import PROCESSED_DATASET_DIR, TRAINING_RUNS_DIR
from src.review.scoped_report import run_scoped_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Les 8 modèles du tableau (identifiés via le champ 'data:' de leur args.yaml,
#     vérifié pour chacun avant d'écrire ce script) ------------------------------
MULTI_DATA_CONFIG = PROJECT_ROOT / "config" / "data_config.yaml"
MULTI_SLICED_DIR = PROCESSED_DATASET_DIR / "4_sliced_dataset"
MULTI_SPLIT_DIR = PROCESSED_DATASET_DIR / "2_split_dataset"

MONO_DATA_CONFIG = PROJECT_ROOT / "config" / "data_config_mono_class.yaml"
MONO_SLICED_DIR = PROCESSED_DATASET_DIR / "4_sliced_dataset_mono_class"
MONO_SPLIT_DIR = PROCESSED_DATASET_DIR / "2_split_dataset_mono_class"

# (label_ligne, taxonomie, nom_du_dossier_de_run)
#
# Mise à jour le 23/09/2026 : 6 des 8 modèles réentraînés suite à l'ajout de
# nouvelles annotations Aldabra/Santa Luzia dans 1_annotated_dataset (voir
# audit_stats_geoloc_experiments.md, section 23/09/2026). Saint Brandon n'a
# reçu aucune nouvelle annotation -> ses 2 lignes restent volontairement sur
# les runs du 11/09/2026 (pas besoin de réentraîner un modèle dont les
# données d'entraînement n'ont pas changé - ses colonnes d'éval croisée sont
# quand même recalculées ci-dessous depuis les datasets COURANTS, voir la
# docstring du module).
MODELS = [
    ("Full", "multi", "reference_multiclasse_v2_yolo11n-seg_20260923_114634"),
    ("Santa Luzia", "multi", "reference_SL_v2_yolo11n-seg_20260923_130239"),
    ("Saint Brandon", "multi", "SB_dedie_yolo11n-seg_20260911_020811"),
    ("Aldabra", "multi", "A_dedie_v2_yolo11n-seg_20260923_132931"),
    ("Full", "mono", "reference_monoclasse_v2_yolo11n-seg_20260923_134351"),
    ("Santa Luzia", "mono", "SL_mono_v2_yolo11n-seg_20260923_150848"),
    # À RÉENTRAÎNER (constat du 23/09/2026) : ce run a été entraîné sur un split
    # "--balance-by items" défectueux (78 imagettes d'entraînement sur 510, AUCUNE sans
    # objet, et 17 de ses images train/val se retrouvent dans le test Full mono) - ses 4
    # cellules sont marquées "!" (fuite) par le contrôle automatique ci-dessous. Refaire
    # 2_split_dataset_SB_mono en mode "images" (défaut) puis réentraîner, comme les autres.
    ("Saint Brandon", "mono", "SB_mono_yolo11n-seg_20260911_022306"),
    ("Aldabra", "mono", "A_mono_v2_yolo11n-seg_20260923_155140"),
]

# (label_colonne, code_de_site) - None => colonne "Full" (aucun filtre). Depuis le
# 23/09/2026, filtre EXACT sur le code de site (raw_dataset.site_of_batch, via
# scoped_report(site_code=...)) plutôt qu'une sous-chaîne insensible à la casse sur le nom
# de lot : "A" devenait "a" et aurait capté tout lot contenant la lettre a (ex. un futur
# site "L Loango" compté dans Aldabra). Résultat identique sur les lots actuels (vérifié).
EVAL_COLUMNS = [
    ("Full", None),
    ("Santa Luzia", "SL"),
    ("Saint Brandon", "SB"),
    ("Aldabra", "A"),
]

ROW_ORDER = ["Full", "Santa Luzia", "Saint Brandon", "Aldabra"]


def _taxonomy_paths(taxonomy: str):
    if taxonomy == "mono":
        return MONO_DATA_CONFIG, MONO_SLICED_DIR, MONO_SPLIT_DIR
    return MULTI_DATA_CONFIG, MULTI_SLICED_DIR, MULTI_SPLIT_DIR


def _macro_f1_from_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    test = data.get("splits", {}).get("test")
    if not test:
        return None
    return test.get("macro", {}).get("f1")


def _cell_metrics_from_json(json_path: Path) -> Dict:
    """Comme `_macro_f1_from_json`, mais capture aussi le nombre d'instances de test et le
    `sliced_data_fingerprint` du run (ajouté le 23/09/2026, voir CSV_FIELDNAMES et la
    docstring de `print_and_export`). Sans ces colonnes, une dérive du test set est
    invisible dans le tableau imprimé (seul le F1 change, sans indice de la cause) - avec
    elles, un `test_instances` qui bouge d'un passage à l'autre pour un MÊME modèle
    non réentraîné signale immédiatement que la donnée de test elle-même a changé.
    Cas réel du 23/09/2026 : 911 -> 2678 instances-tuiles en test, parce que 3 nouvelles
    photos Santa Luzia très denses (lots "SL W1 n"/"SL W3 n") sont entrées en test via le
    tirage par hash (le banc figé étant désactivé par un chemin cassé - voir
    split_dataset.FROZEN_TEST_PATH)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    test = data.get("splits", {}).get("test") or {}
    return {
        "f1_macro_test": test.get("macro", {}).get("f1"),
        "test_instances": test.get("global", {}).get("instances"),
        "sliced_data_fingerprint": data.get("sliced_data_fingerprint"),
    }


def _scoped_column_f1(run_dir: Path, site_code: Optional[str], label: str, data_config_path: Path,
                       sliced_dir: Path, split_dir: Path) -> Dict:
    """Une seule fonction pour les 4 colonnes (voir docstring du module) : `site_code=None`
    pour "Full" ne restreint à rien (`batch_filter=""` : `"" in x` toujours vrai), `"SL"/"SB"/"A"`
    restreint au site par correspondance EXACTE du code de site (23/09/2026). Écrit toujours
    dans un fichier dédié (`rapport_<label>_metrics.json`), jamais dans
    le `rapport_metrics.json` d'origine du run - qui peut représenter tout autre chose
    (voir la mise en garde dans la docstring du module pour les modèles dédiés à un site).

    Retourne un dict (pas juste le F1, depuis le 23/09/2026) - voir `_cell_metrics_from_json`."""
    run_scoped_report(
        run_dir=run_dir,
        batch_filter="",
        site_code=site_code,
        split="test",
        label=label,
        data_config_path=data_config_path,
        sliced_dir=str(sliced_dir),
        split_dir=str(split_dir),
    )
    metrics_path = run_dir / f"rapport_{label}_metrics.json"
    return _cell_metrics_from_json(metrics_path)


def _training_split_dir(run_dir: Path) -> Optional[Path]:
    """Dossier 2_split_dataset* qui a servi à ENTRAÎNER ce run, retrouvé depuis son args.yaml
    (champ `data:` -> `path:` du data_config -> 4_sliced_dataset<suffix> -> 2_split_dataset<suffix>).
    Repli sur `<run>/resolved_data_config.yaml` si le chemin absolu enregistré n'existe plus
    (disque de données monté sous une autre lettre - E: -> F: -, cas prévu par paths_config).
    None si introuvable (le contrôle de fuite est alors signalé comme non vérifié)."""
    import yaml
    from src.paths_config import resolve_data_config_path_field
    from src.training.training_report import resolve_run_class_config_path
    try:
        try:
            cfg_path = resolve_run_class_config_path(run_dir)
        except FileNotFoundError:
            cfg_path = run_dir / "resolved_data_config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = resolve_data_config_path_field(yaml.safe_load(f) or {})
        sliced = Path(cfg["path"])
        split_dir = sliced.parent / sliced.name.replace("4_sliced_dataset", "2_split_dataset", 1)
        return split_dir if (split_dir / ".parent_manifest.json").exists() else None
    except Exception:
        return None


def _seen_parents(run_dir: Path):
    """Images parentes vues par CE modèle (train + val). Priorité à l'instantané écrit par
    train.py au lancement de l'entraînement (`training_parent_manifest.json`, depuis le
    23/09/2026) ; à défaut (runs plus anciens), manifeste ACTUEL de son dataset - exact tant
    que ce split n'a pas été reconstruit depuis l'entraînement. Retourne (ensemble, source)
    ou (None, raison)."""
    snap = run_dir / "training_parent_manifest.json"
    if snap.exists():
        with open(snap, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return {p for p, i in manifest.items() if i["split"] in ("train", "val")}, "instantané du run"
    split_dir = _training_split_dir(run_dir)
    if split_dir is None:
        return None, "split d'entraînement introuvable"
    return _parents(split_dir, ("train", "val")), f"manifeste actuel de {split_dir.name}"


def _parents(split_dir: Path, splits) -> Set[str]:
    with open(split_dir / ".parent_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {pid for pid, info in manifest.items() if info["split"] in splits}


def _test_parents_in_scope(eval_split_dir: Path, site_code: Optional[str]) -> Set[str]:
    from src.data.utils.raw_dataset import site_of_batch
    with open(eval_split_dir / ".parent_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {pid for pid, info in manifest.items()
            if info["split"] == "test" and (site_code is None or site_of_batch(info["batch"]) == site_code)}


def build_matrix():
    results = {"multi": {}, "mono": {}}  # [taxonomy][(row, col)] = dict (voir _cell_metrics_from_json)
    for row_label, taxonomy, run_name in MODELS:
        run_dir = TRAINING_RUNS_DIR / run_name
        data_config_path, sliced_dir, split_dir = _taxonomy_paths(taxonomy)
        print(f"\n=== {taxonomy.upper()} / entraîné sur {row_label} ({run_name}) ===")

        # Contrôle automatique de fuite (23/09/2026) : aucune image parente vue par CE modèle
        # (train ou val de SON propre split) ne doit figurer dans le test évalué. Ajouté
        # après avoir trouvé, à la main, que la ligne Saint Brandon mono (split
        # "--balance-by items") avait 17 parents d'entraînement/val dans le test Full mono.
        seen_parents, seen_source = _seen_parents(run_dir)
        if seen_parents is None:
            print(f"  ⚠️  {seen_source} - contrôle de fuite NON vérifié pour ce modèle ('?' dans le tableau).")
        else:
            print(f"  (contrôle de fuite : {seen_source})")

        for col_label, site_code in EVAL_COLUMNS:
            try:
                label = col_label.replace(" ", "")  # "Santa Luzia" -> "SantaLuzia"
                cell = _scoped_column_f1(run_dir, site_code, label, data_config_path, sliced_dir, split_dir)
                test_parents = _test_parents_in_scope(Path(split_dir), site_code)
                cell["n_test_parents"] = len(test_parents)
                cell["leaked_parents"] = (len(test_parents & seen_parents) if seen_parents is not None else None)
                results[taxonomy][(row_label, col_label)] = cell
                leak_note = (f"  🛑 FUITE : {cell['leaked_parents']} image(s) du test vue(s) à l'entraînement"
                             if cell["leaked_parents"] else "")
                print(f"  -> éval {col_label:<15} F1 macro (test) = {cell['f1_macro_test']} "
                      f"(n={cell['test_instances']} instances-tuiles, {len(test_parents)} images parentes){leak_note}")
            except Exception as e:  # noqa: BLE001 - une cellule en échec ne doit pas bloquer les 31 autres
                print(f"  -> éval {col_label:<15} ÉCHEC : {e}")
                results[taxonomy][(row_label, col_label)] = {
                    "f1_macro_test": None, "test_instances": None, "sliced_data_fingerprint": None,
                    "n_test_parents": None, "leaked_parents": None,
                }

    return results


CSV_FIELDNAMES = [
    "run_timestamp", "taxonomie", "entrainement", "run_name", "evaluation",
    "f1_macro_test", "test_instances", "n_test_parents", "leaked_parents", "sliced_data_fingerprint",
]


def print_and_export(results, models=MODELS):
    """Affiche les 2 grilles puis écrit le CSV - versionné depuis le 23/09/2026 (voir
    ci-dessous) plutôt qu'écrasé à chaque run.

    Pourquoi versionner (retour d'expérience du 23/09/2026) : `evaluation_matrix.csv` était
    jusqu'ici réécrit à l'identique à chaque exécution - impossible de comparer un ancien
    et un nouveau passage sans avoir manuellement recopié les chiffres ailleurs (ce qui est
    arrivé : la comparaison qui a mené à découvrir que le banc de test figé ne
    s'appliquait plus, ce jour-là, n'a été possible que parce que Jame avait reporté les
    anciens chiffres à la main). Chaque run écrit maintenant EN PLUS un instantané horodaté dans
    `evaluation_matrix_history/`, jamais écrasé - `evaluation_matrix.csv` continue d'exister
    tel quel (toujours le dernier run) pour ne rien casser en aval. Les 3 colonnes ajoutées
    (`run_name`, `test_instances`, `sliced_data_fingerprint`) permettent un diff direct
    entre deux instantanés sans avoir à rouvrir les rapports individuels : un
    `test_instances`/`sliced_data_fingerprint` qui bouge pour un modèle NON réentraîné
    (`run_name` identique d'un instantané à l'autre) signale une dérive du test set, pas un
    effet du modèle - exactement le signal qui manquait le 23/09/2026."""
    run_name_by = {(row_label, taxonomy): run_name for row_label, taxonomy, run_name in models}
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_rows = []
    for taxonomy, title in (("multi", "MULTI-CLASSE"), ("mono", "MONO-CLASSE")):
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        col_header_label = "Entrainement \\ Evaluation"
        header = f"{col_header_label:<28}" + "".join(f"{c:<16}" for c in ROW_ORDER)
        print(header)
        for row_label in ROW_ORDER:
            cells = []
            for col_label in ROW_ORDER:
                cell = results[taxonomy].get((row_label, col_label)) or {}
                f1 = cell.get("f1_macro_test")
                leak = cell.get("leaked_parents")
                mark = "!" if leak else ("?" if leak is None else "")
                cells.append((f"{f1:.3f}" + mark) if f1 is not None else "n/a")
                csv_rows.append({
                    "run_timestamp": run_timestamp,
                    "taxonomie": taxonomy,
                    "entrainement": row_label,
                    "run_name": run_name_by.get((row_label, taxonomy), ""),
                    "evaluation": col_label,
                    "f1_macro_test": f1,
                    "test_instances": cell.get("test_instances"),
                    "n_test_parents": cell.get("n_test_parents"),
                    "leaked_parents": cell.get("leaked_parents"),
                    "sliced_data_fingerprint": cell.get("sliced_data_fingerprint"),
                })
            print(f"{row_label:<28}" + "".join(f"{c:<16}" for c in cells))
    print("\n'!' = FUITE : au moins une image du test a été vue à l'entraînement par ce modèle - "
          "cellule NON interprétable (détail dans la sortie ci-dessus et dans le CSV, colonne leaked_parents). "
          "'?' = contrôle de fuite impossible pour ce modèle.")
    print("Rappel : colonnes Saint Brandon / Aldabra = très peu d'objets réels en test (voir n_test_parents "
          "dans le CSV) - un écart de quelques points n'y est pas significatif.")

    script_dir = Path(__file__).resolve().parent
    history_dir = script_dir / "evaluation_matrix_history"
    history_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_csv = history_dir / f"evaluation_matrix_{stamp}.csv"
    latest_csv = script_dir / "evaluation_matrix.csv"

    for out_csv in (versioned_csv, latest_csv):
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n📄 CSV écrit : {latest_csv} (dernier run)")
    print(f"📄 Instantané versionné : {versioned_csv} (jamais écrasé - historique complet dans {history_dir})")


if __name__ == "__main__":
    results = build_matrix()
    print_and_export(results)
