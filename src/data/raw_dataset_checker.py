#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Vérificateur du dataset BRUT (pré-slicing).

À lancer AVANT split_dataset.py (étape 2), directement
sur `1_annotated_dataset` (avant tout découpage en tuiles train/val/test).

Contrairement à `PlasticDatasetChecker` (qui valide 2_split_dataset ou
4_sliced_dataset APRÈS coup, structure images/{train,val,test} + labels/{train,val,test}),
ce script valide la donnée source : un ensemble de lots/batches d'annotation
(un sous-dossier par session terrain / export CVAT), qui n'ont pas encore de
notion de split.

Modèle de classes (révisé le 19/08/2026 - voir class_config.py) : chaque lot
garde son PROPRE data.yaml local (local_id -> nom), qui peut différer
librement d'un lot à l'autre en nombre/ordre/IDs de classes - ce n'est JAMAIS
en soi un problème (ex: SL n'a jamais eu la classe "à déterminer" apparue avec
A LEG1_1/A LEG3_1, et ça n'a rien d'anormal). On NE compare donc plus les
data.yaml des lots entre eux ni à un quelconque "référentiel brut" : on vérifie
uniquement que chaque nom de classe RÉELLEMENT UTILISÉ dans un .txt se résout
(après normalisation/alias) dans `class_taxonomy` (config/data_config.yaml).

Vérifications :
1. Chaque lot doit avoir son propre data.yaml local (sinon: erreur bloquante,
   impossible de savoir ce que ses IDs de classe veulent dire).
2. Un label .txt sans image correspondante est une erreur bloquante (fichier
   orphelin, ne peut correspondre à rien de valide). Une IMAGE sans .txt n'en
   est PAS une : c'est le format normal d'une photo de terrain confirmée sans
   déchet ("background") - volontairement gardée dans le dataset pour
   équilibrer l'entraînement (voir décision du 19/08/2026). Comptée et
   affichée pour information, jamais bloquante.
3. Chaque ID de classe utilisé dans un .txt doit exister dans le data.yaml
   local DE CE LOT (incohérence interne, erreur bloquante sinon), ET le NOM
   correspondant doit se résoudre (via normalisation + class_aliases) dans
   `class_taxonomy` (config/data_config.yaml) - erreur bloquante sinon : c'est
   exactement le cas d'une classe toute nouvelle (ex: "à déterminer") dont
   personne n'a encore décidé le sort. Bloquer ici, plutôt que de la mapper
   par erreur ou de la laisser tomber silencieusement, est ce qui permet à
   class_taxonomy de rester une table flexible : ajouter une classe n'importe
   où ne casse jamais rien en silence, l'oublier de classer se voit
   immédiatement - et un lot qui déclare simplement moins de classes qu'un
   autre ne bloque jamais rien.
4. Format de chaque ligne de label (nombre de coordonnées pair, ≥3 points,
   valeurs dans [0,1]).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import (
    ClassTaxonomy,
    EXCLUDE,
    load_batch_local_names,
    load_class_config,
    resolve_class_name,
)

VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class RawDatasetValidator:
    """Valide l'intégrité de 1_annotated_dataset AVANT le split (étape 2)."""

    def __init__(self, raw_path: str, reference_yaml: str, reference_key: str = "class_taxonomy"):
        """
        Args:
            raw_path: dossier racine des lots d'annotation bruts (1_annotated_dataset).
            reference_yaml: chemin vers le référentiel unique de classes du projet
                (config/data_config.yaml). `reference_key` n'a plus d'usage réel
                dans le nouveau modèle par nom (conservé uniquement pour compatibilité
                de signature/CLI) - la taxonomie est toujours chargée via
                `class_config.load_class_config`, qui lit à la fois `class_taxonomy`
                et `class_aliases`.
        """
        self.raw_root = Path(raw_path)
        self.reference_yaml = Path(reference_yaml)
        self.reference_key = reference_key
        self.class_taxonomy: ClassTaxonomy = {}
        self.target_names: Dict[int, str] = {}
        self.problems: List[str] = []

    def _fail(self, msg: str) -> None:
        print(f"❌ {msg}")
        self.problems.append(msg)

    def _list_batches(self) -> List[Path]:
        """Sous-dossiers de premier niveau = lots d'annotation (ex: SB 1, SL 11-16...)."""
        if not self.raw_root.exists():
            return []
        return sorted(
            p for p in self.raw_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    # ---- 1. Référentiel de classes ----
    def load_reference(self) -> bool:
        if not self.reference_yaml.exists():
            self._fail(f"Référentiel de classes introuvable : {self.reference_yaml}")
            return False
        try:
            self.class_taxonomy, self.target_names = load_class_config(self.reference_yaml)
        except (ValueError, FileNotFoundError) as e:
            self._fail(f"Référentiel de classes invalide ({self.reference_yaml}) : {e}")
            return False

        print(
            f"📖 Référentiel : {len(self.class_taxonomy)} entrée(s) de class_taxonomy/"
            f"class_aliases (résolues), {len(self.target_names)} super-classe(s) cible(s) "
            f"({self.reference_yaml})"
        )
        return True

    # ---- 2, 3 & 4. data.yaml par lot + correspondance image<->label + classes utilisées + format ----
    def check_images_and_labels(self) -> bool:
        print("\n--- 🔍 data.yaml PAR LOT + CORRESPONDANCE IMAGES/LABELS + CLASSES UTILISÉES (récursif) ---")
        ok = True

        for batch_dir in self._list_batches():
            local_yaml = batch_dir / "data.yaml"
            if not local_yaml.exists():
                ok = False
                self._fail(
                    f"[{batch_dir.name}] pas de data.yaml local - impossible de savoir ce que "
                    f"veulent dire les IDs de classe de ce lot. Chaque lot doit déclarer son "
                    f"propre data.yaml (local_id -> nom)."
                )
                continue

            try:
                local_names = load_batch_local_names(local_yaml)
            except Exception as e:
                ok = False
                self._fail(f"[{batch_dir.name}] data.yaml illisible ({local_yaml}) : {e}")
                continue

            img_root = batch_dir / "images"
            lab_root = batch_dir / "labels"
            if not img_root.exists() or not lab_root.exists():
                print(f"  ⚠️  [{batch_dir.name}] pas de sous-dossiers images/ + labels/ standards "
                      f"- vérification manuelle recommandée.")
                continue

            images = {
                p: (lab_root / p.relative_to(img_root)).with_suffix(".txt")
                for p in img_root.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS
                and not any(part.startswith(".") for part in p.relative_to(img_root).parts)
            }
            all_labels = {
                p for p in lab_root.rglob("*.txt")
                if not any(part.startswith(".") for part in p.relative_to(lab_root).parts)
            }
            expected_labels = set(images.values())

            orphan_images = sorted(img for img, lab in images.items() if not lab.exists())
            orphan_labels = sorted(all_labels - expected_labels)

            used_names: Set[str] = set()
            unknown_local_ids: Set[int] = set()
            unresolved_names: Set[str] = set()
            n_malformed = 0

            for lab_path in expected_labels & all_labels:
                for lineno, line in enumerate(lab_path.read_text(encoding="utf-8").splitlines(), start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        local_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                    except ValueError:
                        n_malformed += 1
                        print(f"      ⚠️ ligne non numérique : {lab_path.name} (L{lineno})")
                        continue
                    if len(coords) < 6 or len(coords) % 2 != 0:
                        n_malformed += 1
                        print(f"      ⚠️ polygone invalide (<3 points ou nb impair de coords) : "
                              f"{lab_path.name} (L{lineno})")
                        continue
                    if any(c < -0.01 or c > 1.01 for c in coords):
                        n_malformed += 1
                        print(f"      ⚠️ coordonnée hors [0,1] : {lab_path.name} (L{lineno})")

                    name = local_names.get(local_id)
                    if name is None:
                        unknown_local_ids.add(local_id)
                        continue

                    used_names.add(name)
                    resolved = resolve_class_name(name, self.class_taxonomy)
                    if resolved is None:
                        unresolved_names.add(name)

            print(f"  [{batch_dir.name}] {len(images)} images / {len(all_labels)} labels "
                  f"- classes utilisées : {sorted(used_names)}")

            if orphan_images:
                # PAS une erreur : une image sans .txt est une photo confirmée sans déchet
                # ("background"), volontairement gardée pour équilibrer l'entraînement -
                # voir docstring du module. Juste informatif, ne bloque jamais le pipeline.
                print(f"  ℹ️  [{batch_dir.name}] {len(orphan_images)} image(s) sans .txt "
                      f"(considérées 'sans déchet' / background, pas une erreur).")

            if orphan_labels:
                ok = False
                self._fail(f"[{batch_dir.name}] {len(orphan_labels)} label(s) .txt sans image correspondante.")
                for lab in orphan_labels[:10]:
                    print(f"      • {lab.relative_to(self.raw_root)}")
                if len(orphan_labels) > 10:
                    print(f"      ... (+{len(orphan_labels) - 10} autres)")

            if unknown_local_ids:
                ok = False
                self._fail(
                    f"[{batch_dir.name}] ID(s) de classe utilisé(s) dans les .txt mais absent(s) "
                    f"du data.yaml de ce lot : {sorted(unknown_local_ids)}. Corrige l'annotation "
                    f"ou le data.yaml de ce lot."
                )

            if unresolved_names:
                ok = False
                unresolved_str = ", ".join(f"'{n}'" for n in sorted(unresolved_names))
                self._fail(
                    f"[{batch_dir.name}] classe(s) utilisée(s) mais absente(s) (même après "
                    f"normalisation/alias) de class_taxonomy/class_aliases dans "
                    f"config/data_config.yaml : {unresolved_str}. Ajoute-les à class_taxonomy "
                    f"(ou class_aliases si c'est juste une autre orthographe d'une classe déjà "
                    f"connue), puis relance."
                )

            if n_malformed:
                ok = False
                self._fail(f"[{batch_dir.name}] {n_malformed} ligne(s) de label mal formée(s).")

        return ok

    def run_all(self) -> bool:
        print(f"=== VÉRIFICATION PRÉ-SLICING : {self.raw_root} ===")
        if not self.raw_root.exists():
            self._fail(f"Dossier brut introuvable : {self.raw_root}")
            return False
        if not self.load_reference():
            return False

        all_ok = self.check_images_and_labels()

        print("\n=== RÉSULTAT ===")
        if all_ok:
            print("✅ Dataset brut cohérent — tu peux lancer split_dataset.py (étape 2).")
        else:
            print(f"❌ {len(self.problems)} problème(s) détecté(s) — NE LANCE PAS le slicing avant correction.")
        return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vérificateur pré-slicing PixelOdyssey")
    parser.add_argument(
        "--raw-path",
        default=r"E:\PixelOdyssey\3. Processed dataset\1_annotated_dataset",
        help="Dossier racine des lots d'annotation bruts.",
    )
    parser.add_argument(
        "--reference-yaml",
        default="config/data_config.yaml",
        help="Chemin vers le référentiel unique de classes du projet "
             "(par défaut : config/data_config.yaml, à la racine du repo).",
    )
    parser.add_argument(
        "--reference-key",
        default="class_taxonomy",
        help="Conservé pour compatibilité de CLI - sans effet dans le nouveau modèle par nom "
             "(la taxonomie est toujours chargée via class_config.load_class_config, qui lit "
             "à la fois class_taxonomy et class_aliases).",
    )
    args = parser.parse_args()

    validator = RawDatasetValidator(args.raw_path, args.reference_yaml, args.reference_key)
    success = validator.run_all()
    sys.exit(0 if success else 1)
