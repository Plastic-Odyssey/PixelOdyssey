#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Vérificateur du dataset brut (pré-slicing).

À lancer avant split_dataset.py (étape 2), directement sur
`1_annotated_dataset` (avant tout découpage en tuiles train/val/test).

Contrairement à `PlasticDatasetChecker` (qui valide 2_split_dataset ou
4_sliced_dataset après coup, structure images/{train,val,test} +
labels/{train,val,test}), ce script valide la donnée source : un ensemble de
lots/batches d'annotation (un sous-dossier par session terrain / export
CVAT), qui n'ont pas encore de notion de split.

Modèle de classes : chaque lot garde son propre data.yaml local (local_id ->
nom), qui peut différer librement d'un lot à l'autre en nombre/ordre/IDs de
classes - ce n'est jamais en soi un problème. Les data.yaml des lots ne sont
donc pas comparés entre eux ni à un "référentiel brut" : on vérifie
uniquement que chaque nom de classe réellement utilisé dans un .txt se
résout (après normalisation/alias) dans `class_taxonomy`
(config/data_config.yaml).

Vérifications :
1. Chaque lot doit avoir son propre data.yaml local (sinon : erreur
   bloquante, impossible de savoir ce que ses IDs de classe veulent dire).
2. Un label .txt sans image correspondante est une erreur bloquante (fichier
   orphelin). Une image sans .txt n'en est PAS une : c'est le format normal
   d'une photo de terrain confirmée sans déchet ("background"),
   volontairement gardée dans le dataset pour équilibrer l'entraînement -
   comptée et affichée pour information, jamais bloquante.
3. Chaque ID de classe utilisé dans un .txt doit exister dans le data.yaml
   local DE CE LOT (incohérence interne, erreur bloquante sinon), ET le NOM
   correspondant doit se résoudre (via normalisation + class_aliases) dans
   `class_taxonomy` (config/data_config.yaml) - erreur bloquante sinon. Ça
   permet à class_taxonomy de rester une table flexible : ajouter une classe
   n'importe où ne casse jamais rien en silence, l'oublier de classer se voit
   immédiatement - et un lot qui déclare simplement moins de classes qu'un
   autre ne bloque jamais rien.
4. Format de chaque ligne de label (nombre de coordonnées pair, ≥3 points,
   valeurs dans [0,1]).
5. NON BLOQUANT - doublons probables d'annotation : deux masques de MÊME
   classe cible (après résolution, pas juste même local_id - deux noms
   locaux différents peuvent pointer vers la même super-classe) sur la MÊME
   image parente, dont l'IoU dépasse `--duplicate-iou-threshold` (0.5 par
   défaut). Ça arrive typiquement quand un même déchet a été dessiné deux
   fois (retouche/relecture qui rajoute un contour sans supprimer l'ancien,
   ou fusion de deux exports CVAT) - le contour diffère légèrement mais
   couvre le même objet physique. Volontairement NON bloquant (contrairement
   aux points 1-4) : contrairement à une classe non résolue, il n'y a pas de
   correction automatique évidente - décider si c'est un vrai doublon ou
   deux objets voisins dans un tas nécessite un oeil humain, donc juste
   signalé pour relecture. Important AVANT le slicing : chaque doublon non
   corrigé ici se retrouve démultiplié dans 4_sliced_dataset (une paire
   dupliquée dans l'image parente devient une paire dupliquée dans CHAQUE
   tuile qui recouvre cette zone).

Entrée : dossier racine des lots d'annotation bruts (--raw-path) et le
référentiel de classes du projet (--reference-yaml).
Sortie : rapport sur stdout ; code de sortie 0 si le dataset brut est
cohérent, 1 sinon (le point 5 ci-dessus n'affecte jamais ce code de sortie).

Exemple :
    python -m src.data.raw_dataset_checker --raw-path 1_annotated_dataset
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set

from shapely.geometry import Polygon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.class_config import (
    ClassTaxonomy,
    EXCLUDE,
    load_batch_local_names,
    load_class_config,
    resolve_class_name,
)
# Importé depuis raw_dataset.py (source partagée), pas redéfini ici, pour
# éviter que deux copies identiques divergent silencieusement.
from src.data.raw_dataset import VALID_IMG_EXTS
# Réutilisé tel quel (même calcul que pour comparer GT<->prédiction en revue,
# voir tiled_inference.py/label_review.py) : l'IoU est invariant à l'échelle
# tant que x et y sont mis à l'échelle uniformément, donc valide directement
# sur des coordonnées normalisées [0,1] sans repasser en pixels.
from src.review.matching import polygon_iou


class RawDatasetValidator:
    """Valide l'intégrité de 1_annotated_dataset AVANT le split (étape 2)."""

    def __init__(
        self,
        raw_path: str,
        reference_yaml: str,
        reference_key: str = "class_taxonomy",
        duplicate_iou_threshold: float = 0.5,
    ):
        """
        Args:
            raw_path: dossier racine des lots d'annotation bruts (1_annotated_dataset).
            reference_yaml: chemin vers le référentiel unique de classes du projet
                (config/data_config.yaml). `reference_key` n'a plus d'usage réel
                dans le nouveau modèle par nom (conservé uniquement pour compatibilité
                de signature/CLI) - la taxonomie est toujours chargée via
                `class_config.load_class_config`, qui lit à la fois `class_taxonomy`
                et `class_aliases`.
            duplicate_iou_threshold: seuil au-delà duquel deux masques de même classe
                cible sur la même image sont signalés comme doublon probable (voir
                point 5 de la docstring du module). 0.5 par défaut : assez haut pour
                ne pas confondre deux objets voisins/qui se touchent dans un tas
                (chevauchement partiel plausible) avec un même objet dessiné deux fois
                (chevauchement quasi total attendu).
        """
        self.raw_root = Path(raw_path)
        self.reference_yaml = Path(reference_yaml)
        self.reference_key = reference_key
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.class_taxonomy: ClassTaxonomy = {}
        self.target_names: Dict[int, str] = {}
        self.problems: List[str] = []
        self.warnings: List[str] = []  # non bloquant - voir point 5 de la docstring du module

    def _fail(self, msg: str) -> None:
        print(f"❌ {msg}")
        self.problems.append(msg)

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)

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
            # (lab_path, nom classe cible, ligne A, ligne B, iou) - point 5, non bloquant.
            duplicate_candidates: List[tuple] = []

            for lab_path in expected_labels & all_labels:
                # target_id -> [(lineno, geom)] - remis à zéro à chaque image : un doublon
                # ne compare que des masques de la MÊME image, jamais entre deux images.
                class_polygons: Dict[int, List[tuple]] = {}

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
                    elif isinstance(resolved, int):
                        # EXCLUDE (chaîne) et None déjà écartés par ce elif - seule une
                        # vraie super-classe cible entre dans la comparaison de doublons.
                        try:
                            geom = Polygon(list(zip(coords[0::2], coords[1::2])))
                        except Exception:
                            geom = None
                        if geom is not None:
                            class_polygons.setdefault(resolved, []).append((lineno, geom))

                # Comparaison par PAIRE, seulement entre masques de même classe cible
                # résolue sur cette même image (voir class_polygons ci-dessus).
                for target_id, polys in class_polygons.items():
                    for i in range(len(polys)):
                        lineno_a, geom_a = polys[i]
                        for j in range(i + 1, len(polys)):
                            lineno_b, geom_b = polys[j]
                            iou = polygon_iou(geom_a, geom_b)
                            if iou >= self.duplicate_iou_threshold:
                                duplicate_candidates.append((
                                    lab_path, self.target_names.get(target_id, str(target_id)),
                                    lineno_a, lineno_b, iou,
                                ))

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

            if duplicate_candidates:
                # NON bloquant - ok reste inchangé. Voir point 5 de la docstring du module.
                self._warn(
                    f"[{batch_dir.name}] {len(duplicate_candidates)} paire(s) de masques "
                    f"probablement en double (même classe cible, IoU ≥ "
                    f"{self.duplicate_iou_threshold:.0%}, même image)."
                )
                print(f"  🔁 [{batch_dir.name}] {len(duplicate_candidates)} doublon(s) d'annotation "
                      f"probable(s) (même classe cible, IoU ≥ {self.duplicate_iou_threshold:.0%}, "
                      f"même image) - à relire manuellement (pas corrigé automatiquement) :")
                for lab_path, cls_name, la, lb, iou in duplicate_candidates[:10]:
                    print(f"      • {lab_path.relative_to(self.raw_root)} : lignes {la} & {lb} "
                          f"({cls_name}, IoU={iou:.0%})")
                if len(duplicate_candidates) > 10:
                    print(f"      ... (+{len(duplicate_candidates) - 10} autre(s))")

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
        if self.warnings:
            print(
                f"ℹ️  {len(self.warnings)} avertissement(s) non bloquant(s) - doublons d'annotation "
                f"probables (détail 🔁 ci-dessus). N'empêche pas de lancer le pipeline, mais à relire "
                f"avant de considérer le dataset propre - chaque doublon non corrigé se retrouve "
                f"démultiplié dans 4_sliced_dataset (une paire par tuile qui recouvre la zone)."
            )
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
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.5,
        help="Seuil d'IoU au-delà duquel deux masques de même classe cible sur la même image "
             "sont signalés comme doublon probable (avertissement non bloquant, voir point 5 "
             "de la docstring du module). Défaut : 0.5.",
    )
    args = parser.parse_args()

    validator = RawDatasetValidator(
        args.raw_path, args.reference_yaml, args.reference_key, args.duplicate_iou_threshold
    )
    success = validator.run_all()
    sys.exit(0 if success else 1)
