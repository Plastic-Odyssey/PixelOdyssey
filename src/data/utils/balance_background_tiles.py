#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Test brutal de rééquilibrage : plafonner les tuiles de fond.

Outil PONCTUEL DE TEST, volontairement EN DEHORS de data_pipeline.py : ce
n'est PAS une étape régulière du pipeline (comme freeze_benchmark_test.py,
voir sa docstring pour le même genre d'outil manuel à effet ponctuel). Il ne
lit ni n'écrit jamais un `4_sliced_dataset*` en place : il DUPLIQUE un dossier
déjà tuilé (voir slice_dataset.py) vers un nouveau dossier indépendant, en y
supprimant aléatoirement des tuiles de fond (0 objet) pour ramener leur part
sous un plafond, PAR SPLIT (train/val/test traités indépendamment, comme
partout ailleurs dans le pipeline - voir split_dataset.py).

Pourquoi "brutal", assumé comme tel (voir décision du 08/09/2026, journal de
décisions) : une suppression aléatoire de tuiles déjà écrites ne peut pas
être consciente des classes ni de l'image PARENTE dont chaque tuile est
issue - `background_origin_diagnostic.py` montre que sur 4_sliced_dataset,
~95% des tuiles de fond viennent d'images parentes ENTIÈREMENT sans déchet
(698/735 en train, 211/224 en val, 49/60 en test - voir ce diagnostic pour
le détail), pas du sous-échantillonnage 1 tuile vide sur 10. Un tirage
aléatoire ici peut donc vider certaines de ces images parentes de la quasi-
totalité de leurs tuiles et en laisser d'autres quasi intactes, sans aucun
contrôle sur CE choix. C'est un instrument volontairement grossier pour un
test rapide, pas la correction définitive : si ce test confirme un bénéfice,
le bon levier permanent est un sous-échantillonnage conscient des images
parentes 100% fond DANS slicer.py (avant l'écriture des tuiles - voir sa
docstring, section "Rétention des tuiles sans déchet"), ou un rééquilibrage
à l'étape 3 (augment_dataset.py, encore un simple passe-plat aujourd'hui) -
jamais en itérant sur CE script, qui doit rester un outil de test ponctuel.

Le plafond s'applique PAR SPLIT indépendamment (train, val, test peuvent
avoir des taux de fond très différents au départ - voir tile_density_diagnostic.py) :
soit fg le nombre de tuiles avec au moins 1 objet et bg le nombre de tuiles
vides d'un split, le nombre de tuiles de fond à garder pour rester à
max_empty_pct est plancher(fg * p/(100-p)) - si bg est déjà sous ce plafond,
RIEN n'est supprimé pour ce split (jamais de suppression pour "faire joli",
seulement pour repasser sous le plafond demandé).

Déterminisme : tirage aléatoire seedé (--seed, 42 par défaut) et par split
(un `random.Random` distinct par split, jamais un seul générateur partagé
entre splits - un ajout/retrait de tuiles dans un split ne doit jamais
décaler le tirage d'un autre split) - reproductible exactement, ce qui
compte pour un test dont on veut pouvoir rejouer/documenter le résultat
(cohérent avec la méthodologie "isoler une seule variable à la fois" du
guide d'entraînement).

Le dossier de sortie est un INSTANTANÉ MANUEL, pas géré par le cache
incrémental de slice_dataset.py : le fichier `.slicing_manifest.json` de la
source n'est délibérément PAS copié (il décrirait un contenu qui n'est plus
le bon après suppression), remplacé par `BALANCING_MANIFEST.json`
(paramètres utilisés, comptes avant/après par split) - ne JAMAIS pointer
slice_dataset.py --sliced-dir vers ce dossier de sortie, il n'a pas vocation
à être re-tuilé incrémentalement.

Reprise après interruption (--resume) : chaque paire est copiée sous un nom
temporaire puis renommée atomiquement (os.replace) vers son nom final - une
exécution interrompue en plein milieu (ex: coupure du poste local en cours
de copie, une image PNG peut peser plusieurs Mo sur un support réseau/synchronisé
lent) ne laisse donc jamais de paire à moitié écrite qu'une reprise
confondrait avec une paire déjà terminée. --resume ne recopie que les paires
dont l'image ET le label final manquent encore ; sûr uniquement parce que le
plan est entièrement déterministe (voir plus haut) - une reprise ne peut
jamais mélanger deux tirages différents.

Entrée : --sliced-dir (dossier 4_sliced_dataset* déjà tuilé, défaut :
4_sliced_dataset), --max-empty-pct (défaut 20.0), --seed (défaut 42),
--output-dir (défaut : <sliced-dir>_bg<plafond arrondi>, dossier FRÈRE de
--sliced-dir), --dry-run (affiche le plan sans rien copier), --force
(écrase --output-dir s'il existe déjà), --resume (complète une copie
interrompue sans repartir de zéro).
Sortie : dossier <output-dir>/{images,labels}/{train,val,test} + BALANCING_MANIFEST.json.

Exemple :
    python -m src.data.utils.balance_background_tiles --dry-run
    python -m src.data.utils.balance_background_tiles --max-empty-pct 20 --seed 42
    python -m src.data.utils.balance_background_tiles --sliced-dir "E:\\PixelOdyssey\\3. Processed dataset\\4_sliced_dataset_mono_class"
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.utils.slice_dataset import SLICED_DIR, SPLITS

DEFAULT_MAX_EMPTY_PCT = 20.0
DEFAULT_SEED = 42
MANIFEST_FILENAME = "BALANCING_MANIFEST.json"


def _copy_atomic(src: Path, dest: Path) -> None:
    """Copie via un nom temporaire puis `os.replace` (atomique sur un même
    volume) - voir docstring du module, section reprise : une interruption
    en plein milieu laisse au pire un `.part` orphelin, jamais un fichier
    final tronqué que --resume prendrait à tort pour déjà terminé."""
    tmp = dest.with_name(dest.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def _is_empty_label(label_path: Path) -> bool:
    """Une tuile vide a un .txt de 0 ligne non vide, jamais un .txt absent
    (voir slicer.py, f.writelines(tile_labels) même si tile_labels est vide -
    même hypothèse que tile_density_diagnostic.py). Un .txt manquant (ne
    devrait normalement jamais arriver) est traité comme vide plutôt
    qu'ignoré silencieusement."""
    if not label_path.exists():
        return True
    with open(label_path, "r", encoding="utf-8") as f:
        return not any(line.strip() for line in f)


def _list_split_tiles(sliced_dir: Path, split: str) -> Tuple[List[str], List[str]]:
    """Sortie : (stems foreground, stems background) pour ce split, triés
    (ordre déterministe, condition du tirage aléatoire reproductible)."""
    img_dir = sliced_dir / "images" / split
    lab_dir = sliced_dir / "labels" / split
    fg_stems, bg_stems = [], []
    if not img_dir.exists():
        return fg_stems, bg_stems
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        if _is_empty_label(lab_dir / f"{stem}.txt"):
            bg_stems.append(stem)
        else:
            fg_stems.append(stem)
    return fg_stems, bg_stems


def _max_bg_allowed(n_fg: int, max_empty_pct: float) -> int:
    """Nombre maximal de tuiles de fond compatible avec max_empty_pct une fois
    combiné aux n_fg tuiles foreground (qui ne sont jamais touchées) :
    bg / (fg + bg) <= p/100  =>  bg <= fg * p / (100 - p)."""
    if max_empty_pct >= 100:
        return math.inf
    return math.floor(n_fg * max_empty_pct / (100.0 - max_empty_pct))


def plan_balancing(sliced_dir: Path, max_empty_pct: float, seed: int) -> Dict[str, Dict]:
    """Calcule, PAR SPLIT, les stems de fond à garder/supprimer - ne touche
    à rien sur disque (utilisé aussi bien par --dry-run que juste avant la
    copie réelle, pour que les deux affichent exactement le même plan)."""
    plan: Dict[str, Dict] = {}
    for split in SPLITS:
        fg_stems, bg_stems = _list_split_tiles(sliced_dir, split)
        n_fg, n_bg = len(fg_stems), len(bg_stems)
        max_bg = _max_bg_allowed(n_fg, max_empty_pct)
        n_remove = max(0, n_bg - max_bg) if math.isfinite(max_bg) else 0

        rng = random.Random(f"{seed}_{split}")
        remove_stems = set(rng.sample(bg_stems, n_remove)) if n_remove else set()
        keep_bg_stems = [s for s in bg_stems if s not in remove_stems]

        pct_before = 100.0 * n_bg / (n_fg + n_bg) if (n_fg + n_bg) else 0.0
        n_bg_after = n_bg - n_remove
        pct_after = 100.0 * n_bg_after / (n_fg + n_bg_after) if (n_fg + n_bg_after) else 0.0

        plan[split] = {
            "fg_stems": fg_stems,
            "keep_bg_stems": keep_bg_stems,
            "remove_stems": sorted(remove_stems),
            "n_fg": n_fg,
            "n_bg_before": n_bg,
            "n_bg_after": n_bg_after,
            "n_removed": n_remove,
            "pct_empty_before": round(pct_before, 2),
            "pct_empty_after": round(pct_after, 2),
        }
    return plan


def _print_plan(plan: Dict[str, Dict], max_empty_pct: float) -> None:
    print(f"--- 🔧 PLAN DE RÉÉQUILIBRAGE (plafond {max_empty_pct:.1f}% de tuiles vides / split) ---\n")
    print(f"{'Split':<8}{'fg':>8}{'bg avant':>10}{'% avant':>10}{'à suppr.':>10}{'bg après':>10}{'% après':>10}")
    for split in SPLITS:
        p = plan[split]
        print(
            f"{split:<8}{p['n_fg']:>8}{p['n_bg_before']:>10}{p['pct_empty_before']:>9.1f}%"
            f"{p['n_removed']:>10}{p['n_bg_after']:>10}{p['pct_empty_after']:>9.1f}%"
        )
    print()


def apply_balancing(sliced_dir: Path, output_dir: Path, plan: Dict[str, Dict],
                     max_empty_pct: float, seed: int, skip_existing: bool = False) -> None:
    """`skip_existing=True` (voir --resume) : ne recopie pas une paire déjà
    présente en sortie - sûr uniquement parce que le plan est entièrement
    déterministe (même seed + même source + même plafond => exactement les
    mêmes stems gardés, voir plan_balancing) : une reprise ne peut jamais
    mélanger deux tirages différents, seulement compléter une copie
    interrompue (ex: déconnexion du poste local en cours de copie)."""
    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_sliced_dir": str(sliced_dir),
        "max_empty_pct": max_empty_pct,
        "seed": seed,
        "note": (
            "Instantané manuel issu de balance_background_tiles.py - NE JAMAIS pointer "
            "slice_dataset.py --sliced-dir vers ce dossier (pas de cache incrémental valide ici)."
        ),
        "per_split": {},
    }

    for split in SPLITS:
        p = plan[split]
        src_img_dir = sliced_dir / "images" / split
        src_lab_dir = sliced_dir / "labels" / split
        out_img_dir = output_dir / "images" / split
        out_lab_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lab_dir.mkdir(parents=True, exist_ok=True)

        if skip_existing:
            # Un ".part" orphelin ne peut être qu'une copie interrompue en plein
            # milieu (voir _copy_atomic) - jamais un fichier final valide - donc
            # toujours sûr à supprimer avant de recompter ce qui manque encore.
            for stray in list(out_img_dir.glob("*.part")) + list(out_lab_dir.glob("*.part")):
                stray.unlink()

        kept_stems = p["fg_stems"] + p["keep_bg_stems"]
        n_copied, n_skipped = 0, 0
        for stem in kept_stems:
            out_img = out_img_dir / f"{stem}.png"
            out_lab = out_lab_dir / f"{stem}.txt"
            if skip_existing and out_img.exists() and out_lab.exists():
                n_skipped += 1
                continue
            _copy_atomic(src_img_dir / f"{stem}.png", out_img)
            _copy_atomic(src_lab_dir / f"{stem}.txt", out_lab)
            n_copied += 1

        manifest["per_split"][split] = {
            "n_fg": p["n_fg"],
            "n_bg_before": p["n_bg_before"],
            "n_bg_after": p["n_bg_after"],
            "n_removed": p["n_removed"],
            "pct_empty_before": p["pct_empty_before"],
            "pct_empty_after": p["pct_empty_after"],
            "removed_stems": p["remove_stems"],
        }
        print(f"    [{split}] {len(kept_stems)} tuile(s) au total "
              f"({p['n_fg']} fg + {p['n_bg_after']} bg, {p['n_removed']} supprimée(s)) - "
              f"{n_copied} copiée(s) cette exécution, {n_skipped} déjà présente(s).")

    with open(output_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def run_balancing(sliced_dir: str = SLICED_DIR, max_empty_pct: float = DEFAULT_MAX_EMPTY_PCT,
                   seed: int = DEFAULT_SEED, output_dir: str = None,
                   dry_run: bool = False, force: bool = False, resume: bool = False) -> str:
    sliced_dir_p = Path(sliced_dir)
    if not sliced_dir_p.exists():
        raise RuntimeError(f"Dossier introuvable : {sliced_dir_p}. Lance d'abord slice_dataset.py.")

    if output_dir is None:
        output_dir_p = sliced_dir_p.parent / f"{sliced_dir_p.name}_bg{int(round(max_empty_pct))}"
    else:
        output_dir_p = Path(output_dir)

    plan = plan_balancing(sliced_dir_p, max_empty_pct, seed)
    _print_plan(plan, max_empty_pct)

    if dry_run:
        print(f"[DRY-RUN] Rien n'a été copié. Dossier qui serait créé : {output_dir_p}")
        return str(output_dir_p)

    manifest_path = output_dir_p / MANIFEST_FILENAME
    if output_dir_p.exists():
        if manifest_path.exists() and resume:
            print(f"[REPRISE] {output_dir_p} contient déjà {MANIFEST_FILENAME} - copie déjà terminée, rien à faire.")
            return str(output_dir_p)
        if resume:
            print(f"[REPRISE] {output_dir_p} existe (copie précédente interrompue, pas de "
                  f"{MANIFEST_FILENAME}) - complète les paires manquantes sans toucher à celles déjà là.")
        elif not force:
            raise RuntimeError(
                f"{output_dir_p} existe déjà - passe --force pour l'écraser (le dossier existant "
                f"est entièrement supprimé puis recréé, pas fusionné) ou --resume pour compléter une "
                f"copie interrompue (déterministe, sûr de reprendre - voir docstring de apply_balancing)."
            )
        else:
            shutil.rmtree(output_dir_p)

    output_dir_p.mkdir(parents=True, exist_ok=True)
    apply_balancing(sliced_dir_p, output_dir_p, plan, max_empty_pct, seed, skip_existing=resume)

    print(f"\n[SUCCÈS] Dataset rééquilibré écrit : {output_dir_p}")
    print(f"    Prochaine étape : dupliquer un config/data_config*.yaml, changer son `path` vers ce "
          f"dossier, puis vérifier avec tile_density_diagnostic.py --sliced-dir {output_dir_p}")
    return str(output_dir_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test brutal de rééquilibrage fond/objet PixelOdyssey")
    parser.add_argument(
        "--sliced-dir", default=SLICED_DIR,
        help=f"Dossier 4_sliced_dataset source, déjà tuilé (défaut : {SLICED_DIR}).",
    )
    parser.add_argument(
        "--max-empty-pct", type=float, default=DEFAULT_MAX_EMPTY_PCT,
        help=f"Plafond de tuiles vides par split, en %% (défaut {DEFAULT_MAX_EMPTY_PCT}).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Graine du tirage aléatoire, par split (défaut {DEFAULT_SEED}) - reproductible à l'identique.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Dossier de sortie (défaut : <sliced-dir>_bg<plafond arrondi>, dossier FRÈRE de --sliced-dir).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche le plan (combien supprimer par split) sans rien copier ni créer.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Écrase --output-dir s'il existe déjà (supprimé puis recréé entièrement).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Complète --output-dir s'il existe déjà suite à une exécution interrompue, en ne "
             "copiant que les paires manquantes (voir docstring du module, section reprise) - "
             "incompatible avec --force (l'un écrase, l'autre complète).",
    )
    args = parser.parse_args()
    if args.force and args.resume:
        print("--force et --resume sont incompatibles (l'un écrase tout, l'autre complète).")
        sys.exit(1)
    try:
        run_balancing(
            sliced_dir=args.sliced_dir, max_empty_pct=args.max_empty_pct, seed=args.seed,
            output_dir=args.output_dir, dry_run=args.dry_run, force=args.force, resume=args.resume,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
