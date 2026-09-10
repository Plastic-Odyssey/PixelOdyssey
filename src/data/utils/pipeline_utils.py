#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Utilitaires partagés par les étapes incrémentales du pipeline.

Chaque étape incrémentale (2_split_dataset, 3_augmented_dataset,
4_sliced_dataset) a le même besoin : ne pas retraiter un parent déjà présent,
SAUF si la configuration qui a produit ce qui existe a changé depuis - auquel
cas continuer silencieusement mélangerait deux versions différentes des
données dans le même dossier.

Constantes :
    MANIFEST_FILENAME  Nom par défaut du fichier manifeste de cache incrémental.

Exemple :
    from src.data.utils.pipeline_utils import ensure_cache_is_safe
    ensure_cache_is_safe(output_dir, current_params, force=False, wipe_subdirs=["images", "labels"])
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

MANIFEST_FILENAME = ".pipeline_manifest.json"


def fingerprint(params: Dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def read_upstream_fingerprint(manifest_path: Union[str, Path]) -> Optional[str]:
    """Lit le fingerprint enregistré par une étape AMONT, pour l'inclure dans les
    params (donc dans le fingerprint) de l'étape courante.

    Sans ceci, chaque étape incrémentale n'invaliderait son propre cache que si
    SES PROPRES paramètres changent - elle n'aurait aucune idée que l'étape
    amont a produit un contenu différent (ex: split_dataset.py qui change de
    logique, ce qui change QUELS parents tombent dans train/val/test, sans
    qu'aucun paramètre de augment_dataset.py ou slice_dataset.py n'ait lui-même
    changé). En incluant le fingerprint de l'étape amont dans les params de
    l'étape courante, tout changement amont (même sans changement de paramètre
    local) fait automatiquement mismatcher le fingerprint courant ->
    ensure_cache_is_safe lève l'erreur/wipe comme il faut, à chaque étape en
    aval, en cascade.

    Entrée : chemin du manifeste JSON de l'étape amont.
    Sortie : le fingerprint (str) qu'il contient, ou None si le manifeste
    amont n'existe pas encore (ex: tout premier run, étape amont pas encore
    exécutée) - une valeur absente est un état valide, pas une erreur.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f).get("fingerprint")


def diff_params(old_params: Dict, new_params: Dict) -> List[str]:
    lines = []
    for key in sorted(set(old_params) | set(new_params)):
        old_v, new_v = old_params.get(key, "<absent>"), new_params.get(key, "<absent>")
        if old_v != new_v:
            lines.append(f"  • {key} : {old_v}  ->  {new_v}")
    return lines or ["  (impossible de déterminer le détail - manifest précédent incomplet)"]


class RunConfirmation:
    """Partagée entre plusieurs appels de `ensure_cache_is_safe` au sein d'UN SEUL run de
    `data_pipeline.py`, pour qu'une unique confirmation interactive couvre tout le pipeline
    (split + augment + slice) au lieu d'une invite par étape - la cascade d'empreintes amont
    (voir `read_upstream_fingerprint`) fait qu'un changement en étape 1 déclenche généralement
    un mismatch en cascade sur les étapes suivantes.

    Sans effet quand une étape est lancée seule (`python -m src.data.utils.split_dataset`) : ne pas
    passer d'instance dans ce cas, chaque étape garde alors sa propre invite indépendante.
    `data_pipeline.py` crée UNE instance et la transmet aux 3 étapes de son run.

    Entrée : aucune (état initialisé à "pas encore décidé").
    Sortie : instance à passer en `run_confirmation=` à `ensure_cache_is_safe`.
    """

    def __init__(self):
        self.decided = False
        self.confirmed = False


def ensure_cache_is_safe(
    output_dir: Union[str, Path],
    current_params: Dict,
    force: bool,
    wipe_subdirs: List[str],
    manifest_filename: str = MANIFEST_FILENAME,
    run_confirmation: Optional["RunConfirmation"] = None,
) -> None:
    """Vérifie que `current_params` est compatible avec ce qui a produit le contenu déjà présent
    dans `output_dir` (repéré via un fichier manifeste contenant un fingerprint de la config
    du run précédent).

    - Pas de manifeste (premier run) : on l'écrit et on continue normalement.
    - Manifeste présent, fingerprint identique : le cache incrémental est sûr, on ne fait rien.
    - Manifeste présent, fingerprint différent :
        - `force=True`  -> supprime chaque dossier de `wipe_subdirs` (ex: ["images", "labels"])
          sous `output_dir`, puis réenregistre le nouveau manifeste, sans rien demander (utile pour
          un usage scripté/non interactif qui sait déjà qu'il veut régénérer).
        - `force=False`, `run_confirmation` déjà DÉCIDÉ plus tôt dans ce même run (voir
          `RunConfirmation` ci-dessus) -> réutilise la même réponse sans redemander : écrase si la
          réponse précédente était oui, lève une RuntimeError si elle était non.
        - `force=False`, pas encore décidé, session INTERACTIVE (terminal réel,
          `sys.stdin.isatty()`) -> invite `[y/N]` plutôt que de lever une erreur, pour éviter
          d'avoir à effacer les dossiers à la main ou passer --force à chaque fois qu'une
          décision de taxonomie ou de config change légitimement le contenu attendu. Si
          `run_confirmation` est fourni, la réponse y est mémorisée pour les appels suivants
          dans ce run. Répondre non lève quand même une RuntimeError (rien n'est modifié), pour
          que le code appelant garde un seul chemin d'erreur à gérer.
        - `force=False`, session NON interactive (pas de terminal - cron, script, pipe) -> lève une
          RuntimeError explicite : pas de prompt bloquant qui attendrait indéfiniment une entrée
          qui ne viendra jamais.

    Entrée : dossier de sortie, params courants, flag force, sous-dossiers à effacer en cas
    d'écrasement, nom du fichier manifeste, confirmation partagée optionnelle.
    Sortie : aucune (effet de bord : écrit/réécrit le manifeste, supprime `wipe_subdirs` si
    écrasement) - lève RuntimeError si l'écrasement est refusé ou impossible sans confirmation.
    """
    import shutil

    output_dir = Path(output_dir)
    manifest_path = output_dir / manifest_filename
    current_fp = fingerprint(current_params)

    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": current_fp, "params": current_params}, f, indent=2, ensure_ascii=False)
        print(f"📝 Premier run détecté sur {output_dir} : configuration enregistrée dans "
              f"{manifest_path.name} (fingerprint {current_fp}).")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        old_manifest = json.load(f)
    old_fp = old_manifest.get("fingerprint")

    if old_fp == current_fp:
        return  # Rien n'a changé : le cache incrémental est sûr.

    diff_lines = diff_params(old_manifest.get("params", {}), current_params)

    if not force:
        if run_confirmation is not None and run_confirmation.decided:
            # Déjà répondu plus tôt dans CE run (probablement l'étape amont, via la cascade
            # d'empreintes) - on applique la même décision sans reposer la question.
            if not run_confirmation.confirmed:
                raise RuntimeError(
                    f"Annulé (refusé plus tôt dans ce run) - {output_dir} n'a pas été modifié."
                )
            force = True
        else:
            header = (
                "\n🛑 CONFIGURATION CHANGÉE DEPUIS LE DERNIER RUN "
                f"(fingerprint {old_fp} -> {current_fp}) sur :\n"
                f"  {output_dir}\n\n"
                "Continuer maintenant mélangerait deux versions différentes dans le MÊME dossier.\n\n"
                "Ce qui a changé :\n" + "\n".join(diff_lines) + "\n"
            )

            if not sys.stdin.isatty():
                # Pas de terminal réel (script, cron, pipe) : jamais de prompt bloquant qui
                # attendrait une entrée qui ne viendra jamais - il faut passer --force explicitement.
                raise RuntimeError(
                    header + "\n"
                    f"-> Relance avec --force pour repartir de zéro : ça supprime {', '.join(wipe_subdirs)} "
                    f"sous {output_dir} puis régénère l'intégralité avec la config actuelle.\n"
                )

            print(header)
            scope_note = (
                " (s'applique à tout le pipeline en cours - split/augmentation/slicing - "
                "pas seulement à cette étape)" if run_confirmation is not None else ""
            )
            answer = input(
                f"Écraser {', '.join(wipe_subdirs)} sous {output_dir} et régénérer avec la config "
                f"actuelle{scope_note} ? [y/N] "
            ).strip().lower()
            confirmed = answer in ("y", "yes", "o", "oui")
            if run_confirmation is not None:
                run_confirmation.decided = True
                run_confirmation.confirmed = confirmed
            if not confirmed:
                raise RuntimeError(f"Annulé - {output_dir} n'a pas été modifié.")
            force = True  # confirmé interactivement : rejoint le chemin d'écrasement ci-dessous

    print(f"⚠️  Écrasement confirmé : la config a changé (fingerprint {old_fp} -> {current_fp}).")
    print("Ce qui change :\n" + "\n".join(diff_lines))
    for sub in wipe_subdirs:
        print(f"⚠️  Suppression de {output_dir / sub} avant régénération complète...")
        shutil.rmtree(output_dir / sub, ignore_errors=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": current_fp, "params": current_params}, f, indent=2, ensure_ascii=False)
    print(f"✅ {output_dir} réinitialisé, nouveau fingerprint enregistré : {current_fp}.")


def already_present(parent_id: str, existing_files: set, suffix: str = "") -> bool:
    """Vrai si des fichiers pour CE parent_id précis existent déjà dans `existing_files`.

    Teste `{parent_id}{suffix}` (correspondance exacte - ex: cas d'un fichier
    copié/transformé tel quel) et `{parent_id}_` en préfixe (cas d'un
    découpage en plusieurs sous-fichiers, ex: "{parent_id}_{x}_{y}.png").

    Volontairement plus strict qu'un simple `.startswith(parent_id)` : ça évite
    qu'un parent_id soit confondu avec un autre dont il n'est qu'un préfixe
    littéral (ex: "transect_1" collisionnait avec les tuiles de "transect_11").
    """
    exact = f"{parent_id}{suffix}"
    prefix = f"{parent_id}_"
    return any(f == exact or f.startswith(prefix) for f in existing_files)
