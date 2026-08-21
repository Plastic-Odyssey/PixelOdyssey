#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Utilitaires partagés par les étapes incrémentales du pipeline.

Généralisé à partir de ce qui vivait uniquement dans data_pipeline.py (étape 4).
Chaque étape incrémentale (2_split_dataset, 4_sliced_dataset, et plus tard
3_augmented_dataset une fois implémentée) a le même besoin : ne pas retraiter
un parent déjà présent, SAUF si la configuration qui a produit ce qui existe a
changé depuis - auquel cas continuer silencieusement mélangerait deux versions
différentes des données dans le même dossier.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Union

MANIFEST_FILENAME = ".pipeline_manifest.json"


def fingerprint(params: Dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def diff_params(old_params: Dict, new_params: Dict) -> List[str]:
    lines = []
    for key in sorted(set(old_params) | set(new_params)):
        old_v, new_v = old_params.get(key, "<absent>"), new_params.get(key, "<absent>")
        if old_v != new_v:
            lines.append(f"  • {key} : {old_v}  ->  {new_v}")
    return lines or ["  (impossible de déterminer le détail - manifest précédent incomplet)"]


def ensure_cache_is_safe(
    output_dir: Union[str, Path],
    current_params: Dict,
    force: bool,
    wipe_subdirs: List[str],
    manifest_filename: str = MANIFEST_FILENAME,
) -> None:
    """Vérifie que `current_params` est compatible avec ce qui a produit le contenu déjà présent
    dans `output_dir` (repéré via un fichier manifeste contenant un fingerprint de la config
    du run précédent).

    - Pas de manifeste (premier run) : on l'écrit et on continue normalement.
    - Manifeste présent, fingerprint identique : le cache incrémental est sûr, on ne fait rien.
    - Manifeste présent, fingerprint différent :
        - `force=False` -> lève une RuntimeError explicite (rien n'est modifié sur le disque).
        - `force=True`  -> supprime chaque dossier de `wipe_subdirs` (ex: ["images", "labels"])
          sous `output_dir`, puis réenregistre le nouveau manifeste.
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
        raise RuntimeError(
            "\n🛑 CONFIGURATION CHANGÉE DEPUIS LE DERNIER RUN "
            f"(fingerprint {old_fp} -> {current_fp}) sur :\n"
            f"  {output_dir}\n\n"
            "Continuer maintenant mélangerait deux versions différentes dans le MÊME dossier.\n\n"
            "Ce qui a changé :\n" + "\n".join(diff_lines) + "\n\n"
            f"-> Relance avec --force pour repartir de zéro : ça supprime {', '.join(wipe_subdirs)} "
            f"sous {output_dir} puis régénère l'intégralité avec la config actuelle.\n"
        )

    print(f"⚠️  --force : la config a changé (fingerprint {old_fp} -> {current_fp}).")
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
