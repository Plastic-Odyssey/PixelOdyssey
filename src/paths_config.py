# -*- coding: utf-8 -*-
"""
Racine unique du disque de données PixelOdyssey (dataset annoté et ses
étapes dérivées, résultats, poids de modèles bruts). Le projet sépare
délibérément code (ce repo, versionné) et données (volumineuses,
changeantes, jamais versionnées - voir .gitignore) : ce module est le SEUL
endroit du code qui doit connaître où se trouve ce disque. Tout script qui
a besoin d'un chemin sous ce disque importe une constante d'ici plutôt que
d'écrire un chemin en dur.

Résolution de la racine, dans l'ordre :
1. Variable d'environnement PIXELODYSSEY_DATA_DIR, si définie - à utiliser
   quand le disque change de lettre (Windows ne garantit pas qu'une lettre
   reste stable d'un branchement à l'autre) ou pour faire tourner le
   projet sur une autre machine, sans toucher au code.
2. Valeur par défaut ci-dessous, alignée sur la convention actuelle du
   poste de travail - fonctionne tel quel sans rien configurer.

Si la racine résolue n'existe pas, DATA_ROOT lève une erreur explicite à
l'import, plutôt que de laisser chaque script échouer plus loin avec un
FileNotFoundError générique sans rapport apparent avec la vraie cause
(audit du 08/09/2026, voir journal de décisions).
"""
import os
from pathlib import Path

_DEFAULT_DATA_ROOT = r"E:\PixelOdyssey"
_ENV_VAR = "PIXELODYSSEY_DATA_DIR"


def _resolve_data_root() -> Path:
    raw = os.environ.get(_ENV_VAR, _DEFAULT_DATA_ROOT)
    root = Path(raw)
    if not root.exists():
        raise FileNotFoundError(
            f"Disque de données PixelOdyssey introuvable : {root}\n"
            f"- Vérifie que le disque externe est bien connecté et monté à cet emplacement.\n"
            f"- Si son chemin a changé (nouvelle lettre de lecteur, autre machine), définis "
            f"la variable d'environnement {_ENV_VAR} avec le bon chemin plutôt que d'éditer "
            f"le code (ex. sous PowerShell : $env:{_ENV_VAR} = \"F:\\PixelOdyssey\")."
        )
    return root


DATA_ROOT = _resolve_data_root()

# ---------------------------------------------------------------------------
# Sous-dossiers dérivés, alignés sur la convention de dossiers du disque de
# données (voir le rapport de bilan de projet / journal de décisions pour
# le détail de cette arborescence). Un script garde sa propre logique de
# sous-dossier (1_annotated_dataset, 2_split_dataset...) construite à partir
# de ces racines, exactement comme avant ce module - seule la RACINE change
# de source.
# ---------------------------------------------------------------------------
PROCESSED_DATASET_DIR = DATA_ROOT / "3. Processed dataset"
RESULTS_DIR = DATA_ROOT / "4. Results"
