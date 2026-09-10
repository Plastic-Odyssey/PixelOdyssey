#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Inventaire et validation d'un batch de photos drone brutes
(pipeline application). Rôle équivalent à `src/data/raw_dataset_checker.py`
pour l'étape d'entraînement, mais pour ce pipeline : vérifie que chaque photo
est exploitable pour la géolocalisation AVANT de lancer l'inférence dessus
(mieux vaut échouer tôt et clairement qu'un plantage tardif au milieu de
l'inférence sur 1000 photos, ou pire une position silencieusement fausse).

Politique bloquant/non-bloquant (même esprit que raw_dataset_checker.py) :
- Une photo dont les métadonnées EXIF/XMP échouent à l'extraction (GPS
  manquant, constructeur non supporté, capteur inconnu du registre,
  orientation EXIF anormale) est EXCLUE du batch avec une raison explicite -
  pas fatal pour tout le batch, une seule photo corrompue/atypique ne doit
  pas bloquer 999 autres photos exploitables.
- Une photo dont le tangage caméra dépasse la tolérance du capteur
  (`pitch_within_tolerance` - voir geolocation.py) est également EXCLUE par
  défaut (pas juste flaguée) : l'approximation "vue du dessus plate" utilisée
  pour la géolocalisation devient trop risquée au-delà de cette tolérance -
  mieux vaut une position absente qu'une position silencieusement fausse.
- Si AUCUNE photo n'est exploitable au final, erreur explicite (pas un
  batch vide traité comme normal).

Exemple :
    from src.application.inventory import scan_batch
    entries, report = scan_batch("mon_dossier_photos/")
    print(report)
    for e in entries:
        print(e.metadata.path, e.gsd_cm_per_px)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

from PIL import UnidentifiedImageError

from src.application.exif_metadata import PhotoMetadata, extract_photo_metadata
from src.application.geolocation import compute_gsd_cm_per_px, pitch_within_tolerance
from src.application.sensor_config import SensorSpec, load_sensor_registry, resolve_sensor
from src.data.utils.raw_dataset import VALID_IMG_EXTS

# Erreurs anticipées (levées délibérément par exif_metadata.py/sensor_config.py
# pour un motif d'exclusion connu, ou par PIL pour un fichier illisible/corrompu
# malgré son extension .jpg/.png) - toute AUTRE exception remonte telle quelle,
# elle signale un vrai bug plutôt qu'un cas d'exclusion prévu.
_ANTICIPATED_ERRORS = (ValueError, NotImplementedError, KeyError, FileNotFoundError, UnidentifiedImageError)


@dataclass
class InventoryEntry:
    metadata: PhotoMetadata
    sensor: SensorSpec
    gsd_cm_per_px: float


class InventoryReport(NamedTuple):
    n_total_files: int
    n_usable: int
    n_skipped_metadata_error: int
    n_skipped_pitch_tolerance: int
    skipped_details: List[Tuple[str, str]]  # (path, raison)


def scan_batch(
    batch_dir: Union[str, Path], sensor_registry: Optional[dict] = None,
) -> Tuple[List[InventoryEntry], InventoryReport]:
    """Scanne `batch_dir` (non récursif - un batch = un dossier de photos
    d'une même mission, voir run_application.py), extrait et valide les
    métadonnées de chaque photo. Retourne les entrées exploitables + un
    rapport détaillé des exclusions."""
    batch_dir = Path(batch_dir)
    if not batch_dir.exists() or not batch_dir.is_dir():
        raise NotADirectoryError(f"Dossier batch introuvable : {batch_dir}")

    if sensor_registry is None:
        sensor_registry = load_sensor_registry()

    image_paths = sorted(
        p for p in batch_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS
    )

    entries: List[InventoryEntry] = []
    skipped_details: List[Tuple[str, str]] = []
    n_skipped_pitch = 0

    for img_path in image_paths:
        try:
            meta = extract_photo_metadata(img_path)
            sensor = resolve_sensor(meta.make, meta.model, sensor_registry)
        except _ANTICIPATED_ERRORS as e:
            skipped_details.append((str(img_path), str(e)))
            continue

        if not pitch_within_tolerance(meta, sensor):
            reason = (
                f"tangage caméra {meta.gimbal_pitch_deg}° hors tolérance "
                f"(±{sensor.max_pitch_deviation_deg}° au nadir -90°)"
            )
            skipped_details.append((str(img_path), reason))
            n_skipped_pitch += 1
            continue

        gsd = compute_gsd_cm_per_px(meta, sensor)
        entries.append(InventoryEntry(metadata=meta, sensor=sensor, gsd_cm_per_px=gsd))

    n_skipped_metadata_error = len(skipped_details) - n_skipped_pitch

    if not entries:
        raise RuntimeError(
            f"Aucune photo exploitable dans {batch_dir} ({len(image_paths)} fichier(s) "
            f"trouvé(s), {len(skipped_details)} exclu(s)) - voir le détail des exclusions "
            f"avant de continuer, quelque chose d'anormal affecte probablement tout le lot."
        )

    report = InventoryReport(
        n_total_files=len(image_paths),
        n_usable=len(entries),
        n_skipped_metadata_error=n_skipped_metadata_error,
        n_skipped_pitch_tolerance=n_skipped_pitch,
        skipped_details=skipped_details,
    )
    return entries, report
