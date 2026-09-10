#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Chargement d'image normalisé (toujours 3 canaux BGR).

Point de passage unique pour charger une image en pixels n'importe où dans
le projet (slicing, inférence par tuile, audit) - ne jamais appeler
cv2.imread() directement ailleurs. Les TIFF (.tif/.tiff) sont lus via
rasterio/GDAL, qui lit la bande principale sans se laisser piéger par des
pages d'aperçu de taille différente (pyramide d'orthomosaïque) ; les autres
formats (jpg, png...) sont lus via cv2.imread.

Constantes :
    _TIFF_EXTS  Extensions routées vers le chargeur rasterio plutôt que cv2.

Exemple :
    from src.data.utils.image_io import load_image_bgr
    img = load_image_bgr("lot/images/photo1.jpg")
    if img is not None:
        h, w = img.shape[:2]

Entrée : chemin vers une image (jpg/png/tif/tiff).
Sortie : tableau numpy (H, W, 3) en BGR uint8, ou None si illisible.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

_TIFF_EXTS = {".tif", ".tiff"}


def _load_tiff_bgr_via_rasterio(img_path: Path) -> Optional[np.ndarray]:
    """Lit un TIFF via rasterio/GDAL plutôt que cv2.imread. Retourne toujours
    3 canaux BGR (comme load_image_bgr), ou None si illisible."""
    import rasterio

    try:
        with rasterio.open(str(img_path)) as src:
            band_count = src.count
            if band_count >= 3:
                data = src.read([1, 2, 3])  # RGB - bandes supplémentaires (alpha, proche-infrarouge) ignorées
            elif band_count == 1:
                data = src.read([1, 1, 1])  # niveaux de gris -> répliqué sur 3 canaux
            else:  # 2 bandes, cas rare (gris + alpha) - la 2e est ignorée, même esprit que le cas BGRA
                band = src.read(1)
                data = np.stack([band, band, band], axis=0)
    except rasterio.errors.RasterioIOError:
        return None

    # rasterio retorne (bandes, H, W) en ordre RGB -> (H, W, bandes) BGR, convention du reste du module.
    rgb = np.transpose(data, (1, 2, 0))
    bgr = rgb[:, :, ::-1].copy()

    if bgr.dtype != np.uint8:
        # Filet de sécurité si une image arrive en 16 bits/flottant plutôt qu'en 8 bits.
        print(
            f"⚠️  [image_io] {img_path} : TIFF en {bgr.dtype} (pas 8 bits) - converti en 8 bits "
            f"par mise à l'échelle min/max de CETTE image (pas une calibration radiométrique)."
        )
        lo, hi = float(bgr.min()), float(bgr.max())
        if hi > lo:
            bgr = ((bgr.astype(np.float32) - lo) / (hi - lo) * 255.0).astype(np.uint8)
        else:
            bgr = np.zeros_like(bgr, dtype=np.uint8)

    return bgr


def load_image_bgr(img_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Charge une image et garantit 3 canaux BGR en sortie, quel que soit le
    nombre de bandes du fichier source. Retourne `None` si le fichier est
    illisible - même contrat que `cv2.imread`, pas de RuntimeError levée pour
    ce cas précis.
    """
    img_path = Path(img_path)
    if img_path.suffix.lower() in _TIFF_EXTS:
        return _load_tiff_bgr_via_rasterio(img_path)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    if img.ndim == 2:
        # Défensif seulement : ne devrait jamais arriver avec IMREAD_COLOR,
        # mais autant le couvrir plutôt que laisser un plantage moins clair
        # plus loin dans la chaîne si un jour ça se produit.
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    n_channels = img.shape[2]
    if n_channels == 4:
        print(
            f"⚠️  [image_io] {img_path} : chargée avec 4 canaux malgré IMREAD_COLOR "
            f"(RGB+alpha probable) - canal excédentaire supprimé, 3 canaux BGR conservés."
        )
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if n_channels != 3:
        raise RuntimeError(
            f"{img_path} : image chargée avec {n_channels} canaux (ni 3 ni 4) - "
            f"format inattendu, à inspecter manuellement avant de continuer."
        )

    return img
