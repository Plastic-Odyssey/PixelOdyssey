#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Chargement d'image normalisé (toujours 3 canaux BGR).

Pourquoi ce module existe (bug réel, 23/08/2026) :
---------------------------------------------------
`cv2.imread(path)` utilise par défaut le flag `cv2.IMREAD_COLOR`, censé
TOUJOURS forcer une image en 3 canaux BGR (alpha supprimé) - vrai de façon
fiable pour JPEG/PNG, mais PAS pour certains TIFF multi-bandes selon la
version d'OpenCV/libtiff installée. Or `1_annotated_dataset` accepte belle et
bien `.tif`/`.tiff` comme extension d'image parente valide (voir
`raw_dataset.VALID_IMG_EXTS`) - PAS uniquement les .jpg convertis à la main,
contrairement à ce qu'on pensait au départ. Les orthomosaïques/exports drone
en TIFF ont couramment 4 bandes (RGB + alpha, ou RGB + proche-infrarouge).

Symptôme observé : `label_review.py` a planté avec `Given groups=1, weight of
size [16, 3, 3, 3], expected input[1, 4, 640, 640] to have 3 channels, but got
4 channels instead` - une image .tif à 4 bandes chargée telle quelle,
envoyée directement au modèle (qui attend toujours 3 canaux, peu importe le
format source de l'entraînement).

Pourquoi l'entraînement lui-même n'avait pas planté sur ce même souci :
`slicer.py` (étape 4) écrit ses tuiles en `.png` via `cv2.imwrite` - si
l'image source faisait 4 canaux, la tuile ÉCRITE sur disque en aurait aussi 4
(un PNG supporte l'alpha, `cv2.imwrite` n'y voit rien à corriger). Mais quand
Ultralytics relit ensuite ce PNG DEPUIS LE DISQUE pendant l'entraînement, son
propre chargeur ramène fiablement un PNG à 3 canaux (PNG n'a pas le même
comportement erratique que TIFF sous OpenCV) - le problème passait donc
inaperçu, sauf ici où l'image est utilisée EN MÉMOIRE, directement, sans
jamais repasser par un fichier PNG intermédiaire.

Deuxième bug corrigé (25/08/2026) - TIFF illisible même après le correctif
ci-dessus, sur une VRAIE orthomosaïque (`SL 28-30 avt.tif`, 187 Mo) via
`bootstrap_annotate.py` :
    ValueError: all input arrays must have the same shape
    ... ultralytics/utils/patches.py, in imread
        return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
Cause racine, pas un bug OpenCV cette fois : sous Windows, `ultralytics`
remplace purement et simplement `cv2.imread` par sa propre implémentation dès
qu'il est importé (`cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite,
imshow` dans `ultralytics/utils/__init__.py`, réservé à Windows - support des
chemins non-ASCII). Cette version maison lit un `.tif`/`.tiff` avec
`cv2.imdecodemulti` (pensé pour les TIFF MULTI-PAGES, ex: scans) et empile
toutes les pages trouvées avec `np.stack`. Une orthomosaïque WebODM comme
celle-ci embarque typiquement des vignettes basse résolution en plus de
l'image pleine résolution (pyramide d'aperçus) - des "pages" de tailles
DIFFÉRENTES, que `np.stack` ne peut pas empiler -> crash immédiat. Comme ce
patch est appliqué globalement dès `from ultralytics import YOLO` (déclenché
ici par `make_ultralytics_predict_fn`), N'IMPORTE QUEL appel à `cv2.imread`
ailleurs dans le processus - y compris celui-ci - hérite du même risque dès
qu'une image source est un TIFF pyramidal, pas seulement dans
`bootstrap_annotate.py`. Les TIFF déjà traités sans souci jusqu'ici
(`transect_11.tif` et consorts, ~8 Mo, un export par transect) n'ont
simplement jamais eu cette structure pyramidale - une pleine orthomosaïque de
187 Mo (`SL 28-30 avt.tif`) si.

Correctif : pour tout `.tif`/`.tiff`, ce module ne passe PLUS par
`cv2.imread` du tout (donc jamais exposé au patch Windows d'Ultralytics) -
lecture via `rasterio` à la place, déjà une dépendance du projet
(`geo_density_map.py` l'utilise avec succès sur ce même fichier, justement
parce qu'il évite `cv2.imread` pour cette raison de RAM/fenêtrage - même
bénéfice obtenu ici gratuitement). `rasterio`/GDAL lit la bande PRINCIPALE
sans se laisser piéger par des pages d'aperçu de taille différente. JPEG/PNG
restent chargés via `cv2.imread` comme avant (jamais concernés par ce bug -
pas de notion de pages/pyramide dans ces formats).

Ce module est donc le point de passage UNIQUE pour charger une image en
pixels n'importe où dans le projet (slicing à l'entraînement, inférence par
tuile, audit) - jamais `cv2.imread()` directement dans ces contextes, pour ne
pas réintroduire l'un de ces deux bugs ailleurs à la faveur d'un futur format
d'entrée.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

_TIFF_EXTS = {".tif", ".tiff"}


def _load_tiff_bgr_via_rasterio(img_path: Path) -> Optional[np.ndarray]:
    """Lit un TIFF via rasterio/GDAL plutôt que cv2.imread - voir la
    docstring du module (25/08/2026) pour pourquoi cv2.imread est risqué sur
    un TIFF pyramidal sous Windows avec Ultralytics importé. Retourne
    toujours 3 canaux BGR (comme load_image_bgr), ou None si illisible."""
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

    # rasterio retourne (bandes, H, W) en ordre RGB -> (H, W, bandes) BGR, convention du reste du module.
    rgb = np.transpose(data, (1, 2, 0))
    bgr = rgb[:, :, ::-1].copy()

    if bgr.dtype != np.uint8:
        # Rare pour les orthomosaïques 8 bits de ce projet (voir le bug 4-canaux du 23/08/2026,
        # déjà tous en 8 bits) - filet de sécurité si un futur lot arrive en 16 bits/flottant.
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
    illisible - même contrat que `cv2.imread` (à l'appelant de décider quoi
    faire, voir les `if img is None: ...` déjà en place partout où c'est
    appelé), pas de RuntimeError levée pour ce cas précis.
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
