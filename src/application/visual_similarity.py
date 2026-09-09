#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Similarité visuelle pré-entraînée, pour le dédoublonnage
inter-photos quand la géométrie reprojetée ne suffit plus (voir la limite de
l'IoU seul en tête de src/application/dedup.py).

Modèle choisi : DINOv2 (Meta, "facebook/dinov2-small" via transformers) -
PAS CLIP. Raisonnement (tranché) : CLIP est entraîné pour l'alignement
image/texte, ce qui n'est pas notre besoin ici (comparer deux images entre
elles, jamais de texte) ; DINOv2 est entraîné en auto-supervisé PUREMENT sur
des images, avec un objectif qui pousse spécifiquement à des embeddings
discriminants pour de la similarité/reconnaissance d'instance (retrieval),
exactement notre cas d'usage ("ces deux chips montrent-ils le même objet
réel ?"). La variante "small" (~22M paramètres) suffit ici : on ne classifie
rien de fin, on compare des paires - pas besoin de la variante "large".

Validation empirique rapide (question ouverte, à confirmer sur plus
d'exemples) : sur quelques vues confirmées manuellement d'un même objet,
similarité cosinus 0.55-0.76 entre elles, contre 0.40-0.44 pour un objet
visuellement différent dans le même voisinage. Marge réelle mais pas
énorme - PAS un signal "parfait", à utiliser en complément d'un filtre de
proximité (voir dedup.py), jamais seul sur tout le batch (comparer une image
à une autre à l'autre bout du site n'a de toute façon aucun sens).

Limite connue, pas encore corrigée : l'embedding est calculé sur le chip
ENTIER (objet + marge de fond, voir run_application.py::CROP_MARGIN_PX) - le
fond (sable, végétation) contribue autant que l'objet lui-même au vecteur, ce
qui dilue le signal. Piste d'amélioration si la marge de discrimination
s'avère insuffisante sur plus d'exemples : masquer le fond (ne garder que les
pixels à l'intérieur du contour du masque) avant de calculer l'embedding.

Coût : ~10s de chargement du modèle (une fois, mis en cache par transformers
après le premier appel), puis quelques dizaines de ms par image sur GPU comme
sur CPU (modèle "small") - négligeable comparé à l'inférence YOLO elle-même.

Exemple :
    from src.application.visual_similarity import load_similarity_model, compute_embeddings
    processor, model = load_similarity_model()
    embeddings = compute_embeddings(list_of_pil_images, processor, model)
    similarity = float(embeddings[0] @ embeddings[1])  # cosinus, embeddings déjà normalisés L2
"""

from typing import List, Tuple

import numpy as np
from PIL import Image

DEFAULT_MODEL_NAME = "facebook/dinov2-small"


def load_similarity_model(model_name: str = DEFAULT_MODEL_NAME):
    """Charge le processeur + modèle DINOv2 (téléchargé et mis en cache
    localement par `transformers` au premier appel - nécessite un accès
    réseau une seule fois). Importé ici (pas en tête de module) pour que
    `torch`/`transformers` restent des dépendances OPTIONNELLES : un usage de
    dedup.py en IoU seul (`deduplicate_detections`, sans le "_visual") n'en a
    jamais besoin."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def compute_embeddings(images: List[Image.Image], processor, model) -> np.ndarray:
    """Calcule un embedding L2-normalisé par image (token CLS de la dernière
    couche) - normalisé pour qu'un simple produit scalaire entre deux lignes
    donne directement leur similarité cosinus (voir dedup.py, qui fait
    exactement ça). Traite les images une par une plutôt qu'en un seul batch
    empilé : les chips n'ont pas tous la même taille (voir
    run_application.py::_build_crops, redimensionnement variable selon la
    taille du masque) - le `processor` les redimensionne individuellement à
    la taille attendue par le modèle de toute façon, empiler introduirait une
    complexité de padding pour un gain de vitesse négligeable ici (des
    centaines à quelques milliers d'images, pas des millions)."""
    import torch

    if not images:
        return np.zeros((0, 384), dtype=np.float32)  # 384 = dimension DINOv2-small

    embeddings = []
    with torch.no_grad():
        for img in images:
            inputs = processor(images=img.convert("RGB"), return_tensors="pt")
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0].numpy()[0]
            norm = np.linalg.norm(emb)
            embeddings.append(emb / norm if norm > 0 else emb)
    return np.array(embeddings, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus entre deux embeddings déjà normalisés L2 (simple
    produit scalaire) - fonction utilitaire pour un usage ponctuel/de
    diagnostic ; dedup.py fait ce calcul inline pour éviter l'overhead d'un
    appel de fonction sur potentiellement des millions de paires candidates."""
    return float(np.dot(a, b))
