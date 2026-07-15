#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Pipeline de Préparation des Données
Orchestrateur global : Scanne les exports CVAT bruts et accumule les tuiles
dans le dataset d'entraînement général.
"""

import os
import sys

# Ajout du chemin racine pour l'import de slicer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.slicer import PlasticImageSlicer

# --- CONFIGURATION DES CHEMINS (Sur ton disque E:) ---
BASE_DIR = r"E:\PixelOdyssey\dataset"
RAW_INPUTS_DIR = os.path.join(BASE_DIR, "raw_inputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_dataset")

def run_slicing_pipeline():
    # 1. Initialisation du Slicer avec tes paramètres validés
    slicer = PlasticImageSlicer(
        tile_size=640,
        overlap=256,
        discard_truncated=True
    )

    # Vérification que le dossier source existe
    if not os.path.exists(RAW_INPUTS_DIR):
        print(f"[ERREUR] Le dossier des exports bruts n'existe pas : {RAW_INPUTS_DIR}")
        return

    # 2. Scan des campagnes d'export (ex: SL_6-10_26-27)
    campaigns = [d for d in os.listdir(RAW_INPUTS_DIR) if os.path.isdir(os.path.join(RAW_INPUTS_DIR, d))]
    print(f"[INFO] {len(campaigns)} campagne(s) trouvée(s) dans raw_inputs : {campaigns}")

    for campaign in campaigns:
        print(f"\n--- Traitement de la campagne : {campaign} ---")
        campaign_path = os.path.join(RAW_INPUTS_DIR, campaign)

        # On traite séparément 'train' et 'val' pour respecter ton split d'origine
        for split in ["train", "val"]:
            img_src_dir = os.path.join(campaign_path, "images", split)
            label_src_dir = os.path.join(campaign_path, "labels", split)

            # Cibles globales dans processed_dataset
            img_dst_dir = os.path.join(PROCESSED_DIR, "images", split)
            label_dst_dir = os.path.join(PROCESSED_DIR, "labels", split)

            # Sécurité : Si le dossier d'images de cette campagne existe
            if os.path.exists(img_src_dir):
                os.makedirs(img_dst_dir, exist_ok=True)
                os.makedirs(label_dst_dir, exist_ok=True)

                # Liste des images de la campagne
                images = [f for f in os.listdir(img_src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_name in images:
                    base_name, _ = os.path.splitext(img_name)
                    img_path = os.path.join(img_src_dir, img_name)
                    
                    # On cherche le label correspondant (.txt)
                    label_name = f"{base_name}.txt"
                    label_path = os.path.join(label_src_dir, label_name)

                    # Le préfixe évite les collisions de noms (ex: SL_6-10_26-27_transect_A)
                    prefix = f"{campaign}_{base_name}"

                    # Appel du découpage
                    slicer.slice_single_pair(
                        img_path=img_path,
                        label_path=label_path,
                        output_img_dir=img_dst_dir,
                        output_label_dir=label_dst_dir,
                        prefix=prefix
                    )
            else:
                print(f"[WARN] Pas de dossier '{split}' trouvé pour la campagne {campaign}.")

    print("\n[SUCCÈS] Pipeline de slicing terminé ! Toutes tes tuiles sont cumulées dans processed_dataset.")

if __name__ == "__main__":
    run_slicing_pipeline()