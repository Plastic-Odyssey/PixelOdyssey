"""
PixelOdyssey - Core Training Module.
Initialise le modèle YOLOv11-seg et orchestre le fine-tuning sur le matériel local.
"""

import os
from ultralytics import YOLO

def launch_training():
    print("--- 🏋️ INITIALISATION DE L'ENTRAÎNEMENT PIXELODYSSEY ---")
    
    # 1. Sélection de l'architecture de pointe (YOLOv11-seg version Nano)
    # Le modèle sera automatiquement téléchargé lors du premier run
    model_architecture = "yolo11n-seg.pt"
    model = YOLO(model_architecture)
    
    # 2. Définition du chemin vers notre fichier de configuration des données
    config_path = os.path.join("config", "data_config.yaml")
    
    print(f"Modèle chargé : {model_architecture}")
    print(f"Configuration cible : {config_path}")
    print("Démarrage du fitting...")

    # 3. Lancement de la boucle d'entraînement avec hyperparamètres de contrôle
    # On configure des valeurs minimales pour valider le pipeline sans crash
    results = model.train(
        data=config_path,      # Fichier de config YAML
        epochs=3,              # 3 époques suffisent pour valider que le code fonctionne
        imgsz=640,             # Résolution standard d'entraînement YOLO
        batch=8,               # Taille du batch (8 images par pas pour préserver la VRAM)
        device=0,              # Force l'utilisation du premier GPU Nvidia (met 'cpu' si pas de GPU)
        workers=2,             # Nombre de threads pour charger les images sans saturer le CPU Windows
        project="runs/train",  # Dossier où seront sauvegardés tes graphiques et tes poids 'best.pt'
        name="pixel_odyssey_v1",
        plots=True             # Génère automatiquement les courbes Precision-Recall et les pertes
    )
    
    print("--- ✅ ENTRAÎNEMENT DE VALIDATION TERMINÉ ---")
    print("Les résultats et les poids du modèle sont dans : runs/train/pixel_odyssey_v1/")

if __name__ == "__main__":
    launch_training()