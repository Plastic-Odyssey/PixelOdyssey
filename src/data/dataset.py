"""
PixelOdyssey - Data Pipeline Module.
Gère la vérification et la conformité du dataset de segmentation pour Plastic Odyssey.
"""

import os
from pathlib import Path

class PlasticDatasetChecker:
    def __init__(self, dataset_path: str):
        """
        Initialise le vérificateur de dataset.
        :param dataset_path: Chemin vers la racine du dossier contenant images/ et labels/
        """
        self.dataset_path = Path(dataset_path)
        self.images_train = self.dataset_path / "images" / "train"
        self.labels_train = self.dataset_path / "labels" / "train"

    def verify_splits(self) -> bool:
        """
        Vérifie la cohérence du dossier d'entraînement. 
        S'assure que chaque image possède bien son fichier de label .txt associé.
        """
        if not self.images_train.exists() or not self.labels_train.exists():
            print("❌ Erreur : L'arborescence cible du dataset est vide ou incomplète.")
            return False
            
        # Extraction des noms de fichiers sans l'extension
        images = sorted([f.stem for f in self.images_train.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        labels = sorted([f.stem for f in self.labels_train.glob("*.txt")])
        
        # Détection des images orphelines (sans fichier de coordonnées .txt)
        orphans = set(images) - set(labels)
        
        if orphans:
            print(f"⚠️ Alerte Terrain : {len(orphans)} image(s) n'ont pas de fichier .txt de labellisation !")
            print(f"Exemples d'images orphelines à corriger : {list(orphans)[:3]}")
            return False
            
        if len(images) == 0:
            print("📁 Le dossier est prêt mais actuellement vide (en attente du dataset final).")
            return True
            
        print(f"✅ Cohérence parfaite : {len(images)} images alignées avec {len(labels)} labels.")
        return True

if __name__ == "__main__":
    # Point d'entrée pour tester le script localement
    # On cible le dossier que nous avons créé au format Windows
    base_path = os.path.join("datasets", "plastic_dataset")
    checker = PlasticDatasetChecker(base_path)
    checker.verify_splits() 