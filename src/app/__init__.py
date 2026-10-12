"""PixelOdyssey - application locale de bout en bout (photos de drone -> mosaïque -> détections).

Lancement : python -m src.app            (ouvre l'interface dans le navigateur)
            python -m src.app verifier   (contrôle de l'installation)
            python -m src.app traiter --dossier <photos> --modele <id>   (sans interface)

Indépendant du disque de données et des outils de recherche : n'utilise que les photos
fournies, les poids copiés dans models/app/ et quelques briques communes du dépôt
(tuilage, fusion des tuiles, table de poids).
"""
__version__ = "1.1.0"
