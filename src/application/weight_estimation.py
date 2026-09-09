#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Estimation PROVISOIRE du poids d'un déchet détecté, à partir
de sa surface projetée au sol (m²) et de sa super-classe.

Rôle dans le pipeline : appelé par run_application.py (mode batch, surface
déjà en mètres réels via geolocation.py) et src/review/geo_density_map.py
(mode orthomosaïque, surface dérivée du GSD du GeoTIFF) pour enrichir chaque
détection retenue d'un poids estimé (kg), affiché ensuite dans le panneau de
statistiques agrégées des deux cartes web (voir stats_panel.py).

STATUT : question ouverte - AUCUNE table de conversion surface->poids
calibrée n'existe encore. La calibration réelle (jointure aire<->poids par
famille physique d'objet à partir d'une table de pesée réelle) reste à
faire - aucun coefficient par classe n'a jamais été mesuré à ce jour. Ce
module fournit donc une ESTIMATION D'ORDRE DE GRANDEUR, avec des coefficients
choisis arbitrairement (PAS mesurés), à remplacer entièrement dès que la
calibration réelle sera disponible : voir CALIBRATED ci-dessous (à passer à
True une fois fait) et les deux constantes de coefficient, qui devront très
probablement devenir une table PAR CLASSE plutôt que 2 constantes globales
partagées par famille.

Modèle physique (tranché) : PAS une formule unique pour toutes les classes -
la relation entre surface projetée et masse dépend de la famille physique de
l'objet :
  - objets plats/fins (ex: une semelle de flip-flop) : masse ~ proportionnelle
    à la surface (aire ~ L², l'épaisseur ne varie presque pas d'un exemplaire
    à l'autre).
  - objets compacts 3D (bouteille, bidon, bouée, cagette) : masse ~
    proportionnelle à surface^1.5 (aire ~ L², volume ~ L³, donc masse ~
    aire^(3/2) en supposant une masse volumique apparente à peu près
    constante au sein de la famille).
  - objets allongés/enroulés (cordage/filet) : la surface projetée est un
    MAUVAIS prédicteur - un même cordage enroulé occupe une surface très
    variable selon la façon dont il est posé au sol. AUCUNE estimation n'est
    retournée pour cette famille (voir FAMILY_NOT_ESTIMABLE) plutôt que
    d'afficher un chiffre qui semble précis mais n'a aucune base physique.

Debris_Divers (super-classe fourre-tout) mélange plusieurs familles
physiques à la fois (sacs plats, morceaux rigides, objets compacts,
indéterminé) : impossible de savoir quelle formule appliquer à une détection
de cette classe sans un sous-type précis, que le modèle super-classe ne
prédit justement pas. Traitée en FAMILY_NOT_ESTIMABLE pour la même raison que
Cordage_Filet - extension du même principe, pas une décision séparée.

Exemple :
    from src.application.weight_estimation import estimate_weight_kg
    poids_kg = estimate_weight_kg("Bouteille", area_m2=0.018)  # ordre de grandeur, ou None
"""

from typing import Dict, Optional

# Passer à True le jour où les coefficients ci-dessous sont remplacés par une
# vraie table calibrée (voir docstring du module) - sert de garde-fou lu par
# stats_panel.py pour décider si le panneau affiche l'avertissement "poids
# non calibré" à côté du total agrégé.
CALIBRATED = False

FAMILY_FLAT_THIN = "flat_thin"
FAMILY_COMPACT_3D = "compact_3d"
FAMILY_NOT_ESTIMABLE = "not_estimable"

# Famille physique par super-classe (voir docstring du module pour le
# raisonnement détaillé) - spécifique aux 7 super-classes actuelles de
# config/data_config.yaml, PAS une classification universelle. Si une classe
# est ajoutée/renommée là-bas sans mise à jour ici, `weight_family` la traite
# en FAMILY_NOT_ESTIMABLE par défaut (jamais une classe inconnue assimilée en
# silence à une famille physique qu'elle n'a pas forcément - voir plus bas).
CLASS_WEIGHT_FAMILY: Dict[str, str] = {
    "Bouteille": FAMILY_COMPACT_3D,
    "Bidon": FAMILY_COMPACT_3D,
    "Bouee": FAMILY_COMPACT_3D,
    "Cagette": FAMILY_COMPACT_3D,
    "Flipflops": FAMILY_FLAT_THIN,
    "Cordage_Filet": FAMILY_NOT_ESTIMABLE,
    "Debris_Divers": FAMILY_NOT_ESTIMABLE,
}

# Coefficients ARBITRAIRES (voir CALIBRATED ci-dessus), choisis seulement
# pour donner un ordre de grandeur plausible (quelques dizaines de grammes
# pour une bouteille, quelques centaines pour un bidon) - PAS mesurés sur la
# table de pesée réelle du projet. masse_kg = coefficient * aire_m2^exposant.
_COMPACT_3D_COEFF_KG = 20.0   # exposant 1.5 (masse ~ volume ~ aire^1.5)
_FLAT_THIN_COEFF_KG = 0.3     # exposant 1.0 (masse surfacique, kg/m²)


def weight_family(class_name: str) -> str:
    """Famille physique associée à `class_name` (voir CLASS_WEIGHT_FAMILY) -
    FAMILY_NOT_ESTIMABLE par défaut si la classe n'est pas répertoriée
    (jamais une supposition silencieuse, voir docstring du module)."""
    return CLASS_WEIGHT_FAMILY.get(class_name, FAMILY_NOT_ESTIMABLE)


def estimate_weight_kg(class_name: str, area_m2: Optional[float]) -> Optional[float]:
    """Estimation d'ordre de grandeur du poids (kg) d'une détection, à partir
    de sa surface projetée au sol (m²) et de sa super-classe. Retourne None
    si `area_m2` est absente/nulle OU si la classe appartient à
    FAMILY_NOT_ESTIMABLE (voir docstring du module) - à traiter côté
    appelant comme "détection non incluse dans le total pesé", jamais comme
    un poids de 0 kg (voir stats_panel.compute_aggregate_stats)."""
    if not area_m2 or area_m2 <= 0:
        return None
    family = weight_family(class_name)
    if family == FAMILY_COMPACT_3D:
        return _COMPACT_3D_COEFF_KG * (area_m2 ** 1.5)
    if family == FAMILY_FLAT_THIN:
        return _FLAT_THIN_COEFF_KG * area_m2
    return None
