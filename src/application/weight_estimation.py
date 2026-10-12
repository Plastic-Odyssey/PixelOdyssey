#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelOdyssey - Estimation du poids d'un déchet détecté, à partir de sa surface
projetée au sol (m²) et de sa super-classe.

Rôle dans le pipeline : appelé par run_application.py (mode batch, surface
déjà en mètres réels via geolocation.py) et src/application/geo_density_map.py
(mode orthomosaïque, surface dérivée du GSD du GeoTIFF) pour enrichir chaque
détection retenue d'un poids estimé (kg), affiché ensuite dans le panneau de
statistiques agrégées des cartes web (voir stats_panel.py).

STATUT (table du 09/10/2026) : coefficients tirés de la LITTÉRATURE et de
fiches produits (sources ci-dessous), PAS encore d'une pesée propre au
projet. CALIBRATED reste donc False : le panneau de statistiques continue
d'afficher « ordre de grandeur ». Remplace la table provisoire du 08/09/2026
(deux coefficients arbitraires, cordages et débris non estimés).

Modèle : masse (kg) = densité surfacique apparente (kg par m² de surface vue
du dessus) × surface projetée (m²), avec une densité propre à chaque
super-classe et une fourchette (basse, centrale, haute).

Pourquoi une relation linéaire (et plus surface^1,5 comme dans la table
provisoire) : les déchets plastiques sont presque tous des objets CREUX ou
MINCES (bouteilles, bidons, cagettes, bouées, tongs, fragments). Leur masse
suit la surface de leur paroi, donc à peu près leur surface projetée, et non
leur volume : une bouteille PET de 1,5 L contient 3 fois plus qu'une 0,5 L
mais ne pèse qu'environ 2 fois plus. C'est aussi la méthode « surface ×
épaisseur × masse volumique » (méthode W3) testée par Gonçalves et al. (2024)
sur des images de drone, qui retombe à ±25 % du poids pesé pour une épaisseur
équivalente de 0,3 à 0,6 cm. Sur leur plage, le rapport poids pesé / surface
segmentée vaut 24,72 kg / 5,23 m² = 4,7 kg/m², tous déchets confondus.

Cordages/filets et débris divers sont désormais estimés, avec des fourchettes
larges assumées :
  - Cordage_Filet : un cordage en polypropylène pèse 33 g/m (Ø 8 mm) à
    121 g/m (Ø 16 mm), soit 4 à 8 kg par m² de cordage vu du dessus pour un
    brin bien étalé. Un enchevêtrement annoté d'un seul contour contient des
    vides (sable visible) mais aussi des couches superposées : 1,5 à 7 kg/m²,
    3,5 au centre. C'est la classe la moins sûre (un même cordage occupe une
    surface très variable selon qu'il est étalé ou roulé en boule).
  - Debris_Divers : moyenne pondérée de ses sous-classes, avec les parts de
    surface réellement annotées dans le dataset (voir DEBRIS_COMPOSITION).
Le modèle mono-classe (« Dechet_Plastique ») reçoit lui aussi une densité :
moyenne des super-classes pondérée par leur part de surface annotée dans le
dataset (voir MONO_COMPOSITION), ce qui permet d'estimer un poids même sans
le type de déchet.

Limites connues : objets secs (un filet ou un tissu mouillé pèse 2 à 3 fois
plus), objets entiers et posés à plat ; la densité centrale d'une classe
dépend de la composition du dataset au 09/10/2026 (à recalculer si elle
change beaucoup). La seule vraie validation sera une pesée terrain : peser
les déchets ramassés sur une zone photographiée et comparer au total estimé
(voir CALIBRATED).

Sources (consultées le 09/10/2026) :
  [UAV24] Gonçalves G. et al., 2024. Marine litter weight estimation from UAV
          imagery: three potential methodologies to advance macrolitter
          reports. Marine Pollution Bulletin 202, 116405.
  [ROPE]  SpanSet, cordage polypropylène tressé : 3,34 / 4,26 / 6,16 / 8,66 /
          12,1 kg par 100 m pour Ø 8 / 10 / 12 / 14 / 16 mm.
  [PET]   SIPA d'après IBWA : bouteille PET 0,5 L, 16,9 g (standard) à 9,89 g
          (allégée).
  [HDPE]  ALPLA : flacon PEHD 1 L, 83 g ; bidon PEHD 5 L, 220 g.
  [CRATE] Transoplast : caisse à poisson 600 × 400 × 225 mm, 2,54 kg.
  [BUOY]  Polyform série A : A0 Ø 210 × 280 mm 0,60 kg ; A1 Ø 295 × 380 mm
          1,15 kg ; A2 Ø 390 × 500 mm 2,10 kg.
  [FLIP]  Tongs caoutchouc type Havaianas : 185 à 200 g la paire.
  [BAG]   Sac de caisse PEHD : 5 à 10 g, film de 13 à 25 µm.
  [EPS]   Polystyrène expansé : 15 à 30 kg/m³ ; fragments plastiques rigides
          ~1 g/cm³ (Ryan et al., 2020, Frontiers in Marine Science 7:575395).
Les surfaces projetées des objets de référence (ex. bouteille 0,5 L couchée
≈ 6,5 × 21 cm) sont des dimensions usuelles supposées, pas des mesures.

Exemple :
    from src.application.weight_estimation import estimate_weight_kg, estimate_weight_range_kg
    estimate_weight_kg("Bouteille", area_m2=0.018)        # ≈ 0,027 kg
    estimate_weight_range_kg("Cordage_Filet", 0.25)        # (0,375, 0,875, 1,75) kg
"""

from typing import Dict, Optional, Tuple

# Passer à True le jour où la table est validée par une pesée terrain propre au
# projet (voir docstring) - lu par stats_panel.py pour afficher ou non
# l'avertissement « ordre de grandeur » à côté du total.
CALIBRATED = False
TABLE_VERSION = "litterature_2026-10-09"

Range = Tuple[float, float, float]  # (basse, centrale, haute), en kg par m² de surface projetée

# Familles physiques (gardées pour les appelants existants et pour la lecture).
FAMILY_FLAT_THIN = "flat_thin"        # objets plats et minces (tongs, fragments, films)
FAMILY_COMPACT_3D = "compact_3d"      # contenants creux (bouteilles, bidons, cagettes, bouées)
FAMILY_TANGLED = "tangled"            # cordages et filets enchevêtrés
FAMILY_MIXED = "mixed"                # classes composites (débris divers, mono-classe)
FAMILY_NOT_ESTIMABLE = "not_estimable"  # classe inconnue de la table


def _weighted(parts: Dict[str, Tuple[float, Range]]) -> Range:
    """Moyenne pondérée de fourchettes (poids = part de surface)."""
    total = sum(share for share, _ in parts.values())
    return tuple(round(sum(share * rng[i] for share, rng in parts.values()) / total, 2) for i in range(3))


# Bouteille : PET (76 % de la surface annotée) et plastique rigide (24 %).
#   PET 0,5 L : 9,9-16,9 g [PET] pour ~0,0123 m² couchée -> 0,8-1,4 kg/m².
#   Flacon PEHD 1 L : 40-83 g [HDPE] pour ~0,019 m² -> 2,1-4,4 kg/m².
BOTTLE_COMPOSITION = {
    "bouteille PET": (0.76, (0.6, 1.1, 1.6)),
    "bouteille plastique rigide": (0.24, (1.8, 2.8, 4.4)),
}

# Debris_Divers : parts de surface annotée (1_annotated_dataset, 09/10/2026).
#   Fragments rigides : 1-3,5 mm de paroi à ~1 g/cm³ [EPS] -> 1-3,5 kg/m².
#   Sacs et films : 5-10 g pour ~0,17 m² à plat [BAG], froissés sur la plage -> 0,05-0,5.
#   Tissu : 0,2-1,5 kg/m² (sec à mouillé). Polystyrène : 15-30 kg/m³ [EPS] sur 2-8 cm -> 0,3-2,5.
#   Bouchons : 2-3 g pour ~7 cm² -> 2,5-4,5 kg/m².
DEBRIS_COMPOSITION = {
    "fragments plastique rigide": (0.61, (1.0, 2.0, 3.5)),
    "plastique souple (sac)": (0.28, (0.05, 0.2, 0.5)),
    "tissu": (0.06, (0.2, 0.5, 1.5)),
    "polystyrene": (0.04, (0.3, 1.0, 2.5)),
    "bouchons": (0.01, (2.5, 3.5, 4.5)),
}

AREAL_DENSITY_KG_M2: Dict[str, Range] = {
    "Bouteille": _weighted(BOTTLE_COMPOSITION),   # ≈ (0,9 ; 1,5 ; 2,3)
    # Bidon (91 % petits bidons, 9 % seaux) : bidon PEHD 5 L, 220 g [HDPE] pour ~0,053 m²
    # couché -> 4,1 kg/m² ; les grands bidons et les seaux ont des parois plus épaisses.
    "Bidon": (3.0, 4.5, 8.0),
    # Cagette : caisse à poisson 600 × 400 mm, 2,54 kg [CRATE] -> 10,6 kg/m² ; cagettes
    # plus légères et morceaux cassés plus bas.
    "Cagette": (4.0, 7.0, 11.0),
    # Bouee (93 % de flotteurs de filet) : petits flotteurs ~20 g [UAV24] pour ~0,0045 m² ->
    # ~4,4 kg/m² ; bouées gonflables Polyform A0-A2 [BUOY] -> ~13 kg/m².
    "Bouee": (4.0, 7.0, 13.0),
    # Flipflops : ~95 g la tong [FLIP] pour ~0,02 m² de semelle -> ~4,7 kg/m² ; EVA plus léger.
    "Flipflops": (3.0, 4.5, 5.5),
    # Cordage_Filet : voir docstring [ROPE] - classe la moins sûre.
    "Cordage_Filet": (1.5, 3.5, 7.0),
    "Debris_Divers": _weighted(DEBRIS_COMPOSITION),  # ≈ (0,7 ; 1,4 ; 2,5)
}

# Modèle mono-classe : parts de surface annotée des 7 super-classes (dataset du 09/10/2026).
MONO_COMPOSITION = {
    "Cordage_Filet": 0.354, "Debris_Divers": 0.236, "Bouteille": 0.156, "Cagette": 0.107,
    "Bidon": 0.076, "Flipflops": 0.038, "Bouee": 0.033,
}
AREAL_DENSITY_KG_M2["Dechet_Plastique"] = _weighted(
    {k: (share, AREAL_DENSITY_KG_M2[k]) for k, share in MONO_COMPOSITION.items()})  # ≈ (1,7 ; 3,3 ; 5,9)

CLASS_WEIGHT_FAMILY: Dict[str, str] = {
    "Bouteille": FAMILY_COMPACT_3D,
    "Bidon": FAMILY_COMPACT_3D,
    "Bouee": FAMILY_COMPACT_3D,
    "Cagette": FAMILY_COMPACT_3D,
    "Flipflops": FAMILY_FLAT_THIN,
    "Cordage_Filet": FAMILY_TANGLED,
    "Debris_Divers": FAMILY_MIXED,
    "Dechet_Plastique": FAMILY_MIXED,
}

_BOUND_INDEX = {"low": 0, "central": 1, "high": 2}


def weight_family(class_name: str) -> str:
    """Famille physique associée à `class_name` - FAMILY_NOT_ESTIMABLE si la
    classe n'est pas dans la table (jamais une supposition silencieuse)."""
    return CLASS_WEIGHT_FAMILY.get(class_name, FAMILY_NOT_ESTIMABLE)


def estimate_weight_range_kg(class_name: str, area_m2: Optional[float]) -> Optional[Range]:
    """Fourchette (basse, centrale, haute) du poids en kg. None si la surface est
    absente ou nulle, ou si la classe n'est pas dans la table."""
    if not area_m2 or area_m2 <= 0:
        return None
    rng = AREAL_DENSITY_KG_M2.get(class_name)
    if rng is None:
        return None
    return tuple(d * area_m2 for d in rng)


def estimate_weight_kg(class_name: str, area_m2: Optional[float], bound: str = "central") -> Optional[float]:
    """Poids estimé (kg) d'une détection, valeur centrale par défaut (`bound` =
    "low" ou "high" pour les bornes). Retourne None si la surface est absente
    ou si la classe n'est pas dans la table - à traiter côté appelant comme
    « détection non incluse dans le total pesé », jamais comme 0 kg (voir
    stats_panel.compute_aggregate_stats)."""
    rng = estimate_weight_range_kg(class_name, area_m2)
    if rng is None:
        return None
    return rng[_BOUND_INDEX[bound]]


if __name__ == "__main__":
    print(f"Table {TABLE_VERSION} (kg par m² de surface projetée : basse / centrale / haute)")
    for name, (lo, mid, hi) in AREAL_DENSITY_KG_M2.items():
        print(f"  {name:18s} {lo:5.2f}  {mid:5.2f}  {hi:5.2f}   famille {weight_family(name)}")
