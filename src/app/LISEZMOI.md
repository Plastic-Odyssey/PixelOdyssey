# Application PixelOdyssey

Photos de drone d'une plage → mosaïque géoréférencée → déchets plastiques détectés, comptés une seule
fois, mesurés et pesés (ordre de grandeur), sur une carte.

## Lancer

- Double-clic sur `Lancer_PixelOdyssey.bat` à la racine du dépôt (cherche tout seul l'environnement conda
  du projet ; sinon renseigner `ENV_NAME` dans le fichier), ou depuis le terminal VS Code
  (environnement du projet actif) : `python -m src.app`
- Le navigateur s'ouvre sur `http://127.0.0.1:8765/`. Laisser la fenêtre noire ouverte pendant l'utilisation.
- Contrôle de l'installation : `python -m src.app verifier`
- Sans interface : `python -m src.app traiter --dossier "E:\...\photos" --modele mono_20261008`

## Utiliser

1. **Photos du vol** : choisir le dossier (ou déposer les fichiers). Il faut les JPG d'origine du drone
   (les PNG n'ont plus les métadonnées GPS). La carte montre l'emprise de chaque photo et signale les
   photos isolées, non verticales ou sans GPS.
2. **Modèle** : « Déchets plastiques (1 classe) » pour compter et mesurer, « Types de déchets (7 classes) »
   pour la répartition par type. Le dossier des résultats est proposé à côté du dossier des photos.
3. **Traitement** : la progression s'affiche ; la mosaïque apparaît sur la carte dès qu'elle est prête.
4. **Résultats** : chiffres clés, seuil de confiance réglable, détail par type, fiche de chaque déchet
   (clic sur un contour, avec une vue de la photo d'origine), téléchargements.

Un vol déjà traité se rouvre instantanément (« Ouvrir des résultats… »). Relancer avec un autre modèle
réutilise l'alignement et la mosaïque.

## Ce qui se passe

- Les photos sont recalées entre elles par points homologues puis alignées ensemble (moindres carrés robustes),
  ancrées sur le GPS : précision de 2 à 5 cm entre photos voisines. Le biais de cap commun du drone (souvent
  5 à 8°), l'échelle réelle et les petites déformations de chaque photo (relief) sont estimés au passage.
  Des garde-fous écartent tout alignement aberrant (repli sur un alignement simplifié, ou le GPS seul, signalé
  dans les résultats).
- Détection sur les photos d'origine (pas sur la mosaïque), GPU si disponible. Chaque zone du sol est attribuée
  à la photo qui la voit le plus près de son centre : ses détections sont gardées, un déchet n'est compté qu'une
  fois, et la mosaïque est cousue aux mêmes endroits.
- Fusion des vues (v1.1) : un déchet est vu par 4 à 6 photos et le modèle ne le reconnaît pas sur toutes. Ce que
  les autres photos trouvent là où la photo principale n'a rien vu est regroupé et « rattrapé » si au moins deux
  vues le confirment (ou une seule avec une confiance double). Sur W5 : 7 % -> 30 % des déchets annotés retrouvés
  à précision égale ; sur une photo de test de W1 : +2 points, sans faux positifs en plus.
- Photos ternes (ciel voilé, sable surexposé : écart-type de luminance < 28) : contraste local renforcé avant
  détection. Sur W5 : 7 % -> 45 % ; jamais appliqué aux vols normaux (sur W1 il ferait perdre 9 points).
- Réglages : `DETECTION_MODE` et `CONTRAST_ENHANCE` dans `src/app/config.py`.

## Résultats écrits (dossier `<photos>_pixelodyssey`)

`mosaique.tif` (GeoTIFF UTM, QGIS), `mosaique_apercu.jpg`, `detections_<modèle>.geojson` et `.csv`,
`synthese_<modèle>.json`, `emprises.geojson`, `parametres.json`, `journal.txt`, `cache/` (reprise).

## Limites

- Sol supposé à peu près plan (plage) ; relief marqué ou falaises : mosaïque moins fidèle.
- Le modèle 1 classe retrouve ~55 % des déchets du banc de test : les comptages sont des estimations basses.
- Poids : ordre de grandeur (table de la littérature, pas de pesée terrain).
- Drones reconnus : DJI (métadonnées XMP). Capteur inconnu : résolution estimée depuis l'EXIF.

Modèles proposés : `config/app_models.yaml` (poids copiés dans `models/app/`).
