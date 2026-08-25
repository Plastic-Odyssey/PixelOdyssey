# PixelOdyssey — Journal des décisions d'architecture (pipeline de données)

Ce doc capture les décisions structurantes prises sur le pipeline `1_annotated_dataset → 2_split_dataset → 3_augmented_dataset → 4_sliced_dataset`, pour ne pas avoir à les re-justifier de zéro dans une session future.

**Où vivent ces décisions (24/08/2026)** : ce document EST la source de vérité (project doc claude.ai — visible dans l'onglet du projet PixelOdyssey, pas dans le repo de code). Une copie identique est en plus poussée dans le repo à `docs/decisions/journal_decisions_pipeline.md` pour que tu l'aies sous les yeux sur ta machine et, si tu passes un jour ce dossier sous Git, que son historique se versionne avec le code. Cette copie sur disque est un MIROIR généré depuis ce doc-ci à chaque mise à jour — ne jamais l'éditer à la main côté disque, sous peine de désynchronisation entre les deux (le même risque que le `VALID_IMG_EXTS` dupliqué qu'on a relevé le 24/08 dans `raw_dataset_checker.py` : deux copies qui peuvent diverger silencieusement dès qu'on oublie d'en maintenir une — devenu un exemple concret plutôt qu'un risque théorique, voir plus bas).

## Taxonomie de classes (21/08/2026)

8 super-classes définies à partir d'un audit chiffré (`src/data/dataset_audit.py`) sur 540 images parentes réelles (comptage d'instances + coefficient de variation de l'aire et du ratio largeur/hauteur par classe brute) plutôt qu'à l'intuition :

- `Bouteille`, `Flipflops`, `Cagette`, `Bouee` (bouées+flotteurs fusionnés), `Cordage_Filet`, `Bouchon`, `Bidon`, `Debris_Divers` (fourre-tout).
- Exclues : `Inconnu`, `Morceaux de bois`, `Verre`, `à déterminer`.
- Détail complet et justification de chaque regroupement : voir les commentaires de `config/data_config.yaml`.
- Point de vigilance identifié : `Bidon` (61 instances) — surveiller son rappel après entraînement, faute de données.
- Classes fantômes (0 instance réelle à ce jour, gardées par précaution) : `casques`, `Bouées pieuvre`.

## Split train/val/test stratifié PAR LOT (22/08/2026)

**Problème** : un shuffle global (ancien comportement) traite les 540 images comme interchangeables, alors que de petits lots (ex: A LEG1_1/A LEG3_1) sont la seule source de certaines classes réelles. Le pur hasard du tirage pouvait exclure un lot entier de val ou test.

**Décision** : `split_dataset.py` découpe désormais 70/20/10 séparément À L'INTÉRIEUR de chaque lot, puis fusionne — chaque lot (et ses classes propres) est représenté proportionnellement dans les 3 splits. Un avertissement est émis si un lot est trop petit pour peupler ses 3 portions.

Marqueur de version ajouté (`SPLIT_LOGIC_VERSION` dans `_split_params()`) pour que le cache incrémental invalide correctement `2_split_dataset` lors de ce changement d'algorithme.

### Correctif du biais d'arrondi (22/08/2026, SPLIT_LOGIC_VERSION 3)

**Découvert en aval**, sur le premier entraînement complet (run `baseline_yolo11n-seg_20260822_023343`) : le split test avait MOINS d'images parentes que val (66 vs 102) mais PLUS de tuiles/instances après slicing (836/1375 vs 752/909). Cause : `n_train_b`/`n_val_b` étaient calculés avec `int()` (troncature systématique vers le bas), et `test` récupérait tout le reliquat des deux troncatures — biais systématique qui gonflait test au-delà de ses 10% nominaux, surtout sur les petits lots à résolution variable (A LEG, SL), qui produisent beaucoup de tuiles par image via la fenêtre glissante (contrairement aux imagettes SB, ~1 tuile/image).

Exemple concret : un lot de 4 images visant train=2.8/val=0.8/test=0.4 donnait train=2/val=0/**test=2** (5x sa part théorique).

**Correctif** : `n_val_b = round(n_b * VAL_RATIO)`, `n_test_b = round(n_b * TEST_RATIO)`, et `train` absorbe désormais le reliquat (`n_b - n_val_b - n_test_b`) au lieu de test — train ne sert jamais à mesurer une performance, une distorsion d'arrondi y est donc sans conséquence, contrairement à test. Vérifié sur les tailles réelles des lots : test passe de 66 à 53 images parentes (nominal théorique : 54 — quasi exact), et les tuiles évaluées sur test devraient redevenir cohérentes avec sa taille réelle.

**Compromis accepté** : les lots les plus petits (A LEG1_1/A LEG3_1, 4 images chacun) se retrouvent maintenant avec 0 image en test (contre 2 avant, mais artificiellement) — la seule alternative pour ces lots serait de renoncer à corriger le biais. Le garde-fou de warning existant (`n_train_b/n_val_b/n_test_b == 0`) signale déjà ce cas.

**Action requise** : ce changement invalide `2_split_dataset` ET `4_sliced_dataset` — relancer tout le pipeline avec `--force`, puis ré-entraîner. Les métriques test du run `baseline_yolo11n-seg_20260822_023343` (déjà générées, voir `rapport_lecture.html` du run) sont à considérer comme provisoires : le test set qui les a produites était biaisé.

## Filtre anti-bordure noire (22/08/2026)

**Contexte** : les JPG sont découpés à la main à partir d'orthomosaïques bruts mal orientés → triangles noirs de rotation sur les bords.

**Risque identifié** : ces triangles, s'ils tombent dans une tuile sans objet annoté, étaient retenus comme "exemple de fond" par la logique existante (image parente 100% background, ou sous-échantillonnage 1/10) — diluant la valeur des vrais exemples de fond propres.

**Décision** : `PlasticImageSlicer` (constructeur `max_black_fraction=0.5`) écarte désormais une tuile SANS OBJET ANNOTÉ si sa fraction de pixels ~noirs dépasse 50%. Une tuile contenant un objet réel n'est **jamais** écartée pour cette raison, quel que soit son contenu noir. Appliqué aux deux branches du slicer (image déjà pré-découpée ≤640px, et fenêtre glissante classique).

## Rééquilibrage des classes rares — DIFFÉRÉ (22/08/2026)

Décision explicite : pas de rééquilibrage/duplication des classes rares (`Bidon`, `Bouee`) pour l'instant. À reconsidérer plus tard — piste déjà identifiée si besoin : oversampling par augmentation (pas duplication à l'identique) appliqué APRÈS le slicing, avec un plafond de duplication par tuile source (~5×) pour éviter le surapprentissage sur une poignée d'images uniques.

**Mise à jour (24/08/2026)** : un premier pas concret vers cette piste est fait sans construire de logique dédiée — voir "Retraining diagnostic" plus bas (`copy_paste` d'Ultralytics, natif, activé pour ce run).

## Bug corrigé : images pré-découpées plus petites que `tile_size` (22/08/2026)

`slicer.py` ne déclenchait le passage direct (image = 1 seule tuile) que sur égalité STRICTE avec `tile_size`. Toute image plus petite (ex: "imagettes" du lot SB si elles ne sont pas exactement 640×640) tombait dans la boucle de fenêtre glissante, qui générait plusieurs fenêtres collapsant toutes au même coin (0,0) — comptage de tuiles faux, fichier écrasé plusieurs fois, taille de sortie erronée. Corrigé (`<=` au lieu de `==`, `LOGIC_VERSION` bumpé à 3).

## Outil de rapport val/test — déjà existant

`src/training/training_report.py` génère automatiquement (appelé par `train.py` à la fin de chaque run) un rapport HTML par classe (précision/rappel/taux de FP-FN/mAP/matrice de confusion) sur val ET test, avec un texte explicite rappelant que val est "légèrement optimiste" (vu indirectement pendant l'entraînement) et que test est la seule mesure honnête. Peut être régénéré sans ré-entraîner : `python -m src.training.training_report --run output/runs/<nom_du_run>`. Ne pas réécrire cet outil — il existe déjà et couvre exactement ce besoin.

## Sélection interactive du modèle pour label_review (22/08/2026)

`python -m src.review.label_review` sans `--model` liste désormais les modèles entraînés trouvés sous `output/runs/*/weights/best.pt` (plus récent en premier, avec architecture/epochs/mAP50-95 val extraits en best-effort de `args.yaml`/`results.csv`) et demande lequel utiliser (Entrée seule = le plus récent). `--model <chemin>` reste disponible pour un usage scripté sans interaction.

## Bug corrigé : images TIFF multi-bandes envoyées au modèle avec 4 canaux (23/08/2026)

**Symptôme** : `label_review.py` a planté avec `Given groups=1, weight of size [16, 3, 3, 3], expected input[1, 4, 640, 640] to have 3 channels, but got 4 channels instead`.

**Cause confirmée empiriquement** : les lots SL (et probablement A LEG) contiennent de vrais fichiers `.tif` (orthomosaïques, ex: `SL 11-16/images/train/transect_11.tif`, ~8 Mo) — `1_annotated_dataset` accepte `.tif`/`.tiff` comme extension valide (voir `raw_dataset.VALID_IMG_EXTS`), pas seulement les `.jpg` convertis à la main. Vérifié : `transect_11.tif` a réellement 4 bandes sur disque (RGBA). `cv2.imread()` par défaut (`IMREAD_COLOR`) est censé TOUJOURS forcer 3 canaux, mais ce comportement est connu pour être incohérent selon la version d'OpenCV/libtiff pour ce genre de TIFF (avertissement typique : "Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel") — probablement la version Windows/pip de l'utilisateur laissait passer les 4 canaux, alors que le sandbox de test ne reproduit pas le problème (autre build OpenCV).

Pourquoi l'entraînement lui-même n'avait pas été affecté : `slicer.py` écrit ses tuiles en `.png` (qui, lui, se comporte de façon fiable sous OpenCV) — si une tuile source avait 4 canaux à l'écriture, Ultralytics la relit ensuite correctement à 3 canaux depuis ce PNG intermédiaire pendant l'entraînement. Le problème n'était visible qu'en inférence directe en mémoire (label_review → tiled_inference), sans ce passage par un fichier PNG intermédiaire.

**Correctif** : nouveau module `src/data/image_io.py` (`load_image_bgr()`) — force explicitement 3 canaux BGR après chargement (détecte et corrige tout résultat à 4 canaux), utilisé PARTOUT où une image est chargée en pixels (`slicer.py`, `tiled_inference.py`, `label_review.py`, `dataset_audit.py`, `visualize_predictions.py`) — plus aucun `cv2.imread()` direct dans ces contextes. `slicer.py` `LOGIC_VERSION` bumpé à 4 (une tuile déjà écrite à 4 canaux avant ce correctif doit être régénérée).

**Robustesse ajoutée en même temps** : la boucle principale de `label_review.py` isole maintenant chaque image dans un `try/except` — une image en échec (ce bug ou un autre) est loggée avec son chemin exact et ignorée, sans faire échouer les 167 autres images du batch. Les échecs sont listés dans le résumé final (`counts["failed_images"]`).

## Bug corrigé : data.yaml de l'export de relecture invalide pour l'import CVAT (23/08/2026)

**Découvert en préparant l'import CVAT** (échange du 23/08/2026) : `label_review.py` copiait tel quel le `data.yaml` ORIGINAL du lot (celui de `1_annotated_dataset/<lot>/data.yaml`) dans l'export sous `5_review_dataset/`. Or ce data.yaml original décrit la mise en page de l'export CVAT D'ORIGINE du lot (`train: train.txt` - une liste de chemins, pas de clé `val`/`test` du tout), alors que l'export de relecture utilise une structure `images/<split>/` + `labels/<split>/` (format Ultralytics YOLO Segmentation, celui que CVAT attend à l'import) et peut couvrir train ET/OU val ET/OU test pour un même lot (surtout avec `--scope all`, devenu le défaut). Un data.yaml copié tel quel pointait donc CVAT vers des chemins qui n'existent pas dans cet export, ou omettait val/test si le lot avait des images flaguées dans ces splits — l'import aurait échoué ou été incomplet.

**Correctif** : nouvelle fonction `_write_review_data_yaml()` qui GÉNÈRE (au lieu de copier) le data.yaml de l'export, avec les mêmes `names` (IDs locaux inchangés, ceux des .txt de labels) mais une clé `train`/`val`/`test` UNIQUEMENT pour les splits réellement présents dans l'export de ce lot, pointant vers `images/<split>` (structure réellement écrite sur disque). Régénéré à chaque image traitée (idempotent, coût négligeable) pour toujours refléter l'état réel de l'export.

## Import CVAT de l'export de relecture — mode d'emploi (23/08/2026)

Pour chaque lot flagué sous `5_review_dataset/review_<horodatage>/<lot>/` :
1. Zipper le CONTENU de `<lot>/` (data.yaml + images/ + labels/ à la racine du zip, pas le dossier `<lot>` lui-même — sur Windows, sélectionner les 3 éléments puis "Envoyer vers > Dossier compressé", ne pas compresser le dossier `<lot>` en un clic).
2. Dans CVAT, ouvrir la tâche EXISTANTE correspondant à ce lot (pas une nouvelle tâche) → Actions → Upload annotations → format "Ultralytics YOLO Segmentation" (ou toute entrée contenant "Ultralytics" + "Segmentation" selon la version de CVAT installée) → sélectionner le zip.
3. Le fichier de labels exporté contient déjà les lignes ORIGINALES + les nouvelles lignes du panier A (voir `_translate_and_filter_label`/export non destructif) : l'upload doit donc remplacer l'état d'annotation de la tâche par cet état fusionné, pas juste ajouter par-dessus - à vérifier à l'usage selon le comportement exact de CVAT (remplace vs fusionne par image).
4. Corriger dans CVAT : les détections du panier A portent une sous-classe "placeholder" à reclasser manuellement ; les objets du panier C (voir `review_manifest.csv`) ont un masque à retoucher.
5. Réexporter depuis CVAT vers le dossier du lot correspondant dans `1_annotated_dataset` (geste manuel, jamais automatique).

Non vérifié en conditions réelles avec CVAT installé chez l'utilisateur — si le nom exact du format à l'import ou le comportement remplace/fusionne diverge de ce qui précède, corriger cette note en conséquence après le premier essai réel.

**Statut au 23/08/2026 (fin de journée) : ce sujet est mis en pause à la demande explicite de l'utilisateur** ("on va mettre entre parenthèse ce sujet d'annotation assisté"). Les pistes déjà identifiées mais non lancées — support du cold-start pour un nouveau lot jamais entraîné dessus, intégration d'un modèle externe pour les premières passes d'annotation — ne doivent pas être reprises sans qu'il les remette explicitement sur la table.

**Mise à jour (24/08/2026)** : le sujet est remis sur la table par l'utilisateur lui-même, comme piste possible face à des prédictions décevantes — voir "Prédictions décevantes : diagnostic avant investissement" plus bas. Toujours pas relancé concrètement, mais plus "en pause" au sens strict.

## Nouvel outil : visionneuse de prédictions image par image (23/08/2026)

**Besoin exprimé** : les mosaïques `val_batchN_pred.jpg` générées par Ultralytics en fin d'entraînement compressent plusieurs images dans une grille minuscule — impossible d'inspecter sérieusement une prédiction précise, ou de voir facilement quelles prédictions réussissent vs. ratent. Besoin explicitement disjoint du sujet d'annotation assistée (mis en pause ci-dessus) : ici on veut REGARDER, pas corriger le dataset.

**Décision d'architecture** : nouvel outil `src/review/visualize_predictions.py`, volontairement PUREMENT DIAGNOSTIC — n'écrit jamais dans `1_annotated_dataset` ni dans un export CVAT (contrairement à `label_review.py`). Réutilise au maximum l'existant plutôt que de dupliquer de la logique :
- sélection interactive du modèle : réutilise `_discover_available_models`/`_prompt_model_choice` de `label_review.py`.
- inférence tuile par tuile sur l'image parente entière : réutilise `tiled_inference.predict_parent_image` (même géométrie de tuilage qu'à l'entraînement).
- appariement GT↔prédictions : réutilise `matching.match_gt_to_predictions` (même logique que le panier A/C de `label_review.py`) — pour ne jamais faire diverger silencieusement deux définitions différentes de "ce qui compte comme un succès".

**Fonctionnement** : pour chaque image parente sélectionnée (`--scope train/val/test/all`, filtre optionnel `--batch`, `--limit`), dessine en couleur sur une COPIE de l'image (jamais l'original) :
- vert = GT et prédiction bien appariées (IoU ≥ `--iou-mismatch-threshold`, défaut 0.5) — réussi.
- orange = appariées mais IoU faible — masque à ajuster (même seuil que le panier C).
- rouge = GT sans prédiction correspondante — raté (faux négatif).
- bleu = prédiction sans GT correspondante, confiance ≥ `--conf-threshold` (défaut 0.25) — fausse alerte (faux positif). Sous ce seuil, la prédiction n'est pas dessinée (bruit de tuile).

Écrit un JPG annoté par image (redimensionné à 1600px max pour un affichage rapide) dans `6_prediction_viewer/<horodatage>/images/`, plus un `index.html` autonome (aucun serveur requis, fonctionne en `file://`) avec navigation Précédent/Suivant + flèches clavier, un tri "plus de ratés/fausses alertes/masques à ajuster d'abord" pour aller direct aux pires cas, un filtre texte lot/parent_id, et une légende couleur. Même isolation d'erreur par image que `label_review.py` (`failed_images`, jamais d'arrêt total du batch).

**Testé** (scénario synthétique, faux `predict_tile_fn` injecté + GT synthétique — même pattern de test que le reste du pipeline cette session) : classification TP fort / TP faible / FN / FP vérifiée correcte, génération HTML vérifiée. Poussé chez l'utilisateur (`src/review/visualize_predictions.py`) — non encore testé en conditions réelles avec un vrai modèle entraîné.

**Usage** : `python -m src.review.visualize_predictions` (sélection interactive du modèle, scope val par défaut) ou `python -m src.review.visualize_predictions --scope test --batch "SL"`.

## Relecture de code guidée — dette technique repérée (24/08/2026)

Session de relecture systématique du pipeline avec l'utilisateur (bloc par bloc, du plus en amont au plus en aval), pour consolider sa compréhension de l'architecture. Trois points de dette identifiés ce jour-là ; le premier a depuis été corrigé (voir "Vérification du code avant réentraînement" plus bas) :
- ~~`VALID_IMG_EXTS` dupliqué à l'identique dans `raw_dataset.py` (source censée être partagée) ET `raw_dataset_checker.py`~~ — corrigé le 24/08/2026, voir plus bas (et il s'est avéré qu'il y avait une VRAIE divergence, pas juste un risque théorique).
- `split_dataset.py` : le cache incrémental par parent (`if img_dst.exists(): skip`) ne teste que la présence de l'IMAGE de sortie, pas du label — un label supprimé isolément à la main ne serait jamais régénéré tout seul. Cas limite improbable en usage normal, gardé en mémoire seulement.
- `slice_dataset.py`/`pipeline_utils.already_present()` : un parent qui produit ZÉRO tuile en sortie (ex: imagette pré-découpée entièrement écartée par le filtre anti-bordure noire) n'est jamais reconnu comme "déjà traité" — reste retraité à chaque run. Sans conséquence sur le résultat, juste un peu de calcul récurrent inutile.

## Géolocalisation, orthomosaïques et anticipation d'un changement de matériel drone (24/08/2026)

**Contexte** : objectif final confirmé = produire une carte de densité de déchets sur l'ensemble d'une plage, ce qui nécessite de conserver la position GPS de chaque détection à travers le pipeline — pas seulement de détecter/segmenter. L'utilisateur confirme que les `.tif` actuellement utilisés (issus de WebODM) sont bien des **GeoTIFF** (CRS + transformation affine embarqués), pas de simples TIFF multi-bandes non géoréférencés.

**Constat d'architecture (pas encore un bug, mais un vrai manque pour cet objectif)** : `image_io.load_image_bgr()` (donc tout le pipeline : `slicer.py`, `tiled_inference.py`, `label_review.py`, `dataset_audit.py`, `visualize_predictions.py`) passe par `cv2.imread`, qui NE LIT QUE LES PIXELS — toute métadonnée de géoréférencement (CRS, transformation affine) d'un GeoTIFF source est silencieusement perdue. Sans conséquence aujourd'hui (rien dans le pipeline actuel n'a besoin de la position réelle d'un pixel), mais bloquant pour un futur outil de carte de densité géolocalisée.

**Décision de principe** : ne PAS rendre le pipeline d'entraînement (split/augmentation/slicing) géo-conscient — il n'en a et n'en aura jamais besoin (même logique déjà appliquée partout : chaque étape ne connaît que ce dont elle a besoin). La géolocalisation ne concerne qu'un module d'inférence dédié à la prédiction sur orthomosaïque complète — voir section suivante, construit et démontré le jour même.

**Nouveau développement (24/08/2026) — changement de matériel drone en cours** : le futur matériel produira des images à une résolution sol (GSD, "ground sample distance") d'environ **1 cm/pixel**. Voir section suivante pour la GSD ACTUELLE, désormais mesurée sur une vraie orthomosaïque plutôt qu'inconnue.

Pourquoi ça compte : ce n'est pas la TAILLE DE TUILE (640px) qui doit correspondre entre entraînement et inférence (déjà réglé, voir `tiling_geometry.py`), mais la RÉSOLUTION SOL — un même déchet occupera un nombre de pixels différent selon la GSD de capture, ce qui revient à un changement d'échelle apparente que le modèle doit avoir appris à tolérer.

**Constat déjà présent dans le code, à vérifier avant d'agir** : `train.py` ne passe actuellement AUCUN hyperparamètre d'augmentation explicite à `model.train()` — les valeurs par défaut d'Ultralytics s'appliquent donc implicitement (dont un `scale` par défaut ≈0.5, soit un zoom aléatoire ~0.5x-1.5x, plus le mosaïquage actif par défaut) : une augmentation d'échelle basique existe donc déjà aujourd'hui, sans avoir jamais été consciemment dimensionnée pour un écart de GSD précis.

**Ordre d'action recommandé, à formaliser maintenant que la GSD actuelle est connue (voir section suivante)** :
1. ~~Mesurer/estimer la GSD actuelle des images d'entraînement~~ Fait le 24/08 (≈0.5 cm/px, voir section suivante).
2. Le ratio GSD_actuelle / 1cm (~2×) n'est pas négligeable : à comparer au `scale` par défaut d'Ultralytics (~0.5x-1.5x, qui couvre un facteur 3x max déjà) avant de conclure qu'il faut construire quoi que ce soit dans `3_augmented_dataset` (actuellement un pass-through volontaire, voir sa docstring qui recommande déjà explicitement de vérifier les augmentations natives d'Ultralytics avant de dupliquer cet effort) — probablement suffisant tel quel, à confirmer par un test plutôt que supposé (le run diagnostic ci-dessous est l'occasion de vérifier).
3. Dans tous les cas, dès que le nouveau matériel est disponible : capturer et annoter quelques vraies images à la nouvelle GSD comme un nouveau LOT dans `1_annotated_dataset` (le pipeline le supporte déjà nativement — split stratifié par lot, data.yaml propre au lot) — de la vraie donnée à la nouvelle résolution vaudra toujours mieux qu'une augmentation synthétique pour un changement de matériel permanent (l'augmentation reste utile en attendant, pas comme substitut définitif).
4. Ce point rouvre indirectement le sujet du "cold-start nouveau lot" identifié le 23/08 puis mis en pause avec tout le sujet d'annotation assistée — à reconsidérer explicitement une fois le nouveau matériel disponible, pas avant.

## Nouvel outil : carte de densité géoréférencée — démo validée sur une vraie orthomosaïque (24/08/2026)

**Besoin exprimé** : visualiser rapidement, sur une orthomosaïque GeoTIFF entière (WebODM), soit les détections individuelles avec leur confiance, soit une carte de densité/chaleur, avec zoom interactif et un fond satellite pour situer la zone dans le paysage réel. Démo demandée d'abord sur un petit fichier (`SL 28-30 avt.tif`, 187 Mo) avant de passer à l'échelle sur l'orthomosaïque complète (`SL W1.tif`, ~960 Mo).

**Architecture retenue (première version)** : nouveau module `src/review/geo_density_map.py`.
- `predict_geotiff_windowed()` : lit le GeoTIFF FENÊTRE PAR FENÊTRE via `rasterio` (jamais l'image entière en RAM — condition nécessaire pour un fichier de plusieurs centaines de Mo à plusieurs Go). Réutilise `tiling_geometry.iter_tile_windows` (même géométrie que l'entraînement) et `tiled_inference.nms_merge` (rendu public, plus de `_` — désormais partagé entre les deux outils plutôt que dupliqué).
- `pixels_to_lonlat()` : convertit les coordonnées pixel en lon/lat WGS84 via la transformation affine du GeoTIFF + `rasterio.warp.transform` (reprojection depuis le CRS natif, ici UTM zone 26N/EPSG:32626).
- Sortie : page HTML autonome, Leaflet + plugin `leaflet.heat` **vendorisés en dur** dans `src/review/vendor/` (plus de dépendance à un CDN au chargement - seul le fond satellite reste une requête réseau externe, inévitable pour un fond de carte satellite). Fond satellite Esri World Imagery, bascule "Détections" (points colorés par classe, taille par confiance, popup) / "Densité" (carte de chaleur `leaflet.heat`, pondérée par confiance), légende dynamique.
- Convention de sortie : `7_density_maps/<horodatage>/index.html`, cohérente avec `5_review_dataset/`/`6_prediction_viewer/`.
- Volontairement PUREMENT DIAGNOSTIC comme `visualize_predictions.py` — n'écrit jamais dans le dataset.

**Testé end-to-end sur données réelles** (`SL 28-30 avt.tif`, modèle `baseline_yolo11n-seg_20260822_023343/best.pt`) :
- 12266×12432 px, RGBA, EPSG:32626 (UTM 26N), aucune rotation dans la transformation affine.
- 1056 fenêtres 640×640 (stride 384, overlap 256 — identique à l'entraînement), ~195 ms/tuile sur CPU (pas de GPU dans l'environnement de test) → ~3-4 min de calcul pour ce fichier.
- 501 détections brutes → 255 après fusion des recouvrements de tuiles.

**GSD mesurée pour la première fois sur une vraie donnée** : ≈**0,5 cm/pixel** (61,33 m / 12266 px, transformation affine de ce GeoTIFF). Répond à la question du 24/08 sur la GSD actuelle — écart avec la future GSD ~1 cm/px du nouveau matériel : facteur ~2× (un même déchet occupera environ 4× moins de pixels en aire, ~2× moins en largeur/hauteur, sur les futures images) — à utiliser pour trancher le point 2 de la section précédente sur l'augmentation d'échelle.

**Résultat à interpréter avec précaution** : ce modèle est le run pré-correctifs (avant stratification/arrondi/filtre anti-bordure noire/fix TIFF 4 canaux, voir plus haut) — cette démo valide le PIPELINE DE VISUALISATION, pas la fiabilité scientifique du compte de 255 détections sur cette zone. Le rappel/la précision réels ne seront évaluables qu'après ré-entraînement sur le pipeline corrigé.

### Itération 2 (24/08/2026) — retour utilisateur "je veux zoomer sur le déchet et juger le masque", et pivot d'architecture sur la livraison

**Retour utilisateur sur la V1** : le fond de carte était une image UNIQUE sous-échantillonnée (`build_basemap_overlay`, lecture "decimated" rasterio à `max_dim=2400`) drapée via `L.imageOverlay` — zoomer au-delà de cette résolution ne révélait aucun détail réel (flou d'agrandissement), et seuls des points (pas les masques réels) étaient dessinés. Impossible de juger la qualité d'un masque de segmentation sur un déchet de quelques centimètres.

**Premier essai (abandonné) : pyramide de tuiles XYZ couvrant toute l'orthomosaïque au zoom natif**. Nouvelle fonction `generate_tile_pyramid()` : grille slippy-map standard EPSG:3857 (celle d'OSM/Leaflet/Esri), chaque tuile reprojetée directement depuis le GeoTIFF via `rasterio.warp.reproject` (une bande à la fois, lecture fenêtre par fenêtre — jamais l'image entière en RAM, même logique que `predict_geotiff_windowed` appliquée cette fois au fond de carte). Zoom natif calculé depuis la résolution réelle du GeoTIFF reprojetée en Web Mercator (`calculate_default_transform`) : **zoom 25** sur ce fichier pour atteindre les ~0,5 cm/px natifs.

**Problème découvert à l'exécution** : une pyramide complète (zoom 16→25) sur ce fichier de test (187 Mo) génère **4272 tuiles pour 214 Mo** au total. Bien au-delà de ce qui peut être livré : la pièce jointe de conversation (`SendUserFile`) plafonne à 30 Mio, et le pont vers le disque de l'utilisateur (`device_commit_files`) plafonne à 100 Mio / 50 fichiers par appel. Sur `SL W1.tif` (~5× la surface), ce serait mécaniquement plus lourd encore — cette approche ne serait praticable QUE si le calcul et la livraison se faisaient sur la même machine, ce qui n'est pas le cas ici (calcul dans un environnement cloud, livraison vers le disque Windows de l'utilisateur via un pont explicite).

**Décision finale : séparer "vue d'ensemble" et "inspection fine d'un masque"**, deux besoins différents qui n'ont pas besoin de la même résolution :
- `generate_tile_pyramid()` **conservée mais son zoom plafonné à 21 par défaut** (`ortho_max_zoom_cap`) — vue nette pour naviguer sur la plage (~2 à 4 cm/px), coût de génération négligeable (36 tuiles / 1,6 Mo sur ce fichier). Le fond de carte reste une vraie pyramide de tuiles (pas un retour à l'image unique) : la carte de densité/points bénéficie quand même de zooms progressifs nets jusqu'à ce niveau, juste plus modeste qu'un zoom pixel-à-pixel sur toute la plage.
- **Nouvelle fonction `build_detection_crops()`** : pour CHAQUE détection (donc coût proportionnel au nombre de déchets détectés, pas à la surface totale de l'image), découpe un petit chip à résolution VRAIMENT NATIVE (marge de 60px autour du masque, jamais sous-échantillonné), dessine le contour du masque dessus (jaune vif, choisi pour rester lisible sur sable comme sur plastique coloré), l'agrandit pour l'affichage (LANCZOS) et l'encode en **JPEG** (pas PNG — un chip est une photo, pas un aplat de couleur : gain de compression ~5× mesuré, déterminant car répété une fois par détection). Ce chip s'affiche directement dans le popup Leaflet au clic sur une détection (polygone ou point) — c'est ICI que se juge la qualité réelle d'un masque, pas en zoomant la carte.
- Chaque détection est en plus désormais dessinée comme un VRAI polygone sur la carte (`d.polygon`, contour du masque reprojeté en lon/lat via `L.polygon`), pas seulement un point — situe la forme et la position sur la vue d'ensemble. Un petit point centré reste ajouté en complément (un masque de quelques cm devient quelques pixels écran à faible zoom, difficile à cliquer sans lui).

**Résultat mesuré sur `SL 28-30 avt.tif`** (255 détections) : dossier de sortie **7,2 Mo** (36 tuiles de fond ≈1,6 Mo + `index.html` ≈5,6 Mo avec les 255 chips JPEG embarqués) — confortablement sous les deux limites de livraison. Vérifié par capture d'écran automatisée (Playwright) : vue d'ensemble avec masques réels colorés par classe, bascule densité fonctionnelle, et popup de détection montrant un chip natif net avec le contour jaune du masque bien visible autour de l'objet réel (testé sur une détection `Debris_Divers` à 94% de confiance).

**Point d'architecture à retenir pour la suite** : toute future fonctionnalité de ce type doit distinguer explicitement "résolution nécessaire pour une VUE D'ENSEMBLE" (peu coûteuse, plafonnable) de "résolution nécessaire pour une INSPECTION PONCTUELLE" (coûteuse mais localisée) — les traiter avec la même résolution partout est ce qui a fait exploser la taille de sortie ici. Cette même distinction jouera pour `SL W1.tif` : la vue d'ensemble restera bon marché quelle que soit la taille du fichier (le plafond de zoom ne dépend pas de la surface), et le coût des chips reste proportionnel au nombre de détections, pas à la taille du fichier source — donc pas d'explosion attendue en passant à l'échelle, contrairement à l'approche pyramide-complète abandonnée.

**Non fait à ce stade** : pas encore lancé sur l'orthomosaïque complète (`SL W1.tif`, ~960 Mo) — **mis en attente par l'utilisateur** (voir section suivante) après avoir jugé les prédictions décevantes sur cette démo. La priorité passe à comprendre/corriger la qualité du modèle avant d'aller plus loin sur la visualisation.

## Prédictions décevantes : diagnostic avant investissement (24/08/2026)

**Contexte** : après avoir vu les chips natifs de l'itération 2 (masques réels + zoom), l'utilisateur juge les prédictions décevantes et pense que c'est dû à la faiblesse du dataset. Il propose deux pistes lourdes (plusieurs semaines chacune) : reprendre l'annotation assistée (mise en pause le 23/08), ou construire un dataset synthétique via Unity à partir d'exemples de déchets déjà photographiés, avec contrôle total de la composition/classes/échelle/orientation.

**Point de mentorat soulevé avant de choisir entre les deux** : le jugement "dataset faible" repose sur une démo utilisant `baseline_yolo11n-seg_20260822_023343` — le run **pré-correctifs** (split biaisé, pas de filtre anti-bordure noire, bug TIFF 4 canaux non corrigé au moment de ce run). Une partie de la déception peut venir de bugs déjà réglés, pas nécessairement du volume/de la diversité réelle des données. Recommandation : ré-entraîner sur le pipeline corrigé et lire le rapport val/test PAR CLASSE (`training_report.py`, déjà existant) avant d'attribuer la faiblesse à l'une ou l'autre cause — ça dit QUEL est le problème (classes rares seulement ? toutes les classes ? précision de masque vs rappel ?), ce qui devrait déterminer laquelle des deux pistes (ou les deux, mais pour des raisons différentes) a le plus de valeur :
- Annotation assistée : plus de VRAIES données (conditions réelles), mais ne résout pas le déséquilibre de classes (`Bidon`/`Bouee` resteront rares tant qu'on n'ira pas en photographier plus sur le terrain). Réutilise l'outillage existant (`label_review.py`).
- Synthétique Unity : contrôle total de la composition et de l'échelle - répond directement au déséquilibre de classes ET à la question ouverte sur le GSD (simuler le futur matériel ~1cm/px). Risque principal : écart de domaine sim-to-real, à valider IMPÉRATIVEMENT sur le vrai jeu de test, jamais en confiance aveugle.

**Décision immédiate de l'utilisateur** : relancer l'entraînement sur le pipeline corrigé MAINTENANT (avant de choisir entre les deux pistes ci-dessus) - la piste "diagnostic d'abord" a été suivie.

### Vérification du code avant réentraînement (24/08/2026)

Avant de relancer, vérification effective (pas supposée) de l'état du code par rapport à toutes les décisions listées dans ce journal :

- **Confirmé en place** : `split_dataset.py` (SPLIT_LOGIC_VERSION 3, arrondi corrigé, train absorbe le reliquat), `slicer.py` (LOGIC_VERSION 4, `<=` au lieu de `==`, filtre anti-bordure noire `max_black_fraction`), `image_io.load_image_bgr()` utilisé partout où une image est chargée en pixels (`slicer.py`, `tiled_inference.py`, `label_review.py`, `dataset_audit.py`, `visualize_predictions.py`) — plus aucun `cv2.imread()` direct dans ces contextes.
- **Corrigé à cette occasion : duplication de `VALID_IMG_EXTS`** — en creusant la dette notée le 24/08 (`raw_dataset.py` vs `raw_dataset_checker.py`), découverte qu'il y avait en réalité **QUATRE copies**, pas deux : `raw_dataset.py`, `raw_dataset_checker.py`, `slice_dataset.py`, ET `dataset_sanity_check.py` — cette dernière avec un contenu **différent** (`{".jpg", ".jpeg", ".png"}`, sans `.tif`/`.tiff`) : une vraie divergence silencieuse, pas juste un risque théorique.
  - **Impact réel de cette divergence** : `dataset_sanity_check.py` tourne automatiquement après le split (`2_split_dataset`, via `data_pipeline.py`) - étape qui contient encore les images parentes BRUTES, donc les vrais `.tif` des lots SL/A LEG. Toute image `.tif` y était invisible pour ce checker → son label `.txt` bien réel ressortait comme "label orphelin sans image correspondante" → fausse alerte de bug pipeline à chaque run sur ces lots, alors qu'il n'y avait rien d'anormal. Sans impact sur `4_sliced_dataset` (tout y est en `.png`), donc invisible sur le résultat final - explique pourquoi ça n'avait jamais été remarqué.
  - **Correctif** : les 4 fichiers importent désormais `VALID_IMG_EXTS` depuis `raw_dataset.py` (source unique), plus aucune redéfinition locale.
- **Toujours en l'état, sans impact sur ce run** : `split_dataset.py` (cache incrémental ne teste que l'image, pas le label) et `slice_dataset.py`/`pipeline_utils.already_present()` (parent à 0 tuile jamais marqué "traité") — dette mineure déjà notée, aucune conséquence sur la correction du résultat, seulement un peu de calcul récurrent inutile.

### Choix du modèle : nano vs medium (24/08/2026)

Question posée : tenter un modèle plus puissant (medium) plutôt que nano pour ce réentraînement.

**Point de mentorat** : plus de capacité (nano → medium) n'aide QUE si la faiblesse est un problème de PRÉCISION DU MODÈLE (ex: contours de masque imprécis, généralisation insuffisante avec assez de données) - pas un problème de VOLUME/DIVERSITÉ DE DONNÉES (ex: classe rare sous-représentée), que plus de paramètres ne fait qu'aggraver côté surapprentissage sur un dataset encore petit (~540 images parentes avant slicing). Sans diagnostic par classe déjà en main, sauter direct à medium est un pari, pas une décision informée.

**Recommandation, notée dans `train.py` lui-même** : regarder le rapport val/test par classe de CE run (nano, pipeline corrigé) avant de sauter à "m". Si le nano généralise déjà bien mais plafonne en précision, tester `yolo11s-seg.pt` d'abord (saut de capacité plus mesuré) plutôt que `m` directement - `train.py` documentait déjà ces deux options comme prochaines étapes à tester (`MODEL_WEIGHTS`, une seule ligne à changer), donc aucun changement de code nécessaire pour tester, juste une décision informée par les résultats.

### Hyperparamètres d'augmentation Ultralytics rendus explicites (24/08/2026)

Question posée : quels paramètres Ultralytics (rotation, luminosité, etc.) sont utiles à modifier. Jusqu'ici, `train.py` ne passait AUCUN hyperparamètre d'augmentation explicite - les défauts d'Ultralytics 8.4.122 s'appliquaient silencieusement (vérifiés programmatiquement plutôt que supposés : `degrees=0.0`, `flipud=0.0`, `fliplr=0.5`, `scale=0.5`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `mosaic=1.0`, `mixup=0.0`, `copy_paste=0.0`).

**Constat clé, spécifique à ce projet** : nos images sont des vues NADIR (drone à la verticale) - contrairement à une photo "normale" avec un horizon et un "haut" naturel (le cas pour lequel les défauts Ultralytics, pensés COCO, sont calibrés), un déchet vu du dessus peut apparaître à N'IMPORTE QUELLE orientation. Deux changements en découlent directement (pas des suppositions) :
- `degrees=180.0` (défaut 0.0, donc aucune rotation jusqu'ici) - rotation aléatoire sur tout le cercle.
- `flipud=0.5` (défaut 0.0 ; `fliplr` était déjà à 0.5 par défaut mais pas le retournement vertical - illogique en nadir où les deux sont équivalents).

**Rééquilibrage des classes rares, sans construire de logique dédiée** : `copy_paste=0.3` activé (défaut 0.0) - mécanisme natif Ultralytics qui colle des instances segmentées d'une image sur une autre du batch. C'est exactement la piste "oversampling par augmentation" notée comme DIFFÉRÉE le 22/08 pour `Bidon`/`Bouee`, obtenue ici gratuitement plutôt qu'en construisant du code - valeur modérée pour un premier essai, à surveiller sur le rapport par classe plutôt qu'à monter à l'aveugle.

**Laissé tel quel, avec justification explicite (pas un oubli)** :
- `scale=0.5` (défaut) - répond déjà à l'écart de GSD actuel/futur (~2×, le défaut couvre un facteur ~3×) ; ce run est justement l'occasion de le VÉRIFIER sur le rapport test plutôt que de continuer à le supposer suffisant (voir section GSD plus haut, point 2).
- `hsv_h`/`hsv_s`/`hsv_v`, `mosaic` : défauts déjà raisonnables (`hsv_v=0.4` couvre une variation de luminosité significative, pertinente pour du sable au soleil/à l'ombre) - pas de raison de les changer sans preuve qu'ils ne suffisent pas.
- `mixup` : laissé désactivé - mélange deux images entières, utile sur des scènes complexes/multi-objets ; sur de petits déchets isolés sur fond de sable, le risque de brouiller des masques déjà petits dépasse probablement le bénéfice.

Modifié dans `train.py` (constantes `DEGREES`/`FLIPUD`/`COPY_PASTE` en haut du fichier, passées explicitement à `model.train()`), avec les mêmes commentaires de justification directement dans le code plutôt que seulement ici.

### Bug corrigé : le cache incrémental ne propageait pas les changements amont (24/08/2026)

**Découvert en relançant `data_pipeline.py`** juste après la vérification ci-dessus : l'étape 1 (split) a levé l'erreur de config changée attendue (`split_logic_version 2 -> 3`, le correctif d'arrondi du 22/08/2026, jamais réellement réappliqué sur disque depuis) — légitime, pas un nouveau problème.

**Point de mentorat avant de simplement relancer avec `--force`** : chaque étape (`ensure_cache_is_safe`) n'invalide son propre cache que si SES PROPRES paramètres locaux ont changé — elle n'a aucune notion que l'étape AMONT a produit un contenu différent. Or `data_pipeline.py --force` transmet le même booléen `force=True` aux 3 étapes, mais chacune ne wipe QUE si son propre fingerprint local ne correspond plus : ici, seul le split (dont `split_logic_version` a changé) aurait été régénéré ; `3_augmented_dataset` et `4_sliced_dataset` (dont aucun paramètre local n'a changé) auraient gardé leur ANCIEN contenu tel quel, et se seraient contentés d'AJOUTER les fichiers manquants dans leur nouvel emplacement.

**Risque concret identifié avant d'agir** (pas théorique - le split v2→v3 change réellement l'affectation train/val/test de certains lots, via l'arrondi) : une image réassignée de train à val par le nouveau split se serait retrouvée copiée dans le nouveau `val/` de `3_augmented_dataset` (fichier absent, donc copié) SANS que son ancienne copie dans `train/` soit supprimée (le pass-through ne regarde que ce qui doit être ajouté, jamais ce qui doit être retiré) — la même image dans train ET val, donc dans les tuiles dérivées des deux côtés après slicing. Fuite train/val silencieuse, qui aurait faussé les métriques sans qu'aucune erreur ne soit levée.

**Correctif structurel (pas un contournement manuel)** : chaque étape inclut désormais dans ses propres paramètres de cache le fingerprint enregistré par l'étape amont (nouvelle fonction `read_upstream_fingerprint()` dans `pipeline_utils.py`, qui lit le fingerprint déjà écrit dans le manifeste de l'étape précédente) :
- `augment_dataset.py` inclut le fingerprint de `2_split_dataset/.split_manifest.json`.
- `slice_dataset.py` inclut le fingerprint de `3_augmented_dataset/.augment_manifest.json`.

Effet : tout changement amont (même sans changement de paramètre local à l'étape elle-même) fait automatiquement mismatcher le fingerprint de CHAQUE étape en aval, en cascade — `--force` sur `data_pipeline.py` redevient donc réellement sûr : une seule commande wipe et régénère les 3 étapes de façon cohérente, sans mélange de versions. Comme cette modification change elle-même la structure des paramètres suivis, elle déclenche par construction un rebuild complet dès ce prochain `--force` (les anciens manifestes n'ont pas cette clé) — occasion propre d'établir la nouvelle référence.

**Point d'architecture à retenir** : un garde-fou de cache par étape qui ne regarde que sa PROPRE configuration est nécessaire mais pas suffisant dans un pipeline à étapes chaînées - il doit aussi être informé de ce que l'étape amont a produit. Cette même logique de propagation devra être appliquée à toute future étape ajoutée au pipeline.

## Résultats du run diagnostic `diag_pipeline_fix_yolo11n-seg_20260824_155822` (24/08/2026)

**Contexte** : premier ré-entraînement sur le pipeline corrigé (stratification, arrondi, filtre anti-bordure noire, fix TIFF 4 canaux, dédup `VALID_IMG_EXTS`, augmentation nadir + `copy_paste`) — le diagnostic qu'on avait décidé de faire avant de choisir entre annotation assistée et dataset synthétique Unity (voir "Prédictions décevantes" plus haut). 100 epochs complétées (patience=20 jamais déclenché, mais courbes en plateau depuis ~epoch 80-85 - `mAP50-95` masque oscille 0,41-0,43 sans tendance nette, pas un arrêt prématuré).

**Résultat global (test, 1019 instances, jamais vu pendant l'entraînement)** : précision masque 67,9%, rappel 62,9%, mAP50 65,3%, mAP50-95 45,0% - nettement meilleur que ce que suggérait la démo décevante du 24/08 sur `SL 28-30 avt.tif` (modèle pré-correctifs). Confirme la lecture du point de mentorat "diagnostic avant investissement" : une bonne partie de la déception initiale était bien un artefact des bugs déjà corrigés, pas une preuve de faiblesse structurelle du dataset sur l'ensemble des classes.

**Par classe (test)** : 6 classes sur 8 dans une fourchette solide et exploitable (60-83% précision/rappel) - Bouteille, Flipflops, Cagette, Bouee, Bidon, Debris_Divers. Le point de vigilance noté le 21/08/2026 sur **Bidon** (61 instances, crainte de sous-représentation) semble réglé (67,5% précision / 66,2% rappel en test).

**Deux classes restent clairement mauvaises, sur val ET test (donc pas du bruit d'échantillon)** :
- `Bouchon` : rappel ~18% sur les deux splits, précision qui s'effondre en test (66,8% → 23,3%, mais n=39 seulement, à interpréter avec prudence sur la précision).
- `Cordage_Filet` : rappel 31,6% (val) / 47,0% (test) - plus stable mais chroniquement faible.

**Diagnostic de la cause, via `confusion_matrix_normalized.png` (le graphique Ultralytics natif, PAS le tableau "Matrice de confusion" du rapport HTML - voir bug ci-dessous)** : pour les vrais `Bouchon`, le modèle prédit "background" **66% du temps** ; pour `Cordage_Filet`, **50% du temps**. Ce n'est pas une confusion avec une autre classe de déchet - le modèle ne les détecte quasiment pas, signature typique d'un problème de TAILLE D'OBJET À LA RÉSOLUTION DE CAPTURE plutôt que d'un manque de données (`Bidon`, volume d'instances comparable, s'en sort bien).

**Connexion avec la section GSD/matériel drone (24/08/2026, plus haut)** : GSD actuelle mesurée ≈0,5 cm/px → un bouchon de bouteille standard (~3 cm) ne fait qu'environ 5-6 pixels de diamètre sur ces images, à la limite de ce qu'un YOLO peut représenter (tête de détection la plus fine = cellules de 8px). Le futur matériel visant ~1 cm/px (2× plus grossier, déjà noté plus haut) **réduirait encore la taille apparente de Bouchon**, pas l'inverse - implication stratégique importante : si le problème est bien une limite physique de résolution, ni l'annotation assistée (ne crée pas de pixels absents des photos actuelles) ni un dataset synthétique Unity simulant fidèlement la vraie distance de prise de vue ne le résoudraient. Un synthétique qui simulerait une prise de vue plus rapprochée que la réalité de terrain rouvrirait le risque d'écart sim-to-real déjà signalé le 24/08.

**Vérification recommandée avant tout investissement lourd (pas encore faite)** :
1. Relancer `src/data/dataset_audit.py` (déjà existant, calcule déjà l'aire moyenne par classe) et comparer l'aire en pixels de `Bouchon`/`Cordage_Filet` à celle de `Bidon`/`Bouee` - pour chiffrer l'hypothèse de taille plutôt que la garder comme intuition.
2. Inspecter concrètement des exemples avec `src/review/visualize_predictions.py --scope test` (jamais testé en conditions réelles jusqu'ici) trié par "plus de ratés" - pour voir si les `Bouchon` manqués sont bien minuscules/peu contrastés, ou si autre chose est en jeu (mélange de types d'objets sous un même label, éclairage, etc.).

**Sur nano vs medium/small** : pas de saut de capacité recommandé pour l'instant. L'échec de `Bouchon`/`Cordage_Filet` est un problème de détection ("le modèle ne voit pas l'objet"), pas de précision de masque sur un objet déjà détecté - plus de paramètres n'aide pas à voir des pixels qui ne sont pas assez nombreux. Les courbes d'entraînement confirment par ailleurs un plateau (pas un manque de capacité qui bloquerait une progression encore active). `yolo11s-seg` reste l'étape mesurée déjà documentée dans `train.py` si on veut valider empiriquement, mais après l'investigation sur `Bouchon`, pas avant.

**Bug trouvé dans l'outillage (pas encore corrigé)** : le tableau "Matrice de confusion" DANS `rapport_lecture.html` affiche des zéros partout sur ce run - bug de parsing dans `training_report.py`, pas un vrai résultat. Le PNG `confusion_matrix_normalized.png` généré directement par Ultralytics reste fiable (c'est lui qui a permis le diagnostic ci-dessus). À corriger à l'occasion, sans urgence puisque le contournement (lire le PNG) fonctionne.

## Vérifications post-run : audit des tailles par classe + inspection visuelle (25/08/2026)

Suite aux deux actions recommandées dans la section précédente, faites par l'utilisateur le jour même :

### `dataset_audit.py` confirme chiffrentiquement l'hypothèse de taille sur `Bouchon`

Sortie sur 535 images parentes réelles : `bouchons` a une aire médiane normalisée de **0,0032** - plusieurs fois plus petite que TOUTE autre classe réelle de la taxonomie (la plus proche parmi les classes gardées, `Petit bidon`, est à 0,0292 - presque 10× plus grande). Ce n'est plus une intuition tirée de la GSD mesurée sur un seul fichier : c'est confirmé sur l'ensemble du dataset annoté. `Cordage`, en revanche, a une aire médiane normale (0,0373, comparable aux classes qui marchent bien) mais un CV de ratio largeur/hauteur élevé (0,568, parmi les plus hauts) - son problème est la variabilité de FORME (objet fin, courbé, jamais la même silhouette), pas la taille. Deux causes différentes pour deux classes différentes, confirmées par les chiffres plutôt que supposées.

### `visualize_predictions.py --scope test` : premier usage réel, et une vraie découverte

Premier test en conditions réelles de cet outil (jusqu'ici seulement testé en synthétique, voir plus haut). Résumé sur 53 images test : 123 réussies, 9 masques à ajuster, 46 ratées, **117 fausses alertes**.

**Point de départ de l'investigation** : l'utilisateur a remarqué que le mAP50-95 (masque, val) de ce run diagnostic (0,427) est PLUS BAS que celui du run pré-correctifs `baseline_yolo11n-seg_20260822_023343` (0,557). Deux réserves avant d'interpréter ce chiffre :
- Le split v2→v3 a changé quelles images tombent en val entre les deux runs - ce n'est pas la même évaluation.
- Ce run a changé 4 choses à la fois (filtre anti-bordure noire, fix TIFF 4 canaux, ET 3 hyperparamètres d'augmentation) - impossible d'attribuer un écart à une cause précise avec un seul avant/après groupé. Leçon pour la suite : isoler les changements un par un pour mesurer l'effet de chacun.

**Cause réelle trouvée en ouvrant les images** : 101 des 117 fausses alertes viennent de seulement 3 images sur 53 (les 3 grandes orthomosaïques SL - `transect_13`, `transect_18`, `transect9` - contre quasiment aucune sur les 50 tuiles SB). En regardant `transect_13.jpg` et `transect_18.jpg` annotées : le modèle hallucine des détections ("Debris_Divers", "Bouteille", "Bidon"...) à confiance faible-moyenne (0,26-0,72) sur du sable quasiment vide, sans aucun déchet réel dessus. Ce n'est pas un artefact de tuilage/fusion NMS (détections isolées, pas de doublons qui se chevauchent) - le modèle est devenu plus sensible au bruit de fond.

**Suspect principal : `copy_paste=0.3`, pas `degrees`/`flipud`.** `copy_paste` colle des instances segmentées d'une image sur une autre - si le collage n'est pas parfaitement cohérent avec l'éclairage/l'ombre/le grain du fond receveur, le modèle peut apprendre à repérer des "indices de collage" plutôt qu'un vrai objet, et devenir trop sensible à de simples variations de texture du sable. `degrees=180`/`flipud=0.5` ne font que tourner la MÊME photo réelle - ils ne peuvent pas inventer un objet qui n'existe pas, donc suspects bien moins probables pour ce type de faux positifs. C'est exactement le risque anticipé dans le commentaire de `train.py` en ajoutant ce paramètre le 24/08 ("à surveiller sur le rapport par classe plutôt qu'à monter à l'aveugle").

**Vérification recommandée, pas encore faite** : relancer un entraînement avec UNIQUEMENT `copy_paste=0` (garder `degrees=180`/`flipud=0.5`, toujours justifiés par la géométrie nadir) - si les faux positifs sur fond vide disparaissent et que le mAP50-95 remonte vers 0,5+, ça confirme `copy_paste` comme responsable plutôt que de laisser planer le doute.

**Point pratique, à ne pas oublier avant de juger le taux de fausses alertes trop alarmant** : plusieurs des faux positifs relevés sont à confiance basse (0,26-0,43) - `visualize_predictions.py` utilise un seuil de confiance par défaut de 0,25. Les courbes `MaskP_curve.png`/`BoxP_curve.png` du run (précision en fonction du seuil) permettent de choisir un seuil de déploiement réaliste plutôt que de juger sur le seuil le plus permissif de l'outil de diagnostic.

Ce diagnostic ne remet pas en cause le constat sur `Bouchon`/`Cordage_Filet` (confirmé, pas affaibli, par l'audit) - c'est un problème SÉPARÉ, propre à cette expérience d'augmentation, qui s'ajoute à la liste des choses à vérifier avant de ré-entraîner à nouveau.

**Mise à jour (25/08/2026)** : `COPY_PASTE` remis à `0.0` dans `train.py` (`RUN_TAG = "ablation_no_copy_paste"`), commentaire daté ajouté dans le code — changement appliqué, ré-entraînement pas encore relancé par l'utilisateur à ce stade.

## Table de conversion masque → poids : cadrage initial (25/08/2026)

**Besoin exprimé** : convertir une aire de masque de segmentation en une estimation de masse (g), pour produire au final une masse totale de déchets sur une zone plutôt qu'un simple comptage. Point de départ : un tableau de pesée terrain (classe, poids en g, plus plus tard un numéro de photo + un code de cellule de grille 5×5 + une description libre de l'objet).

**Modèle physique retenu, PAR FAMILLE D'OBJET plutôt qu'une formule unique** — un déchet n'a pas la même relation aire↔masse selon sa géométrie :
- Objets plats/fins (fragments de film, sacs) : masse ≈ proportionnelle à l'AIRE (épaisseur/densité ~constantes).
- Objets compacts 3D (bouteilles, bidons, seaux) : masse ≈ proportionnelle à AIRE^1.5 (aire ~ L², volume ~ L³).
- Objets allongés/enroulés (cordage) : l'aire 2D d'un cordage enroulé est un MAUVAIS prédicteur de sa masse (un même cordage peut occuper une aire très différente selon son enroulement, sans changer de masse) - nécessitera probablement un proxy différent (longueur estimée par squelettisation) une fois l'échec du modèle par aire démontré empiriquement pour cette classe, pas supposé d'avance.

**Défaut de qualité de données identifié AVANT toute modélisation** (analyse statistique du tableau brut fourni) : ~76% des lignes appartiennent à des séries de ≥3 valeurs identiques consécutives par classe - signature arithmétique d'une PESÉE PAR LOT (poids total ÷ nombre d'objets, ex: 1,24137931 × 29 = 36,0 exactement) plutôt que des pesées individuelles indépendantes. Confirmé par l'utilisateur (une photo = un lot pesé ensemble, pas un objet). Conséquence : la MOYENNE par classe calculée sur les lignes brutes reste valide (équivaut à une pondération par taille de lot), mais la VARIANCE naïve est sous-estimée (les pseudo-répétitions d'un même lot ont une variance intra-lot nulle qui ne reflète pas la vraie variabilité inter-fragments), et un appariement aire↔poids INDIVIDUEL n'est valide que pour les lignes pesées individuellement - pour les lots, seul un appariement AIRE TOTALE DU LOT ↔ POIDS TOTAL DU LOT a un sens.

**Calibration photo par photo, pas une constante globale** : le plateau de la balance (23,5cm × 19cm, confirmé par l'utilisatrice) sert de référence d'échelle connue visible (au moins partiellement) sur chaque photo - mais le cadrage varie d'une photo à l'autre (confirmé), donc chaque photo nécessite SA PROPRE conversion pixel→cm (détecter la portion visible du plateau → ratio px/cm propre à cette photo → segmenter l'objet → aire en cm²). Une constante d'échelle unique pour tout le lot de photos serait incorrecte. Un repérage semi-manuel (clic utilisateur sur les coins visibles du plateau) est prévu comme filet de sécurité pour les photos où la détection automatique du plateau échouerait, plutôt que de faire une confiance aveugle à l'automatisation sur l'ensemble des ~178 photos.

## Reconstitution du lien numéro de photo ↔ nom de fichier (25/08/2026)

**Problème posé** : le tableau de pesée indique un "numéro de photo" par ligne (1 à ~178), mais les fichiers réels dans `E:\PixelOdyssey\2. Raw data\3. Santa Luzia\5 x 5 qualitatif\Balance\` suivent la convention Android brute `PXL_YYYYMMDD_HHMMSSmmm.jpg` (horodatage de capture, aucun numéro manuel visible dans le nom).

**Inventaire du dossier (`device_list_dir`)** : 183 fichiers au total.
- 179 fichiers `PXL_...jpg` "normaux".
- 4 fichiers `.trashed-<horodatage_suppression>-PXL_...jpg` : convention de la corbeille système Android (scoped storage) - le fichier N'EST PAS supprimé, juste renommé et gardé ~30-60 jours. Sur ce dossier synchronisé, son contenu est donc très probablement toujours présent et récupérable à ce chemin même si Windows/l'explorateur ne le montre pas comme une photo normale.
- 1 fichier `PXL_20260305_152718005.NIGHT.jpg` : cliché "compagnon" Night Sight, généré automatiquement par l'appareil en basse lumière en plus (pas à la place) de la photo normale - PAS un objet pesé distinct.

**Hypothèse testée et confirmée numériquement** : `numéro de photo` = rang chronologique de capture, parmi les photos "propres" UNIQUEMENT (hors les 4 `.trashed` ET hors le compagnon `.NIGHT`).
- En triant les 183 fichiers par l'horodatage embarqué dans le nom (et non par l'ordre retourné par le lister, qui regroupe d'abord les `.trashed-` par tri alphabétique du préfixe) : 179 non-trashed, dont 1 `.NIGHT` → **178 photos "propres" exactement**, cohérent avec le maximum de "numéro de photo" observé dans le tableau fourni par l'utilisatrice.
- Vérification croisée avec l'exemple cité par l'utilisatrice elle-même (`PXL_20260305_100342624.jpg`) : ce fichier est le **5ᵉ** de la séquence chronologique propre - à confronter à ce que dit la ligne "numéro de photo = 5" du tableau (objet/poids) pour une confirmation définitive avant de généraliser la règle à tout le dossier.

**Deux réserves à vérifier avant de généraliser cette règle aux ~178 lignes** :
1. Les 4 photos `.trashed` ont des horodatages de SUPPRESSION très récents (autour d'aujourd'hui, pas de mars 2026) mais des horodatages de CAPTURE bien intercalés dans la séquence normale (espacement de quelques minutes avec leurs voisines, cohérent avec le rythme du reste de la session - pas des doublons immédiats). Elles ont donc probablement été prises pendant la session terrain de mars, puis supprimées récemment (raison inconnue) - et le fait que leur exclusion tombe pile sur 178 suggère qu'elles n'ont VRAISEMBLABLEMENT jamais eu de numéro de photo attribué (photos ratées/écartées sur le moment), mais ce n'est qu'une déduction numérique, pas une certitude. Si l'utilisatrice se souvient avoir supprimé des photos par erreur ou tardivement APRÈS la session (plutôt que sur le terrain), la règle se complique : il faudrait alors les réintégrer à leur place chronologique, ce qui décale tous les numéros après leur position.
2. Quatre écarts de temps anormalement longs dans la séquence (10 min, 76 min, 138 min, 12 min) correspondent probablement à des pauses (déjeuner, déplacement entre zones de la grille 5×5) - à confirmer, sans impact sur la numérotation si ce sont bien de simples pauses.

**Table de correspondance générée** (`balance_crosswalk_rang_fichier.csv`, rang chronologique → nom de fichier → horodatage) et livrée à l'utilisatrice pour vérification croisée manuelle avec son tableau de pesée - en priorité sur les rangs qui suivent immédiatement un fichier exclu `.trashed`/`.NIGHT` (c'est là qu'un décalage se révélerait en premier si l'hypothèse d'exclusion était fausse pour l'un de ces cas).

**Confirmation empirique (25/08/2026)** : l'utilisatrice a fait le tri manuel de son côté (suppression d'une photo supplémentaire, `PXL_20260305_101911041.jpg`, en plus des 4 `.trashed` et du `.NIGHT` déjà identifiés) — le dossier `Balance` contient maintenant exactement **177 photos**, dont l'ordre alphabétique coïncide désormais avec l'ordre chronologique. Vérifié sur deux repères donnés par l'utilisatrice : `numéro de photo 43` = `PXL_20260305_110755655.jpg` (rang 43 dans la liste triée) et `numéro de photo 12` = 12ᵉ photo de la séquence — les deux correspondent exactement. **Règle définitivement validée** : `numéro de photo` = rang chronologique (1 à 177) dans le dossier `Balance` nettoyé, sans aucune exception à gérer. Table de correspondance finale régénérée (177 lignes, colonnes `numero_photo`/`fichier`/`horodatage_capture`) et repoussée à l'utilisatrice.

**Recommandation de process, indépendante du résultat de cette vérification** : cesser de coller le tableau de pesée dans le chat sous forme de texte - la version à 360 lignes de ce tableau a été perdue lors d'une compaction de contexte entre deux messages (mécanisme de résumé automatique de la conversation), ce qui a empêché de vérifier directement le contenu du tableau au moment de reconstruire cette correspondance. Un fichier Excel/CSV réel, uploadé en pièce jointe, est lu et traité programmatiquement (gestion propre des virgules décimales françaises, déduplication des lots pesés ensemble, jointure avec la table de correspondance ci-dessus) sans ce risque de perte ni de retranscription manuelle - à faire pour la suite de ce chantier.

## Nouvel outil : validation assistée pour démarrer un nouveau lot (25/08/2026)

**Besoin exprimé** : une petite interface pour passer en revue, une à une, les prédictions du modèle sur une image/mosaïque choisie (confiance >50%), avec la possibilité de valider le masque+classe, corriger la classe, ou rejeter une fausse alerte - puis écrire directement les nouvelles annotations. Reprend directement la piste "annotation assistée CVAT - cold-start pour un nouveau lot jamais entraîné dessus" notée dans *En rade*.

**Différence structurelle avec `label_review.py`** (pas juste une question d'interface) : `label_review.py` suppose toujours un lot DÉJÀ annoté - il complète/corrige une vérité terrain existante, et sa docstring parle de "human-in-the-loop" alors qu'en réalité il n'y a AUCUNE interface : il injecte automatiquement les prédictions confiantes dans un export CVAT, et c'est CVAT qui sert d'interface de validation. Le besoin exprimé ici est l'inverse - une image encore vierge de toute annotation - donc un vrai nouvel outil, pas un doublon.

**Décision d'architecture - écriture directe dans `1_annotated_dataset/<nouveau_lot>/`** : plutôt qu'un format de sortie maison, le lot produit (images/ + labels/ + data.yaml) est immédiatement ingérable par `split_dataset.py` et tout le pipeline standard - vérifié par un test qui confirme que `raw_dataset.collect_parent_images` le détecte correctement.

**Simplification rendue possible par le cold-start** : un nouveau lot n'a par construction AUCUNE taxonomie fine existante (contrairement à SB/SL/A LEG, ~20 sous-classes chacun) - inventer une sous-classe qui n'existerait que pour ce lot n'aurait aucun sens. Son `data.yaml` déclare donc directement les 8 SUPER-classes de `config/data_config.yaml` comme classes locales. Pour que `raw_dataset_checker.py` accepte ces noms (ex: "Bouteille") sans les rejeter comme inconnus (seules les sous-classes fines comme "bouteille PET" étaient jusqu'ici des entrées de `class_taxonomy`), **8 entrées d'auto-mapping ont été ajoutées à `class_taxonomy`** (`"Bouteille": 0`, etc. - sans effet sur les autres lots). Conséquence directe : pas de "placeholder" à corriger plus tard dans CVAT comme pour `label_review.py` - la classe validée dans l'interface EST la classe finale, tout de suite. Contrepartie assumée : ce lot n'aura jamais de sous-classe fine tant que quelqu'un ne le réannote pas plus précisément à la main.

**Décision d'interface (25/08/2026, discutée avant de coder)** : pas de 3e bouton "classe à revoir" séparé comme envisagé initialement - un bandeau de classe ÉDITABLE (menu déroulant pré-rempli avec la classe prédite) + 2 boutons seulement, Valider (écrit le masque avec la classe ACTUELLEMENT affichée dans le bandeau, prédite ou changée) et Supprimer. Plus simple qu'un 3e état à gérer séparément, et un seul geste suffit même pour reclasser.

**Architecture technique** : nouveau module `src/review/bootstrap_annotate.py`. Réutilise au maximum l'existant plutôt que de dupliquer : sélection interactive du modèle (`_discover_available_models`/`_prompt_model_choice` de `label_review.py`), inférence tuilée (`tiled_inference.predict_parent_image`, même géométrie qu'à l'entraînement), chargement image (`image_io.load_image_bgr`). Double seuil de confiance identique au pattern déjà établi dans `label_review.py` (filtre large 25% par tuile avant fusion des recouvrements, seuil métier 50% après fusion - configurable). Interface : petite page web locale servie par le serveur HTTP intégré à Python (`http.server`, aucune dépendance ajoutée) plutôt qu'une fenêtre OpenCV au clavier - chip à résolution native avec contour du masque en surimpression (même esprit que les chips de `geo_density_map.py`), raccourcis clavier (Entrée = Valider, Suppr = Supprimer) en plus des boutons. Écrit le lot à la fin de la revue, ou plus tôt via un bouton "Enregistrer et terminer maintenant".

**Testé** (scénario synthétique, `predict_tile_fn` injecté avec 4 fausses détections dont une sous le seuil - même pattern de test que le reste des outils de `src/review/`) : filtrage par seuil de confiance vérifié, reclassement via le bandeau vérifié (une détection prédite "Bouchon" validée sous "Bidon"), suppression vérifiée (exclue du fichier de labels), coordonnées normalisées vérifiées dans [0,1], et confirmation que `raw_dataset.collect_parent_images` détecte bien le nouveau lot produit. Poussé chez l'utilisateur (`src/review/bootstrap_annotate.py` + `config/data_config.yaml` mis à jour) - non encore testé en conditions réelles avec un vrai modèle/vraie image.

**Usage** : `python -m src.review.bootstrap_annotate --image "chemin/vers/image.tif" --lot-name "nom_du_nouveau_lot"`, puis ouvrir l'URL affichée (http://127.0.0.1:8765/ par défaut, ouverte automatiquement).

## Bug corrigé : TIFF pyramidal illisible via cv2.imread sous Windows (25/08/2026)

**Découvert au premier vrai essai de `bootstrap_annotate.py`**, sur `SL 28-30 avt.tif` (187 Mo) :

```
ValueError: all input arrays must have the same shape
  ... ultralytics/utils/patches.py, in imread
    return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
```

**Cause racine, pas un bug OpenCV cette fois** : sous Windows, `ultralytics` remplace purement et simplement `cv2.imread`/`cv2.imwrite`/`cv2.imshow` par sa propre implémentation dès qu'il est importé (`cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow` dans `ultralytics/utils/__init__.py`, réservé à Windows - support des chemins non-ASCII). Sa version d'`imread` lit tout `.tif`/`.tiff` avec `cv2.imdecodemulti` (pensé pour des TIFF MULTI-PAGES, ex: scans) et empile toutes les "pages" trouvées avec `np.stack`. Une orthomosaïque WebODM comme celle-ci embarque typiquement une pyramide d'aperçus basse résolution en plus de l'image pleine résolution - des pages de tailles DIFFÉRENTES, que `np.stack` ne peut pas empiler → crash immédiat.

**Portée du bug, plus large que `bootstrap_annotate.py`** : ce patch Windows s'applique GLOBALEMENT dès qu'un modèle Ultralytics est chargé n'importe où dans le process - donc n'importe quel outil du projet qui appelle `image_io.load_image_bgr()` sur un TIFF pyramidal aurait pu planter de la même façon, pas seulement ce nouvel outil. Les TIFF déjà traités sans souci jusqu'ici (`transect_11.tif` et consorts, ~8 Mo, un export par transect) n'ont simplement jamais eu cette structure pyramidale - une pleine orthomosaïque de 187 Mo si. Un bug latent depuis la création d'`image_io.py` (23/08/2026), jamais déclenché avant faute d'avoir pointé le chargement cv2 sur un fichier de cette nature.

**Correctif structurel** : `image_io.load_image_bgr()` ne passe plus du tout par `cv2.imread` pour un `.tif`/`.tiff` - lecture via `rasterio`/GDAL à la place (déjà une dépendance du projet, c'est exactement pour cette même raison - RAM/fenêtrage - que `geo_density_map.py` l'utilisait déjà avec succès sur ce même fichier). `rasterio` comprend la structure interne du GeoTIFF (bande principale vs pyramide d'aperçus) au lieu de tout traiter comme des pages génériques équivalentes. JPEG/PNG restent chargés via `cv2.imread` comme avant, jamais concernés par ce bug (pas de notion de pages/pyramide dans ces formats). Filet de sécurité ajouté en même temps : conversion 8 bits si un futur TIFF arrivait en 16 bits/flottant (pas le cas aujourd'hui, tous les lots sont en 8 bits).

**Testé** (5 cas synthétiques via rasterio, dont le déclencheur exact du bug) : RGB + pyramide d'aperçus (le cas qui plantait), RGBA 4 bandes (régression du bug du 23/08 - toujours géré), TIFF niveaux de gris 1 bande, JPEG inchangé (toujours via cv2), TIFF 16 bits (conversion 8 bits). Tout passe. Non re-testé avec le vrai fichier `SL 28-30 avt.tif` sur la machine de l'utilisatrice à ce stade.

**Confirmation empirique (25/08/2026)** : le correctif tient - `bootstrap_annotate.py` tourne jusqu'au bout sur `SL 28-30 avt.tif` sans replanter. Question soulevée ensuite par l'utilisatrice : seulement 4 déchets trouvés, alors qu'elle en attendait davantage. Vérification de `tiling_geometry.iter_tile_windows` et `tiled_inference.predict_parent_image` : le balayage couvre bien l'image entière (fenêtres en butée recalées, jamais tronquées/abandonnées) et la fusion de recouvrement (`nms_merge`) ne fusionne que des détections de même classe à IoU ≥ 0.5 - la géométrie de tuilage n'est PAS en cause. Diagnostic renvoyé à l'utilisatrice pour trancher entre "le seuil de confiance 50% coupe légitimement beaucoup de détections brutes" vs "le modèle est réellement peu sensible sur ce type de grande orthomosaïque" (ligne `N détection(s) brute(s), M au-dessus de X% de confiance` demandée) - hypothèse GSD/résolution de `SL 28-30 avt.tif` (187 Mo) vs les TIFF transect déjà traités (~8 Mo) également soulevée comme piste. Pas encore tranché - à reprendre si l'utilisatrice revient avec ces chiffres.

## Bug corrigé : import CVAT de `bootstrap_annotate.py` n'importe RIEN (25/08/2026)

**Découvert en important le lot produit par `bootstrap_annotate.py` dans un nouveau projet CVAT dédié** (guidage donné le même jour : nouveau projet séparé, 8 labels dans l'ordre exact de `target_names`, plutôt qu'une tâche dans un des projets existants par lot - cohérent avec la pratique déjà en place d'un projet CVAT par lot/groupe, et avec la décision du 25/08 selon laquelle ce lot cold-start reste volontairement aux 8 super-classes) : l'import du zip se déroule sans erreur visible, mais **zéro annotation** apparaît, ni sur l'image ni dans le panneau Objects.

**Cause racine** : `_write_new_lot()` (dans `bootstrap_annotate.py`) écrivait un `data.yaml` avec seulement la clé `names` - sans clé `path` ni `train`/`val`/`test` pointant vers le dossier `images/`. Sans cette clé, l'importeur CVAT (format "Ultralytics YOLO Segmentation") n'a aucun moyen de savoir où se trouvent les images à associer aux fichiers de labels - il n'échoue pas bruyamment, il importe simplement zéro objet. **Exactement la même classe de bug** que celle déjà rencontrée et corrigée le 23/08/2026 dans l'export de `label_review.py` (voir "Bug corrigé : data.yaml de l'export de relecture invalide pour l'import CVAT" plus haut, résolue par `_write_review_data_yaml`) - `_write_new_lot()` avait été écrit indépendamment le 25/08 sans réutiliser ce correctif déjà connu, et a donc réintroduit le même problème.

**Correctif** : `_write_new_lot()` écrit maintenant `path: "."` et `train: "images"` en plus de `names` - sans changer la structure de dossiers déjà écrite (`images/`, `labels/` à plat, PAS de sous-dossier `images/train/`) puisque `train: "images"` pointe directement dessus, et que CVAT retrouve les labels en substituant `images`→`labels` dans ce même chemin (convention standard Ultralytics). Choix déféré à une restructuration en `images/<split>/` (qui aurait mécaniquement copié la convention de `_write_review_data_yaml`) car cette structure plate est déjà celle testée et vérifiée compatible avec `raw_dataset.collect_parent_images`/`class_config.load_batch_local_names` pour l'ingestion dans le pipeline standard - `load_batch_local_names` ne lit que la clé `names`, confirmé par relecture du code, donc ajouter `path`/`train` ne risque aucune régression côté pipeline.

**Testé** : régénération d'un lot factice via `_write_new_lot()`, `data.yaml` produit vérifié (`path`/`train`/`names` présents), et `load_batch_local_names` revérifié pour confirmer qu'il retrouve toujours exactement les mêmes 8 classes malgré les 2 clés ajoutées. Poussé chez l'utilisatrice. **Non encore reconfirmé par un import CVAT réel** - à valider au prochain essai (si le zéro-import persiste, la piste suivante serait l'ordre exact des labels créés manuellement dans le projet CVAT, potentiellement pas aligné avec les ID de `target_names` malgré la consigne donnée).

## État au 25/08/2026 — prochaine action

Trois chantiers ouverts en parallèle, indépendants les uns des autres :

1. **Ablation `copy_paste`** : les deux vérifications recommandées sont faites (audit de taille + inspection visuelle) - la taille de `Bouchon` est confirmée par les chiffres, `Cordage_Filet` est un problème de forme plutôt que de taille, et une cause concrète et séparée (probablement `copy_paste`) a été trouvée pour l'explosion de fausses alertes sur les grandes images SL. `COPY_PASTE=0.0` déjà appliqué dans `train.py` - prochaine action : relancer l'entraînement pour confirmer que c'est bien la cause, ET revérifier `Bouchon`/`Cordage_Filet` sur un run sans ce bruit de fond avant de conclure définitivement sur eux. Toujours pas de décision à prendre sur annotation assistée vs dataset synthétique tant que ce cycle de vérification n'est pas bouclé.
2. **Table de conversion masque → poids** : correspondance numéro de photo ↔ fichier **confirmée définitivement** (voir section ci-dessus) — `numéro de photo` = rang chronologique dans le dossier `Balance` nettoyé (177 photos), plus aucune ambiguïté. Prochaine étape : l'utilisatrice upload le tableau de pesée comme fichier réel (pas collé en texte, pour éviter de reproduire la perte de données déjà vécue), puis prototyper la détection du plateau de balance (23,5×19cm) sur les 5 photos d'exemple pour établir la conversion pixel→cm par photo, avant de joindre aire↔poids pour construire le modèle par famille d'objet (linéaire / aire^1.5 / longueur pour le cordage).
3. **Validation assistée nouveau lot (`bootstrap_annotate.py`)** : construit et testé en synthétique, puis confronté au réel en plusieurs vagues successives. TIFF pyramidal (bug du 25/08 dans `image_io.py`) : corrigé et confirmé tenir sur `SL 28-30 avt.tif` en conditions réelles. Question ouverte "seulement 4 détections" : tuilage/fusion vérifiés sains, pas encore tranché entre seuil de confiance légitime et modèle peu sensible sur ce type de grande orthomosaïque (voir section TIFF ci-dessus, paragraphe "Confirmation empirique") - en attente des chiffres bruts de l'utilisatrice. Import CVAT du lot produit : guidage donné (nouveau projet CVAT dédié, 8 labels dans l'ordre de `target_names`), premier essai réel a révélé un `data.yaml` incomplet (0 annotation importée) - corrigé (voir section "Bug corrigé : import CVAT de `bootstrap_annotate.py` n'importe RIEN"), **pas encore reconfirmé par un import réussi**. Prochaine étape : l'utilisatrice retente l'import CVAT avec le `data.yaml` corrigé et confirme que les annotations apparaissent bien cette fois.
