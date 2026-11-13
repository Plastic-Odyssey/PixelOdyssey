# Orthomosaïques ODM / WebODM (Santa Luzia)

Scripts qui ont produit les orthomosaïques des vols 1 à 10 de Santa Luzia (26 au 28/10/2026, PC DESKTOP-67RRNLE).
Copie fidèle de `C:\odm_runs\` : ils appellent directement le moteur **ODX 3.8.2** livré avec WebODM
(`C:\WebODM\resources\app\apps\ODX\run.bat`), pas l'interface web.

## Fichiers

| Fichier | Rôle |
|---|---|
| `lot_odx.ps1` | Script principal : enchaîne les vols (copie des photos sans la photo de décollage, calcul ODX, compression DEFLATE + aperçus, contrôles, copie sur D: avec vérification MD5, suppression du projet sur C:). Gère le mode « split » (sous-modèles masqués puis assemblés par `gdalwarp`). Un vol qui a déjà un `FAIT.txt` est sauté. |
| `w1b_odx.ps1` | Vol W1 : reconstruction en un seul bloc, arrêtée à `odm_dem`. |
| `troncons_w1.ps1` | Vol W1 (une seule ligne de ~900 m) : découpe du modèle texturé en tronçons, `odm_orthophoto` par tronçon, assemblage `gdalwarp`, masque d'emprise. |
| `decouper_obj.py` | Découpe le modèle texturé 2,5D (`odm_textured_model_geo.obj`) en tronçons. |
| `masque_emprise.py` | Masque l'ortho sur l'emprise des photos, écrit `emprise.geojson`, mode `--controle`. |
| `controle_ortho.py` | Contrôles finaux : surface, trous, photos perdues, aperçus. |
| `suivi_odm.ps1`, `Suivi_ODX.bat` | Console de suivi pendant le calcul. |
| `creer_projet_qgis.py`, `vue_ensemble.py` | Projet QGIS et vue d'ensemble de Santa Luzia. |

Les scripts Python tournent avec le Python de l'ODX (`...\ODX\venv\Scripts\python.exe`), qui fournit
rasterio, pyproj, shapely, scipy et PIL.

## Recette ODX

```
--fast-orthophoto --feature-quality high --min-num-features 20000 --matcher-neighbors 24
--orthophoto-resolution 0.49 --ignore-gsd --crop 0 --skip-3dmodel --max-concurrency 12
```

- W9 et W10 : `--split 220` / `--split 210` avec `--split-overlap 20 --end-with split`.
- W1 : le découpage en sous-modèles a été écarté (sous-modèles mal recalés, décalage d'environ 14 m),
  d'où la méthode par tronçons.

## Avant de relancer

- **Chemin de sortie périmé** : `$resultSL` (`lot_odx.ps1`) et `$dst` (`troncons_w1.ps1`) pointent vers
  `D:\PixelOdyssey\4. Results\ODM Santa Luzia\`. Les mosaïques ont depuis été déplacées dans
  `D:\PixelOdyssey\2. Raw data\3. Santa Luzia\Mosaic Santa Luzia\`. Sans correction, les `FAIT.txt`
  ne sont pas vus et tous les vols sont recalculés.
- **Liste des vols** : `$vols` dans `lot_odx.ps1` ne contient plus que le dernier lot (W1). Ajouter
  les vols voulus avec leurs photos à exclure et la taille de `split`.
- **Chemins en dur** : `C:\odm_runs`, `C:\WebODM\...`, zone UTM 26N (`troncons_w1.ps1`).
- Seul Santa Luzia est scripté ; aucun script pour Aldabra ou Saint Brandon.
