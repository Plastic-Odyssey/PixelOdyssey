# W1 (vol manuel sur une seule ligne) : ortho calculée tronçon par tronçon à partir d'UN SEUL modèle ODX.
# Prérequis : C:\odm_runs\sl_w1b calculé en un bloc avec --end-with odm_dem (modèle texturé 2,5D prêt).
# 1. découpe du modèle texturé en tronçons (decouper_obj.py) ; 2. odm_orthophoto sur chaque tronçon ;
# 3. assemblage gdalwarp (tronçons identiques dans les recouvrements) ; 4. masque d'emprise, aperçus internes,
# contrôles ; 5. remplacement de l'ancienne ortho W1 (mal raccordée) sur D:, vérification MD5.
$ErrorActionPreference = 'Continue'
$runs   = 'C:\odm_runs'
$projet = "$runs\sl_w1b"
$final  = "$runs\sl_w1b_final"
$tron   = "$projet\troncons"
$dst    = 'D:\PixelOdyssey\4. Results\ODM Santa Luzia\Mosaic Santa Luzia flight 1'
$nomMosaique = 'Mosaic Santa Luzia flight 1'
$odx    = 'C:\WebODM\resources\app\apps\ODX'
$py     = "$odx\venv\Scripts\python.exe"
$osgeo  = "$odx\venv\Lib\site-packages\osgeo"
$sbbin  = "$odx\SuperBuild\install\bin"
$env:PATH = "$osgeo;$sbbin;$env:PATH"
$env:GDAL_DATA = "$osgeo\data\gdal"; $env:GDAL_DRIVER_PATH = "$osgeo\gdalplugins"
$env:PROJ_LIB = "$odx\venv\Lib\site-packages\rasterio\proj_data"; $env:PROJ_DATA = $env:PROJ_LIB
$env:OMP_NUM_THREADS = '12'
$journal = "$runs\w1b_journal.txt"
$co = @('-co', 'TILED=YES', '-co', 'BLOCKXSIZE=512', '-co', 'BLOCKYSIZE=512', '-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2', '-co', 'BIGTIFF=YES')

function Note([string]$m) {
    $l = (Get-Date -Format 'dd/MM HH:mm:ss') + '  ' + $m; Write-Host $l
    for ($i = 0; $i -lt 5; $i++) { try { Add-Content -Path $journal -Value $l -Encoding UTF8 -ErrorAction Stop; break } catch { Start-Sleep -Milliseconds 500 } }
}

Note ("W1 troncons : debut (C: {0:N0} Go libres)" -f ((Get-PSDrive C).Free / 1GB))
if (-not (Test-Path "$projet\odm_texturing_25d\odm_textured_model_geo.obj")) { Note 'ECHEC : modele texture absent'; exit 1 }

# 1. Decoupe
& $py -I "$runs\decouper_obj.py" $projet --photos-par-troncon 45 --recouvrement 5 2>&1 | ForEach-Object { Note "decoupe : $_" }

# 2. Ortho de chaque troncon (meme commande qu'ODX, a 0,49 cm/px)
$coords = Get-Content "$projet\odm_georeferencing\coords.txt"
$est, $nord = ($coords[1] -split '\s+')[0..1]
New-Item -ItemType Directory -Force $tron | Out-Null
$objs = Get-ChildItem "$projet\odm_texturing_25d" -Filter 'troncon_*.obj' | Sort-Object Name
foreach ($o in $objs) {
    $t = Join-Path $tron ($o.BaseName + '.tif')
    if (Test-Path $t) { Note "$($o.BaseName) deja calcule"; continue }
    Note "$($o.BaseName) : odm_orthophoto"
    & "$sbbin\odm_orthophoto.exe" -inputFiles $o.FullName -logFile (Join-Path $tron ($o.BaseName + '_log.txt')) -outputFile $t `
        -resolution 204.08163265306123 -verbose -outputCornerFile (Join-Path $tron ($o.BaseName + '_corners.txt')) -inpaintThreshold 1.0 `
        -utm_north_offset $nord -utm_east_offset $est -a_srs '+proj=utm +zone=26 +datum=WGS84 +units=m +no_defs +type=crs' `
        -co TILED=YES -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=IF_SAFER -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 -co NUM_THREADS=12 `
        --config GDAL_CACHEMAX 4294967296 2>&1 | Where-Object { $_ -match 'Model bounds|Creating output file|rror|xception' } | ForEach-Object { Note "$($o.BaseName) : $_" }
    if (Test-Path $t) { Note ("{0} : fait ({1:N2} Go)" -f $o.BaseName, ((Get-Item $t).Length / 1GB)) } else { Note "$($o.BaseName) : ECHEC" }
}
$tifs = @(Get-ChildItem $tron -Filter 'troncon_*.tif' | Sort-Object Name | ForEach-Object FullName)
if ($tifs.Count -ne $objs.Count) { Note "ECHEC : $($tifs.Count) orthos de troncon sur $($objs.Count)"; exit 1 }

# 3. Assemblage puis 4. masque d'emprise des photos, apercus, controles
New-Item -ItemType Directory -Force $final | Out-Null
$brut = "$final\ortho_brute.tif"; $ortho = "$final\ortho_odm.tif"
& "$osgeo\gdalwarp.exe" -overwrite -srcalpha -dstalpha -r near -tr 0.0049 0.0049 -multi -wm 2048 -wo NUM_THREADS=ALL_CPUS -wo SKIP_NOSOURCE=YES --config GDAL_CACHEMAX 2048 @co @tifs $brut 2>&1 | Select-Object -Last 1 | ForEach-Object { Note "gdalwarp : $_" }
& $py -I "$runs\masque_emprise.py" $brut "$projet\images" --sortie $ortho 2>&1 | ForEach-Object { Note "masque : $_" }
& "$osgeo\gdaladdo.exe" -r average --config COMPRESS_OVERVIEW DEFLATE --config PREDICTOR_OVERVIEW 2 --config GDAL_NUM_THREADS ALL_CPUS --config GDAL_CACHEMAX 4096 $ortho 2 4 8 16 32 64 128 256 | Out-Null
$emprise = & $py -I "$runs\masque_emprise.py" $ortho "$projet\images" --controle --geojson "$final\emprise.geojson" 2>&1
$stats   = & $py -I "$runs\controle_ortho.py" $ortho $final --projet $projet --emprise "$final\emprise.geojson" 2>&1
($emprise + $stats) | Where-Object { $_ -notmatch '^crs:' } | ForEach-Object { Note "controle : $_" }

# 5. Remplacement sur D:
$orthoD = Join-Path $dst "$nomMosaique.tif"
Copy-Item $ortho $orthoD -Force
foreach ($f in 'apercu_1sur32.png', 'zoom_centre.jpg', 'controle_emprise.jpg', 'emprise.geojson') { if (Test-Path "$final\$f") { Copy-Item "$final\$f" $dst -Force } }
Copy-Item "$runs\sl_w1b_console.log" (Join-Path $dst 'odm_console.log') -Force
Copy-Item $journal (Join-Path $dst 'troncons_journal.txt') -Force
$ok = (Get-FileHash $ortho -Algorithm MD5).Hash -eq (Get-FileHash $orthoD -Algorithm MD5).Hash
@(
    "$nomMosaique (ancien nom Santa Luzia W1 - mar (manuel)) : ortho refaite le $(Get-Date -Format 'dd/MM/yyyy HH:mm') (PC DESKTOP-67RRNLE, ODX 3.8.2)",
    'Vol manuel sur une seule ligne d environ 900 m, nacelle a -90 deg ; photo exclue : DJI_0287.JPG (decollage, aussi presente dans le vol 2)',
    'Premiere version (split 50 en 6 sous-modeles) ECARTEE : sous-modeles mal recales (objets dupliques, decalage d environ 14 m).',
    'Methode : reconstruction ODX en UN seul bloc (--end-with odm_dem), modele texture 2,5D decoupe en troncons (decouper_obj.py),',
    '  odm_orthophoto sur chaque troncon a 0,49 cm/px, assemblage gdalwarp (troncons identiques dans les recouvrements), masque d emprise.',
    'Commande ODX : run.bat --project-path C:\odm_runs sl_w1b --fast-orthophoto --feature-quality high --min-num-features 20000 --matcher-neighbors 24 --orthophoto-resolution 0.49 --ignore-gsd --crop 0 --skip-3dmodel --max-concurrency 12 --end-with odm_dem',
    'Controles :'
) + (($emprise + $stats) | Where-Object { $_ -notmatch '^crs:' } | ForEach-Object { '  ' + $_ }) + @("Copie identique a l'original (MD5) : $ok") | Out-File (Join-Path $dst 'FAIT.txt') -Encoding utf8
Note "W1 : copie sur D: verifiee = $ok"
Note 'W1 troncons : termine'
