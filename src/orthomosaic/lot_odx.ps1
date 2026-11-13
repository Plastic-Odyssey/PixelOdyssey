# Lot ODX Santa Luzia : enchaîne plusieurs vols avec la recette validée sur W2 à W7.
# Pour chaque vol : copie des photos (sans les photos de décollage), calcul ODX à 0,49 cm/px,
# finalisation (DEFLATE + aperçus), contrôles, copie sur D:, vérification MD5, puis suppression du projet sur C:.
# Vol « split » : découpage ODX en sous-modèles (--end-with split), chaque ortho de sous-modèle est masquée
# sur l'emprise de ses propres photos, puis assemblage gdalwarp (le dernier sous-modèle l'emporte).
# Un vol dont D:\...\FAIT.txt existe est sauté. Journal : C:\odm_runs\lot_journal.txt
$ErrorActionPreference = 'Continue'

# Lot du 28/10 : W1, vol manuel sur une seule ligne d'environ 900 m (nacelle à -90 deg), découpé en sous-modèles d'environ 50 photos.
# Lots précédents : W8 (un bloc), W9 (split 220, sans DJI_0422), W10 (split 210, sans DJI_0856).
$vols = @(
    @{ nom = 'W1'; exclure = @('DJI_0287.JPG'); split = 50 }
)

$runs     = 'C:\odm_runs'
$photosSL = 'D:\PixelOdyssey\2. Raw data\3. Santa Luzia\Raw pictures'
$resultSL = 'D:\PixelOdyssey\4. Results\ODM Santa Luzia'
$odx      = 'C:\WebODM\resources\app\apps\ODX'
$py       = "$odx\venv\Scripts\python.exe"
$osgeo    = "$odx\venv\Lib\site-packages\osgeo"
$env:GDAL_DATA = "$osgeo\data\gdal"; $env:PROJ_LIB = "$osgeo\data\proj"; $env:PROJ_DATA = "$osgeo\data\proj"
Remove-Item Env:NoDefaultCurrentDirectoryInExePath -ErrorAction SilentlyContinue
$journal  = "$runs\lot_journal.txt"
$recette  = '--fast-orthophoto --feature-quality high --min-num-features 20000 --matcher-neighbors 24 --orthophoto-resolution 0.49 --ignore-gsd --crop 0 --skip-3dmodel --max-concurrency 12'
$co       = @('-co', 'TILED=YES', '-co', 'BLOCKXSIZE=512', '-co', 'BLOCKYSIZE=512', '-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2', '-co', 'BIGTIFF=YES')

function Note([string]$m) {
    $l = (Get-Date -Format 'dd/MM HH:mm:ss') + '  ' + $m; Write-Host $l
    # le journal peut être momentanément ouvert par une console de suivi : quelques essais, sans bloquer le lot
    for ($i = 0; $i -lt 5; $i++) { try { Add-Content -Path $journal -Value $l -Encoding UTF8 -ErrorAction Stop; break } catch { Start-Sleep -Milliseconds 500 } }
}

foreach ($vol in $vols) {
    $nom = $vol.nom; $id = 'sl_' + $nom.ToLower(); $numero = $nom.Substring(1)
    # noms des dossiers depuis le renommage du 28/10/2026
    $src = Join-Path $photosSL "Raw pictures Santa Luzia flight $numero"
    $nomMosaique = "Mosaic Santa Luzia flight $numero"
    $dst = Join-Path $resultSL $nomMosaique
    $projet = Join-Path $runs $id; $final = Join-Path $runs ($id + '_final'); $log = Join-Path $runs ($id + '_console.log')
    if (Test-Path (Join-Path $dst 'FAIT.txt')) { Note "$nom deja fait, saute"; continue }
    Note ("$nom : debut (C: {0:N0} Go libres)" -f ((Get-PSDrive C).Free / 1GB))

    # 1. Photos
    $img = Join-Path $projet 'images'
    New-Item -ItemType Directory -Force $img | Out-Null
    $aCopier = Get-ChildItem -LiteralPath $src -File -Filter *.JPG | Where-Object { $vol.exclure -notcontains $_.Name }
    if (@(Get-ChildItem $img -File).Count -ne @($aCopier).Count) { $aCopier | Copy-Item -Destination $img -Force }
    Note ("$nom : {0} photos (exclues : {1})" -f @(Get-ChildItem $img -File).Count, ($(if ($vol.exclure) { $vol.exclure -join ', ' } else { 'aucune' })))

    # 2. Calcul ODX
    $opts = $recette
    if ($vol.split -gt 0) { $opts += " --split $($vol.split) --split-overlap 20 --end-with split" }
    $cmd = "run.bat --project-path $runs $id $opts"
    Note "$nom : ODX $cmd"
    Push-Location $odx
    cmd /c "$cmd > $log 2>&1"
    Pop-Location
    Note "$nom : ODX termine"

    # 3. Ortho brute (un bloc) ou assemblage des sous-modeles
    New-Item -ItemType Directory -Force $final | Out-Null
    $ortho = Join-Path $final 'ortho_odm.tif'
    if ($vol.split -gt 0) {
        $sous = Get-ChildItem (Join-Path $projet 'submodels') -Directory -Filter 'submodel_*' -ErrorAction SilentlyContinue | Sort-Object Name
        $morceaux = @()
        foreach ($s in $sous) {
            $t = Join-Path $s.FullName 'odm_orthophoto\odm_orthophoto_feathered.tif'
            if (-not (Test-Path $t)) { $t = Join-Path $s.FullName 'odm_orthophoto\odm_orthophoto.tif' }
            if (-not (Test-Path $t)) { Note "$nom : ATTENTION pas d'ortho pour $($s.Name)"; continue }
            $m = Join-Path $final ($s.Name + '_masque.tif')
            & $py -I "$runs\masque_emprise.py" $t (Join-Path $s.FullName 'images') --sortie $m 2>&1 | ForEach-Object { Note "$nom : $($s.Name) $_" }
            if (Test-Path $m) { $morceaux += $m }
        }
        if ($morceaux.Count -eq 0) { Note "$nom : ECHEC aucune ortho de sous-modele, vol abandonne"; continue }
        Note "$nom : assemblage de $($morceaux.Count) sous-modeles"
        & "$osgeo\gdalwarp.exe" -overwrite -srcalpha -dstalpha -r near -tr 0.0049 0.0049 -multi -wm 2048 -wo NUM_THREADS=ALL_CPUS -wo SKIP_NOSOURCE=YES --config GDAL_CACHEMAX 2048 @co @morceaux $ortho 2>&1 | Select-Object -Last 2 | ForEach-Object { Note "$nom : gdalwarp $_" }
    } else {
        $brute = Join-Path $projet 'odm_orthophoto\odm_orthophoto.tif'
        if (-not (Test-Path $brute)) { Note "$nom : ECHEC pas d'ortho (voir $log), vol abandonne"; continue }
        & "$osgeo\gdal_translate.exe" @co -co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 4096 $brute $ortho 2>&1 | Select-Object -Last 1 | ForEach-Object { Note "$nom : gdal_translate $_" }
    }
    if (-not (Test-Path $ortho)) { Note "$nom : ECHEC ortho finale absente, vol abandonne"; continue }
    & "$osgeo\gdaladdo.exe" -r average --config COMPRESS_OVERVIEW DEFLATE --config PREDICTOR_OVERVIEW 2 --config GDAL_NUM_THREADS ALL_CPUS --config GDAL_CACHEMAX 4096 $ortho 2 4 8 16 32 64 128 256 | Out-Null

    # 4. Controles
    $emprise = & $py -I "$runs\masque_emprise.py" $ortho $img --controle --geojson (Join-Path $final 'emprise.geojson') 2>&1
    $stats   = & $py -I "$runs\controle_ortho.py" $ortho $final --projet $projet --emprise (Join-Path $final 'emprise.geojson') 2>&1
    ($emprise + $stats) | ForEach-Object { Note "$nom : $_" }

    # 5. Copie sur D: et verification
    New-Item -ItemType Directory -Force $dst | Out-Null
    $orthoD = Join-Path $dst "$nomMosaique.tif"
    Copy-Item $ortho $orthoD -Force
    foreach ($f in 'apercu_1sur32.png', 'zoom_centre.jpg', 'controle_emprise.jpg', 'emprise.geojson') {
        if (Test-Path (Join-Path $final $f)) { Copy-Item (Join-Path $final $f) $dst -Force }
    }
    Copy-Item $log (Join-Path $dst 'odm_console.log') -Force
    $rapport = Get-ChildItem $projet -Recurse -Filter report.pdf -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($rapport) { Copy-Item $rapport.FullName (Join-Path $dst 'rapport_odm.pdf') -Force }
    $ok = (Get-FileHash $ortho -Algorithm MD5).Hash -eq (Get-FileHash $orthoD -Algorithm MD5).Hash
    $mode = if ($vol.split -gt 0) { "decoupe en sous-modeles (--split $($vol.split) --split-overlap 20), orthos des sous-modeles masquees sur leur emprise puis assemblees par gdalwarp" } else { 'un seul bloc, sans split' }
    @(
        "$nomMosaique (ancien nom Santa Luzia $nom - mar) : ortho ODX terminee le $(Get-Date -Format 'dd/MM/yyyy HH:mm') (PC DESKTOP-67RRNLE, ODX 3.8.2, lot_odx.ps1)",
        "Photos exclues (decollage) : $(if ($vol.exclure) { $vol.exclure -join ', ' } else { 'aucune' })",
        "Commande : $cmd",
        "Mode : $mode",
        'Controles :'
    ) + (($emprise + $stats) | ForEach-Object { '  ' + $_ }) + @("Copie identique a l'original (MD5) : $ok") | Out-File (Join-Path $dst 'FAIT.txt') -Encoding utf8
    if ($ok) {
        Get-ChildItem $final -Filter 'submodel_*_masque.tif' | Remove-Item -Force
        Remove-Item -Recurse -Force $projet -Confirm:$false
        Note ("$nom : FAIT, copie sur D: verifiee, projet supprime (C: {0:N0} Go libres)" -f ((Get-PSDrive C).Free / 1GB))
    } else {
        Note "$nom : ATTENTION copie sur D: differente, projet garde"
    }
}
Note 'Lot termine'
