# Console de suivi d'un calcul ODX : étapes, progression, RAM, disque, erreurs.
# Usage : powershell -NoProfile -ExecutionPolicy Bypass -File C:\odm_runs\suivi_odm.ps1 [-Log <journal>] [-Intervalle 5]
# Sans -Log : suit en continu le *_console.log le plus récent de C:\odm_runs (pratique pour un lot de vols).
param([string]$Log = '', [int]$Intervalle = 5)

$auto = -not $Log
function DernierJournal {
    $d = Get-ChildItem C:\odm_runs -Filter *_console.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($d) { $d.FullName }
}
if ($auto) {
    $Log = DernierJournal
    if (-not $Log) { Write-Host 'Aucun journal *_console.log dans C:\odm_runs' -ForegroundColor Red; exit 1 }
}

$etapes = 'dataset','split','merge','opensfm','openmvs','odm_filterpoints','odm_meshing','mvs_texturing',
          'odm_georeferencing','odm_dem','odm_orthophoto','odm_report','odm_postprocess'
$debut = (Get-Item $Log).CreationTime
$vuEtape = @{}   # heure à laquelle cette console a vu chaque étape démarrer
$histRam = New-Object System.Collections.Generic.List[double]
$histDisque = New-Object System.Collections.Generic.List[double]
$histMax = 60    # nombre de mesures gardées pour les courbes

# Courbe texte : un caractère par mesure, du plus bas au plus haut de l'échelle 0..max
function Courbe($valeurs, [double]$max) {
    $niveaux = ' _.-=+*#'
    ($valeurs | ForEach-Object { $niveaux[[int][math]::Min(7, [math]::Max(0.0, [math]::Round(7 * $_ / $max)))] }) -join ''
}
function Jauge([double]$utilise, [double]$total, [int]$l = 30) {
    $p = [math]::Min(1.0, [math]::Max(0.0, $utilise / $total))
    '[' + ('#' * [int]($p * $l)) + ('.' * ($l - [int]($p * $l))) + ']' + (' {0:P0}' -f $p)
}

function Barre([int]$n, [int]$total) {
    if ($total -le 0) { return '' }
    $p = [math]::Min(1.0, $n / $total); $l = 30
    '[' + ('#' * [int]($p * $l)) + ('.' * ($l - [int]($p * $l))) + ']  ' + $n + ' / ' + $total + ('  ({0:P0})' -f $p)
}
function Duree([timespan]$t) { '{0:00}h{1:00}m{2:00}s' -f [int][math]::Floor($t.TotalHours), $t.Minutes, $t.Seconds }

while ($true) {
    if ($auto) {
        $nouveau = DernierJournal
        if ($nouveau -and $nouveau -ne $Log) { $Log = $nouveau; $debut = (Get-Item $Log).CreationTime; $vuEtape = @{} }
    }
    $lignes = Get-Content -LiteralPath $Log -ErrorAction SilentlyContinue
    if (-not $lignes) { $lignes = @() }
    $texte = $lignes -join "`n"

    $nImages = 0
    if ($texte -match 'Loading (\d+) images') { $nImages = [int]$Matches[1] }
    else {
        # à la reprise d'un calcul, ODX lit images.json sans réécrire le nombre : on compte le dossier images du projet
        $dossierImages = Join-Path (Join-Path (Split-Path $Log) ((Split-Path $Log -Leaf) -replace '_console\.log$', '')) 'images'
        if (Test-Path $dossierImages) { $nImages = @(Get-ChildItem $dossierImages -File).Count }
    }
    $lances = [regex]::Matches($texte, 'Running (\w+) stage') | ForEach-Object { $_.Groups[1].Value }
    $finis  = [regex]::Matches($texte, 'Finished (\w+) stage') | ForEach-Object { $_.Groups[1].Value }
    $courante = $lances | Select-Object -Last 1
    if ($courante -and -not $vuEtape.ContainsKey($courante)) { $vuEtape[$courante] = Get-Date }
    $termine = $texte -match 'OD[MX] app finished'
    $erreurs = $lignes | Where-Object { $_ -match 'Traceback|\[ERROR\]|Exception|MemoryError|bad_alloc|Killed|n''est pas reconnu|Child returned [1-9]' }

    $os = Get-CimInstance Win32_OperatingSystem
    $ramTot = $os.TotalVisibleMemorySize / 1MB; $ramLibre = $os.FreePhysicalMemory / 1MB
    $procs = Get-Process | Where-Object { $_.ProcessName -match '^(python|odm_|opensfm|OpenMVS|Densify|Recon|texrecon|gdal|pdal|PoissonRecon|dem2|obj2|orthophoto)' }
    $ramOdm = ($procs | Measure-Object WorkingSet64 -Sum).Sum / 1GB
    $cLibre = (Get-PSDrive C).Free / 1GB
    $cTotal = ((Get-PSDrive C).Used + (Get-PSDrive C).Free) / 1GB
    $histRam.Add($ramLibre); $histDisque.Add($cLibre)
    if ($histRam.Count -gt $histMax) { $histRam.RemoveAt(0); $histDisque.RemoveAt(0) }

    # Avancement global : étapes terminées + part faite de l'étape OpenSfM (la plus longue)
    $nFaites = @($etapes | Where-Object { $finis -contains $_ }).Count
    $nFeatG = @($lignes | Where-Object { $_ -match 'Found \d+ features for' }).Count
    $nReconG = @([regex]::Matches($texte, 'Adding (\S+) to the reconstruction') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique).Count
    $partOsfm = 0
    if ($courante -eq 'opensfm' -and $nImages -gt 0) { $partOsfm = [math]::Min(1.0, ($nFeatG + $nReconG) / (2 * $nImages)) }
    $global = if ($termine) { 1 } else { [math]::Min(0.99, ($nFaites + $partOsfm) / $etapes.Count) }

    Clear-Host
    Write-Host ('  Suivi ODX  -  ' + (Split-Path $Log -Leaf) + '   (' + (Get-Date -Format 'HH:mm:ss') + ')') -ForegroundColor Cyan
    Write-Host ('  Démarré le ' + $debut.ToString('dd/MM HH:mm') + '  |  écoulé ' + (Duree ((Get-Date) - $debut)) + '  |  ' + $nImages + ' photos')
    Write-Host ('  Avancement global (approximatif) ' + (Jauge $global 1)) -ForegroundColor Cyan
    Write-Host ''

    foreach ($e in $etapes) {
        if ($finis -contains $e)        { $sym = '[ok]'; $coul = 'Green' }
        elseif ($e -eq $courante)       { $sym = '[>>]'; $coul = 'Yellow' }
        else                            { $sym = '[  ]'; $coul = 'DarkGray' }
        $extra = ''
        if ($e -eq $courante -and -not $termine -and $vuEtape.ContainsKey($e)) { $extra = '   depuis ' + (Duree ((Get-Date) - $vuEtape[$e])) }
        Write-Host ('  ' + $sym + ' ' + $e + $extra) -ForegroundColor $coul
    }
    Write-Host ''

    # Progression détaillée dans OpenSfM
    $nFeat = @($lignes | Where-Object { $_ -match 'Found \d+ features for' }).Count
    # le journal arrive par paquets : on compte aussi les fichiers déjà écrits dans opensfm\features
    $dossierFeat = Join-Path (Join-Path (Split-Path $Log) ((Split-Path $Log -Leaf) -replace '_console\.log$', '')) 'opensfm\features'
    if (Test-Path $dossierFeat) { $nFeat = [math]::Max($nFeat, @(Get-ChildItem $dossierFeat -File).Count) }
    $nRecon = @([regex]::Matches($texte, 'Adding (\S+) to the reconstruction') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique).Count
    $nUndist = @($lignes | Where-Object { $_ -match '(?i)undistort.*\.(jpg|tif)' }).Count
    if ($nImages -gt 0 -and ($courante -eq 'opensfm' -or $nFeat -gt 0)) {
        Write-Host ('  Points caractéristiques  ' + (Barre $nFeat $nImages))
        if ($nRecon -gt 0) { Write-Host ('  Photos reconstruites     ' + (Barre $nRecon $nImages)) }
        if ($nUndist -gt 0) { Write-Host ('  Photos redressées        ' + (Barre $nUndist $nImages)) }
    }
    $bornes = $lignes | Where-Object { $_ -match 'Model bounds|Creating output file' } | Select-Object -Last 2
    foreach ($b in $bornes) { Write-Host ('  ' + $b.Trim()) -ForegroundColor Magenta }
    Write-Host ''

    $coulRam = if ($ramLibre -lt 3) { 'Red' } elseif ($ramLibre -lt 6) { 'Yellow' } else { 'White' }
    $coulC = if ($cLibre -lt 10) { 'Red' } elseif ($cLibre -lt 25) { 'Yellow' } else { 'White' }
    Write-Host ('  RAM utilisée  ' + (Jauge ($ramTot - $ramLibre) $ramTot) + ('   {0:N1} Go libres sur {1:N1}  (ODX : {2:N1} Go)' -f $ramLibre, $ramTot, $ramOdm)) -ForegroundColor $coulRam
    Write-Host ('  Disque C:     ' + (Jauge ($cTotal - $cLibre) $cTotal) + ('   {0:N1} Go libres sur {1:N0}' -f $cLibre, $cTotal)) -ForegroundColor $coulC
    Write-Host ''
    Write-Host ('  Historique (1 mesure / {0} s, la plus récente à droite, caractère plus haut = plus de place libre) :' -f $Intervalle) -ForegroundColor DarkCyan
    Write-Host ('    RAM libre  |' + (Courbe $histRam $ramTot) + ('|  min {0:N1} Go' -f ($histRam | Measure-Object -Minimum).Minimum))
    Write-Host ('    C: libre   |' + (Courbe $histDisque ([math]::Max(100, ($histDisque | Measure-Object -Maximum).Maximum))) + ('|  min {0:N1} Go' -f ($histDisque | Measure-Object -Minimum).Minimum))
    Write-Host ''

    Write-Host '  Dernières lignes du journal :' -ForegroundColor DarkCyan
    $lignes | Where-Object { $_.Trim() } | Select-Object -Last 8 | ForEach-Object {
        $l = $_; if ($l.Length -gt 150) { $l = $l.Substring(0, 147) + '...' }; Write-Host ('    ' + $l) -ForegroundColor Gray }

    if ($erreurs) {
        Write-Host ''
        Write-Host '  ERREURS / ALERTES :' -ForegroundColor Red
        $erreurs | Select-Object -Last 5 | ForEach-Object { Write-Host ('    ' + $_) -ForegroundColor Red }
    }
    if ($termine) {
        Write-Host ''
        Write-Host ('  CALCUL TERMINÉ  (durée totale ' + (Duree ((Get-Item $Log).LastWriteTime - $debut)) + ')') -ForegroundColor Green
        if (-not $auto) { break }
    }
    # Lot de vols en cours : dernières lignes du journal du lot
    $jl = 'C:\odm_runs\lot_journal.txt'
    if ($auto -and (Test-Path $jl) -and ((Get-Date) - (Get-Item $jl).LastWriteTime).TotalHours -lt 24) {
        Write-Host ''
        Write-Host '  Lot de vols (lot_journal.txt) :' -ForegroundColor DarkCyan
        Get-Content $jl -Tail 5 -Encoding UTF8 | ForEach-Object { $l = $_; if ($l.Length -gt 150) { $l = $l.Substring(0, 147) + '...' }; Write-Host ('    ' + $l) -ForegroundColor Gray }
    }
    Start-Sleep -Seconds $Intervalle
}
