Remove-Item Env:NoDefaultCurrentDirectoryInExePath -ErrorAction SilentlyContinue
Set-Location C:\WebODM\resources\app\apps\ODX
cmd /c "run.bat --project-path C:\odm_runs sl_w1b --fast-orthophoto --feature-quality high --min-num-features 20000 --matcher-neighbors 24 --orthophoto-resolution 0.49 --ignore-gsd --crop 0 --skip-3dmodel --max-concurrency 12 --end-with odm_dem > C:\odm_runs\sl_w1b_console.log 2>&1"
