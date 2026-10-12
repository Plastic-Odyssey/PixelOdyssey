@echo off
chcp 65001 >nul
title PixelOdyssey
cd /d "%~dp0"

REM ---------------------------------------------------------------------------
REM  Lance l'application PixelOdyssey (interface dans le navigateur).
REM  L'environnement Python du projet est cherché automatiquement (conda).
REM  S'il n'est pas trouvé : indiquer son nom ci-dessous (celui choisi dans
REM  VS Code pour entraîner les modèles), ex. set "ENV_NAME=plastic_env"
REM ---------------------------------------------------------------------------
set "ENV_NAME="

REM 1) Python déjà prêt (lancé depuis un terminal où l'environnement est actif)
python -c "import ultralytics, scipy, rasterio" >nul 2>nul && goto :run

REM 2) Recherche d'une installation conda et de l'environnement du projet
for %%B in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge") do (
  if exist "%%~B\Scripts\activate.bat" (
    for %%E in (%ENV_NAME% plastic_env pixelodyssey PixelOdyssey yolo) do (
      if exist "%%~B\envs\%%E\python.exe" (
        call "%%~B\Scripts\activate.bat" "%%~B\envs\%%E"
        python -c "import ultralytics, scipy, rasterio" >nul 2>nul && goto :run
      )
    )
    call "%%~B\Scripts\activate.bat" "%%~B"
    python -c "import ultralytics, scipy, rasterio" >nul 2>nul && goto :run
  )
)

echo.
echo  Environnement Python du projet introuvable (ultralytics, scipy, rasterio).
echo  Solution 1 : ouvrir ce fichier avec un editeur et renseigner ENV_NAME.
echo  Solution 2 : dans le terminal VS Code ^(environnement du projet actif^) :
echo               python -m src.app
echo.
pause
exit /b 1

:run
python -m src.app %*
if errorlevel 1 pause
