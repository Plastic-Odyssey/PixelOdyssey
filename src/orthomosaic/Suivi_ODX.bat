@echo off
title Suivi ODX
mode con cols=130 lines=50
powershell -NoProfile -ExecutionPolicy Bypass -File C:\odm_runs\suivi_odm.ps1
pause
