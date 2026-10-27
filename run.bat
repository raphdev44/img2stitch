@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERREUR] Environnement absent : lancez d'abord install.bat
    pause
    exit /b 1
)

set "HOST=127.0.0.1"
set "PORT=5000"

rem Ouvre le navigateur quelques secondes apres, le temps que le serveur demarre
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://%HOST%:%PORT%'"

echo Serveur img2stitch sur http://%HOST%:%PORT%  (Ctrl+C pour arreter)
".venv\Scripts\python.exe" -m flask --app webapp run --host %HOST% --port %PORT%
pause
