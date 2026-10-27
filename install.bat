@echo off
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERREUR] Python introuvable. Installez Python 3.9+ depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creation de l'environnement virtuel .venv ...
    %PY% -m venv .venv || goto :error
)

rem Reseau d'entreprise : pypi.org n'est pas resolu en direct, on passe par le proxy (surcharge possible via PROXY)
powershell -NoProfile -Command "try { [Net.Dns]::GetHostEntry('pypi.org') | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    if not defined PROXY set "PROXY=http://10.242.191.81:80"
)
rem Variable d'environnement (et non option) pour que le sous-processus de build de pip en herite
if defined PROXY (
    echo Utilisation du proxy %PROXY%
    set "PIP_PROXY=%PROXY%"
)

echo Installation des dependances ...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -e ".[web]" || goto :error

echo.
echo Installation terminee. Lancez l'application avec run.bat
pause
exit /b 0

:error
echo.
echo [ERREUR] L'installation a echoue.
pause
exit /b 1
