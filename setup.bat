@echo off
rem Zimo Studio - one-step setup (Windows)
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [x] Python not found. Install Python 3.10+ and tick "Add to PATH".
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 exit /b 1
) else (
    echo [1/3] Reusing existing .venv
)

echo [2/3] Installing dependencies (this can take a few minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements-studio.txt
if errorlevel 1 (
    echo [x] Dependency install failed. Check your network / proxy.
    exit /b 1
)

echo.
echo [3/3] Done. Start the Studio with:
echo     run.bat
echo.
echo Then open http://127.0.0.1:8769
endlocal
