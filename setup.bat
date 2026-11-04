@echo off
rem Zimo Studio - one-step setup (Windows)
rem
rem   setup.bat            auto: CUDA build when an NVIDIA GPU is present
rem   setup.bat cpu        force the small CPU-only wheel
rem   setup.bat cu124      force a specific CUDA build (cu121/cu124/cu126)
rem   setup.bat --check    print what would be installed, change nothing
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=auto"

where python >nul 2>nul
if errorlevel 1 (
    echo [x] Python not found. Install Python 3.10+ and tick "Add to PATH".
    exit /b 1
)

rem ---- detect the accelerator ------------------------------------------------
set "GPU_NAME="
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    for /f "usebackq delims=" %%g in (`nvidia-smi --query-gpu^=name --format^=csv,noheader 2^>nul`) do (
        if not defined GPU_NAME set "GPU_NAME=%%g"
    )
)

set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
if /i "%MODE%"=="cpu" set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if /i "%MODE%"=="auto" if not defined GPU_NAME set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if /i "%MODE%"=="cu121" set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
if /i "%MODE%"=="cu124" set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
if /i "%MODE%"=="cu126" set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
if defined ZIMO_TORCH_INDEX set "TORCH_INDEX=%ZIMO_TORCH_INDEX%"

echo Python      : & python -V
if defined GPU_NAME (echo GPU         : %GPU_NAME%) else (echo GPU         : none detected)
echo torch index : %TORCH_INDEX%

if /i "%MODE%"=="--check" (
    echo ^(check only, nothing installed^)
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 exit /b 1
) else (
    echo [1/3] Reusing existing .venv
)

echo [2/3] Installing PyTorch from the index above ^(CUDA wheels are ~2.5 GB^) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install torch --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [x] PyTorch install failed. Check network / proxy, or run: setup.bat cpu
    exit /b 1
)

echo [3/3] Installing the Studio dependencies ...
".venv\Scripts\python.exe" -m pip install -r requirements-studio.txt
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -c "import torch; print(); print('torch      :', torch.__version__); print('cuda build :', torch.version.cuda or 'cpu-only'); print('available  :', torch.cuda.is_available()); print('device     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo.
echo Done. Start the Studio with:
echo     run.bat
echo.
echo Then open http://127.0.0.1:8769
endlocal
