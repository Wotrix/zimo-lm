@echo off
rem Zimo Studio - uninstall (Windows)
rem
rem   uninstall.bat           remove .venv (the only thing setup.bat created)
rem   uninstall.bat --purge   also remove training runs and caches
rem
rem Nothing outside this folder is touched: setup.bat never writes to your
rem system Python, installs no service and creates no registry entries.
setlocal
cd /d "%~dp0"

set "PURGE=0"
if /i "%~1"=="--purge" set "PURGE=1"

echo This will remove:
echo   - .venv\            (the isolated environment created by setup.bat)
if "%PURGE%"=="1" echo   - runs\  __pycache__\  (training artifacts)
echo.

if exist ".venv" (
    rmdir /s /q ".venv"
    echo [ok] removed .venv
) else (
    echo [--] .venv not present
)

if "%PURGE%"=="1" (
    if exist "runs" rmdir /s /q "runs"
    for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
    echo [ok] removed runs and caches
) else (
    echo [--] kept runs\ ^(pass --purge to delete training artifacts^)
)

echo.
echo Done. The code itself is untouched; delete this folder whenever you like.
endlocal
