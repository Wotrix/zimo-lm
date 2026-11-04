@echo off
rem Launch Zimo Studio. Once this folder is on PATH, `zimo` works anywhere.
rem (Batch files stay ASCII-only: cmd reads them in the OEM codepage.)
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" zimo_run.py %*
) else (
    python zimo_run.py %*
)
endlocal
