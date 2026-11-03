@echo off
rem Zimo Studio - launch (Windows). Run setup.bat once first.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [x] .venv not found. Run setup.bat first.
    exit /b 1
)

echo Starting Zimo Studio on http://127.0.0.1:8769 ...
".venv\Scripts\python.exe" -m streamlit run app.py ^
    --server.port 8769 --server.address 127.0.0.1 ^
    --browser.gatherUsageStats false
endlocal
