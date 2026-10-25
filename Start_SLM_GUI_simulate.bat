@echo off
REM Start the SLM GUI in offline SIMULATE mode (no Meadowlark hardware / Blink SDK)
cd /d "%~dp0"

if not exist "SLM_venv\Scripts\python.exe" (
    echo Virtual environment not found: "%~dp0SLM_venv\Scripts\python.exe"
    echo Create it or adjust the path in this script.
    pause
    exit /b 1
)

echo Starting SLM GUI in SIMULATE mode...
"SLM_venv\Scripts\python.exe" GUI_SLM.py --simulate
if errorlevel 1 (
    echo SLM GUI exited with an error.
    pause
)
exit /b 0
