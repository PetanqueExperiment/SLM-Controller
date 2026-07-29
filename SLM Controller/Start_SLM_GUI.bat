@echo off
REM Start the SLM 3D pattern creator (PyQt GUI)
cd /d "%~dp0"

if not exist "SLM_venv\Scripts\python.exe" (
    echo Virtual environment not found: "%~dp0SLM_venv\Scripts\python.exe"
    echo Create it or adjust the path in this script.
    pause
    exit /b 1
)

echo Starting SLM GUI...
"SLM_venv\Scripts\python.exe" GUI_SLM.py
if errorlevel 1 (
    echo SLM GUI exited with an error.
    pause
)
exit /b 0
