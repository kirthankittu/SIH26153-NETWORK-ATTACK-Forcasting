@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Create it with: py -3.12 -m venv .venv
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run dashboard\app.py
endlocal
