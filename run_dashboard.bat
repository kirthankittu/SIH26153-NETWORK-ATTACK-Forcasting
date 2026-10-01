@echo off
setlocal
cd /d "%~dp0"

set PORT=8502
set PYTHON=.venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo.
    echo Virtual environment not found.
    echo Please create .venv first.
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo        ThreatCast AI - SIH26153
echo ==========================================
echo.

REM Check whether ThreatCast is already running on PORT
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"

if %ERRORLEVEL% EQU 0 (
    echo ThreatCast AI is already running.
    echo Opening dashboard...
    start "" "http://localhost:%PORT%"
    exit /b 0
)

echo Starting ThreatCast AI...
echo.

start "" /b "%PYTHON%" -m streamlit run dashboard\app.py --server.port %PORT%

echo Waiting for dashboard to start...

:WAIT
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"

if %ERRORLEVEL% EQU 0 goto READY

timeout /t 1 /nobreak >nul
goto WAIT

:READY
echo.
echo ThreatCast AI is ready.
echo Opening dashboard...
start "" "http://localhost:%PORT%"

endlocal