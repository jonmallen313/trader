@echo off
REM AI Trader - With Dashboard
REM Runs trader + web dashboard

echo ========================================
echo   AI TRADER - WITH DASHBOARD
echo ========================================
echo.
echo Starting trader with web dashboard...
echo Dashboard will be available at http://127.0.0.1:8000
echo.
echo The trader runs in background even if you close the browser.
echo Press Ctrl+C in this window to stop.
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Run the trader with dashboard
python run.py

pause
