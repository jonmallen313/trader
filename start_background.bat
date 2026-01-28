@echo off
REM AI Trader - Background Service
REM Runs continuously, even when browser is closed

echo ========================================
echo   AI TRADER - BACKGROUND SERVICE
echo ========================================
echo.
echo Starting trader in background mode...
echo The trader will run continuously.
echo You can close this window and it will keep running.
echo.
echo Press Ctrl+C to stop the trader.
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Run the background trader
python background_trader.py

pause
