@echo off
REM Quick deployment script for Railway (Windows)

echo 🚀 Deploying AI Trader to Railway
echo.

REM Check for changes
git status --porcelain > nul 2>&1
if errorlevel 1 (
    echo ❌ Not a git repository. Run 'git init' first.
    exit /b 1
)

echo 📝 Staging changes...
git add .

echo 💾 Committing...
git commit -m "Deploy working railway_app.py with live trading"

echo 📤 Pushing to Railway...
git push

echo.
echo ✅ Deployed!
echo.
echo Next steps:
echo 1. Check Railway dashboard for build status
echo 2. Set environment variables:
echo    - BYBIT_API_KEY
echo    - BYBIT_API_SECRET
echo 3. Open your Railway URL to see dashboard
echo.
pause
