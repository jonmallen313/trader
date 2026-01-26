#!/bin/bash
# Quick deployment script for Railway

echo "🚀 Deploying AI Trader to Railway"
echo ""

# Check if we're in a git repo
if [ ! -d .git ]; then
    echo "❌ Not a git repository. Run 'git init' first."
    exit 1
fi

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    echo "✅ No changes to commit"
else
    echo "📝 Staging changes..."
    git add .
    
    echo "💾 Committing..."
    git commit -m "Deploy working railway_app.py with live trading"
fi

echo "📤 Pushing to Railway..."
git push

echo ""
echo "✅ Deployed!"
echo ""
echo "Next steps:"
echo "1. Check Railway dashboard for build status"
echo "2. Set environment variables:"
echo "   - BYBIT_API_KEY"
echo "   - BYBIT_API_SECRET"
echo "3. Open your Railway URL to see dashboard"
echo ""
echo "Logs command: railway logs"
