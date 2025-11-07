#!/bin/bash

# Railway Deployment Script for AI Trading System
# This script prepares the project for Railway deployment

echo "🚄 Preparing AI Trading System for Railway deployment..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial AI Trading System commit"
else
    echo "✅ Git repository already exists"
fi

# Check for Railway CLI
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "   npm install -g @railway/cli"
    echo "   or visit: https://railway.app/cli"
    exit 1
fi

echo "✅ Railway CLI found"

# Login to Railway
echo "🔐 Logging into Railway..."
railway login

# Create new project or link existing
echo "📋 Setting up Railway project..."
railway link

# Set essential environment variables
echo "⚙️  Setting up environment variables..."

railway variables set PAPER_MODE=true
railway variables set LOG_LEVEL=INFO
railway variables set INITIAL_CAPITAL=100.0
railway variables set GLOBAL_TAKE_PROFIT=2000.0

# Prompt for API keys
echo ""
echo "🔑 API Key Setup:"
echo "For TESTING, you can use dummy keys. For LIVE trading, use real keys."
echo ""

read -p "Enter Binance API Key (or press Enter for demo): " BINANCE_KEY
if [ ! -z "$BINANCE_KEY" ]; then
    railway variables set BINANCE_API_KEY="$BINANCE_KEY"
fi

read -p "Enter Binance Secret (or press Enter for demo): " BINANCE_SECRET
if [ ! -z "$BINANCE_SECRET" ]; then
    railway variables set BINANCE_SECRET="$BINANCE_SECRET"
fi

read -p "Enter Webhook Secret Key (recommended): " WEBHOOK_KEY
if [ ! -z "$WEBHOOK_KEY" ]; then
    railway variables set SECRET_WEBHOOK_KEY="$WEBHOOK_KEY"
else
    railway variables set SECRET_WEBHOOK_KEY=$(openssl rand -hex 32)
fi

# Deploy
echo ""
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Your AI Trading System is now running on Railway!"
echo "🌐 Check your Railway dashboard for the public URL"
echo "📡 Use the webhook URL with TradingView alerts"
echo "💡 Monitor logs with: railway logs --tail"
echo ""
echo "Next steps:"
echo "1. Visit your Railway dashboard to get the public URL"
echo "2. Test the webhook endpoint"
echo "3. Set up TradingView alerts"
echo "4. Monitor the trading performance"