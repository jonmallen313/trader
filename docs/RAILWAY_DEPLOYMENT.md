# Railway Deployment Guide

This guide shows how to deploy the AI Trading System to Railway cloud platform.

## 🚄 Railway Setup

### 1. Prepare Your Repository

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial AI Trading System"
git remote add origin https://github.com/YOUR_USERNAME/ai-trader.git
git push -u origin main
```

### 2. Deploy to Railway

1. **Visit**: https://railway.app
2. **Sign up** with your GitHub account
3. **Create New Project** → **Deploy from GitHub repo**
4. **Select** your ai-trader repository
5. **Deploy** - Railway will automatically detect Python and install dependencies

### 3. Environment Variables

In Railway dashboard, go to **Variables** tab and add:

#### Required Variables:
```env
# Trading Mode
PAPER_MODE=true

# Binance API (for crypto)
BINANCE_API_KEY=your_binance_testnet_key
BINANCE_SECRET=your_binance_testnet_secret

# Alpaca API (for stocks) - Optional
ALPACA_API_KEY=your_alpaca_paper_key
ALPACA_SECRET=your_alpaca_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Webhook Security
SECRET_WEBHOOK_KEY=your_random_secret_key

# System Configuration
INITIAL_CAPITAL=100.0
GLOBAL_TAKE_PROFIT=2000.0
LOG_LEVEL=INFO
```

#### Optional Variables:
```env
# Database (Railway provides PostgreSQL)
DATABASE_URL=${{RAILWAY_DATABASE_URL}}

# Redis (if you add Redis service)
REDIS_URL=${{RAILWAY_REDIS_URL}}

# Custom Trading Settings
DEFAULT_TP_PCT=0.02
DEFAULT_SL_PCT=0.01
PREDICTION_THRESHOLD=0.6
```

### 4. Domain & URLs

After deployment, Railway provides:
- **App URL**: `https://your-app-name.railway.app`
- **Webhook URL**: `https://your-app-name.railway.app/webhook/tradingview`
- **Dashboard**: Access via the app URL
- **Health Check**: `https://your-app-name.railway.app/health`

## 🔧 Configuration Files

The system includes these Railway-specific files:

- **`railway.json`**: Railway service configuration
- **`Procfile`**: Process definition
- **`runtime.txt`**: Python version specification
- **`config/railway.py`**: Railway environment handling

## 📊 Monitoring on Railway

### Built-in Monitoring:
- **Logs**: Real-time logs in Railway dashboard
- **Metrics**: CPU, Memory, Network usage
- **Health Checks**: Automatic endpoint monitoring
- **Deployments**: View deployment history

### Custom Monitoring:
Access your deployed endpoints:
```bash
# System Status
curl https://your-app.railway.app/status

# Health Check
curl https://your-app.railway.app/health

# Recent Signals
curl https://your-app.railway.app/signals/history
```

## 🎯 TradingView Integration

### Webhook URL
Use this in your TradingView alerts:
```
https://your-app.railway.app/webhook/tradingview
```

### Pine Script Alert Example:
```pinescript
//@version=5
strategy("AI Trader Signal", overlay=true)

// Your strategy logic here
long_condition = ta.crossover(ta.sma(close, 10), ta.sma(close, 20))
short_condition = ta.crossunder(ta.sma(close, 10), ta.sma(close, 20))

if long_condition
    strategy.entry("Long", strategy.long)
    alert('{"symbol":"' + syminfo.ticker + '","action":"BUY","tp_pct":0.02,"sl_pct":0.01}', alert.freq_once_per_bar)

if short_condition
    strategy.entry("Short", strategy.short)  
    alert('{"symbol":"' + syminfo.ticker + '","action":"SELL","tp_pct":0.02,"sl_pct":0.01}', alert.freq_once_per_bar)
```

## 🔄 Deployment Modes

### Paper Trading (Default):
```bash
# Railway automatically runs: python main.py paper --capital 100 --target 2000
```

### Live Trading:
Change the start command in `railway.json`:
```json
{
  "deploy": {
    "startCommand": "python main.py live --capital 100 --target 2000"
  }
}
```

### Backtesting:
```json
{
  "deploy": {
    "startCommand": "python main.py backtest --symbols BTC/USDT ETH/USDT"
  }
}
```

## 🚨 Safety on Railway

### Paper Mode Security:
- Always start with `PAPER_MODE=true`
- Use testnet API keys
- Monitor logs carefully
- Test webhook integration thoroughly

### Live Mode Checklist:
- [ ] Thorough paper testing completed
- [ ] Real API keys configured securely
- [ ] Stop-loss limits appropriate
- [ ] Monitoring alerts set up
- [ ] Backup plans in place

## 📈 Scaling & Performance

### Railway Resources:
- **Memory**: 512MB-8GB (auto-scaling)
- **CPU**: Shared to dedicated cores
- **Storage**: Ephemeral (use database for persistence)
- **Bandwidth**: Generous limits

### Optimization Tips:
1. Use Railway's PostgreSQL for trade history
2. Add Redis for caching market data
3. Enable Railway's metrics monitoring
4. Set up log aggregation
5. Use Railway's built-in SSL/TLS

## 🔧 Troubleshooting

### Common Issues:

1. **Port Binding Error**:
   - Railway automatically sets `$PORT` environment variable
   - System uses this automatically

2. **API Key Errors**:
   - Verify keys in Railway Variables tab
   - Check testnet vs live environment
   - Ensure proper key permissions

3. **Webhook Not Receiving**:
   - Check Railway logs for incoming requests
   - Verify TradingView alert URL
   - Test with manual curl request

4. **Memory Issues**:
   - Monitor Railway metrics
   - Upgrade plan if needed
   - Optimize ML model memory usage

### Debug Commands:
```bash
# View live logs
railway logs --tail

# Check service status  
railway status

# Redeploy
railway up
```

## 💰 Railway Pricing

- **Hobby Plan**: $5/month (perfect for paper trading)
- **Pro Plan**: $20/month (for live trading with better resources)
- **Custom**: Enterprise pricing for high-frequency trading

## 🎯 Production Checklist

Before going live:
- [ ] Paper trading results satisfactory
- [ ] All API keys secured in Railway Variables
- [ ] Webhook URL tested with TradingView
- [ ] Monitoring and alerts configured
- [ ] Stop-loss and take-profit validated
- [ ] Backup and recovery plan established
- [ ] Railway plan appropriate for trading volume

---

## 🚀 Quick Deploy Commands

```bash
# 1. Connect to Railway
railway login

# 2. Link your project  
railway link

# 3. Set environment variables
railway variables set PAPER_MODE=true
railway variables set BINANCE_API_KEY=your_key
railway variables set BINANCE_SECRET=your_secret

# 4. Deploy
railway up
```

Your AI trading system is now running 24/7 on Railway! 🎉