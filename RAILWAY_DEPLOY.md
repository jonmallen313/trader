# 🚀 RAILWAY DEPLOYMENT - AI TRADER

## What's Fixed

✅ **Live Market Data** - Real BTC/ETH/SOL prices updating every 2 seconds  
✅ **Real Trading Logic** - Actually executes trades based on momentum  
✅ **High-Tech Dashboard** - Modern WebSocket-powered UI  
✅ **Proper Bybit Integration** - Uses testnet for safe trading  

## Deploy to Railway

1. **Push this code to your Railway project**
2. **Set environment variables:**
   ```
   BYBIT_API_KEY=your_testnet_key
   BYBIT_API_SECRET=your_testnet_secret
   ```
3. **Railway will automatically:**
   - Build from Dockerfile
   - Run `railway_app.py`
   - Expose port 8000
   - Start trading

## Access Dashboard

Once deployed, open your Railway URL:
```
https://your-app.railway.app
```

You'll see:
- **Live market prices** for BTC, ETH, SOL
- **Active positions** with entry, TP, SL
- **Recent trades** with P&L
- **Real-time stats** (balance, win rate, etc.)

## How It Works

### 1. Market Data Loop
- Fetches live prices every 2 seconds
- Displays on dashboard via WebSocket
- Calculates 24h change, volume, highs/lows

### 2. Trading Logic
```python
# Simple momentum strategy
if 24h_change > 2%:
    confidence = change / 10
    if confidence >= 55%:
        execute_trade(long/short based on direction)
```

### 3. Position Management
- TP: 1.5% profit
- SL: 1% loss
- Time limit: 30 minutes
- Leverage: 5x-15x based on confidence

### 4. Live Dashboard
- WebSocket updates every 2 seconds
- No page refresh needed
- Real-time market data display
- Trade history tracking

## Files

- **railway_app.py** - Main application (FastAPI + WebSocket)
- **Procfile** - Railway startup command
- **Dockerfile** - Container build
- **requirements.txt** - Python dependencies

## Test Locally

```bash
# Install deps
pip install -r requirements.txt

# Set env vars
export BYBIT_API_KEY=your_key
export BYBIT_API_SECRET=your_secret

# Run
python railway_app.py

# Open browser
http://localhost:8000
```

## Logs

Check Railway logs to see:
```
🚀 Starting Live Trader
✅ Connected to Bybit
🎯 OPENED BTCUSDT LONG @ $95,147.00 | 10x | Conf: 67%
✅ CLOSED BTCUSDT long | $+2.15 | TP
```

## Troubleshooting

**No trades executing?**
- Check Bybit API keys are set
- Verify testnet mode is enabled
- Look for "Connected to Bybit" in logs

**Dashboard not updating?**
- WebSocket must be working
- Check browser console for errors
- Try hard refresh (Ctrl+Shift+R)

**Market data not showing?**
- Takes 10-15 seconds to load initially
- Check Railway logs for errors
- Verify ccxt is installed

## Why This Works vs Before

| Before | Now |
|--------|-----|
| Complex multi-file setup | Single `railway_app.py` |
| No market data display | Live prices on dashboard |
| Random "AI" | Simple momentum strategy |
| Terminal only | WebSocket dashboard |
| No trades | Trades every ~10 seconds |

**This actually executes trades and shows live data.** 🎯
