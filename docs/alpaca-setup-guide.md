# 🚀 Quick Setup: Alpaca Paper Trading (FREE - No Deposit!)

## Why Alpaca?
- ✅ **Not geo-restricted** (works on Railway)
- ✅ **Free paper trading** with $100K virtual cash
- ✅ **Real-time market data** from US stock market
- ✅ **No deposit required** ever
- ✅ **2-minute setup**

## Step 1: Sign Up (1 minute)

1. Go to: https://app.alpaca.markets/signup
2. Create a free account
3. No credit card or deposit needed!

## Step 2: Get Paper Trading API Keys (30 seconds)

1. Log in to Alpaca
2. Click on **"Paper Trading"** in the top menu
3. Go to: https://app.alpaca.markets/paper/dashboard/overview
4. Scroll down to **"Your API Keys"**
5. Click **"Generate"** or **"View"** if keys exist
6. Copy your:
   - **API Key** (starts with `PK...`)
   - **Secret Key** (starts with letters/numbers)

## Step 3: Add to Railway (1 minute)

1. Go to your Railway project: https://railway.app
2. Click on your **trader** service
3. Go to **"Variables"** tab
4. Click **"+ New Variable"**
5. Add these **two** variables:

```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
ALPACA_API_SECRET=yyyyyyyyyyyyyyyyyyyy
```

6. Click **"Deploy"** or it will auto-redeploy

## Step 4: Verify It's Working

Once redeployed, check the Railway logs. You should see:

```
✅ Added Alpaca paper trading broker (real market data)
🚀 AI Trading System running in paper mode
📊 Target: $2000 ($100 → 20x multiplier)
```

## What You Get

### Trading Capabilities
- ✅ Trade **US stocks** (AAPL, TSLA, NVDA, etc.)
- ✅ Trade **ETFs** (SPY, QQQ, etc.)
- ✅ Trade **crypto** (BTC, ETH via BTCUSD, ETHUSD)
- ✅ **Real-time prices** from US markets
- ✅ **Realistic execution** simulation

### Your Virtual Account
- 💰 **$100,000** virtual cash (but our system starts with $100)
- 📊 **Real market hours** (9:30 AM - 4:00 PM ET)
- 📈 **Extended hours** trading available
- 🔄 **Unlimited** paper trades

## Supported Symbols

Update your symbols in Railway environment variables:

```
TRADING_SYMBOLS=AAPL,TSLA,NVDA,SPY,QQQ
```

Or for crypto:
```
TRADING_SYMBOLS=BTCUSD,ETHUSD,SOLUSD
```

## Dashboard Access

After deployment:
- **Main Dashboard**: `https://your-app.railway.app/`
- **Status API**: `https://your-app.railway.app/status`
- **Health Check**: `https://your-app.railway.app/health`
- **Alpaca Dashboard**: https://app.alpaca.markets/paper/dashboard/overview

## Troubleshooting

### "Invalid API credentials"
- Make sure you copied the **Paper Trading** keys (not Live keys)
- Keys should start with `PK...` for paper trading
- Check for extra spaces when pasting

### "Market closed"
- US stock market: Mon-Fri, 9:30 AM - 4:00 PM ET
- Crypto trading: 24/7
- System will queue orders until market opens

### Want to test right now?
Use crypto symbols (24/7 trading):
```
TRADING_SYMBOLS=BTCUSD,ETHUSD
```

## Cost: $0 Forever

Alpaca paper trading is:
- ✅ **100% free** forever
- ✅ **No deposit** ever required
- ✅ **Unlimited** paper trades
- ✅ **Real market data** included

Perfect for testing your AI trading system before going live! 🎉

---

## Next: TradingView Integration (Optional)

Once your system is running with Alpaca, you can add TradingView alerts:

1. Your webhook URL: `https://your-app.railway.app/webhook/tradingview`
2. Set up Pine Script alerts in TradingView
3. Send signals to your AI trader

See `docs/tradingview-setup.md` for details.
