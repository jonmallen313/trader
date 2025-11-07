# 📝 Paper Trading Setup Guide (No Deposit Required!)

Your AI trading system is configured for **100% paper trading** with **live market data**. No real money needed!

## 🎯 Available Paper Trading Options

### Option 1: Alpaca (Recommended for Stocks)
**Best for**: US stocks, ETFs, crypto
**Live Data**: ✅ Real-time market data
**Deposit Required**: ❌ No deposit needed

#### Setup Steps:
1. **Sign up for free**: https://app.alpaca.markets/signup
2. **Get Paper Trading Keys**:
   - Go to https://app.alpaca.markets/paper/dashboard/overview
   - Click "View" under "Your API Keys"
   - Generate new keys (or use existing)
   - Copy the **API Key** and **Secret Key**

3. **Add to Railway**:
   - Go to your Railway project
   - Click "Variables"
   - Add these variables:
   ```
   ALPACA_API_KEY=PKxxxxxxxxx
   ALPACA_API_SECRET=yyyyyyyyyyyy
   USE_EXCHANGE=alpaca
   PAPER_MODE=true
   ```

**Features**:
- $100,000 virtual cash (you'll start with $100 in our system)
- Real-time market data
- Trade stocks, ETFs, crypto
- Full order book and execution simulation

---

### Option 2: Binance Testnet (Recommended for Crypto)
**Best for**: Cryptocurrency trading
**Live Data**: ✅ Real-time crypto prices
**Deposit Required**: ❌ No deposit needed

#### Setup Steps:
1. **Sign up for free**: https://testnet.binance.vision/
2. **Get Testnet API Keys**:
   - Login to Binance Testnet
   - Go to API Management
   - Create new API key
   - Copy the **API Key** and **Secret Key**

3. **Get Test Funds**:
   - Use the testnet faucet to get fake BTC, ETH, etc.
   - You can "deposit" unlimited test funds

4. **Add to Railway**:
   - Go to your Railway project
   - Click "Variables"
   - Add these variables:
   ```
   BINANCE_API_KEY=your_testnet_key
   BINANCE_API_SECRET=your_testnet_secret
   USE_EXCHANGE=binance
   PAPER_MODE=true
   BINANCE_TESTNET=true
   ```

**Features**:
- Unlimited fake crypto
- Real-time market data
- All Binance trading pairs
- Full order execution simulation

---

## 🚀 Quick Start (No APIs Required)

If you just want to test the system without setting up APIs, the system includes a **MockBroker** that simulates trading with realistic price movements:

```bash
# In Railway, just set:
USE_EXCHANGE=mock
PAPER_MODE=true
```

The MockBroker will:
- Generate realistic price data
- Simulate order execution
- Track positions and P&L
- Perfect for testing the AI models

---

## 📊 Verify It's Working

Once deployed to Railway:

1. **Check the Dashboard**: `https://your-app.railway.app/`
2. **View Status**: `https://your-app.railway.app/status`
3. **Check Logs**: Railway dashboard → "Deployments" → "View Logs"

You should see:
```
Connected to Alpaca Paper Trading
or
Connected to Binance Testnet
or
Using Mock Broker for simulation
```

---

## 🔒 Safety Features

Your system is configured with multiple safety layers:

1. **Paper Trading Only**: `PAPER_MODE=true` (hardcoded in settings.py)
2. **Alpaca URL**: Always uses `paper-api.alpaca.markets`
3. **Binance Testnet**: Always uses testnet when `BINANCE_TESTNET=true`
4. **Mock Mode**: Fallback to simulation if APIs not configured

**You cannot accidentally trade real money** - the system defaults to paper trading!

---

## 💡 Recommended Configuration

For best results, use **both** exchanges:

```env
# Railway Environment Variables
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
USE_EXCHANGE=binance  # or alpaca, depending on what you want to trade
PAPER_MODE=true
BINANCE_TESTNET=true
INITIAL_CAPITAL=100
GLOBAL_TAKE_PROFIT=2000
```

This gives you:
- **Real market data** from live exchanges
- **Zero risk** (no real money involved)
- **Full trading capabilities** to test your AI

---

## 🎓 Next Steps

1. **Deploy to Railway** with paper trading credentials
2. **Monitor the dashboard** to see AI predictions
3. **Watch it learn** from live market data
4. **Reach the $2000 goal** in simulation
5. **Review performance** before considering live trading (much later!)

The system will run continuously, learning from real market patterns without any financial risk! 🚀
