# 🚀 Quick Start - Restructured Trading System

## What Changed?

**Old System:** Messy, vibecoded, hard to control
**New System:** Clean architecture, analytical, professional

---

## 🎯 Your Goal

Turn **any amount** into **any target** using **leverage**:
- Start with $50, $100, $500, or any amount
- Set custom target (20x, 50x, 100x)
- Use 1x to 100x leverage
- FREE paper trading (Bybit testnet)

---

## ⚡ Get Started in 3 Steps

### 1. Get FREE Bybit Testnet Credentials

**No credit card, no deposit, unlimited fake money:**

```bash
# Visit testnet
https://testnet.bybit.com

# Steps:
1. Sign up (email only, instant access)
2. Click "API Management" → "Create New Key"
3. Enable "Read-Write" permissions
4. Copy API Key and Secret
```

You'll get **unlimited fake USDT** for testing!

### 2. Set Environment Variables

**Windows (PowerShell):**
```powershell
$env:BYBIT_API_KEY="your_api_key_here"
$env:BYBIT_API_SECRET="your_secret_here"
```

**Linux/Mac:**
```bash
export BYBIT_API_KEY="your_api_key_here"
export BYBIT_API_SECRET="your_secret_here"
```

**Or create `.env` file:**
```bash
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_secret_here
```

### 3. Start Trading!

```bash
# Basic: $100 → $2000 with 10x leverage
python cli/trader.py start

# Custom: $500 → $10,000 with 20x leverage
python cli/trader.py start --capital 500 --target 10000 --leverage 20

# Conservative: $100 → $500 with 5x leverage
python cli/trader.py start --capital 100 --target 500 --leverage 5

# Aggressive: $50 → $5000 with 50x leverage (risky!)
python cli/trader.py start --capital 50 --target 5000 --leverage 50
```

---

## 📊 Monitor Your Trading

### View Performance Metrics
```bash
python cli/trader.py metrics
```

**Shows:**
- Win rate
- Profit factor
- Total P&L
- Sharpe ratio
- Liquidation safety
- Margin usage

### View Recent Trades
```bash
# Last 20 trades
python cli/trader.py trades

# Last 50 trades
python cli/trader.py trades --last 50

# Only winning trades
python cli/trader.py trades --wins-only

# Filter by symbol
python cli/trader.py trades --symbol BTC/USDT
```

### Analytics API Server
```bash
# Start API server
python cli/trader.py serve --port 8000

# Query endpoints:
curl http://localhost:8000/metrics
curl http://localhost:8000/trades
curl http://localhost:8000/positions
```

---

## 🎛️ Advanced Control

### Full Command Options

```bash
python cli/trader.py start \
  --capital 100 \              # Starting capital ($)
  --target 2000 \              # Profit target ($)
  --leverage 10 \              # Leverage (1-100x)
  --broker bybit \             # Broker (bybit/alpaca/mock)
  --strategy microtrend \      # Strategy (microtrend/momentum)
  --symbols BTC/USDT ETH/USDT \ # Symbols to trade
  --max-positions 10 \         # Max open positions
  --risk-per-trade 0.02        # Risk 2% per trade
```

### Emergency Stop

```bash
# Close all positions immediately
python cli/trader.py stop
```

---

## 💡 Understanding Leverage

### What is Leverage?

- **1x leverage** = Trade with your actual capital ($100 controls $100)
- **10x leverage** = $100 controls $1,000 of positions
- **100x leverage** = $100 controls $10,000 of positions

### Risk vs Reward

| Leverage | Capital | Controls | Risk |
|----------|---------|----------|------|
| 1x | $100 | $100 | Very Low |
| 5x | $100 | $500 | Low |
| 10x | $100 | $1,000 | Medium |
| 20x | $100 | $2,000 | High |
| 50x | $100 | $5,000 | Very High |
| 100x | $100 | $10,000 | Extreme |

**Higher leverage = Faster gains BUT faster liquidation!**

### Example Scenarios

**Conservative (5x):**
```bash
python cli/trader.py start --capital 100 --target 500 --leverage 5
```
- Safer: 20% price move to liquidation
- Slower: Need more trades to reach target
- Win rate needed: ~60%

**Balanced (10x):**
```bash
python cli/trader.py start --capital 100 --target 2000 --leverage 10
```
- Moderate: 10% price move to liquidation  
- Moderate speed: Reasonable path to target
- Win rate needed: ~55%

**Aggressive (50x):**
```bash
python cli/trader.py start --capital 100 --target 5000 --leverage 50
```
- Risky: 2% price move to liquidation
- Fast: Can reach target quickly
- Win rate needed: ~70%+

---

## 📈 What Symbols Can I Trade?

### Crypto Perpetual Futures (Bybit)

Popular pairs with high liquidity:
- BTC/USDT (Bitcoin)
- ETH/USDT (Ethereum)
- SOL/USDT (Solana)
- XRP/USDT (Ripple)
- DOGE/USDT (Dogecoin)
- AVAX/USDT (Avalanche)
- MATIC/USDT (Polygon)

**Testnet has ALL symbols that live trading has!**

---

## 🛡️ Safety Features

### Automatic Protections

1. **Liquidation Guard** - Monitors distance to liquidation
2. **Max Drawdown** - Stops trading if losses exceed limit
3. **Position Limits** - Prevents over-leveraging
4. **Emergency Stop** - Instant close all positions

### Risk Settings

Modify in code or config:
```python
MAX_DRAWDOWN = -200  # Stop if down $200
MAX_DAILY_LOSS = -100  # Stop if daily loss > $100
LIQUIDATION_BUFFER = 0.20  # Keep 20% safety margin
```

---

## 🔧 Troubleshooting

### "API credentials not found"
```bash
# Check environment variables
echo $BYBIT_API_KEY
echo $BYBIT_API_SECRET

# If empty, set them:
export BYBIT_API_KEY="your_key"
export BYBIT_API_SECRET="your_secret"
```

### "Connection failed"
- Verify API key is correct
- Check testnet.bybit.com is accessible
- Make sure API permissions include "Read-Write"

### "Insufficient balance"
- Testnet gives unlimited USDT - just request more on their website
- Visit: https://testnet.bybit.com → Wallet → Get Test Funds

---

## 📚 Next Steps

1. **Test with small leverage first** (5x-10x)
2. **Monitor liquidation distance** in analytics
3. **Tune strategy parameters** once profitable
4. **Scale up gradually** as confidence grows

**Remember: This is TESTNET - experiment freely!**

---

## 🆘 Get Help

Questions? Issues?
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Review logs in `logs/trades/`
- Enable debug mode: `--log-level DEBUG`

**Have fun and trade smart! 🚀**
