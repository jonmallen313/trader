# AI Trader - Always Running Mode 🤖

## Overview
The AI Trader now operates in **ALWAYS RUNNING** mode with **15 concurrent positions** to maximize profits. The trading engine runs continuously in the background, independent of any web interface.

## Key Features ✨

### 🔥 15 Concurrent Positions
- Entire $100 balance split across 15 different trades
- Each position: ~6.7% of balance (~$6.67)
- Diversified across 15+ crypto symbols
- Maximum capital utilization

### ⚡ Always Trading
- Runs continuously in background
- No need to keep webpage open
- Automatically restarts on errors
- Maintains position quota at all times

### 🧠 AI-Powered
- XGBoost microtrend detection
- Real-time market analysis
- Confidence-based trade filtering
- Continuous learning from results

### 📊 Multiple Trading Symbols
BTC, ETH, SOL, BNB, ADA, DOGE, AVAX, MATIC, DOT, UNI, LINK, ATOM, LTC, XRP, TRX

## Quick Start 🚀

### Option 1: Background Only (No Dashboard)
```bash
# Windows
start_background.bat

# Linux/Mac
chmod +x start_background.sh
./start_background.sh
```

This runs the trader in pure background mode. No web interface - just continuous trading.

### Option 2: With Dashboard
```bash
# Windows
start_dashboard.bat

# Linux/Mac
chmod +x start_dashboard.sh
./start_dashboard.sh
```

This runs the trader AND a web dashboard at http://127.0.0.1:8000

**Important:** The trader runs in background even if you close the browser!

### Option 3: Direct Python
```bash
# Background only
python background_trader.py

# With dashboard
python run.py
```

## How It Works 🔧

### Position Management
1. **Initial Scan**: Analyzes all 15 symbols for trading opportunities
2. **Open Positions**: Opens trades up to 15 concurrent positions
3. **Continuous Monitoring**: Checks prices every second
4. **Auto-Close**: Closes positions when TP/SL hit
5. **Auto-Replace**: Immediately opens new position to maintain 15 total

### Trading Logic
```
Balance: $100
Positions: 15 concurrent
Size per trade: $100 / 15 = ~$6.67
Leverage: 10x
TP: 0.2-0.3% (based on confidence)
SL: 0.2% (tight risk management)
```

### Example Scenario
```
Total Capital: $100
Position 1: BTC/USDT LONG  $6.67  (TP: $6.71, SL: $6.66)
Position 2: ETH/USDT SHORT $6.67  (TP: $6.69, SL: $6.65)
Position 3: SOL/USDT LONG  $6.67  (TP: $6.70, SL: $6.66)
...
Position 15: TRX/USDT LONG $6.67  (TP: $6.69, SL: $6.66)

When Position 1 closes → Immediately opens Position 16
Always maintains exactly 15 open positions
```

## Configuration ⚙️

### Environment Variables
```bash
# Capital
INITIAL_CAPITAL=100

# API Keys (choose one)
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret

# OR
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
```

### Settings in Code
Edit `config/settings.py`:
```python
MAX_POSITIONS = 15           # Number of concurrent positions
POSITION_SIZE_PCT = 0.067    # ~6.7% per trade
```

Edit `background_trader.py` for advanced settings:
```python
self.max_positions = 15      # Concurrent positions
self.min_confidence = 0.52   # AI confidence threshold
self.leverage = 10           # Leverage multiplier
```

## Monitoring 📈

### Console Output
The trader logs every action:
```
✅ OPENED LONG: BTC/USDT @ $45,123.00 | Size: $6.67 | Conf: 62%
🟢 CLOSED LONG: ETH/USDT @ $2,345.67 | Reason: TP | PnL: +$0.15
📊 Open: 15/15 | Need: 0 more positions
💰 Balance: $100.43 (+0.4%)
```

### Performance Reports
Every minute, get a full report:
```
================================================================================
📊 PERFORMANCE REPORT
================================================================================
💰 Balance: $102.34 (+2.3%)
🎯 Target: $2000.00 (5.1% complete)
📈 Open Positions: 15/15
📊 Total Trades: 47 | Win Rate: 61.7%
⏱️  Runtime: 0:15:32
================================================================================
```

### Web Dashboard
If using `start_dashboard.bat/sh`, access:
- http://127.0.0.1:8000 - Live dashboard
- Real-time balance updates
- Position tracking
- Trade history

## Stopping the Trader 🛑

### Graceful Shutdown
Press `Ctrl+C` in the terminal window

The trader will:
1. Stop accepting new positions
2. Close all open positions at market price
3. Display final balance and stats
4. Clean up connections

### Force Stop
- Close terminal window (positions may remain open!)
- Use Task Manager (Windows) or `kill` (Linux/Mac)

**Warning:** Force stopping may leave positions open on the exchange!

## Troubleshooting 🔧

### No Trades Executing
1. Check API keys are set
2. Check internet connection
3. Verify exchange is accessible
4. Lower `min_confidence` threshold

### Too Slow
1. Reduce number of symbols in `background_trader.py`
2. Increase `min_confidence` for fewer trades
3. Check API rate limits

### Positions Not Closing
1. TP/SL prices may be too wide/narrow
2. Market moved too fast
3. Check broker connection

### Module Not Found
```bash
pip install -r requirements.txt
```

## Files Changed 📝

### Updated Files
- `config/settings.py` - MAX_POSITIONS = 15
- `core/autopilot.py` - max_positions = 15
- `trade_live.py` - 15 positions, adjusted sizing
- `railway_app.py` - Auto-start trader on load
- `complete_trader.py` - 15 positions
- `run.py` - Uses background trader

### New Files
- `background_trader.py` - **Main background service**
- `start_background.bat` - Windows launcher (no dashboard)
- `start_dashboard.bat` - Windows launcher (with dashboard)
- `start_background.sh` - Linux/Mac launcher (no dashboard)
- `start_dashboard.sh` - Linux/Mac launcher (with dashboard)
- `ALWAYS_RUNNING_README.md` - This file

## Architecture 🏗️

```
background_trader.py (MAIN)
│
├── Market Data Loop (2s)
│   └── Fetch prices for all 15 symbols
│
├── Aggressive Trading Loop (5s)
│   ├── Check position count (target: 15)
│   ├── Find best opportunities
│   └── Open new positions
│
├── Position Monitor (1s)
│   ├── Check TP/SL for all positions
│   └── Close positions when triggered
│
└── Performance Reporter (60s)
    └── Display stats and progress
```

## Performance Expectations 📊

### Conservative Estimate
- Win Rate: 55-60%
- Avg Win: +0.2%
- Avg Loss: -0.2%
- Trades/Hour: ~20-30
- Daily Return: 1-3%

### Aggressive Estimate (Good Conditions)
- Win Rate: 60-65%
- Avg Win: +0.3%
- Avg Loss: -0.2%
- Trades/Hour: 30-50
- Daily Return: 3-7%

### Time to 20x Goal
- Conservative: 30-60 days
- Aggressive: 14-21 days
- Best case: 7-10 days

## Safety Features 🛡️

1. **Position Limits**: Max 15 positions
2. **Stop Loss**: Every position has SL
3. **Take Profit**: Every position has TP
4. **Balance Checks**: Won't trade if insufficient funds
5. **Error Recovery**: Auto-restarts on crashes
6. **Graceful Shutdown**: Closes positions on Ctrl+C

## Next Steps 🎯

1. **Set API Keys**
   ```bash
   # Windows
   set BYBIT_API_KEY=your_key
   set BYBIT_API_SECRET=your_secret
   
   # Linux/Mac
   export BYBIT_API_KEY=your_key
   export BYBIT_API_SECRET=your_secret
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Trading**
   ```bash
   # Background only
   python background_trader.py
   
   # OR with dashboard
   python run.py
   ```

4. **Monitor Progress**
   - Watch console output
   - Check web dashboard (if enabled)
   - Review logs in `logs/background_trader.log`

## Support 💬

- Check logs: `logs/background_trader.log`
- Review console output for errors
- Verify API keys are valid
- Test with paper trading first

## Disclaimer ⚠️

This is an automated trading system. Trading carries risk:
- Start with small capital
- Use paper trading first
- Monitor regularly
- Don't invest more than you can afford to lose

**Paper Trading Recommended:**
- Bybit Testnet
- Alpaca Paper Trading
- No real money at risk

---

**Ready to trade? Run `start_background.bat` or `python background_trader.py`!** 🚀
