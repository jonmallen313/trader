# 🚀 QUICK START - Always Running Trader

## What Changed?

### ✅ 15 Concurrent Positions (was 5)
- Entire $100 balance now split across 15 trades
- Each trade: ~$6.67 (6.7% of balance)
- Maximizes capital utilization

### ✅ Always Running (Background Mode)
- No longer requires webpage to be loaded
- Runs continuously as background service
- Automatically maintains 15 open positions
- Restarts on errors

## How to Run

### Option 1: Background Only (Recommended)
```bash
# Windows
start_background.bat

# Linux/Mac  
./start_background.sh

# OR direct
python background_trader.py
```

### Option 2: With Dashboard
```bash
# Windows
start_dashboard.bat

# Linux/Mac
./start_dashboard.sh

# OR direct
python run.py
```

## What Happens Now

1. **Trader starts** → Analyzes 15 crypto symbols
2. **Opens 15 positions** → Spreads $100 across all
3. **Monitors continuously** → Checks prices every second
4. **Closes on TP/SL** → Takes profits or cuts losses
5. **Opens new position** → Immediately replaces closed position
6. **Repeat forever** → Always maintains 15 positions

## Example
```
Starting Balance: $100
Position 1:  BTC/USDT  LONG   $6.67
Position 2:  ETH/USDT  SHORT  $6.67
Position 3:  SOL/USDT  LONG   $6.67
...
Position 15: TRX/USDT  LONG   $6.67

Total Deployed: $100 (100% utilization)
Positions: 15/15 ✅
Status: ALWAYS TRADING
```

## Files Created/Modified

### New Files
- `background_trader.py` - **Main trading engine (use this!)**
- `start_background.bat` - Windows launcher
- `start_dashboard.bat` - Windows launcher with UI
- `start_background.sh` - Linux launcher
- `start_dashboard.sh` - Linux launcher with UI
- `ALWAYS_RUNNING_README.md` - Full documentation

### Modified Files
- `config/settings.py` - MAX_POSITIONS = 15
- `core/autopilot.py` - max_positions = 15
- `trade_live.py` - 15 positions
- `railway_app.py` - Auto-start trader
- `complete_trader.py` - 15 positions
- `run.py` - Uses background trader

## Before You Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API keys** (optional for testing)
   ```bash
   # Windows
   set BYBIT_API_KEY=your_key
   set BYBIT_API_SECRET=your_secret
   
   # Linux/Mac
   export BYBIT_API_KEY=your_key
   export BYBIT_API_SECRET=your_secret
   ```

3. **Run the trader**
   ```bash
   python background_trader.py
   ```

## What You'll See

```
================================================================================
🤖 BACKGROUND AI TRADER - ALWAYS RUNNING
================================================================================
💰 Starting Capital: $100.00
🎯 Target Profit: $2,000.00 (20x)
📊 Max Positions: 15 (always trading)
💹 Position Size: 6.7% per trade
🎲 Trading Universe: 15 symbols
🧠 AI Confidence Threshold: 52%
================================================================================

✅ OPENED LONG: BTC/USDT @ $45,123.00 | Size: $6.67 | Conf: 58%
✅ OPENED SHORT: ETH/USDT @ $2,345.67 | Size: $6.67 | Conf: 61%
📊 Open: 2/15 | Need: 13 more positions
✅ OPENED LONG: SOL/USDT @ $98.45 | Size: $6.67 | Conf: 55%
...
📊 Open: 15/15 | Need: 0 more positions

🟢 CLOSED LONG: BTC/USDT @ $45,178.00 | Reason: TP | PnL: +$0.34 | Balance: $100.34
✅ OPENED LONG: BNB/USDT @ $312.45 | Size: $6.68 | Conf: 57%
📊 Open: 15/15 | Need: 0 more positions

================================================================================
📊 PERFORMANCE REPORT
================================================================================
💰 Balance: $101.23 (+1.2%)
🎯 Target: $2,000.00 (5.1% complete)
📈 Open Positions: 15/15
📊 Total Trades: 23 | Win Rate: 60.9%
⏱️  Runtime: 0:10:45
================================================================================
```

## Stopping

Press `Ctrl+C` - Will gracefully close all positions and exit.

## Need Help?

Read the full documentation: [ALWAYS_RUNNING_README.md](ALWAYS_RUNNING_README.md)

---

**Ready? Just run:** `python background_trader.py` 🚀
