# 🚀 Dead Simple Trading

## How It Works Now

### **Old Way (Complex):**
```bash
python cli/trader.py start --capital 100 --target 2000 --leverage 10 \
  --broker bybit --strategy microtrend --symbols BTC/USDT ETH/USDT \
  --max-positions 10 --risk-per-trade 0.02
```

### **New Way (Simple):**
```bash
python trade.py
```

That's it! Just run and enter your capital amount.

---

## What You Control

- **Capital amount** - How much to trade with

## What AI Controls (Everything Else)

- ✅ Which symbols to trade (BTC, ETH, SOL, AVAX)
- ✅ When to enter trades
- ✅ What leverage to use (5x-20x based on confidence)
- ✅ Where to set TP/SL
- ✅ When to exit
- ✅ Position sizing
- ✅ Risk management
- ✅ Auto-adjusts based on performance

---

## Quick Start

### 1. Get FREE Bybit Testnet Keys (2 min)
```
Visit: https://testnet.bybit.com
Sign up → API Management → Create Key
```

### 2. Set Credentials
**Windows:**
```powershell
$env:BYBIT_API_KEY="your_key"
$env:BYBIT_API_SECRET="your_secret"
```

**Linux/Mac:**
```bash
export BYBIT_API_KEY="your_key"
export BYBIT_API_SECRET="your_secret"
```

### 3. Install Dependencies
```bash
pip install pybit aiohttp python-dotenv
```

### 4. Run!
```bash
python trade.py
```

---

## Example Session

```bash
$ python trade.py

╔═══════════════════════════════════════════════════════════╗
║          🤖 AI AUTONOMOUS TRADER                          ║
║                                                           ║
║  YOU control:  Capital amount                            ║
║  AI controls:  Everything else                           ║
║                                                           ║
║  ✓ Symbol selection      ✓ Entry/exit timing            ║
║  ✓ Leverage calculation  ✓ TP/SL levels                 ║
║  ✓ Position sizing       ✓ Risk management              ║
╚═══════════════════════════════════════════════════════════╝

How much capital do you want to trade with?
(Use testnet fake money - get it from https://testnet.bybit.com)

💰 Enter capital (default: $100): 100

✅ Capital: $100.00
🎯 Target: $2,000.00 (20x)

Start AI autopilot? [Y/n]: y

🚀 Starting AI autopilot...

🤖 Starting Full AutoPilot Mode
💰 Capital: $100.00
🎯 Target: $2,000.00
🧠 AI controls: Symbols, Leverage, Timing, TP/SL, Exits

🔌 Connecting to Bybit testnet...
✅ Connected to Bybit | Balance: $10,000.00
🔍 Market scanner started (AI selecting symbols)
👁️  Position monitor started (AI manages exits)

======================================================================
🤖 AI AUTOPILOT - LIVE STATUS
======================================================================
💰 Capital: $100.00 / $2,000.00
📊 Progress: 5.0% █
📈 Total P&L: $+0.00 (+0.00%)
🎯 Win Rate: 0.0% (0W / 0L)
⚡ Open Positions: 0 / 5

======================================================================

🎯 AI TRADE: BTC/USDT LONG 10x | Confidence: 67.3%
✅ CLOSED: BTC/USDT long | P&L: $+3.50 (+0.35%) | TP_HIT
🎯 AI TRADE: ETH/USDT SHORT 15x | Confidence: 82.1%
✅ CLOSED: ETH/USDT short | P&L: $+4.20 (+0.42%) | TP_HIT

======================================================================
🤖 AI AUTOPILOT - LIVE STATUS
======================================================================
💰 Capital: $107.70 / $2,000.00
📊 Progress: 5.4% █
📈 Total P&L: $+7.70 (+7.70%)
🎯 Win Rate: 100.0% (2W / 0L)
⚡ Open Positions: 1 / 5

📋 Active Positions:
  SOL/USDT     long   12x | P&L: $+2.30 (+0.23%) | 🟢 Liq: 18.3%

======================================================================
```

---

## What Happens Automatically

### AI Analyzes Markets Every 5 Seconds
- Scans BTC, ETH, SOL, AVAX
- Calculates entry confidence
- Decides long/short direction

### AI Calculates Leverage Based on Confidence
- 85%+ confidence → 20x leverage (aggressive)
- 70%-85% confidence → 10x leverage (balanced)
- 55%-70% confidence → 5x leverage (conservative)

### AI Sets TP/SL Based on Volatility
- High confidence → Tight scalp (0.4% TP, 0.2% SL)
- Medium confidence → Standard (0.3% TP, 0.25% SL)
- Lower confidence → Conservative (0.2% TP, 0.3% SL)

### AI Manages Risk Automatically
- Exits if too close to liquidation (< 5%)
- Closes positions after 5 minutes (reduce exposure)
- Reduces leverage after losing streak
- Increases leverage after winning streak
- Emergency stop at -50% drawdown

### AI Displays Live Updates
- Updates every 10 seconds
- Shows current P&L, win rate, open positions
- Color-coded liquidation risk (🟢🟡🔴)
- Progress bar to target

---

## Stop Trading

Press `Ctrl+C` to stop anytime. AI will gracefully shut down.

---

## Why This Is Better

### Before:
- 10+ command-line arguments
- Hard to understand what's happening
- Manual parameter tuning
- Separate commands for monitoring
- Complex configuration files

### Now:
- One question: "How much capital?"
- AI does everything else
- Live updates built-in
- Auto-adjusts to performance
- Just works!

---

## Advanced (Optional)

If you want to modify AI behavior, edit `core/autopilot.py`:

```python
# Line 25: Change target multiplier
target_profit = capital * 50  # 50x instead of 20x

# Line 41: Change max positions
max_positions: int = 10  # Manage 10 positions instead of 5

# Line 55: Change confidence threshold
ai_confidence_threshold = 0.60  # Only trade 60%+ confidence

# Line 56: Change base leverage
base_leverage = 5  # Start with 5x instead of 10x
```

---

## Troubleshooting

**"Bybit credentials not found"**
→ Set BYBIT_API_KEY and BYBIT_API_SECRET environment variables

**"Connection failed"**
→ Check API key has "Read-Write" permissions on testnet.bybit.com

**"Insufficient balance"**
→ Request test funds at https://testnet.bybit.com (unlimited fake USDT)

**"No trades happening"**
→ AI confidence threshold not met. Market conditions may be unfavorable.

---

## That's It!

No complex configs. No parameter hell. Just:

```bash
python trade.py
```

Enter your capital and watch the AI work. 🚀
