# 🤖 AI Trading System

An automated AI-powered trading system that starts with $100 and uses machine learning to detect micro trends for achieving a 20x multiplier goal ($2000). The system supports both paper trading and live trading with real-time data analysis and continuous learning.

## 🎯 Project Goals

- **Starting Capital**: $100
- **Target Goal**: $2,000 (20x multiplier)
- **Strategy**: Micro-trend detection using AI/ML
- **Risk Management**: Per-trade TP/SL + global profit target
- **Learning**: Continuous model adaptation to market conditions

## 🚀 Features

- ✅ **Real-time Data Feeds**: Binance and Alpaca websocket integration
- ✅ **AI Micro-trend Detection**: XGBoost + Online Learning ensemble
- ✅ **Automated Trading**: Full autopilot with position management
- ✅ **Risk Management**: Per-position and global TP/SL controls
- ✅ **TradingView Integration**: Webhook for Pine Script signals
- ✅ **Paper Trading**: Risk-free testing mode
- ✅ **Live Dashboard**: Real-time monitoring with Streamlit
- ✅ **Backtesting**: Historical strategy validation
- ✅ **Continuous Learning**: Models retrain on live data

## 📁 Project Structure

```
trader/
├── src/
│   ├── autopilot.py          # Core trading engine
│   ├── brokers.py            # Broker integrations (Binance, Alpaca)
│   ├── webhook.py            # TradingView webhook server
│   └── dashboard.py          # Streamlit monitoring dashboard
├── models/
│   └── microtrend_ai.py      # AI/ML prediction models
├── data/
│   └── market_data.py        # Real-time data feeds
├── config/
│   ├── settings.py           # Configuration parameters
│   └── .env.example          # Environment variables template
├── tests/
│   └── backtesting.py        # Testing and validation framework
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## � Quick Start - Get Trading in 2 Minutes!

### ⚡ Alpaca Paper Trading (Recommended - Works Everywhere!)

**FREE paper trading with real market data - no geo-restrictions!**

1. **Sign up** (free): https://app.alpaca.markets/signup
2. **Get API keys**: Go to Paper Trading → API Keys
3. **Add to Railway**:
   ```bash
   ALPACA_API_KEY=PKxxxxxxxxx
   ALPACA_API_SECRET=yyyyyyyyyyyy
   ```
4. **Deploy** - Your AI trader starts immediately!

📖 **Full guide**: `docs/alpaca-setup-guide.md`

### What You Get:
- ✅ Real-time US stock & crypto prices
- ✅ $100K virtual cash (our system uses $100)
- ✅ Trade stocks: AAPL, TSLA, NVDA, SPY, QQQ
- ✅ Trade crypto: BTCUSD, ETHUSD, SOLUSD
- ✅ **Zero geo-restrictions** (works on Railway)
- ✅ **100% free forever**

---

### Alternative Options

**Option 2: Binance Testnet** (Crypto only - may be geo-restricted)
- Sign up: https://testnet.binance.vision/
- Get unlimited fake crypto
- Note: Blocked in some regions (including Railway servers)

**Option 3: Mock Mode** (Testing only)
- No API keys needed
- Built-in simulation
- Perfect for testing AI models

📖 **Full setup guide**: `docs/paper-trading-setup.md`

### Local Development

The easiest way to run this system 24/7 is on Railway:

1. **Fork/Clone this repository**
2. **Deploy to Railway**:
   - Visit [railway.app](https://railway.app)
   - Connect your GitHub account
   - Deploy from your repository
   - Set environment variables (see below)

3. **Set Environment Variables in Railway**:
```env
PAPER_MODE=true
BINANCE_API_KEY=your_testnet_key
BINANCE_SECRET=your_testnet_secret
SECRET_WEBHOOK_KEY=your_random_key
INITIAL_CAPITAL=100.0
GLOBAL_TAKE_PROFIT=2000.0
```

4. **Get Your URLs**:
   - App: `https://your-app.railway.app`
   - Webhook: `https://your-app.railway.app/webhook/tradingview`

📖 **Detailed Railway Guide**: [docs/RAILWAY_DEPLOYMENT.md](docs/RAILWAY_DEPLOYMENT.md)

### 💻 Local Development

For local testing and development:

```bash
git clone <repository-url>
cd trader
pip install -r requirements.txt
cp config/.env.example config/.env
# Edit .env with your API keys
python main.py paper --capital 100 --target 2000
```

## 🎮 Usage

### Paper Trading Mode (Recommended for testing)
```bash
python main.py paper --capital 100 --target 2000
```

### Live Trading Mode (Real money)
```bash
python main.py live --capital 100 --target 2000
```

### Backtesting Mode
```bash
python main.py backtest --symbols BTC/USDT ETH/USDT
```

### With Custom Symbols
```bash
python main.py paper --symbols BTC/USDT ETH/USDT ADA/USDT DOT/USDT
```

## 📊 Monitoring Dashboard

Your Railway deployment includes a built-in dashboard accessible via your app URL:

```
https://your-app-name.railway.app/
```

### Dashboard Features:
- Real-time P&L tracking
- Open positions monitoring  
- Signal history
- Performance metrics
- Manual signal injection
- System controls

### API Endpoints:
- **Status**: `GET /status` - System status and metrics
- **Positions**: `GET /positions` - Current open positions
- **History**: `GET /signals/history` - Recent signal history
- **Health**: `GET /health` - Health check endpoint
- **Manual Signal**: `POST /webhook/manual` - Send test signals

## 📡 TradingView Integration

### 1. Webhook Server
Your Railway deployment automatically provides a webhook URL:
```
https://your-app-name.railway.app/webhook/tradingview
```

### 2. Pine Script Template
Use this template for your TradingView strategy:

```pinescript
//@version=5
strategy("AI Trader Signal", overlay=true)

// Your trading strategy logic
sma_fast = ta.sma(close, 10)
sma_slow = ta.sma(close, 20)

long_condition = ta.crossover(sma_fast, sma_slow)
short_condition = ta.crossunder(sma_fast, sma_slow)

// Send signals to Railway
webhook_url = "https://your-app.railway.app/webhook/tradingview"

if long_condition
    strategy.entry("Long", strategy.long)
    alert('{"symbol":"' + syminfo.ticker + '","action":"BUY","tp_pct":0.02,"sl_pct":0.01}', alert.freq_once_per_bar)

if short_condition
    strategy.entry("Short", strategy.short)
    alert('{"symbol":"' + syminfo.ticker + '","action":"SELL","tp_pct":0.02,"sl_pct":0.01}', alert.freq_once_per_bar)

plot(sma_fast, "Fast SMA", color=color.blue)
plot(sma_slow, "Slow SMA", color=color.red)
```

### 3. TradingView Alert Setup
1. Add the Pine Script to your TradingView chart
2. Create an alert on the strategy
3. Set webhook URL to your Railway app
4. Use JSON format for the message
5. Enable "Once Per Bar Close"

## 🧠 AI Strategy Components

### Ensemble Predictor
- **XGBoost Model**: Fast gradient boosting for pattern recognition
- **Online Learning**: Continuous adaptation using River library
- **Feature Engineering**: 17+ technical indicators and microstructure features

### Key Features:
- Price momentum (1, 5, 10 periods)
- Moving averages (SMA 5, 10, 20)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volume analysis and order flow
- Volatility measures
- Time-based features

### Signal Generation:
```python
# Confidence threshold
if prediction.confidence > 0.6:
    # Execute trade with dynamic TP/SL
    tp_pct = base_tp_pct * confidence
    sl_pct = base_sl_pct * confidence
```

## 💰 Risk Management

### Position-Level Controls:
- **Take Profit**: 2% default (configurable)
- **Stop Loss**: 1% default (configurable)
- **Position Size**: 5% of available capital per trade
- **Maximum Positions**: 20 concurrent trades

### Global Controls:
- **Global Take Profit**: $2,000 (system stops)
- **Global Stop Loss**: -$50 (system stops)
- **Daily Loss Limit**: -$100
- **Maximum Drawdown**: -$200

### Capital Management:
```python
# Split capital into micro-positions
total_capital = $100
split_count = 20
position_size = $5 per trade

# Dynamic sizing based on confidence
position_size *= prediction.confidence
```

## 📈 Performance Metrics

### Trading Metrics:
- Win Rate
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Average Win/Loss
- Risk-Adjusted Returns

### Real-time Monitoring:
- Realized P&L
- Unrealized P&L
- Progress to target (%)
- Open positions count
- System uptime
- Signal accuracy

## 🔧 Configuration

### Key Settings (`config/settings.py`):
```python
# Trading Parameters
INITIAL_CAPITAL = 100.0
GLOBAL_TAKE_PROFIT = 2000.0
SPLIT_COUNT = 20
DEFAULT_TP_PCT = 0.02  # 2%
DEFAULT_SL_PCT = 0.01  # 1%

# AI Parameters
PREDICTION_THRESHOLD = 0.6
MODEL_RETRAIN_INTERVAL = 100
FEATURE_WINDOW_SIZE = 50

# Risk Management
MAX_POSITIONS = 20
POSITION_SIZE_PCT = 0.05
MAX_DAILY_LOSS = -100.0
```

## 🧪 Testing

### Run All Tests:
```bash
# Quick backtest
python -m tests.backtesting

# Paper trading test
python main.py paper --capital 1000

# Strategy validation
python -c "from tests.backtesting import TestRunner; import asyncio; print(asyncio.run(TestRunner.run_quick_backtest()))"
```

### Manual Testing:
```bash
# Test webhook endpoint
curl -X POST http://localhost:8000/webhook/manual \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","action":"BUY","tp_pct":0.02,"sl_pct":0.01}'

# Check system status
curl http://localhost:8000/status
```

## 🚨 Important Notes

### ⚠️ Risk Warning
- This system is for educational purposes
- Always test in paper mode first
- Never risk more than you can afford to lose
- Past performance doesn't guarantee future results

### 🔐 Security Best Practices
- Keep API keys secure
- Use environment variables
- Enable IP restrictions on exchanges
- Use paper trading for testing
- Regular security audits

### 📊 Expected Performance
- **Realistic Win Rate**: 55-65%
- **Target Achievement Time**: Weeks to months
- **Daily Trades**: 10-50 micro-positions
- **Risk per Trade**: 1-2% of position

## 🔄 Continuous Learning

The system continuously improves through:

1. **Online Learning**: Models update with each trade result
2. **Feature Evolution**: New indicators added based on performance
3. **Parameter Optimization**: TP/SL percentages adapt to market conditions
4. **Strategy Ensemble**: Multiple models vote on decisions

## 📞 Support & Troubleshooting

### Common Issues:
1. **Connection Errors**: Check API keys and network
2. **Import Errors**: Install requirements.txt dependencies
3. **Webhook Issues**: Verify firewall and port settings
4. **Model Errors**: Ensure sufficient training data

### Logs:
```bash
tail -f logs/trader.log
```

### Health Check:
```bash
curl http://localhost:8000/health
```

## 🎯 Roadmap

- [ ] Go execution engine for ultra-low latency
- [ ] Advanced ML models (transformers, deep RL)
- [ ] Multi-exchange arbitrage
- [ ] Portfolio rebalancing
- [ ] Social sentiment integration
- [ ] Mobile app monitoring
- [ ] Advanced backtesting with slippage models

## 📄 License

This project is for educational purposes. Use at your own risk.

---

**💡 Ready to start your 100 → 2000 journey? Begin with paper trading mode!**

```bash
python main.py paper --capital 100 --target 2000
```