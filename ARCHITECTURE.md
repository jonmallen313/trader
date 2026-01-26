# 🏗️ AI Trading System - Clean Architecture

## Design Principles
- **Separation of Concerns**: Each module has ONE responsibility
- **Dependency Injection**: Easy to test and swap implementations
- **Event-Driven**: Components communicate via events, not tight coupling
- **Analytics-First**: Every decision is logged and measurable

---

## 📁 New Directory Structure

```
trader/
├── core/                          # Core business logic (no dependencies)
│   ├── engine.py                  # Main trading engine orchestrator
│   ├── position.py                # Position models and lifecycle
│   ├── signal.py                  # Trading signal definitions
│   └── events.py                  # Event bus for system communication
│
├── brokers/                       # Broker integrations (swappable)
│   ├── base.py                    # BrokerInterface ABC
│   ├── bybit.py                   # Bybit futures with leverage ⭐ NEW
│   ├── alpaca.py                  # Alpaca spot trading
│   ├── binance.py                 # Binance spot/futures
│   └── mock.py                    # Paper trading simulator
│
├── strategies/                    # Trading strategies (pluggable)
│   ├── base.py                    # Strategy interface
│   ├── microtrend.py              # AI microtrend scalping
│   ├── momentum.py                # Momentum following
│   └── mean_reversion.py          # Mean reversion
│
├── risk/                          # Risk management (isolated)
│   ├── position_sizer.py          # Position sizing with leverage
│   ├── liquidation_guard.py       # Liquidation prevention
│   ├── drawdown_monitor.py        # Drawdown tracking
│   └── risk_calculator.py         # Risk metrics
│
├── analytics/                     # Analytics and reporting ⭐ NEW
│   ├── metrics.py                 # Performance metrics calculator
│   ├── reporter.py                # Report generation
│   ├── logger.py                  # Structured trade logging
│   └── visualizer.py              # Chart generation
│
├── api/                           # API layer (clean interface)
│   ├── trading.py                 # Trading endpoints
│   ├── analytics.py               # Analytics endpoints
│   └── control.py                 # System control endpoints
│
├── cli/                           # Command-line interface
│   ├── trader.py                  # Main CLI controller
│   ├── analytics.py               # Analytics CLI
│   └── config.py                  # Configuration management
│
└── config/                        # Configuration
    ├── trading.yaml               # Trading parameters
    ├── brokers.yaml               # Broker configurations
    └── strategies.yaml            # Strategy settings
```

---

## 🎯 Core Components

### 1. **TradingEngine** (core/engine.py)
- Orchestrates all components
- Event-driven architecture
- Clean start/stop lifecycle
- No business logic (delegates to strategies)

### 2. **BrokerInterface** (brokers/base.py)
- Abstract interface for all brokers
- Standardized order execution
- Leverage support built-in
- Position tracking

### 3. **Strategy** (strategies/base.py)
- Generates signals
- No direct market access
- Testable in isolation

### 4. **RiskManager** (risk/)
- Position sizing with leverage
- Liquidation price calculation
- Max drawdown enforcement
- Emergency stop loss

### 5. **Analytics** (analytics/)
- Real-time metrics
- Historical performance
- Risk reports
- Trade journal

---

## 🔄 Data Flow

```
Market Data → Strategy → Signal → Risk Check → Broker → Position
     ↓           ↓          ↓          ↓          ↓         ↓
  Analytics ← Analytics ← Analytics ← Analytics ← Analytics ← Analytics
```

**Everything is logged and measurable**

---

## ⚙️ Usage Examples

### Simple Start
```bash
python cli/trader.py start --capital 100 --target 2000 --leverage 10 --broker bybit
```

### Advanced Control
```bash
python cli/trader.py start \
  --capital 500 \
  --target 10000 \
  --leverage 20 \
  --broker bybit \
  --strategy microtrend \
  --max-positions 10 \
  --risk-per-trade 0.02
```

### Analytics
```bash
python cli/analytics.py summary           # Get current stats
python cli/analytics.py trades --last 24h # Show recent trades
python cli/analytics.py risk              # Risk assessment
```

### API Server
```bash
python cli/trader.py serve                # Start API server
curl http://localhost:8000/analytics/metrics
curl http://localhost:8000/trading/positions
```

---

## 🎛️ Configuration

### trading.yaml
```yaml
capital:
  initial: 100
  target: 2000
  stop_loss: -50

leverage:
  enabled: true
  max: 20
  default: 10

positions:
  max_open: 10
  size_pct: 0.05
  tp_pct: 0.002
  sl_pct: 0.003

risk:
  max_drawdown: -200
  max_daily_loss: -100
  liquidation_buffer: 0.2  # Keep 20% margin buffer
```

### brokers.yaml
```yaml
bybit:
  testnet: true
  api_key: ${BYBIT_API_KEY}
  api_secret: ${BYBIT_API_SECRET}
  leverage: 10
  
alpaca:
  paper: true
  api_key: ${ALPACA_API_KEY}
  api_secret: ${ALPACA_API_SECRET}
```

---

## 📊 Analytics Dashboard

Instead of messy Streamlit, we'll build:
- **CLI Analytics** - Fast, terminal-based
- **REST API** - Query metrics programmatically  
- **Optional Web UI** - Clean, professional dashboard

### Real-time Metrics
- Win rate
- Profit factor
- Sharpe ratio
- Max drawdown
- Average trade duration
- Liquidation distance
- Margin usage

### Trade Journal
- Every trade logged with context
- Entry/exit reasons
- AI confidence scores
- Market conditions
- Performance attribution

---

## 🚀 Migration Plan

1. ✅ Create new structure (preserve old code)
2. Build core components (engine, events, models)
3. Migrate brokers (add Bybit with leverage)
4. Refactor strategies (clean separation)
5. Add risk management (leverage-aware)
6. Build analytics (professional metrics)
7. Create CLI (unified interface)
8. Test everything
9. Archive old code

Old code stays functional during migration.
