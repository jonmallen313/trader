# AI Trader Project Instructions

## Project Overview
This project is an automated AI trading system that:
- Starts with $100 capital
- Trades micro-positions to reach a 20x multiplier goal ($2000)
- Uses real-time data analysis and machine learning for microtrend detection
- Supports both paper trading and live trading modes
- Integrates with TradingView for signal alerts
- Continuously learns and adapts to market conditions

## Current Status
- [x] Project structure created
- [x] Core trading engine implemented
- [x] Real-time data feeds configured
- [x] AI microtrend detection built
- [x] Broker integrations added
- [x] TradingView webhook created
- [x] Monitoring dashboard setup
- [x] Testing framework implemented

## Technical Stack
- **Language**: Python (with potential Go execution engine)
- **ML Libraries**: scikit-learn, xgboost, tensorflow
- **Trading APIs**: Alpaca, Binance, CCXT
- **Web Framework**: FastAPI
- **Database**: SQLite/PostgreSQL
- **Monitoring**: Streamlit/Dash

## Key Features
1. **Capital Management**: Splits $100 into multiple micro-positions
2. **Risk Management**: Per-trade TP/SL + global profit target
3. **Continuous Learning**: Models retrain on live trading data
4. **Multi-Mode**: Paper trading for testing, live trading for execution
5. **TradingView Integration**: Receives Pine Script alerts via webhook
6. **Real-time Monitoring**: Live dashboard for positions and P&L

## Project Structure
- `/src/` - Core trading logic
- `/models/` - AI/ML models for trend detection
- `/data/` - Market data processing
- `/config/` - Configuration files
- `/tests/` - Testing and validation
- `/docs/` - Documentation