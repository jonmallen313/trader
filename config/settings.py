import os

# Trading Configuration
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100.0))
GLOBAL_TAKE_PROFIT = float(os.getenv('GLOBAL_TAKE_PROFIT', 2000.0))
GLOBAL_STOP_LOSS = float(os.getenv('GLOBAL_STOP_LOSS', -50.0))
SPLIT_COUNT = int(os.getenv('SPLIT_COUNT', 20))

# Position Management
DEFAULT_TP_PCT = float(os.getenv('DEFAULT_TP_PCT', 0.002))  # 0.2% take profit (MICRO SCALPING)
DEFAULT_SL_PCT = float(os.getenv('DEFAULT_SL_PCT', 0.003))  # 0.3% stop loss (tight)
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 15))
POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', 0.067))  # ~6.7% of available capital per position (100/15)
MIN_POSITION_SIZE = float(os.getenv('MIN_POSITION_SIZE', 10.0))  # Minimum $10 (Alpaca crypto requirement)

# Trading Parameters
CHECK_INTERVAL = float(os.getenv('CHECK_INTERVAL', 0.5))  # seconds between monitor checks
SIGNAL_TIMEOUT = int(os.getenv('SIGNAL_TIMEOUT', 10))    # 10 seconds (FASTER)
MIN_PRICE_CHANGE = float(os.getenv('MIN_PRICE_CHANGE', 0.0001))  # 0.01% minimum (SENSITIVE)

# Exchange Settings
USE_EXCHANGE = os.getenv('USE_EXCHANGE', "binance")  # or "alpaca"
PAPER_MODE = os.getenv('PAPER_MODE', 'True').lower() == 'true'
API_RATE_LIMIT = os.getenv('API_RATE_LIMIT', 'True').lower() == 'true'

# Model Settings
MODEL_RETRAIN_INTERVAL = int(os.getenv('MODEL_RETRAIN_INTERVAL', 100))  # retrain after N trades
FEATURE_WINDOW_SIZE = int(os.getenv('FEATURE_WINDOW_SIZE', 50))     # number of ticks for features
PREDICTION_THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', 0.51))   # 51% confidence (AGGRESSIVE)

# Risk Management
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', -100.0))
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', -200.0))
RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', 0.02))

# Webhook Security
SECRET_WEBHOOK_KEY = os.getenv('SECRET_WEBHOOK_KEY', "your_secret_key_here")
WEBHOOK_PORT = int(os.getenv('WEBHOOK_PORT', os.getenv('PORT', 8000)))

# Broker API Keys (Paper Trading - No Deposit Required!)
# Alpaca Paper Trading: https://app.alpaca.markets/paper/dashboard/overview
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
# Note: alpaca-py library handles paper/live switching via paper=True parameter

# Binance Testnet (Paper Trading): https://testnet.binance.vision/
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET = True  # Use testnet for paper trading with live market data