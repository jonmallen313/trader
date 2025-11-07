"""
Main application entry point for the AI Trading System.
This script ties together all components and provides different run modes.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Railway configuration first
from config.railway import railway_config

from src.autopilot import AutoPilotController, TradingSignal, PositionSide
from src.brokers import BrokerManager, BinanceBroker, AlpacaBroker, MockBroker
from data.market_data import DataFeedManager, BinanceWebSocketFeed, AlpacaWebSocketFeed
from models.microtrend_ai import EnsemblePredictor, XGBoostMicroTrend, OnlineLearningModel
from src.webhook import WebhookServer
from tests.backtesting import BacktestEngine, PaperTradingMode, TestRunner
from config.settings import *


class AITradingSystem:
    """Main AI Trading System orchestrator."""
    
    def __init__(self, mode: str = "paper", config_override: dict = None):
        """
        Initialize the trading system.
        
        Args:
            mode: "paper", "live", or "backtest"
            config_override: Optional config overrides
        """
        self.mode = mode
        self.config = config_override or {}
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.broker_manager = None
        self.data_feed_manager = None
        self.predictor = None
        self.autopilot = None
        self.webhook_server = None
        
        self.logger.info(f"AI Trading System initialized in {mode} mode")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('LOG_LEVEL', os.getenv('LOG_LEVEL', 'INFO'))
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/trader.log', mode='a') if Path('logs').exists() else logging.NullHandler()
            ]
        )
    
    async def initialize_components(self):
        """Initialize all system components with error handling."""
        self.logger.info("Initializing system components...")
        
        try:
            # Initialize broker manager
            await self.setup_brokers()
        except Exception as e:
            self.logger.error(f"Broker setup failed: {e}. Using mock broker fallback.")
            self.broker_manager = BrokerManager()
            mock_broker = MockBroker(INITIAL_CAPITAL)
            self.broker_manager.add_broker(mock_broker, is_primary=True)
        
        try:
            # Initialize data feeds
            await self.setup_data_feeds()
        except Exception as e:
            self.logger.warning(f"Data feed setup failed: {e}. Continuing without live data.")
        
        try:
            # Initialize AI predictor
            await self.setup_ai_predictor()
        except Exception as e:
            self.logger.warning(f"AI predictor setup failed: {e}. System will run with basic logic.")
        
        try:
            # Initialize autopilot controller
            await self.setup_autopilot()
        except Exception as e:
            self.logger.error(f"Autopilot setup failed: {e}")
            raise
        
        try:
            # Initialize webhook server (critical for Railway health checks)
            await self.setup_webhook_server()
        except Exception as e:
            self.logger.error(f"Webhook server setup failed: {e}")
            raise
        
        self.logger.info("All components initialized successfully")
    
    async def setup_brokers(self):
        """Setup broker connections with error handling."""
        self.broker_manager = BrokerManager()
        
        try:
            if self.mode == "live":
                # Add Binance if keys available
                api_key = os.getenv('BINANCE_API_KEY')
                secret_key = os.getenv('BINANCE_SECRET')
                
                if api_key and secret_key:
                    binance = BinanceBroker(api_key, secret_key, paper_mode=False)
                    self.broker_manager.add_broker(binance, is_primary=True)
                    self.logger.info("Added Binance live broker")
                
                # Add Alpaca if keys available
                alpaca_key = os.getenv('ALPACA_API_KEY')
                alpaca_secret = os.getenv('ALPACA_SECRET')
                
                if alpaca_key and alpaca_secret:
                    alpaca = AlpacaBroker(alpaca_key, alpaca_secret, paper_mode=False)
                    self.broker_manager.add_broker(alpaca)
                    self.logger.info("Added Alpaca live broker")
                    
            elif self.mode == "paper":
                # Priority 1: Try Alpaca Paper Trading (not geo-restricted, free, real data)
                alpaca_key = os.getenv('ALPACA_API_KEY')
                alpaca_secret = os.getenv('ALPACA_API_SECRET')
                
                if alpaca_key and alpaca_secret:
                    try:
                        alpaca_paper = AlpacaBroker(alpaca_key, alpaca_secret, paper_mode=True)
                        self.broker_manager.add_broker(alpaca_paper, is_primary=True)
                        self.logger.info("✅ Added Alpaca paper trading broker (real market data)")
                    except Exception as e:
                        self.logger.warning(f"Alpaca setup failed: {e}")
                else:
                    self.logger.warning("⚠️  No ALPACA_API_KEY found. Using mock broker.")
                    self.logger.warning("📖 Get free Alpaca paper trading keys: https://app.alpaca.markets/signup")
                
                # Priority 2: Try Binance (might be geo-restricted on Railway)
                if not self.broker_manager.brokers:
                    api_key = os.getenv('BINANCE_API_KEY', 'test_key')
                    secret_key = os.getenv('BINANCE_SECRET', 'test_secret')
                    
                    try:
                        binance_paper = BinanceBroker(api_key, secret_key, paper_mode=True)
                        self.broker_manager.add_broker(binance_paper, is_primary=True)
                        self.logger.info("Added Binance paper broker")
                    except Exception as e:
                        self.logger.warning(f"Binance not available (geo-restricted): {e}")
                
                # Priority 3: Fallback to MockBroker
                if not self.broker_manager.brokers:
                    self.logger.warning("⚠️  No real brokers available - using MockBroker for simulation")
                    self.logger.warning("🔑 Add ALPACA_API_KEY and ALPACA_API_SECRET for real paper trading")
                    mock_broker = MockBroker(INITIAL_CAPITAL)
                    self.broker_manager.add_broker(mock_broker, is_primary=True)
                    self.logger.info("Using MockBroker with simulated data")
                
            else:  # backtest or testing
                mock_broker = MockBroker(INITIAL_CAPITAL)
                self.broker_manager.add_broker(mock_broker, is_primary=True)
                self.logger.info("Added mock broker for testing")
            
            # Connect all brokers (with error handling)
            try:
                await self.broker_manager.connect_all()
            except Exception as e:
                self.logger.warning(f"Broker connection failed: {e}")
                # If no brokers connected, ensure we have at least a mock broker
                if not any(b.is_connected for b in self.broker_manager.brokers):
                    self.logger.info("No brokers connected - adding mock broker fallback")
                    mock_broker = MockBroker(INITIAL_CAPITAL)
                    self.broker_manager.add_broker(mock_broker, is_primary=True)
                    await mock_broker.connect()
                    
        except Exception as e:
            self.logger.error(f"Error in setup_brokers: {e}. Using mock broker as fallback.")
            # Emergency fallback
            mock_broker = MockBroker(INITIAL_CAPITAL)
            self.broker_manager.add_broker(mock_broker, is_primary=True)
            await mock_broker.connect()
        """Setup broker connections."""
        self.broker_manager = BrokerManager()
        
        if self.mode == "live":
            # Add live brokers
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET')
            
            if api_key and secret_key:
                binance = BinanceBroker(api_key, secret_key, paper_mode=False)
                self.broker_manager.add_broker(binance, is_primary=True)
                self.logger.info("Added Binance live broker")
            
            # Add Alpaca if keys available
            alpaca_key = os.getenv('ALPACA_API_KEY')
            alpaca_secret = os.getenv('ALPACA_SECRET')
            
            if alpaca_key and alpaca_secret:
                alpaca = AlpacaBroker(alpaca_key, alpaca_secret, paper_mode=False)
                self.broker_manager.add_broker(alpaca)
                self.logger.info("Added Alpaca live broker")
                
        elif self.mode == "paper":
            # Add paper trading brokers
            api_key = os.getenv('BINANCE_API_KEY', 'test_key')
            secret_key = os.getenv('BINANCE_SECRET', 'test_secret')
            
            binance_paper = BinanceBroker(api_key, secret_key, paper_mode=True)
            self.broker_manager.add_broker(binance_paper, is_primary=True)
            self.logger.info("Added Binance paper broker")
            
        else:  # backtest or testing
            mock_broker = MockBroker(INITIAL_CAPITAL)
            self.broker_manager.add_broker(mock_broker, is_primary=True)
            self.logger.info("Added mock broker for testing")
        
        # Connect all brokers
        await self.broker_manager.connect_all()
    
    async def setup_data_feeds(self):
        """Setup market data feeds."""
        self.data_feed_manager = DataFeedManager()
        
        # Define symbols to track
        symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        
        if self.mode in ["live", "paper"]:
            # Add live data feeds
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET')
            
            if True:  # Binance websocket doesn't require API keys for market data
                binance_feed = BinanceWebSocketFeed(symbols)
                self.data_feed_manager.add_feed(binance_feed)
                self.logger.info(f"Added Binance data feed for {len(symbols)} symbols")
            
            # Add Alpaca feed if available
            alpaca_key = os.getenv('ALPACA_API_KEY')
            alpaca_secret = os.getenv('ALPACA_SECRET')
            
            if alpaca_key and alpaca_secret:
                stock_symbols = ['AAPL', 'TSLA', 'SPY']  # Example stock symbols
                alpaca_feed = AlpacaWebSocketFeed(stock_symbols, alpaca_key, alpaca_secret)
                self.data_feed_manager.add_feed(alpaca_feed)
                self.logger.info(f"Added Alpaca data feed for {len(stock_symbols)} symbols")
    
    async def setup_ai_predictor(self):
        """Setup AI prediction models."""
        self.predictor = EnsemblePredictor()
        
        # Add XGBoost model
        xgb_model = XGBoostMicroTrend()
        self.predictor.add_model(xgb_model, weight=1.0)
        
        # Add online learning model
        online_model = OnlineLearningModel()
        self.predictor.add_model(online_model, weight=0.5)
        
        self.logger.info("AI predictor ensemble created")
        
        # Try to load pre-trained models
        models_dir = Path("models/saved")
        if models_dir.exists():
            try:
                self.predictor.load_ensemble(str(models_dir))
                self.logger.info("Loaded pre-trained models")
            except Exception as e:
                self.logger.warning(f"Could not load pre-trained models: {e}")
    
    async def setup_autopilot(self):
        """Setup the autopilot controller."""
        active_broker = await self.broker_manager.get_active_broker()
        if not active_broker:
            raise RuntimeError("No active broker available")
        
        self.autopilot = AutoPilotController(
            exchange_client=active_broker,
            data_feed=self.data_feed_manager,
            model_predictor=self.predictor,
            initial_capital=INITIAL_CAPITAL
        )
        
        # Set up event callbacks
        self.autopilot.on_position_opened = self._on_position_opened
        self.autopilot.on_position_closed = self._on_position_closed
        self.autopilot.on_global_target_hit = self._on_global_target_hit
        
        self.logger.info("Autopilot controller ready")
    
    async def setup_webhook_server(self):
        """Setup the webhook server."""
        if self.mode != "backtest":
            # Use Railway port if available
            port = railway_config.port if railway_config.is_railway else WEBHOOK_PORT
            self.webhook_server = WebhookServer(self.autopilot, port=port)
            self.logger.info(f"Webhook server ready on port {port}")
            
            # Log the webhook URL for Railway deployment
            if railway_config.is_railway:
                webhook_url = railway_config.get_webhook_url()
                self.logger.info(f"🌐 Public webhook URL: {webhook_url}")
                self.logger.info(f"📊 Dashboard URL: {railway_config.get_dashboard_url()}")
    
    def _on_position_opened(self, position):
        """Handle position opened event."""
        self.logger.info(f"Position opened: {position.symbol} {position.side.value} @ ${position.entry_price:.4f}")
    
    def _on_position_closed(self, position):
        """Handle position closed event."""
        pnl_str = f"+${position.realized_pnl:.2f}" if position.realized_pnl >= 0 else f"-${abs(position.realized_pnl):.2f}"
        self.logger.info(f"Position closed: {position.symbol} P&L: {pnl_str}")
    
    def _on_global_target_hit(self, target_type: str, amount: float):
        """Handle global target hit event."""
        self.logger.info(f"🎯 GLOBAL {target_type} HIT: ${amount:.2f}")
        
        if target_type == "TAKE_PROFIT":
            self.logger.info("🎉 CONGRATULATIONS! Trading goal achieved!")
        
    async def start_system(self):
        """Start the complete trading system."""
        self.logger.info("Starting AI Trading System...")
        
        try:
            # Initialize all components
            await self.initialize_components()
            
            if self.mode == "backtest":
                # Run backtest mode
                await self.run_backtest()
            else:
                # Start live/paper trading
                await self.run_live_trading()
                
        except Exception as e:
            self.logger.error(f"System error: {e}")
            raise
    
    async def run_live_trading(self):
        """Run live or paper trading mode."""
        
        # Start webhook server FIRST (Railway needs health check immediately)
        webhook_task = None
        if self.webhook_server:
            webhook_task = asyncio.create_task(self.webhook_server.start_server())
            # Give server a moment to start and be ready for health checks
            await asyncio.sleep(2)
            self.logger.info("✅ Webhook server started and ready for health checks")
        
        tasks = []
        
        # Start data feeds
        if self.data_feed_manager:
            tasks.append(self.data_feed_manager.start_all())
        
        # Start autopilot
        tasks.append(self.autopilot.start())
        
        self.logger.info(f"🚀 AI Trading System running in {self.mode} mode")
        self.logger.info(f"📊 Target: ${GLOBAL_TAKE_PROFIT} (${INITIAL_CAPITAL} → 20x multiplier)")
        self.logger.info(f"🛡️ Global Stop Loss: ${abs(GLOBAL_STOP_LOSS)}")
        
        # Railway-specific logging
        if railway_config.is_railway:
            self.logger.info(f"☁️  Running on Railway cloud platform")
            self.logger.info(f"🌐 Environment: {railway_config.environment}")
            webhook_url = railway_config.get_webhook_url()
            self.logger.info(f"📡 TradingView Webhook: {webhook_url}")
        
        # Wait for all tasks (including webhook server)
        if webhook_task:
            tasks.append(webhook_task)
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def run_backtest(self):
        """Run backtest mode."""
        self.logger.info("Running backtest...")
        
        # Use TestRunner for quick backtest
        results = await TestRunner.run_quick_backtest("BTCUSDT", days=7)
        
        self.logger.info("Backtest Results:")
        self.logger.info(f"Total Trades: {results['total_trades']}")
        self.logger.info(f"Win Rate: {results['win_rate']:.1f}%")
        self.logger.info(f"Total P&L: ${results['total_pnl']:.2f}")
        self.logger.info(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        self.logger.info(f"Max Drawdown: {results.get('max_drawdown', 0)*100:.1f}%")
    
    async def stop_system(self):
        """Stop the trading system gracefully."""
        self.logger.info("Stopping AI Trading System...")
        
        # Stop autopilot
        if self.autopilot and self.autopilot.is_running:
            await self.autopilot.stop()
        
        # Stop data feeds
        if self.data_feed_manager:
            await self.data_feed_manager.stop_all()
        
        # Disconnect brokers
        if self.broker_manager:
            await self.broker_manager.disconnect_all()
        
        self.logger.info("System stopped")


async def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="AI Trading System")
    parser.add_argument('mode', choices=['paper', 'live', 'backtest'], 
                       help='Trading mode: paper (testing), live (real money), or backtest')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'ETH/USDT'],
                       help='Trading symbols (default: BTC/USDT ETH/USDT)')
    parser.add_argument('--capital', type=float, default=100.0,
                       help='Initial capital (default: $100)')
    parser.add_argument('--target', type=float, default=2000.0,
                       help='Take profit target (default: $2000)')
    parser.add_argument('--config', type=str,
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Create config override
    config_override = {
        'symbols': args.symbols,
        'INITIAL_CAPITAL': args.capital,
        'GLOBAL_TAKE_PROFIT': args.target
    }
    
    # Load config file if provided
    if args.config and Path(args.config).exists():
        import json
        with open(args.config) as f:
            file_config = json.load(f)
            config_override.update(file_config)
    
    # Create and start system
    system = AITradingSystem(mode=args.mode, config_override=config_override)
    
    try:
        await system.start_system()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received")
        await system.stop_system()
    except Exception as e:
        print(f"❌ System error: {e}")
        await system.stop_system()
        sys.exit(1)


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the system
    asyncio.run(main())