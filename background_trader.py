"""
BACKGROUND AI TRADER
Runs continuously in the background without needing a web server.
Trades 15 positions simultaneously, always running.
"""

import asyncio
import os
import logging
import signal
import sys
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

# Broker
try:
    from brokers.bybit import BybitBroker
except ImportError:
    BybitBroker = None

# AI Model
try:
    from models.microtrend_ai import XGBoostMicroTrend, Prediction
except ImportError:
    XGBoostMicroTrend = None
    Prediction = None

# Data Feed
import ccxt.async_support as ccxt


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/background_trader.log', mode='a') if Path('logs').exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Track an open position."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float
    leverage: int
    tp_price: float
    sl_price: float
    timestamp: datetime
    id: Optional[str] = None
    pnl: float = 0.0


class BackgroundAITrader:
    """
    Autonomous AI trader that runs continuously in background.
    - Always trading 15 positions
    - No web interface needed
    - Restarts automatically on crash
    """
    
    def __init__(self, capital: float = 100.0):
        self.capital = capital
        self.target = capital * 20  # 20x multiplier goal
        self.balance = capital
        self.positions: List[Position] = []
        self.closed_trades = []
        
        # Real components
        self.broker = None
        self.ai_model = None
        self.exchange = None  # For market data
        
        # Trading universe - diversified across crypto
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
            'DOT/USDT', 'UNI/USDT', 'LINK/USDT', 'ATOM/USDT',
            'LTC/USDT', 'XRP/USDT', 'TRX/USDT'
        ]
        
        # Market data buffers
        self.price_history = {symbol: deque(maxlen=100) for symbol in self.symbols}
        self.last_prices = {}
        
        # Trading settings
        self.max_positions = 15  # Always aim for 15 positions
        self.min_confidence = 0.52  # Aggressive threshold
        self.position_size_pct = 0.067  # ~6.7% per position (100/15)
        self.leverage = 10  # Conservative leverage
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.start_time = datetime.now()
        self.last_trade_time = None
        
        # Signal handler for graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\n🛑 Shutdown signal received. Closing positions...")
        self.running = False
        
    async def start(self):
        """Start the background trader."""
        logger.info("=" * 80)
        logger.info("🤖 BACKGROUND AI TRADER - ALWAYS RUNNING")
        logger.info("=" * 80)
        logger.info(f"💰 Starting Capital: ${self.capital:,.2f}")
        logger.info(f"🎯 Target Profit: ${self.target:,.2f} (20x)")
        logger.info(f"📊 Max Positions: {self.max_positions} (always trading)")
        logger.info(f"💹 Position Size: {self.position_size_pct:.1%} per trade")
        logger.info(f"🎲 Trading Universe: {len(self.symbols)} symbols")
        logger.info(f"🧠 AI Confidence Threshold: {self.min_confidence:.0%}")
        logger.info("=" * 80)
        logger.info("")
        
        # Initialize
        await self._setup()
        
        # Main loop - runs forever
        while self.running:
            try:
                await asyncio.gather(
                    self._market_data_loop(),
                    self._aggressive_trading_loop(),
                    self._position_monitor(),
                    self._performance_reporter(),
                    return_exceptions=True
                )
            except Exception as e:
                logger.error(f"💥 Error in main loop: {e}")
                logger.error("🔄 Restarting in 10 seconds...")
                await asyncio.sleep(10)
                
        # Cleanup on exit
        await self._cleanup()
        
    async def _setup(self):
        """Initialize components."""
        logger.info("🔧 Setting up components...")
        
        # Try to load AI model
        if XGBoostMicroTrend:
            try:
                self.ai_model = XGBoostMicroTrend()
                logger.info("✅ AI Model loaded (XGBoost)")
            except Exception as e:
                logger.warning(f"⚠️  AI Model failed to load: {e}")
                logger.warning("📉 Using fallback strategy")
        
        # Setup broker and data feed
        bybit_key = os.getenv('BYBIT_API_KEY')
        bybit_secret = os.getenv('BYBIT_API_SECRET')
        alpaca_key = os.getenv('ALPACA_API_KEY')
        alpaca_secret = os.getenv('ALPACA_API_SECRET')
        
        if bybit_key and bybit_secret and BybitBroker:
            try:
                self.broker = BybitBroker(bybit_key, bybit_secret, testnet=True)
                await self.broker.connect()
                
                self.exchange = ccxt.bybit({
                    'apiKey': bybit_key,
                    'secret': bybit_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'linear'}
                })
                self.exchange.set_sandbox_mode(True)
                
                logger.info("✅ Connected to Bybit Testnet")
                
                # Get actual balance
                balance_info = await self.broker.get_balance()
                if balance_info:
                    self.balance = balance_info.get('total', self.capital)
                    logger.info(f"💵 Current Balance: ${self.balance:,.2f}")
                    
            except Exception as e:
                logger.error(f"❌ Bybit setup failed: {e}")
                self.broker = None
                
        elif alpaca_key and alpaca_secret:
            # Use Kraken for crypto data (no restrictions)
            try:
                self.exchange = ccxt.kraken({
                    'enableRateLimit': True
                })
                logger.info("✅ Using Kraken for market data (no API keys needed)")
            except Exception as e:
                logger.error(f"❌ Kraken setup failed: {e}")
        
        if not self.exchange:
            logger.warning("⚠️  Running in SIMULATION MODE (no broker connection)")
            logger.warning("⚠️  Add BYBIT_API_KEY/SECRET or ALPACA_API_KEY/SECRET to environment")
            
    async def _market_data_loop(self):
        """Continuously fetch market data for all symbols."""
        while self.running:
            try:
                if not self.exchange:
                    await asyncio.sleep(5)
                    continue
                
                # Fetch data for all symbols (in batches to avoid rate limits)
                for symbol in self.symbols:
                    try:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        
                        self.last_prices[symbol] = price
                        self.price_history[symbol].append({
                            'price': price,
                            'volume': ticker.get('quoteVolume', 0),
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        logger.debug(f"Error fetching {symbol}: {e}")
                        
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Market data error: {e}")
                await asyncio.sleep(5)
                
    async def _aggressive_trading_loop(self):
        """Aggressively maintain 15 open positions at all times."""
        await asyncio.sleep(15)  # Wait for initial market data
        
        while self.running:
            try:
                # Check if we've hit target
                if self.balance >= self.target:
                    logger.info(f"🎉 TARGET REACHED! ${self.balance:,.2f} / ${self.target:,.2f}")
                    logger.info("🏆 20x Multiplier Achieved!")
                    self.running = False
                    break
                
                # Calculate how many positions we need
                open_positions = len(self.positions)
                positions_needed = self.max_positions - open_positions
                
                if positions_needed <= 0:
                    await asyncio.sleep(3)  # Check again soon
                    continue
                
                logger.info(f"📊 Open: {open_positions}/{self.max_positions} | Need: {positions_needed} more positions")
                
                # Open new positions to fill quota
                for _ in range(positions_needed):
                    if not self.running:
                        break
                        
                    # Find best trading opportunity
                    best_signal = await self._find_best_opportunity()
                    
                    if best_signal:
                        await self._open_position(best_signal)
                        await asyncio.sleep(2)  # Pace ourselves
                    else:
                        logger.debug("No strong signals found, waiting...")
                        break
                        
                await asyncio.sleep(5)  # Check for new opportunities every 5 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)
                
    async def _find_best_opportunity(self) -> Optional[Dict]:
        """Find the best trading opportunity across all symbols."""
        best_signal = None
        best_confidence = self.min_confidence
        
        for symbol in self.symbols:
            # Skip if already have position in this symbol
            if any(p.symbol == symbol for p in self.positions):
                continue
                
            # Check if we have enough price history
            if len(self.price_history[symbol]) < 20:
                continue
                
            # Get prediction from AI or fallback strategy
            signal = await self._analyze_symbol(symbol)
            
            if signal and signal['confidence'] > best_confidence:
                best_confidence = signal['confidence']
                best_signal = signal
                
        return best_signal
        
    async def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a symbol and generate trading signal."""
        try:
            history = list(self.price_history[symbol])
            if len(history) < 20:
                return None
                
            prices = [h['price'] for h in history[-20:]]
            current_price = prices[-1]
            
            # Try AI model first
            if self.ai_model:
                try:
                    # Prepare features for AI
                    features = self._calculate_features(prices)
                    prediction = self.ai_model.predict(features)
                    
                    if prediction.confidence >= self.min_confidence:
                        return {
                            'symbol': symbol,
                            'side': 'long' if prediction.direction == 1 else 'short',
                            'confidence': prediction.confidence,
                            'entry_price': current_price
                        }
                except Exception as e:
                    logger.debug(f"AI prediction failed for {symbol}: {e}")
                    
            # Fallback: Simple momentum strategy
            sma_short = sum(prices[-5:]) / 5
            sma_long = sum(prices[-20:]) / 20
            momentum = (sma_short - sma_long) / sma_long
            
            # Strong momentum signals
            if abs(momentum) > 0.002:  # 0.2% momentum
                confidence = min(0.65, 0.52 + abs(momentum) * 10)
                return {
                    'symbol': symbol,
                    'side': 'long' if momentum > 0 else 'short',
                    'confidence': confidence,
                    'entry_price': current_price
                }
                
        except Exception as e:
            logger.debug(f"Analysis error for {symbol}: {e}")
            
        return None
        
    def _calculate_features(self, prices: List[float]) -> Dict:
        """Calculate technical features for AI model."""
        if len(prices) < 20:
            return {}
            
        return {
            'momentum': (prices[-1] - prices[-5]) / prices[-5],
            'volatility': sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices))) / len(prices),
            'sma_ratio': prices[-1] / (sum(prices) / len(prices)),
            'price_change': (prices[-1] - prices[0]) / prices[0]
        }
        
    async def _open_position(self, signal: Dict):
        """Open a new trading position."""
        try:
            symbol = signal['symbol']
            side = signal['side']
            entry_price = signal['entry_price']
            confidence = signal['confidence']
            
            # Calculate position size
            position_value = self.balance * self.position_size_pct
            position_size = position_value / entry_price
            
            # Calculate TP/SL based on confidence
            tp_pct = 0.003 if confidence > 0.60 else 0.002  # 0.3% or 0.2%
            sl_pct = 0.002  # 0.2% stop loss
            
            if side == 'long':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
                
            # Execute trade (if broker connected)
            position_id = None
            if self.broker:
                try:
                    order = await self.broker.place_order(
                        symbol=symbol.replace('/', ''),
                        side=side,
                        size=position_size,
                        leverage=self.leverage,
                        tp_price=tp_price,
                        sl_price=sl_price
                    )
                    position_id = order.id
                    entry_price = order.fill_price  # Use actual fill price
                except Exception as e:
                    logger.error(f"Order placement failed: {e}")
                    return
                    
            # Create position record
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=position_size,
                leverage=self.leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                timestamp=datetime.now(),
                id=position_id
            )
            
            self.positions.append(position)
            self.last_trade_time = datetime.now()
            
            logger.info(
                f"✅ OPENED {side.upper()}: {symbol} @ ${entry_price:,.2f} | "
                f"Size: ${position_value:.2f} | Conf: {confidence:.1%} | "
                f"TP: ${tp_price:,.2f} SL: ${sl_price:,.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            
    async def _position_monitor(self):
        """Monitor and close positions when TP/SL hit."""
        while self.running:
            try:
                if not self.positions:
                    await asyncio.sleep(5)
                    continue
                    
                for position in list(self.positions):
                    # Get current price
                    current_price = self.last_prices.get(position.symbol)
                    if not current_price:
                        continue
                        
                    # Update PnL
                    if position.side == 'long':
                        pnl_pct = (current_price - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - current_price) / position.entry_price
                        
                    position.pnl = pnl_pct * position.size * position.entry_price * position.leverage
                    
                    # Check TP/SL
                    should_close = False
                    reason = ""
                    
                    if position.side == 'long':
                        if current_price >= position.tp_price:
                            should_close = True
                            reason = "TP"
                        elif current_price <= position.sl_price:
                            should_close = True
                            reason = "SL"
                    else:
                        if current_price <= position.tp_price:
                            should_close = True
                            reason = "TP"
                        elif current_price >= position.sl_price:
                            should_close = True
                            reason = "SL"
                            
                    if should_close:
                        await self._close_position(position, current_price, reason)
                        
                await asyncio.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(5)
                
    async def _close_position(self, position: Position, close_price: float, reason: str):
        """Close a position and update balance."""
        try:
            # Close on broker if connected
            if self.broker and position.id:
                try:
                    await self.broker.close_position(position.id)
                except Exception as e:
                    logger.error(f"Broker close failed: {e}")
                    
            # Calculate actual PnL
            if position.side == 'long':
                pnl_pct = (close_price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - close_price) / position.entry_price
                
            pnl = pnl_pct * position.size * position.entry_price * position.leverage
            
            # Update balance
            self.balance += pnl
            
            # Track stats
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
                
            # Remove from open positions
            self.positions.remove(position)
            self.closed_trades.append(position)
            
            logger.info(
                f"{'🟢' if pnl > 0 else '🔴'} CLOSED {position.side.upper()}: {position.symbol} @ ${close_price:,.2f} | "
                f"Reason: {reason} | PnL: ${pnl:+.2f} | Balance: ${self.balance:,.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            
    async def _performance_reporter(self):
        """Periodically report performance metrics."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                runtime = datetime.now() - self.start_time
                win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                total_return = ((self.balance - self.capital) / self.capital * 100)
                
                logger.info("")
                logger.info("=" * 80)
                logger.info("📊 PERFORMANCE REPORT")
                logger.info("=" * 80)
                logger.info(f"💰 Balance: ${self.balance:,.2f} ({total_return:+.1f}%)")
                logger.info(f"🎯 Target: ${self.target:,.2f} ({self.balance/self.target*100:.1f}% complete)")
                logger.info(f"📈 Open Positions: {len(self.positions)}/{self.max_positions}")
                logger.info(f"📊 Total Trades: {self.total_trades} | Win Rate: {win_rate:.1f}%")
                logger.info(f"⏱️  Runtime: {runtime}")
                logger.info("=" * 80)
                logger.info("")
                
            except Exception as e:
                logger.error(f"Reporter error: {e}")
                
    async def _cleanup(self):
        """Cleanup on shutdown."""
        logger.info("🧹 Cleaning up...")
        
        # Close all positions
        for position in list(self.positions):
            current_price = self.last_prices.get(position.symbol, position.entry_price)
            await self._close_position(position, current_price, "SHUTDOWN")
            
        # Close exchange connection
        if self.exchange:
            await self.exchange.close()
            
        logger.info("✅ Cleanup complete")
        logger.info(f"💰 Final Balance: ${self.balance:,.2f}")
        logger.info(f"📊 Total Trades: {self.total_trades}")


async def main():
    """Main entry point."""
    capital = float(os.getenv('INITIAL_CAPITAL', '100'))
    
    trader = BackgroundAITrader(capital=capital)
    await trader.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
