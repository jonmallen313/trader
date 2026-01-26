"""
REAL AI TRADER - Actually works this time.
Uses real market data + real AI model = real money (or losses).
"""

import asyncio
import os
import logging
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

# Broker
from brokers.bybit import BybitBroker

# AI Model
from models.microtrend_ai import XGBoostMicroTrend, Prediction
from src.autopilot import PositionSide

# Data Feed
import ccxt.async_support as ccxt


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
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


class RealAITrader:
    """
    Actually intelligent trader.
    - Uses real market data
    - Uses real AI model
    - Makes real (paper) trades
    """
    
    def __init__(self, capital: float, target: float = None):
        self.capital = capital
        self.target = target or capital * 20
        self.balance = capital
        self.positions: List[Position] = []
        self.closed_trades = []
        
        # Real components
        self.broker = None
        self.ai_model = XGBoostMicroTrend()
        self.exchange = None  # For market data
        
        # Market data buffers
        self.price_history = {
            'BTC/USDT': deque(maxlen=100),
            'ETH/USDT': deque(maxlen=100),
            'SOL/USDT': deque(maxlen=100)
        }
        
        # Settings
        self.max_positions = 5
        self.min_confidence = 0.60  # Only trade when AI is confident
        self.position_size_pct = 0.15  # 15% of balance per trade
        
    async def start(self):
        """Start the real trader."""
        logger.info("=" * 60)
        logger.info("🤖 REAL AI TRADER")
        logger.info("=" * 60)
        logger.info(f"💰 Capital: ${self.capital:,.2f}")
        logger.info(f"🎯 Target: ${self.target:,.2f}")
        logger.info(f"🧠 AI Model: XGBoost Microtrend Detector")
        logger.info(f"📊 Minimum Confidence: {self.min_confidence:.0%}")
        logger.info("=" * 60)
        logger.info("")
        
        # Initialize
        await self._setup()
        
        # Run
        await asyncio.gather(
            self._market_data_loop(),
            self._trading_loop(),
            self._monitor_loop(),
            return_exceptions=True
        )
    
    async def _setup(self):
        """Initialize broker and exchange."""
        # Setup broker
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError(
                "Missing Bybit credentials!\n"
                "1. Get free testnet keys: https://testnet.bybit.com\n"
                "2. Set BYBIT_API_KEY and BYBIT_API_SECRET in .env"
            )
        
        self.broker = BybitBroker(api_key, api_secret, testnet=True)
        await self.broker.connect()
        
        # Setup exchange for market data
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'},
        })
        self.exchange.set_sandbox_mode(True)
        
        logger.info("✅ Connected to Bybit Testnet")
        logger.info("")
    
    async def _market_data_loop(self):
        """Continuously fetch real market data."""
        while True:
            try:
                for symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    
                    # Store price history
                    self.price_history[symbol].append({
                        'price': price,
                        'volume': ticker.get('baseVolume', 0),
                        'timestamp': datetime.now()
                    })
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Market data error: {e}")
                await asyncio.sleep(5)
    
    async def _trading_loop(self):
        """Main trading logic - find and execute trades."""
        await asyncio.sleep(10)  # Wait for initial data
        
        while True:
            try:
                # Check if we should trade
                if len(self.positions) >= self.max_positions:
                    await asyncio.sleep(5)
                    continue
                
                if self.balance < self.capital * 0.3:  # Stop at 70% loss
                    logger.warning("🛑 Balance too low - stopping new trades")
                    await asyncio.sleep(60)
                    continue
                
                # Scan each symbol for opportunities
                for symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
                    if len(self.positions) >= self.max_positions:
                        break
                    
                    # Get AI prediction
                    prediction = await self._get_ai_prediction(symbol)
                    
                    if prediction and prediction.confidence >= self.min_confidence:
                        await self._execute_trade(symbol, prediction)
                
                await asyncio.sleep(5)  # Scan every 5 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)
    
    async def _get_ai_prediction(self, symbol: str) -> Optional[Prediction]:
        """Get AI prediction for a symbol."""
        try:
            # Need sufficient price history
            if len(self.price_history[symbol]) < 20:
                return None
            
            # Build market data dict for AI
            history = list(self.price_history[symbol])
            current = history[-1]
            
            market_data = {
                'symbol': symbol.replace('/', ''),
                'current_price': current['price'],
                'current_volume': current['volume'],
                'timestamp': current['timestamp'],
                
                # Calculate features from history
                'price_change_1': self._calc_change(history, 1),
                'price_change_5': self._calc_change(history, 5),
                'price_change_10': self._calc_change(history, 10),
                'price_volatility': self._calc_volatility(history, 10),
                'volume_ratio': self._calc_volume_ratio(history, 10),
                
                # Moving averages
                'sma_5': self._calc_sma(history, 5),
                'sma_10': self._calc_sma(history, 10),
                'sma_20': self._calc_sma(history, 20),
            }
            
            # Get AI prediction
            prediction = self.ai_model.predict(market_data)
            return prediction
            
        except Exception as e:
            logger.debug(f"Prediction error for {symbol}: {e}")
            return None
    
    def _calc_change(self, history: list, periods: int) -> float:
        """Calculate price change over periods."""
        if len(history) <= periods:
            return 0.0
        current = history[-1]['price']
        past = history[-periods-1]['price']
        return (current - past) / past if past > 0 else 0.0
    
    def _calc_volatility(self, history: list, periods: int) -> float:
        """Calculate price volatility."""
        if len(history) <= periods:
            return 0.0
        prices = [h['price'] for h in history[-periods:]]
        import numpy as np
        return float(np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0.0
    
    def _calc_volume_ratio(self, history: list, periods: int) -> float:
        """Calculate volume ratio."""
        if len(history) <= periods:
            return 1.0
        volumes = [h['volume'] for h in history[-periods:]]
        current_vol = history[-1]['volume']
        import numpy as np
        avg_vol = np.mean(volumes)
        return float(current_vol / avg_vol) if avg_vol > 0 else 1.0
    
    def _calc_sma(self, history: list, periods: int) -> float:
        """Calculate simple moving average."""
        if len(history) < periods:
            return history[-1]['price'] if history else 0.0
        prices = [h['price'] for h in history[-periods:]]
        import numpy as np
        return float(np.mean(prices))
    
    async def _execute_trade(self, symbol: str, prediction: Prediction):
        """Execute a trade based on AI prediction."""
        try:
            # Calculate position size
            position_value = self.balance * self.position_size_pct
            
            # Calculate leverage based on confidence
            if prediction.confidence >= 0.80:
                leverage = 15
            elif prediction.confidence >= 0.70:
                leverage = 10
            else:
                leverage = 5
            
            # Get current price
            current_price = self.price_history[symbol][-1]['price']
            
            # Calculate TP/SL
            if prediction.side == PositionSide.LONG:
                tp_price = current_price * (1 + prediction.tp_pct)
                sl_price = current_price * (1 - prediction.sl_pct)
            else:
                tp_price = current_price * (1 - prediction.tp_pct)
                sl_price = current_price * (1 + prediction.sl_pct)
            
            # Execute on broker
            symbol_clean = symbol.replace('/', '')
            await self.broker.set_leverage(symbol_clean, leverage)
            
            order = await self.broker.place_order(
                symbol=symbol_clean,
                side=prediction.side.value,
                amount=position_value,
                price=current_price,
                take_profit=tp_price,
                stop_loss=sl_price
            )
            
            # Track position
            position = Position(
                symbol=symbol,
                side=prediction.side.value,
                entry_price=current_price,
                size=position_value,
                leverage=leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                timestamp=datetime.now(),
                id=order.get('id')
            )
            self.positions.append(position)
            
            logger.info(
                f"🎯 OPENED {symbol} {prediction.side.value.upper()} | "
                f"${position_value:.2f} @ {leverage}x | "
                f"Entry: ${current_price:,.2f} | "
                f"Confidence: {prediction.confidence:.0%}"
            )
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def _monitor_loop(self):
        """Monitor positions and close when needed."""
        await asyncio.sleep(15)  # Wait for initial setup
        
        while True:
            try:
                for position in list(self.positions):
                    symbol = position.symbol
                    current_price = self.price_history[symbol][-1]['price']
                    
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
                    else:  # short
                        if current_price <= position.tp_price:
                            should_close = True
                            reason = "TP"
                        elif current_price >= position.sl_price:
                            should_close = True
                            reason = "SL"
                    
                    # Check time limit (30 min max)
                    age = (datetime.now() - position.timestamp).total_seconds() / 60
                    if age > 30:
                        should_close = True
                        reason = "TIME"
                    
                    if should_close:
                        await self._close_position(position, current_price, reason)
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)
    
    async def _close_position(self, position: Position, close_price: float, reason: str):
        """Close a position and update balance."""
        try:
            # Close on broker
            symbol_clean = position.symbol.replace('/', '')
            await self.broker.close_position(symbol_clean)
            
            # Calculate P&L
            if position.side == 'long':
                pnl_pct = (close_price - position.entry_price) / position.entry_price
            else:  # short
                pnl_pct = (position.entry_price - close_price) / position.entry_price
            
            pnl = position.size * pnl_pct * position.leverage
            
            # Update balance
            self.balance += pnl
            
            # Track
            self.closed_trades.append({
                'position': position,
                'pnl': pnl,
                'reason': reason
            })
            self.positions.remove(position)
            
            # Log
            emoji = "✅" if pnl > 0 else "❌"
            logger.info(
                f"{emoji} CLOSED {position.symbol} {position.side} | "
                f"${pnl:+.2f} | {reason}"
            )
            
            # Display stats every close
            self._display_stats()
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
    
    def _display_stats(self):
        """Display current performance."""
        wins = sum(1 for t in self.closed_trades if t['pnl'] > 0)
        total = len(self.closed_trades)
        win_rate = (wins / total * 100) if total > 0 else 0
        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"💰 Balance: ${self.balance:.2f} | P&L: ${total_pnl:+.2f}")
        logger.info(f"📊 Active: {len(self.positions)} | Closed: {total} | Win: {win_rate:.0f}%")
        logger.info("=" * 60)
        logger.info("")


async def main():
    """Entry point."""
    print("╔" + "═" * 60 + "╗")
    print("║" + "  🤖 REAL AI TRADER - ACTUALLY USES AI  ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    
    capital = input("💵 Enter capital (default $100): ").strip()
    capital = float(capital) if capital else 100.0
    
    trader = RealAITrader(capital)
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        logger.info("\n\n👋 Shutting down...")
        if trader.exchange:
            await trader.exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
