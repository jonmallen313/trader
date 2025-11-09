"""
Core trading engine for automated AI trading system.
Handles position management, TP/SL enforcement, and global profit targets.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from datetime import datetime
from enum import Enum

from config.settings import *


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class Position:
    """Represents a trading position with all necessary tracking information."""
    id: str
    symbol: str
    side: PositionSide
    size: float  # Position size in base currency
    entry_price: float
    tp_price: float  # Take profit price
    sl_price: float  # Stop loss price
    opened_at: datetime
    status: PositionStatus = PositionStatus.OPEN
    closed_at: Optional[datetime] = None
    close_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0


@dataclass
class TradingSignal:
    """Represents a trading signal from AI model or TradingView."""
    symbol: str
    side: PositionSide
    confidence: float
    tp_pct: float = DEFAULT_TP_PCT
    sl_pct: float = DEFAULT_SL_PCT
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class AutoPilotController:
    """
    Main autopilot trading controller that manages:
    - Position lifecycle
    - TP/SL enforcement
    - Global profit targets
    - Risk management
    """
    
    def __init__(self, 
                 exchange_client,
                 data_feed,
                 model_predictor,
                 initial_capital: float = INITIAL_CAPITAL):
        self.exchange = exchange_client
        self.data_feed = data_feed
        self.predictor = model_predictor
        self.initial_capital = initial_capital
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.realized_profit = 0.0
        self.unrealized_profit = 0.0
        self.available_capital = initial_capital
        self.is_running = False
        
        # Track pending orders to prevent duplicates
        self.pending_orders: Dict[str, float] = {}  # {symbol: timestamp}
        self.last_trade_time: Dict[str, float] = {}  # {symbol_side: timestamp}
        
        # Queues for async communication
        self.signal_queue = asyncio.Queue()
        self.price_updates = asyncio.Queue()
        
        # Callbacks for events
        self.on_position_opened: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        self.on_global_target_hit: Optional[Callable] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Start the autopilot trading system."""
        self.is_running = True
        self.logger.info("Starting AutoPilot Controller...")
        
        # Start all async tasks
        tasks = [
            asyncio.create_task(self._signal_processor()),
            asyncio.create_task(self._position_monitor()),
            asyncio.create_task(self._data_processor()),
            asyncio.create_task(self._global_monitor()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Shutting down AutoPilot Controller...")
            await self.stop()
            
    async def stop(self):
        """Stop the trading system and close all positions."""
        self.is_running = False
        
        # Close all open positions
        open_positions = [p for p in self.positions.values() 
                         if p.status == PositionStatus.OPEN]
        
        for position in open_positions:
            await self._close_position(position, "SYSTEM_SHUTDOWN")
            
        self.logger.info(f"AutoPilot stopped. Final P&L: {self.realized_profit:.2f}")
        
    async def add_signal(self, signal: TradingSignal):
        """Add a trading signal to the processing queue."""
        await self.signal_queue.put(signal)
        
    async def _signal_processor(self):
        """Process trading signals and open positions."""
        while self.is_running:
            try:
                signal = await self.signal_queue.get()
                
                # Check if we can open a new position
                if await self._can_open_position(signal):
                    await self._open_position(signal)
                    
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}")
                
            await asyncio.sleep(0.1)
            
    async def _can_open_position(self, signal: TradingSignal) -> bool:
        """Check if we can open a new position based on risk rules."""
        # Check for pending order on same symbol
        current_time = time.time()
        if signal.symbol in self.pending_orders:
            order_time = self.pending_orders[signal.symbol]
            if current_time - order_time < 5.0:  # 5 second cooldown
                self.logger.debug(f"⏳ Order pending for {signal.symbol}, skipping")
                return False
        
        # Check for existing position in same direction
        symbol_side_key = f"{signal.symbol}_{signal.side.value}"
        if symbol_side_key in self.last_trade_time:
            last_time = self.last_trade_time[symbol_side_key]
            if current_time - last_time < 30.0:  # 30 second cooldown
                self.logger.debug(f"⏳ Recent {signal.side.value} trade for {signal.symbol}, cooldown active")
                return False
        
        # Check for existing open position in same direction
        for position in self.positions.values():
            if (position.symbol == signal.symbol and 
                position.side == signal.side and 
                position.status == PositionStatus.OPEN):
                self.logger.debug(f"📍 Already have {signal.side.value} position for {signal.symbol}")
                return False
        
        # Check if we've hit global targets
        if self.realized_profit >= GLOBAL_TAKE_PROFIT:
            self.logger.info("Global take profit reached, no new positions")
            return False
            
        if self.realized_profit <= GLOBAL_STOP_LOSS:
            self.logger.info("Global stop loss hit, no new positions")
            return False
            
        # Check maximum positions
        open_count = sum(1 for p in self.positions.values() 
                        if p.status == PositionStatus.OPEN)
        if open_count >= MAX_POSITIONS:
            self.logger.info("Maximum positions reached")
            return False
            
        # Check available capital
        position_size = self.available_capital * POSITION_SIZE_PCT
        if position_size < MIN_POSITION_SIZE:  # Minimum $10 per position (Alpaca crypto requirement)
            self.logger.info(f"Insufficient capital for new position (need ${MIN_POSITION_SIZE}, have ${position_size:.2f})")
            return False
            
        return True
        
    async def _open_position(self, signal: TradingSignal):
        """Open a new position based on the signal."""
        try:
            # Mark order as pending to prevent duplicates
            self.pending_orders[signal.symbol] = time.time()
            
            # Get current price
            current_price = await self.exchange.get_price(signal.symbol)
            
            # Calculate position size with minimum enforcement
            position_value = self.available_capital * POSITION_SIZE_PCT
            position_value = max(position_value, MIN_POSITION_SIZE)  # Enforce $10 minimum for Alpaca crypto
            position_size = position_value / current_price
            
            # Ensure minimum position size (at least 1 share for stocks)
            if position_size < 1.0 and not signal.symbol.endswith('/USD'):  # Stock, not crypto
                # Use notional (dollar) amount instead for small positions
                self.logger.info(f"📊 Position too small ({position_size:.4f} shares), using notional ${position_value:.2f}")
                position_size = None  # Will use notional instead
                actual_position_value = position_value
            else:
                actual_position_value = position_size * current_price
            
            # Calculate TP and SL prices
            if signal.side == PositionSide.LONG:
                tp_price = current_price * (1 + signal.tp_pct)
                sl_price = current_price * (1 - signal.sl_pct)
            else:
                tp_price = current_price * (1 - signal.tp_pct)
                sl_price = current_price * (1 + signal.sl_pct)
                
            # Execute the trade
            order_result = await self.exchange.place_order(
                symbol=signal.symbol,
                side=signal.side.value,
                size=position_size if position_size else None,
                notional=position_value if not position_size else None,  # Use notional if size too small
                order_type="market"
            )
            
            # Record trade time to enforce cooldown
            symbol_side_key = f"{signal.symbol}_{signal.side.value}"
            self.last_trade_time[symbol_side_key] = time.time()
            
            # Remove from pending
            if signal.symbol in self.pending_orders:
                del self.pending_orders[signal.symbol]
            
            # Create position record
            position = Position(
                id=order_result["id"],
                symbol=signal.symbol,
                side=signal.side,
                size=position_size,
                entry_price=order_result["fill_price"],
                tp_price=tp_price,
                sl_price=sl_price,
                opened_at=datetime.now()
            )
            
            self.positions[position.id] = position
            self.available_capital -= position_value
            
            self.logger.info(f"Opened position: {position.symbol} {position.side.value} "
                           f"Size: {position.size:.4f} Entry: {position.entry_price:.4f}")
            
            if self.on_position_opened:
                self.on_position_opened(position)
                
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            # Clean up pending order on error
            if signal.symbol in self.pending_orders:
                del self.pending_orders[signal.symbol]
            
    async def _close_position(self, position: Position, reason: str = "TP_SL"):
        """Close an open position."""
        try:
            # Get current price for final calculation
            current_price = await self.exchange.get_price(position.symbol)
            
            # Execute closing trade
            close_side = "sell" if position.side == PositionSide.LONG else "buy"
            order_result = await self.exchange.place_order(
                symbol=position.symbol,
                side=close_side,
                size=position.size,
                order_type="market"
            )
            
            # Update position
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.now()
            position.close_price = order_result["fill_price"]
            
            # Calculate P&L
            if position.side == PositionSide.LONG:
                position.realized_pnl = (position.close_price - position.entry_price) * position.size
            else:
                position.realized_pnl = (position.entry_price - position.close_price) * position.size
                
            # Update global P&L
            self.realized_profit += position.realized_pnl
            self.available_capital += (position.entry_price * position.size) + position.realized_pnl
            
            self.logger.info(f"Closed position: {position.symbol} {reason} "
                           f"P&L: {position.realized_pnl:.2f}")
            
            # LIVE FEEDBACK LOOP: Teach AI from actual trade result
            try:
                # Get the market data snapshot from when position opened
                market_data = await self.data_feed.get_latest_data(position.symbol)
                if market_data and self.predictor:
                    # Feed the REAL outcome back to the AI
                    await self.predictor.learn_from_trade(
                        market_data=market_data,
                        pnl=position.realized_pnl,
                        side=position.side.value
                    )
            except Exception as e:
                self.logger.warning(f"Could not update AI from trade result: {e}")
            
            if self.on_position_closed:
                self.on_position_closed(position)
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            
    async def _position_monitor(self):
        """Monitor open positions for TP/SL triggers."""
        while self.is_running:
            try:
                open_positions = [p for p in self.positions.values() 
                               if p.status == PositionStatus.OPEN]
                
                for position in open_positions:
                    current_price = await self.exchange.get_price(position.symbol)
                    
                    # Calculate unrealized P&L
                    if position.side == PositionSide.LONG:
                        position.unrealized_pnl = (current_price - position.entry_price) * position.size
                        
                        # Check TP/SL
                        if current_price >= position.tp_price:
                            await self._close_position(position, "TAKE_PROFIT")
                        elif current_price <= position.sl_price:
                            await self._close_position(position, "STOP_LOSS")
                            
                    else:  # SHORT
                        position.unrealized_pnl = (position.entry_price - current_price) * position.size
                        
                        # Check TP/SL
                        if current_price <= position.tp_price:
                            await self._close_position(position, "TAKE_PROFIT")
                        elif current_price >= position.sl_price:
                            await self._close_position(position, "STOP_LOSS")
                            
            except Exception as e:
                self.logger.error(f"Error in position monitor: {e}")
                
            await asyncio.sleep(CHECK_INTERVAL)
            
    async def _data_processor(self):
        """Process incoming market data and generate signals."""
        self.logger.info("🤖 AI Data Processor started - analyzing market data...")
        iteration = 0
        
        while self.is_running:
            try:
                iteration += 1
                
                # Get latest market data
                market_data = await self.data_feed.get_latest_data()
                
                if market_data:
                    if iteration % 10 == 0:  # Log every 10th iteration
                        self.logger.info(f"📈 Processing market data: {len(market_data) if isinstance(market_data, dict) else 1} symbols")
                    
                    # Generate prediction using AI model
                    prediction = await self.predictor.predict(market_data)
                    
                    if prediction:
                        self.logger.info(f"🎯 AI Prediction: {prediction.symbol} {prediction.side.value} (confidence: {prediction.confidence:.2%})")
                        
                        if prediction.confidence > PREDICTION_THRESHOLD:
                            self.logger.info(f"✅ Confidence above threshold ({PREDICTION_THRESHOLD:.2%}) - generating signal")
                            signal = TradingSignal(
                                symbol=prediction.symbol,
                                side=prediction.side,
                                confidence=prediction.confidence,
                                tp_pct=prediction.tp_pct or DEFAULT_TP_PCT,
                                sl_pct=prediction.sl_pct or DEFAULT_SL_PCT
                            )
                            await self.add_signal(signal)
                        else:
                            if iteration % 10 == 0:
                                self.logger.debug(f"⏸️ Low confidence ({prediction.confidence:.2%}) - skipping trade")
                    else:
                        if iteration % 100 == 0:  # Only log every 100 iterations
                            self.logger.debug("⚠️ No prediction generated by AI model")
                else:
                    if iteration % 100 == 0:  # Only log every 100 iterations
                        self.logger.debug("⚠️ No market data available from feed")
                        
            except Exception as e:
                self.logger.error(f"❌ Error in data processor: {e}", exc_info=True)
                
            await asyncio.sleep(0.1)
            
    async def _global_monitor(self):
        """Monitor global P&L and enforce take profit/stop loss."""
        while self.is_running:
            try:
                # Update unrealized profit
                self.unrealized_profit = sum(p.unrealized_pnl for p in self.positions.values() 
                                           if p.status == PositionStatus.OPEN)
                
                total_profit = self.realized_profit + self.unrealized_profit
                
                # Check global targets
                if total_profit >= GLOBAL_TAKE_PROFIT:
                    self.logger.info(f"Global take profit hit: {total_profit:.2f}")
                    if self.on_global_target_hit:
                        self.on_global_target_hit("TAKE_PROFIT", total_profit)
                    await self.stop()
                    
                elif total_profit <= GLOBAL_STOP_LOSS:
                    self.logger.info(f"Global stop loss hit: {total_profit:.2f}")
                    if self.on_global_target_hit:
                        self.on_global_target_hit("STOP_LOSS", total_profit)
                    await self.stop()
                    
            except Exception as e:
                self.logger.error(f"Error in global monitor: {e}")
                
            await asyncio.sleep(CHECK_INTERVAL * 2)
            
    def get_status(self) -> Dict:
        """Get current status of the trading system."""
        open_positions = [p for p in self.positions.values() 
                         if p.status == PositionStatus.OPEN]
        
        return {
            "is_running": self.is_running,
            "realized_profit": self.realized_profit,
            "unrealized_profit": self.unrealized_profit,
            "total_profit": self.realized_profit + self.unrealized_profit,
            "available_capital": self.available_capital,
            "open_positions": len(open_positions),
            "total_positions": len(self.positions),
            "progress_to_target": (self.realized_profit / GLOBAL_TAKE_PROFIT) * 100
        }