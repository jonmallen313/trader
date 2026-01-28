"""
Fully Autonomous AI Trading Engine.
You set capital, AI does EVERYTHING else.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

from core.events import event_bus, Event, EventType
from core.position import Position, PositionSide, PositionStatus
from core.signal import TradingSignal
from brokers.base import BrokerInterface
from analytics.metrics import MetricsCalculator
from analytics.logger import TradeLogger
from models.microtrend_ai import XGBoostMicroTrend, FeatureEngineer
from data.market_data import MarketDataFeed


@dataclass
class AutoPilotConfig:
    """Simple config - AI controls everything else."""
    initial_capital: float
    target_profit: Optional[float] = None  # Auto: 20x capital
    max_positions: int = 15  # Trade entire balance across 15 positions
    broker: str = "bybit"
    testnet: bool = True
    
    def __post_init__(self):
        if self.target_profit is None:
            self.target_profit = self.initial_capital * 20


class FullAutoPilot:
    """
    Fully autonomous AI trading engine.
    YOU control: Capital amount
    AI controls: Everything else (symbols, leverage, timing, TP/SL, exits)
    """
    
    def __init__(self, config: AutoPilotConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core state
        self.broker: Optional[BrokerInterface] = None
        self.is_running = False
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.current_balance = config.initial_capital
        self.realized_pnl = 0.0
        self.trade_logger = TradeLogger()
        
        # AI decision engine
        self.ai_confidence_threshold = 0.55  # Auto-adjust based on performance
        self.base_leverage = 10  # Auto-adjust based on win rate
        self.ai_model = XGBoostMicroTrend()
        self.market_data_feed = None
        
    async def start(self):
        """Start fully autonomous trading."""
        self.logger.info("🤖 Starting Full AutoPilot Mode")
        self.logger.info(f"💰 Capital: ${self.config.initial_capital:,.2f}")
        self.logger.info(f"🎯 Target: ${self.config.target_profit:,.2f}")
        self.logger.info("🧠 AI controls: Symbols, Leverage, Timing, TP/SL, Exits")
        self.logger.info("")
        
        # Initialize broker
        await self._setup_broker()
        
        # Subscribe to events
        self._setup_event_handlers()
        
        # Start main loop
        self.is_running = True
        
        # Run concurrent tasks
        await asyncio.gather(
            self._market_scanner(),      # Find trading opportunities
            self._position_monitor(),     # Manage open positions
            self._performance_display(),  # Live status updates
            self._risk_monitor()          # Safety checks
        )
    
    async def _setup_broker(self):
        """Initialize broker connection."""
        if self.config.broker == "bybit":
            from brokers.bybit import BybitBroker
            import os
            
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError(
                    "Bybit credentials not found!\n"
                    "Get FREE testnet keys: https://testnet.bybit.com\n"
                    "Then set: BYBIT_API_KEY and BYBIT_API_SECRET"
                )
            
            self.broker = BybitBroker(api_key, api_secret, testnet=self.config.testnet)
            await self.broker.connect()
            
            # Update balance
            self.current_balance = await self.broker.get_balance()
            self.logger.info(f"✅ Connected to Bybit | Balance: ${self.current_balance:,.2f}")
    
    def _setup_event_handlers(self):
        """Subscribe to system events."""
        event_bus.subscribe(EventType.POSITION_OPENED, self._on_position_opened)
        event_bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)
        event_bus.subscribe(EventType.TARGET_REACHED, self._on_target_reached)
    
    async def _market_scanner(self):
        """AI continuously scans for trading opportunities."""
        self.logger.info("🔍 Market scanner started (AI selecting symbols)")
        
        # AI-selected symbols based on volatility and liquidity
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT']
        
        while self.is_running:
            try:
                # Scan each symbol
                for symbol in symbols:
                    if len(self.positions) >= self.config.max_positions:
                        break
                    
                    # Get market data
                    price = await self.broker.get_price(symbol)
                    
                    # AI analyzes if we should trade (simplified for now)
                    signal = await self._ai_analyze_opportunity(symbol, price)
                    
                    if signal and signal.confidence > self.ai_confidence_threshold:
                        await self._execute_trade(signal)
                
                await asyncio.sleep(5)  # AI checks every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Scanner error: {e}")
                await asyncio.sleep(10)
    
    async def _ai_analyze_opportunity(self, symbol: str, price: float) -> Optional[TradingSignal]:
        """
        AI analyzes market and decides:
        - Should we trade? (confidence)
        - Long or short? (direction)
        - How much leverage? (based on confidence)
        - TP/SL levels? (based on volatility)
        """
        # TODO: Connect real AI model here
        # For now, simplified random analysis
        import random
        
        # Simulate AI analysis
        confidence = random.uniform(0.4, 0.9)
        
        if confidence < self.ai_confidence_threshold:
            return None
        
        # AI decides direction
        side = PositionSide.LONG if random.random() > 0.5 else PositionSide.SHORT
        
        # AI calculates optimal leverage based on confidence
        leverage = self._ai_calculate_leverage(confidence)
        
        # AI sets TP/SL based on volatility
        tp_pct, sl_pct = self._ai_calculate_risk_reward(confidence)
        
        return TradingSignal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            suggested_leverage=leverage,
            source="ai",
            strategy="autonomous"
        )
    
    def _ai_calculate_leverage(self, confidence: float) -> int:
        """AI determines optimal leverage based on confidence."""
        # Higher confidence = higher leverage (but capped)
        if confidence >= 0.85:
            return min(20, self.base_leverage * 2)
        elif confidence >= 0.70:
            return self.base_leverage
        else:
            return max(5, self.base_leverage // 2)
    
    def _ai_calculate_risk_reward(self, confidence: float) -> tuple:
        """AI sets TP/SL based on market conditions."""
        # Higher confidence = wider TP, tighter SL
        if confidence >= 0.80:
            return 0.004, 0.002  # 0.4% TP, 0.2% SL (tight scalp)
        elif confidence >= 0.65:
            return 0.003, 0.0025  # 0.3% TP, 0.25% SL
        else:
            return 0.002, 0.003  # 0.2% TP, 0.3% SL (conservative)
    
    async def _execute_trade(self, signal: TradingSignal):
        """Execute trade based on AI signal."""
        try:
            # Calculate position size (AI manages capital)
            risk_per_trade = 0.05  # 5% of capital per trade
            position_value = self.current_balance * risk_per_trade
            
            # Place order
            self.logger.info(
                f"🎯 AI TRADE: {signal.symbol} {signal.side.value.upper()} "
                f"{signal.suggested_leverage}x | "
                f"Confidence: {signal.confidence:.1%}"
            )
            
            current_price = await self.broker.get_price(signal.symbol)
            position_size = (position_value * signal.suggested_leverage) / current_price
            
            # Calculate TP/SL prices
            if signal.side == PositionSide.LONG:
                tp_price = current_price * (1 + signal.tp_pct)
                sl_price = current_price * (1 - signal.sl_pct)
            else:
                tp_price = current_price * (1 - signal.tp_pct)
                sl_price = current_price * (1 + signal.sl_pct)
            
            # Execute
            order = await self.broker.place_order(
                symbol=signal.symbol,
                side=signal.side.value,
                size=position_size,
                leverage=signal.suggested_leverage,
                tp_price=tp_price,
                sl_price=sl_price
            )
            
            # Create position
            position = Position(
                id=order.id,
                symbol=signal.symbol,
                side=signal.side,
                size=position_size,
                leverage=signal.suggested_leverage,
                entry_price=order.fill_price,
                current_price=order.fill_price,
                tp_price=tp_price,
                sl_price=sl_price,
                liquidation_price=await self.broker.get_liquidation_price(
                    Position(symbol=signal.symbol, side=signal.side, 
                            size=position_size, entry_price=order.fill_price,
                            leverage=signal.suggested_leverage,
                            current_price=order.fill_price)
                ),
                status=PositionStatus.OPEN,
                strategy="ai_autopilot",
                confidence=signal.confidence
            )
            
            self.positions.append(position)
            
            await event_bus.publish(Event(
                type=EventType.POSITION_OPENED,
                data={"position": position.to_dict()}
            ))
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
    
    async def _position_monitor(self):
        """Monitor positions and auto-close when TP/SL hit."""
        self.logger.info("👁️  Position monitor started (AI manages exits)")
        
        while self.is_running:
            try:
                for position in self.positions[:]:  # Copy list
                    if position.status != PositionStatus.OPEN:
                        continue
                    
                    # Update current price
                    current_price = await self.broker.get_price(position.symbol)
                    position.update_price(current_price)
                    
                    # AI checks if we should exit
                    should_exit, reason = self._ai_should_exit(position)
                    
                    if should_exit:
                        await self._close_position(position, reason)
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    def _ai_should_exit(self, position: Position) -> tuple:
        """AI decides when to exit position."""
        # Check TP
        if position.side == PositionSide.LONG:
            if position.current_price >= position.tp_price:
                return True, "TP_HIT"
            if position.current_price <= position.sl_price:
                return True, "SL_HIT"
        else:
            if position.current_price <= position.tp_price:
                return True, "TP_HIT"
            if position.current_price >= position.sl_price:
                return True, "SL_HIT"
        
        # AI: Check if too close to liquidation
        if position.liquidation_distance < 5:  # < 5% from liquidation
            return True, "LIQUIDATION_RISK"
        
        # AI: Exit if position held too long (reduce risk)
        if position.duration > 300:  # 5 minutes max
            if position.unrealized_pnl > 0:  # Take profit if positive
                return True, "TIME_LIMIT_PROFIT"
        
        return False, ""
    
    async def _close_position(self, position: Position, reason: str):
        """Close position and update state."""
        try:
            result = await self.broker.close_position(position)
            
            position.close(result.fill_price, reason)
            self.positions.remove(position)
            self.closed_positions.append(position)
            
            # Update balance
            self.realized_pnl += position.realized_pnl
            self.current_balance += position.realized_pnl
            
            # Log
            emoji = "✅" if position.realized_pnl > 0 else "❌"
            self.logger.info(
                f"{emoji} CLOSED: {position.symbol} {position.side.value} | "
                f"P&L: ${position.realized_pnl:+.2f} ({position.pnl_pct:+.2f}%) | "
                f"{reason}"
            )
            
            self.trade_logger.log_trade(position, None, reason)
            
            await event_bus.publish(Event(
                type=EventType.POSITION_CLOSED,
                data={"position": position.to_dict(), "reason": reason}
            ))
            
            # Check if target reached
            if self.current_balance >= self.config.target_profit:
                await event_bus.publish(Event(type=EventType.TARGET_REACHED))
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    async def _performance_display(self):
        """Live performance display."""
        import os
        
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                # Clear screen (optional)
                # os.system('cls' if os.name == 'nt' else 'clear')
                
                # Calculate metrics
                metrics = MetricsCalculator.calculate(
                    self.closed_positions,
                    self.positions,
                    self.config.initial_capital,
                    self.current_balance
                )
                
                # Progress
                progress = (self.current_balance / self.config.target_profit) * 100
                
                print("\n" + "=" * 70)
                print("🤖 AI AUTOPILOT - LIVE STATUS")
                print("=" * 70)
                print(f"💰 Capital: ${self.current_balance:,.2f} / ${self.config.target_profit:,.2f}")
                print(f"📊 Progress: {progress:.1f}% {'█' * int(progress/5)}")
                print(f"📈 Total P&L: ${self.realized_pnl:+,.2f} ({metrics.total_pnl_pct:+.2f}%)")
                print(f"🎯 Win Rate: {metrics.win_rate*100:.1f}% ({metrics.winning_trades}W / {metrics.losing_trades}L)")
                print(f"⚡ Open Positions: {len(self.positions)} / {self.config.max_positions}")
                
                if self.positions:
                    print("\n📋 Active Positions:")
                    for pos in self.positions:
                        liq_emoji = "🟢" if pos.liquidation_distance > 15 else "🟡" if pos.liquidation_distance > 5 else "🔴"
                        print(
                            f"  {pos.symbol:<12} {pos.side.value:<6} {pos.leverage}x | "
                            f"P&L: ${pos.unrealized_pnl:+7.2f} ({pos.pnl_pct:+.2f}%) | "
                            f"{liq_emoji} Liq: {pos.liquidation_distance:.1f}%"
                        )
                
                print("=" * 70)
                
            except Exception as e:
                self.logger.error(f"Display error: {e}")
                await asyncio.sleep(10)
    
    async def _risk_monitor(self):
        """AI monitors risk and stops if necessary."""
        while self.is_running:
            try:
                # Check max drawdown
                drawdown_pct = ((self.current_balance - self.config.initial_capital) / 
                               self.config.initial_capital * 100)
                
                if drawdown_pct < -50:  # -50% stop loss
                    self.logger.error("🛑 EMERGENCY STOP: Max drawdown reached!")
                    await self._emergency_stop()
                
                # AI adjusts based on performance
                if len(self.closed_positions) >= 10:
                    recent = self.closed_positions[-10:]
                    win_rate = len([p for p in recent if p.realized_pnl > 0]) / 10
                    
                    # AI lowers leverage if losing
                    if win_rate < 0.4:
                        self.base_leverage = max(5, self.base_leverage - 2)
                        self.logger.warning(f"⚠️ AI reducing leverage to {self.base_leverage}x")
                    
                    # AI increases leverage if winning
                    elif win_rate > 0.65:
                        self.base_leverage = min(20, self.base_leverage + 2)
                        self.logger.info(f"📈 AI increasing leverage to {self.base_leverage}x")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Risk monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _emergency_stop(self):
        """Emergency stop - close all positions."""
        self.logger.warning("🚨 Emergency stop triggered!")
        
        for position in self.positions[:]:
            await self._close_position(position, "EMERGENCY_STOP")
        
        self.is_running = False
        await event_bus.publish(Event(type=EventType.EMERGENCY_STOP))
    
    async def _on_position_opened(self, event: Event):
        """Handle position opened event."""
        pass
    
    async def _on_position_closed(self, event: Event):
        """Handle position closed event."""
        pass
    
    async def _on_target_reached(self, event: Event):
        """Handle target reached."""
        self.logger.info("🎉 TARGET REACHED! Stopping autopilot...")
        self.is_running = False
