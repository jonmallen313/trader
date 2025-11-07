"""
Backtesting and testing framework for the AI trading system.
Provides paper trading mode, historical backtesting, and validation tools.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from src.autopilot import AutoPilotController, Position, PositionSide, TradingSignal
from src.brokers import MockBroker
from data.market_data import MarketDataPoint, DataBuffer
from models.microtrend_ai import EnsemblePredictor, XGBoostMicroTrend
from config.settings import *


class BacktestResults:
    """Container for backtest results and metrics."""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.daily_pnl: List[float] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_consecutive_losses = 0
        
    def calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return
            
        # Basic metrics
        self.total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # P&L metrics
        self.total_pnl = sum(t['pnl'] for t in self.trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1
        
        self.profit_factor = (avg_win * self.winning_trades) / (avg_loss * self.losing_trades) if avg_loss > 0 and self.losing_trades > 0 else 0
        
        # Drawdown calculation
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0
            for equity in self.equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            self.max_drawdown = max_dd
        
        # Sharpe ratio (simplified)
        if self.daily_pnl and len(self.daily_pnl) > 1:
            daily_returns = np.array(self.daily_pnl)
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            self.sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive = 0
        for trade in self.trades:
            if trade['pnl'] < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        self.max_consecutive_losses = max_consecutive
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'max_consecutive_losses': self.max_consecutive_losses,
            'trades': self.trades[-10:],  # Last 10 trades for preview
        }


class HistoricalDataGenerator:
    """Generate realistic historical market data for backtesting."""
    
    @staticmethod
    def generate_ohlc_data(
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int = 1,
        base_price: float = 100.0,
        volatility: float = 0.02
    ) -> pd.DataFrame:
        """Generate OHLC data with realistic price movements."""
        
        # Calculate number of periods
        total_minutes = int((end_date - start_date).total_seconds() / 60)
        periods = total_minutes // interval_minutes
        
        # Generate timestamps
        timestamps = pd.date_range(start_date, periods=periods, freq=f'{interval_minutes}min')
        
        # Generate price data using geometric brownian motion
        dt = interval_minutes / (24 * 60)  # Time step in days
        drift = 0.0001  # Small positive drift
        
        # Random walk
        random_changes = np.random.normal(drift * dt, volatility * np.sqrt(dt), periods)
        log_prices = np.log(base_price) + np.cumsum(random_changes)
        close_prices = np.exp(log_prices)
        
        # Generate OHLC from close prices
        ohlc_data = []
        for i, close in enumerate(close_prices):
            if i == 0:
                open_price = close
            else:
                open_price = close_prices[i-1]
            
            # Random high/low around open and close
            high_factor = np.random.uniform(1.0, 1.001 + volatility/2)
            low_factor = np.random.uniform(1.0 - volatility/2, 1.0)
            
            high = max(open_price, close) * high_factor
            low = min(open_price, close) * low_factor
            
            # Volume (random but realistic)
            volume = np.random.lognormal(10, 1.5)
            
            ohlc_data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(ohlc_data)
    
    @staticmethod
    def convert_to_market_data_points(df: pd.DataFrame, symbol: str) -> List[MarketDataPoint]:
        """Convert OHLC dataframe to MarketDataPoint objects."""
        data_points = []
        
        for _, row in df.iterrows():
            # Create data point for each tick within the bar
            for tick_price in [row['open'], row['high'], row['low'], row['close']]:
                data_point = MarketDataPoint(
                    symbol=symbol,
                    price=float(tick_price),
                    volume=float(row['volume']) / 4,  # Distribute volume across ticks
                    bid=float(tick_price) * 0.9999,
                    ask=float(tick_price) * 1.0001,
                    timestamp=row['timestamp']
                )
                data_points.append(data_point)
        
        return data_points


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.broker = MockBroker(initial_capital)
        self.logger = logging.getLogger(__name__)
        
    async def run_backtest(
        self,
        strategy_predictor,
        historical_data: List[MarketDataPoint],
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResults:
        """Run a complete backtest."""
        
        results = BacktestResults()
        
        # Connect broker
        await self.broker.connect()
        
        # Create data buffer for feature calculation
        data_buffer = DataBuffer(symbol)
        
        # Initialize autopilot controller (but don't start the async loops)
        class MockDataFeed:
            async def get_latest_data(self):
                return None
                
        autopilot = AutoPilotController(
            exchange_client=self.broker,
            data_feed=MockDataFeed(),
            model_predictor=strategy_predictor,
            initial_capital=self.initial_capital
        )
        
        self.logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Process each data point
        current_equity = self.initial_capital
        daily_pnl = 0
        last_date = start_date.date()
        
        for i, data_point in enumerate(historical_data):
            # Add to buffer and get features
            data_buffer.add(data_point)
            features = data_buffer.get_features()
            
            if features is None:
                continue
                
            # Get prediction from strategy
            prediction = strategy_predictor.predict(features)
            
            if prediction and prediction.confidence > PREDICTION_THRESHOLD:
                # Convert prediction to signal
                signal = TradingSignal(
                    symbol=symbol,
                    side=prediction.side,
                    confidence=prediction.confidence,
                    tp_pct=prediction.tp_pct,
                    sl_pct=prediction.sl_pct,
                    timestamp=data_point.timestamp
                )
                
                # Simulate trade execution
                await self._execute_backtest_trade(autopilot, signal, results, data_point.price)
            
            # Update daily tracking
            current_date = data_point.timestamp.date()
            if current_date != last_date:
                # New day - record daily P&L
                new_equity = await self.broker.get_balance()
                daily_pnl = new_equity - current_equity
                results.daily_pnl.append(daily_pnl)
                results.equity_curve.append(new_equity)
                results.timestamps.append(data_point.timestamp)
                
                current_equity = new_equity
                last_date = current_date
            
            # Progress logging
            if i % 1000 == 0:
                progress = i / len(historical_data) * 100
                self.logger.info(f"Backtest progress: {progress:.1f}%")
        
        # Final calculations
        results.calculate_metrics()
        
        self.logger.info(f"Backtest completed. Total trades: {results.total_trades}, P&L: ${results.total_pnl:.2f}")
        
        return results
    
    async def _execute_backtest_trade(
        self,
        autopilot: AutoPilotController,
        signal: TradingSignal,
        results: BacktestResults,
        current_price: float
    ):
        """Execute a single trade in backtest mode."""
        try:
            # Check if we can open position
            if not await autopilot._can_open_position(signal):
                return
            
            # Calculate position size
            available_capital = await self.broker.get_balance()
            position_value = available_capital * POSITION_SIZE_PCT
            position_size = position_value / current_price
            
            # Calculate TP and SL prices
            if signal.side == PositionSide.LONG:
                tp_price = current_price * (1 + signal.tp_pct)
                sl_price = current_price * (1 - signal.sl_pct)
                side_str = "buy"
            else:
                tp_price = current_price * (1 - signal.tp_pct)
                sl_price = current_price * (1 + signal.sl_pct)
                side_str = "sell"
            
            # Place entry order
            order_result = await self.broker.place_order(
                symbol=signal.symbol,
                side=side_str,
                size=position_value,  # Use dollar amount
                order_type="market"
            )
            
            # Simulate immediate TP/SL execution (simplified)
            # In reality, you'd track the position until TP/SL is hit
            exit_price = tp_price  # Assume TP hit (optimistic for testing)
            
            # Calculate P&L
            if signal.side == PositionSide.LONG:
                pnl = (exit_price - current_price) * position_size
            else:
                pnl = (current_price - exit_price) * position_size
            
            # Record trade
            trade_record = {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'side': signal.side.value,
                'entry_price': current_price,
                'exit_price': exit_price,
                'size': position_size,
                'pnl': pnl,
                'confidence': signal.confidence,
                'tp_pct': signal.tp_pct,
                'sl_pct': signal.sl_pct
            }
            
            results.trades.append(trade_record)
            
        except Exception as e:
            self.logger.error(f"Error executing backtest trade: {e}")


class PaperTradingMode:
    """Paper trading mode for live testing without real money."""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.broker = MockBroker(initial_capital)
        self.is_running = False
        self.trades_log = []
        self.logger = logging.getLogger(__name__)
    
    async def start(self, autopilot: AutoPilotController):
        """Start paper trading mode."""
        self.is_running = True
        self.logger.info("Paper trading mode started")
        
        # Replace the broker in autopilot
        autopilot.exchange = self.broker
        await self.broker.connect()
        
        # Set callbacks for trade logging
        autopilot.on_position_opened = self._log_position_opened
        autopilot.on_position_closed = self._log_position_closed
        
        # Start autopilot
        await autopilot.start()
    
    async def stop(self):
        """Stop paper trading mode."""
        self.is_running = False
        await self.broker.disconnect()
        self.logger.info("Paper trading mode stopped")
    
    def _log_position_opened(self, position: Position):
        """Log when a position is opened."""
        self.logger.info(f"PAPER TRADE OPENED: {position.symbol} {position.side.value} @ ${position.entry_price:.4f}")
    
    def _log_position_closed(self, position: Position):
        """Log when a position is closed."""
        self.logger.info(f"PAPER TRADE CLOSED: {position.symbol} P&L: ${position.realized_pnl:.2f}")
        
        # Add to trades log
        self.trades_log.append({
            'timestamp': position.closed_at.isoformat() if position.closed_at else datetime.now().isoformat(),
            'symbol': position.symbol,
            'side': position.side.value,
            'entry_price': position.entry_price,
            'exit_price': position.close_price,
            'size': position.size,
            'pnl': position.realized_pnl,
            'duration': (position.closed_at - position.opened_at).total_seconds() if position.closed_at else 0
        })
    
    def get_performance_report(self) -> Dict:
        """Get performance report for paper trading."""
        if not self.trades_log:
            return {"message": "No trades executed yet"}
        
        total_trades = len(self.trades_log)
        winning_trades = sum(1 for t in self.trades_log if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in self.trades_log)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades * 100,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades,
            'current_balance': self.broker.balance,
            'return_pct': (self.broker.balance - self.initial_capital) / self.initial_capital * 100,
            'recent_trades': self.trades_log[-5:]  # Last 5 trades
        }


class TestRunner:
    """Test runner for various trading system tests."""
    
    @staticmethod
    async def run_quick_backtest(symbol: str = "BTCUSDT", days: int = 7) -> Dict:
        """Run a quick backtest for testing."""
        # Generate test data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = HistoricalDataGenerator.generate_ohlc_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval_minutes=1,
            base_price=50000.0 if symbol.startswith('BTC') else 100.0,
            volatility=0.02
        )
        
        historical_data = HistoricalDataGenerator.convert_to_market_data_points(df, symbol)
        
        # Create simple predictor
        predictor = XGBoostMicroTrend()
        
        # Mock training with some data
        training_data = []
        for i in range(min(200, len(historical_data))):
            data_point = historical_data[i]
            buffer = DataBuffer(symbol)
            buffer.add(data_point)
            features = buffer.get_features()
            if features:
                training_data.append(features)
        
        if training_data:
            predictor.train(training_data)
        
        # Run backtest
        engine = BacktestEngine()
        results = await engine.run_backtest(
            strategy_predictor=predictor,
            historical_data=historical_data,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        return results.to_dict()
    
    @staticmethod
    async def test_paper_trading(duration_minutes: int = 5) -> Dict:
        """Test paper trading mode for a short duration."""
        paper_trader = PaperTradingMode()
        
        # Create mock autopilot
        from data.market_data import DataFeedManager
        from models.microtrend_ai import EnsemblePredictor
        
        data_feed = DataFeedManager()
        predictor = EnsemblePredictor()
        
        autopilot = AutoPilotController(
            exchange_client=paper_trader.broker,
            data_feed=data_feed,
            model_predictor=predictor
        )
        
        # Start paper trading
        start_task = asyncio.create_task(paper_trader.start(autopilot))
        
        # Let it run for specified duration
        await asyncio.sleep(duration_minutes * 60)
        
        # Stop and get results
        await paper_trader.stop()
        return paper_trader.get_performance_report()


# Example usage and testing
if __name__ == "__main__":
    async def main():
        print("AI Trading System - Testing Framework")
        print("=====================================")
        
        # Run quick backtest
        print("\n1. Running quick backtest...")
        backtest_results = await TestRunner.run_quick_backtest("BTCUSDT", days=1)
        print(f"Backtest Results: {json.dumps(backtest_results, indent=2)}")
        
        # Test paper trading
        print("\n2. Testing paper trading mode...")
        paper_results = await TestRunner.test_paper_trading(duration_minutes=1)
        print(f"Paper Trading Results: {json.dumps(paper_results, indent=2)}")
        
    asyncio.run(main())