"""
Performance metrics calculator.
Clean, testable, comprehensive analytics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from core.position import Position, PositionStatus


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Capital
    initial_capital: float
    current_capital: float
    total_pnl: float
    total_pnl_pct: float
    
    # Trading stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    gross_profit: float
    gross_loss: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Position metrics
    average_trade_duration: float  # seconds
    average_leverage: float
    max_leverage_used: float
    
    # Liquidation safety
    liquidations: int
    close_calls: int  # < 10% from liquidation
    average_liquidation_distance: float
    
    # Time-based
    trading_days: int
    daily_pnl: float
    best_day: float
    worst_day: float
    
    # Current state
    open_positions: int
    pending_orders: int
    available_margin: float
    used_margin: float
    margin_usage_pct: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "total_pnl": self.total_pnl,
                "total_pnl_pct": self.total_pnl_pct,
            },
            "trading": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
            },
            "pnl": {
                "gross_profit": self.gross_profit,
                "gross_loss": self.gross_loss,
                "profit_factor": self.profit_factor,
                "average_win": self.average_win,
                "average_loss": self.average_loss,
                "largest_win": self.largest_win,
                "largest_loss": self.largest_loss,
            },
            "risk": {
                "max_drawdown": self.max_drawdown,
                "max_drawdown_pct": self.max_drawdown_pct,
                "current_drawdown": self.current_drawdown,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
            },
            "positions": {
                "average_duration": self.average_trade_duration,
                "average_leverage": self.average_leverage,
                "max_leverage": self.max_leverage_used,
            },
            "safety": {
                "liquidations": self.liquidations,
                "close_calls": self.close_calls,
                "avg_liquidation_distance": self.average_liquidation_distance,
            },
            "current": {
                "open_positions": self.open_positions,
                "pending_orders": self.pending_orders,
                "available_margin": self.available_margin,
                "used_margin": self.used_margin,
                "margin_usage_pct": self.margin_usage_pct,
            }
        }


class MetricsCalculator:
    """
    Calculate comprehensive trading metrics.
    Stateless - pass in data, get metrics out.
    """
    
    @staticmethod
    def calculate(
        closed_positions: List[Position],
        open_positions: List[Position],
        initial_capital: float,
        current_balance: float,
        risk_free_rate: float = 0.02
    ) -> PerformanceMetrics:
        """Calculate all metrics from position history."""
        
        # Filter closed positions
        closed = [p for p in closed_positions if p.status == PositionStatus.CLOSED]
        
        if not closed:
            # No trades yet
            return MetricsCalculator._empty_metrics(initial_capital, current_balance)
        
        # Basic stats
        total_trades = len(closed)
        winners = [p for p in closed if p.realized_pnl > 0]
        losers = [p for p in closed if p.realized_pnl < 0]
        
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        gross_profit = sum(p.realized_pnl for p in winners)
        gross_loss = abs(sum(p.realized_pnl for p in losers))
        total_pnl = sum(p.realized_pnl for p in closed)
        total_pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        average_win = gross_profit / winning_trades if winning_trades > 0 else 0
        average_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        
        largest_win = max([p.realized_pnl for p in winners]) if winners else 0
        largest_loss = min([p.realized_pnl for p in losers]) if losers else 0
        
        # Drawdown
        equity_curve = MetricsCalculator._calculate_equity_curve(closed, initial_capital)
        max_drawdown, max_drawdown_pct = MetricsCalculator._calculate_max_drawdown(equity_curve)
        current_drawdown = equity_curve[-1] - max(equity_curve) if equity_curve else 0
        
        # Risk-adjusted returns
        returns = [p.realized_pnl / initial_capital for p in closed]
        sharpe = MetricsCalculator._calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = MetricsCalculator._calculate_sortino_ratio(returns, risk_free_rate)
        
        # Position metrics
        avg_duration = np.mean([p.duration for p in closed]) if closed else 0
        avg_leverage = np.mean([p.leverage for p in closed]) if closed else 1
        max_leverage = max([p.leverage for p in closed]) if closed else 1
        
        # Safety metrics
        liquidations = len([p for p in closed_positions if p.status == PositionStatus.LIQUIDATED])
        
        # Calculate close calls (positions that got within 10% of liquidation)
        close_calls = 0
        liq_distances = []
        for p in closed:
            if p.liquidation_price:
                min_dist = p.liquidation_distance
                liq_distances.append(min_dist)
                if min_dist < 10:  # Within 10%
                    close_calls += 1
        
        avg_liq_distance = np.mean(liq_distances) if liq_distances else 100
        
        # Time-based
        if closed:
            first_trade = min(p.opened_at for p in closed)
            last_trade = max(p.closed_at for p in closed if p.closed_at)
            trading_days = max(1, (last_trade - first_trade).days)
        else:
            trading_days = 1
        
        daily_pnl = total_pnl / trading_days if trading_days > 0 else 0
        
        # Best/worst days (simplified - would need daily grouping)
        best_day = largest_win
        worst_day = largest_loss
        
        # Current state
        used_margin = sum(p.margin_required for p in open_positions)
        margin_usage = (used_margin / current_balance * 100) if current_balance > 0 else 0
        
        return PerformanceMetrics(
            initial_capital=initial_capital,
            current_capital=current_balance,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            average_trade_duration=avg_duration,
            average_leverage=avg_leverage,
            max_leverage_used=max_leverage,
            liquidations=liquidations,
            close_calls=close_calls,
            average_liquidation_distance=avg_liq_distance,
            trading_days=trading_days,
            daily_pnl=daily_pnl,
            best_day=best_day,
            worst_day=worst_day,
            open_positions=len(open_positions),
            pending_orders=0,
            available_margin=current_balance - used_margin,
            used_margin=used_margin,
            margin_usage_pct=margin_usage
        )
    
    @staticmethod
    def _empty_metrics(initial_capital: float, current_balance: float) -> PerformanceMetrics:
        """Return empty metrics when no trades."""
        return PerformanceMetrics(
            initial_capital=initial_capital,
            current_capital=current_balance,
            total_pnl=0, total_pnl_pct=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            gross_profit=0, gross_loss=0, profit_factor=0,
            average_win=0, average_loss=0, largest_win=0, largest_loss=0,
            max_drawdown=0, max_drawdown_pct=0, current_drawdown=0,
            sharpe_ratio=0, sortino_ratio=0,
            average_trade_duration=0, average_leverage=1, max_leverage_used=1,
            liquidations=0, close_calls=0, average_liquidation_distance=100,
            trading_days=1, daily_pnl=0, best_day=0, worst_day=0,
            open_positions=0, pending_orders=0,
            available_margin=current_balance, used_margin=0, margin_usage_pct=0
        )
    
    @staticmethod
    def _calculate_equity_curve(positions: List[Position], initial: float) -> List[float]:
        """Calculate equity curve from position history."""
        equity = [initial]
        for pos in sorted(positions, key=lambda p: p.closed_at or datetime.now()):
            equity.append(equity[-1] + pos.realized_pnl)
        return equity
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: List[float]) -> tuple:
        """Calculate maximum drawdown (absolute and percentage)."""
        if not equity_curve:
            return 0, 0
        
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = equity - peak
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            
            if dd < max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        return max_dd, max_dd_pct
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free
        
        if np.std(excess_returns) == 0:
            return 0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    @staticmethod
    def _calculate_sortino_ratio(returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)
        
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0
        
        downside_dev = np.std(downside_returns)
        if downside_dev == 0:
            return 0
        
        sortino = np.mean(excess_returns) / downside_dev * np.sqrt(252)
        return sortino
