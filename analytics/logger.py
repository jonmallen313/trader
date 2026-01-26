"""
Structured trade logging for analytics.
Every action is tracked and queryable.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from core.position import Position
from core.signal import TradingSignal
from core.events import Event


@dataclass
class TradeLog:
    """Complete record of a trade."""
    # Timestamps
    timestamp: str
    opened_at: str
    closed_at: Optional[str]
    
    # Trade details
    id: str
    symbol: str
    side: str
    size: float
    leverage: int
    
    # Pricing
    entry_price: float
    exit_price: Optional[float]
    tp_price: Optional[float]
    sl_price: Optional[float]
    liquidation_price: Optional[float]
    
    # P&L
    realized_pnl: float
    pnl_pct: float
    fees: float
    
    # Context
    strategy: str
    confidence: float
    entry_reason: str
    exit_reason: str
    
    # Performance
    duration: float  # seconds
    win: bool
    
    # Market context
    market_conditions: Dict
    features: Dict


class TradeLogger:
    """
    Structured logging for all trading activity.
    Writes JSON logs for easy analysis.
    """
    
    def __init__(self, log_dir: str = "logs/trades"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Current session log file
        self.session_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.session_file.touch()
        
    def log_trade(self, position: Position, signal: TradingSignal, exit_reason: str = ""):
        """Log a completed trade."""
        trade_log = TradeLog(
            timestamp=datetime.now().isoformat(),
            opened_at=position.opened_at.isoformat(),
            closed_at=position.closed_at.isoformat() if position.closed_at else None,
            id=position.id,
            symbol=position.symbol,
            side=position.side.value,
            size=position.size,
            leverage=position.leverage,
            entry_price=position.entry_price,
            exit_price=position.close_price,
            tp_price=position.tp_price,
            sl_price=position.sl_price,
            liquidation_price=position.liquidation_price,
            realized_pnl=position.realized_pnl,
            pnl_pct=position.pnl_pct,
            fees=position.fees,
            strategy=position.strategy,
            confidence=position.confidence,
            entry_reason=signal.notes if signal else "manual",
            exit_reason=exit_reason,
            duration=position.duration,
            win=position.realized_pnl > 0,
            market_conditions={},
            features=signal.features if signal else {}
        )
        
        # Write to JSON lines file
        with open(self.session_file, 'a') as f:
            f.write(json.dumps(asdict(trade_log)) + '\n')
        
        # Log summary
        result = "WIN" if trade_log.win else "LOSS"
        self.logger.info(
            f"[{result}] {trade_log.symbol} {trade_log.side} | "
            f"PnL: ${trade_log.realized_pnl:+.2f} ({trade_log.pnl_pct:+.2f}%) | "
            f"Duration: {trade_log.duration:.0f}s | "
            f"{exit_reason}"
        )
    
    def log_event(self, event: Event):
        """Log a system event."""
        event_data = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.type.value,
            "source": event.source,
            "data": event.data
        }
        
        with open(self.session_file, 'a') as f:
            f.write(json.dumps(event_data) + '\n')
    
    def query_trades(self, symbol: Optional[str] = None, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> List[TradeLog]:
        """Query trade history with filters."""
        trades = []
        
        with open(self.session_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Skip non-trade events
                    if 'side' not in data:
                        continue
                    
                    # Apply filters
                    if symbol and data.get('symbol') != symbol:
                        continue
                    
                    # Convert back to TradeLog
                    trade = TradeLog(**data)
                    trades.append(trade)
                except:
                    continue
        
        return trades
    
    def get_summary(self) -> Dict:
        """Get summary of current session."""
        trades = self.query_trades()
        
        if not trades:
            return {"total_trades": 0}
        
        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        
        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_pnl": sum(t.realized_pnl for t in trades),
            "average_pnl": sum(t.realized_pnl for t in trades) / len(trades) if trades else 0,
            "average_duration": sum(t.duration for t in trades) / len(trades) if trades else 0,
        }
