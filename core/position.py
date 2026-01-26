"""
Position models and lifecycle management.
Clean, focused, testable.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class PositionSide(Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position lifecycle status."""
    PENDING = "pending"      # Order placed, not filled
    OPEN = "open"            # Active position
    CLOSED = "closed"        # Closed normally
    LIQUIDATED = "liquidated"  # Force closed by exchange
    CANCELLED = "cancelled"  # Order cancelled before fill


@dataclass
class Position:
    """
    Represents a trading position with leverage support.
    Immutable after creation (use update methods).
    """
    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    
    # Position details
    side: PositionSide = PositionSide.LONG
    size: float = 0.0  # Position size in base currency
    leverage: int = 1  # Leverage multiplier
    
    # Pricing
    entry_price: float = 0.0
    current_price: float = 0.0
    tp_price: Optional[float] = None  # Take profit
    sl_price: Optional[float] = None  # Stop loss
    liquidation_price: Optional[float] = None  # Liquidation price
    
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0
    
    # Timestamps
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    
    # Status
    status: PositionStatus = PositionStatus.PENDING
    
    # Metadata
    strategy: str = "unknown"
    confidence: float = 0.0
    notes: str = ""
    
    @property
    def notional_value(self) -> float:
        """Total notional value (size * price * leverage)."""
        return self.size * self.current_price * self.leverage
    
    @property
    def margin_required(self) -> float:
        """Margin required for position."""
        return self.size * self.entry_price / self.leverage if self.leverage > 0 else 0
    
    @property
    def duration(self) -> float:
        """Position duration in seconds."""
        if self.closed_at:
            return (self.closed_at - self.opened_at).total_seconds()
        return (datetime.now() - self.opened_at).total_seconds()
    
    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of entry."""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.size * self.entry_price)) * 100
    
    @property
    def liquidation_distance(self) -> float:
        """Distance to liquidation as percentage."""
        if not self.liquidation_price or self.current_price == 0:
            return 100.0
        return abs((self.liquidation_price - self.current_price) / self.current_price) * 100
    
    def update_price(self, new_price: float):
        """Update current price and recalculate unrealized P&L."""
        self.current_price = new_price
        self.unrealized_pnl = self.calculate_unrealized_pnl(new_price)
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.size * self.leverage
        else:
            return (self.entry_price - current_price) * self.size * self.leverage
    
    def close(self, exit_price: float, reason: str = "manual"):
        """Close the position."""
        self.status = PositionStatus.CLOSED if reason != "liquidation" else PositionStatus.LIQUIDATED
        self.closed_at = datetime.now()
        self.current_price = exit_price
        self.realized_pnl = self.calculate_unrealized_pnl(exit_price) - self.fees
        self.notes = f"{self.notes} | Closed: {reason}".strip(" |")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/API."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "size": self.size,
            "leverage": self.leverage,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "liquidation_price": self.liquidation_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "pnl_pct": self.pnl_pct,
            "liquidation_distance": self.liquidation_distance,
            "margin_required": self.margin_required,
            "status": self.status.value,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "duration": self.duration,
            "strategy": self.strategy,
            "confidence": self.confidence,
        }
    
    def __repr__(self):
        return (f"Position({self.symbol} {self.side.value} {self.size:.4f} "
                f"@{self.entry_price:.2f} {self.leverage}x | "
                f"PnL: {self.unrealized_pnl:+.2f} ({self.pnl_pct:+.2f}%))")
