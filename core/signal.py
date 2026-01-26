"""
Trading signal models.
Clean separation from execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from core.position import PositionSide


class SignalStrength(Enum):
    """Signal confidence levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TradingSignal:
    """
    Represents a trading signal from any source.
    No execution logic - pure data.
    """
    # What to trade
    symbol: str
    side: PositionSide
    
    # Confidence
    confidence: float  # 0.0 to 1.0
    strength: SignalStrength = SignalStrength.MODERATE
    
    # Risk parameters (suggestions, not mandates)
    tp_pct: float = 0.002  # 0.2%
    sl_pct: float = 0.003  # 0.3%
    suggested_leverage: int = 1
    
    # Source information
    source: str = "unknown"  # "ai", "tradingview", "manual", etc.
    strategy: str = "unknown"
    timeframe: str = "1m"
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    features: Dict = field(default_factory=dict)
    notes: str = ""
    
    @property
    def strength_from_confidence(self) -> SignalStrength:
        """Derive strength from confidence score."""
        if self.confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif self.confidence >= 0.65:
            return SignalStrength.STRONG
        elif self.confidence >= 0.55:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def is_valid(self) -> bool:
        """Check if signal is valid."""
        return (
            self.symbol and
            0 <= self.confidence <= 1.0 and
            self.tp_pct > 0 and
            self.sl_pct > 0 and
            self.suggested_leverage >= 1
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "confidence": self.confidence,
            "strength": self.strength.value,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "suggested_leverage": self.suggested_leverage,
            "source": self.source,
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
        }
    
    def __repr__(self):
        return (f"Signal({self.symbol} {self.side.value} | "
                f"Confidence: {self.confidence:.1%} | "
                f"TP: {self.tp_pct:.2%} SL: {self.sl_pct:.2%} | "
                f"{self.source})")
