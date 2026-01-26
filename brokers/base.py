"""
Abstract broker interface.
All brokers must implement this contract.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from core.position import Position, PositionSide


class OrderResult:
    """Standardized order result."""
    def __init__(self, order_id: str, symbol: str, side: str, size: float,
                 fill_price: float, status: str, leverage: int = 1, fees: float = 0.0):
        self.id = order_id
        self.symbol = symbol
        self.side = side
        self.size = size
        self.fill_price = fill_price
        self.status = status
        self.leverage = leverage
        self.fees = fees
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "fill_price": self.fill_price,
            "status": self.status,
            "leverage": self.leverage,
            "fees": self.fees,
        }


class BrokerInterface(ABC):
    """
    Abstract base class for all broker integrations.
    Provides standardized interface for trading operations.
    """
    
    def __init__(self, name: str, paper_mode: bool = True):
        self.name = name
        self.paper_mode = paper_mode
        self.is_connected = False
        self.supports_leverage = False
        self.max_leverage = 1
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker. Returns True if successful."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> float:
        """Get available balance in USD."""
        pass
    
    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        pass
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        price: Optional[float] = None,
        leverage: int = 1,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None
    ) -> OrderResult:
        """Place an order. Returns OrderResult."""
        pass
    
    @abstractmethod
    async def close_position(self, position: Position) -> OrderResult:
        """Close an existing position."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        pass
    
    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol. Returns True if successful."""
        pass
    
    @abstractmethod
    async def get_liquidation_price(self, position: Position) -> Optional[float]:
        """Calculate liquidation price for position."""
        pass
    
    def __repr__(self):
        mode = "PAPER" if self.paper_mode else "LIVE"
        status = "CONNECTED" if self.is_connected else "DISCONNECTED"
        return f"{self.name} [{mode}] ({status})"
