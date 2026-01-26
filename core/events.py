"""
Event bus for system-wide communication.
Decouples components via publish-subscribe pattern.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import logging
import asyncio


class EventType(Enum):
    """System event types."""
    # Market events
    MARKET_DATA_RECEIVED = "market_data_received"
    PRICE_UPDATE = "price_update"
    
    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    
    # Order events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_LIQUIDATED = "position_liquidated"
    POSITION_TP_HIT = "position_tp_hit"
    POSITION_SL_HIT = "position_sl_hit"
    
    # Risk events
    RISK_WARNING = "risk_warning"
    LIQUIDATION_WARNING = "liquidation_warning"
    DRAWDOWN_LIMIT_HIT = "drawdown_limit_hit"
    
    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    TARGET_REACHED = "target_reached"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class Event:
    """Base event class."""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    
    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.type.value}: {self.data}"


class EventBus:
    """
    Event bus for publish-subscribe messaging.
    Thread-safe, async-capable, with filtering support.
    """
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        self.logger.debug(f"Subscribed {handler.__name__} to {event_type.value}")
        
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from an event type."""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
            
    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        async with self._lock:
            # Store in history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
        
        # Notify subscribers
        if event.type in self.subscribers:
            for handler in self.subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler {handler.__name__}: {e}")
                    
    def get_history(self, event_type: Optional[EventType] = None, 
                    limit: int = 100) -> List[Event]:
        """Get event history, optionally filtered by type."""
        if event_type:
            filtered = [e for e in self.event_history if e.type == event_type]
        else:
            filtered = self.event_history
        return filtered[-limit:]
    
    def clear_history(self):
        """Clear event history."""
        self.event_history.clear()


# Global event bus instance
event_bus = EventBus()
