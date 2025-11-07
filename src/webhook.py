"""
TradingView webhook integration for receiving Pine Script signals.
Provides a secure FastAPI endpoint for external trading signals.
"""

import asyncio
import hmac
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import uvicorn

from src.autopilot import AutoPilotController, TradingSignal, PositionSide
from config.settings import *


# Pydantic models for request validation
class TradingViewSignal(BaseModel):
    """TradingView webhook signal model."""
    symbol: str
    action: str  # "BUY" or "SELL"
    price: Optional[float] = None
    timestamp: Optional[str] = None
    tp_pct: Optional[float] = DEFAULT_TP_PCT
    sl_pct: Optional[float] = DEFAULT_SL_PCT
    confidence: Optional[float] = 1.0
    strategy: Optional[str] = "TradingView"
    timeframe: Optional[str] = "1m"
    
    @validator('action')
    def validate_action(cls, v):
        if v.upper() not in ['BUY', 'SELL', 'LONG', 'SHORT']:
            raise ValueError('action must be BUY, SELL, LONG, or SHORT')
        return v.upper()
        
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 3:
            raise ValueError('symbol must be valid')
        return v.upper()


class SystemStatus(BaseModel):
    """System status response model."""
    status: str
    uptime: float
    total_positions: int
    open_positions: int
    realized_pnl: float
    unrealized_pnl: float
    progress_to_target: float


class WebhookServer:
    """FastAPI webhook server for receiving trading signals."""
    
    def __init__(self, autopilot: AutoPilotController, port: int = WEBHOOK_PORT):
        self.autopilot = autopilot
        self.port = port
        self.app = FastAPI(title="AI Trading Webhook", version="1.0.0")
        self.security = HTTPBearer()
        self.start_time = datetime.now()
        self.signal_history = []
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/webhook/tradingview")
        async def receive_tradingview_signal(
            signal: TradingViewSignal,
            request: Request,
            background_tasks: BackgroundTasks
        ):
            """Receive TradingView webhook signal."""
            try:
                # Verify webhook signature if configured
                if SECRET_WEBHOOK_KEY:
                    body = await request.body()
                    signature = request.headers.get("X-Signature")
                    if not self._verify_signature(body, signature):
                        raise HTTPException(status_code=401, detail="Invalid signature")
                
                # Convert to internal signal format
                trading_signal = TradingSignal(
                    symbol=signal.symbol,
                    side=PositionSide.LONG if signal.action in ['BUY', 'LONG'] else PositionSide.SHORT,
                    confidence=signal.confidence,
                    tp_pct=signal.tp_pct,
                    sl_pct=signal.sl_pct,
                    timestamp=datetime.now(),
                    metadata={
                        'source': 'TradingView',
                        'strategy': signal.strategy,
                        'timeframe': signal.timeframe,
                        'entry_price': signal.price
                    }
                )
                
                # Add to autopilot queue
                await self.autopilot.add_signal(trading_signal)
                
                # Log the signal
                self.signal_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'signal': signal.dict(),
                    'processed': True
                })
                
                # Keep only last 100 signals
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
                    
                self.logger.info(f"Received TradingView signal: {signal.symbol} {signal.action}")
                
                return {
                    "status": "success",
                    "message": "Signal received and queued",
                    "signal_id": len(self.signal_history)
                }
                
            except Exception as e:
                self.logger.error(f"Error processing TradingView signal: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/webhook/manual")
        async def manual_signal(signal: TradingViewSignal):
            """Manual signal endpoint for testing."""
            return await receive_tradingview_signal(signal, request=None, background_tasks=None)
        
        @self.app.get("/status", response_model=SystemStatus)
        async def get_status():
            """Get system status."""
            status_data = self.autopilot.get_status()
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return SystemStatus(
                status="running" if status_data["is_running"] else "stopped",
                uptime=uptime,
                total_positions=status_data["total_positions"],
                open_positions=status_data["open_positions"],
                realized_pnl=status_data["realized_profit"],
                unrealized_pnl=status_data["unrealized_profit"],
                progress_to_target=status_data["progress_to_target"]
            )
        
        @self.app.get("/signals/history")
        async def get_signal_history(limit: int = 20):
            """Get recent signal history."""
            return {
                "signals": self.signal_history[-limit:],
                "total_count": len(self.signal_history)
            }
        
        @self.app.get("/positions")
        async def get_positions():
            """Get current positions."""
            return {
                "positions": [
                    {
                        "id": pos.id,
                        "symbol": pos.symbol,
                        "side": pos.side.value,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                        "current_pnl": pos.unrealized_pnl,
                        "tp_price": pos.tp_price,
                        "sl_price": pos.sl_price,
                        "opened_at": pos.opened_at.isoformat()
                    }
                    for pos in self.autopilot.positions.values()
                    if pos.status.value == "open"
                ]
            }
        
        @self.app.post("/control/start")
        async def start_system():
            """Start the trading system."""
            if not self.autopilot.is_running:
                asyncio.create_task(self.autopilot.start())
                return {"status": "success", "message": "System started"}
            else:
                return {"status": "info", "message": "System already running"}
        
        @self.app.post("/control/stop")
        async def stop_system():
            """Stop the trading system."""
            if self.autopilot.is_running:
                await self.autopilot.stop()
                return {"status": "success", "message": "System stopped"}
            else:
                return {"status": "info", "message": "System already stopped"}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": (datetime.now() - self.start_time).total_seconds()
            }
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "AI Trading Webhook Server",
                "version": "1.0.0",
                "status": "running",
                "endpoints": {
                    "webhook": "/webhook/tradingview",
                    "manual": "/webhook/manual",
                    "status": "/status",
                    "positions": "/positions",
                    "history": "/signals/history",
                    "health": "/health"
                }
            }
    
    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        if not SECRET_WEBHOOK_KEY or not signature:
            return True  # Skip verification if not configured
            
        try:
            expected = hmac.new(
                SECRET_WEBHOOK_KEY.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            # Handle different signature formats
            if signature.startswith('sha256='):
                signature = signature[7:]
                
            return hmac.compare_digest(expected, signature)
            
        except Exception as e:
            self.logger.error(f"Signature verification error: {e}")
            return False
    
    async def start_server(self):
        """Start the webhook server."""
        # Use 0.0.0.0 for Railway deployment to bind to all interfaces
        host = "0.0.0.0" if os.getenv('RAILWAY_ENVIRONMENT') else "127.0.0.1"
        
        config = uvicorn.Config(
            self.app,
            host=host,
            port=self.port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        
        self.logger.info(f"Starting webhook server on {host}:{self.port}")
        await server.serve()


# Pine Script Template for TradingView
PINE_SCRIPT_TEMPLATE = '''
//@version=5
strategy("AI Trading Signal", overlay=true)

// Strategy Parameters
tp_pct = input.float(2.0, "Take Profit %") / 100
sl_pct = input.float(1.0, "Stop Loss %") / 100
webhook_url = input.string("https://your-server.com/webhook/tradingview", "Webhook URL")

// Your trading logic here
// Example: Simple moving average crossover
sma_fast = ta.sma(close, 10)
sma_slow = ta.sma(close, 20)

long_condition = ta.crossover(sma_fast, sma_slow)
short_condition = ta.crossunder(sma_fast, sma_slow)

// Execute trades and send webhooks
if long_condition
    strategy.entry("Long", strategy.long)
    alert('{"symbol":"' + syminfo.ticker + '","action":"BUY","price":' + str.tostring(close) + ',"tp_pct":' + str.tostring(tp_pct) + ',"sl_pct":' + str.tostring(sl_pct) + ',"timestamp":"' + str.tostring(timenow) + '","strategy":"SMA_Cross","timeframe":"' + timeframe.period + '"}', alert.freq_once_per_bar)

if short_condition
    strategy.entry("Short", strategy.short)
    alert('{"symbol":"' + syminfo.ticker + '","action":"SELL","price":' + str.tostring(close) + ',"tp_pct":' + str.tostring(tp_pct) + ',"sl_pct":' + str.tostring(sl_pct) + ',"timestamp":"' + str.tostring(timenow) + '","strategy":"SMA_Cross","timeframe":"' + timeframe.period + '"}', alert.freq_once_per_bar)

// Plot indicators
plot(sma_fast, "Fast SMA", color=color.blue)
plot(sma_slow, "Slow SMA", color=color.red)
'''


class TradingViewIntegration:
    """Helper class for TradingView integration setup."""
    
    @staticmethod
    def get_pine_script_template() -> str:
        """Get Pine Script template."""
        return PINE_SCRIPT_TEMPLATE
    
    @staticmethod
    def setup_webhook_alert(webhook_url: str, symbol: str = "BTCUSDT") -> Dict:
        """Generate webhook alert configuration."""
        return {
            "webhook_url": webhook_url,
            "message_template": {
                "symbol": "{{ticker}}",
                "action": "{{strategy.order.action}}",
                "price": "{{close}}",
                "timestamp": "{{time}}",
                "strategy": "TradingView",
                "timeframe": "{{interval}}"
            },
            "instructions": [
                "1. Copy the Pine Script template to TradingView",
                "2. Add your trading logic between the comments",
                "3. Set up an alert on the strategy",
                "4. Use the webhook URL in the alert notification",
                "5. Set the message to JSON format with the provided template"
            ]
        }
    
    @staticmethod
    def validate_signal_format(signal_data: Dict) -> bool:
        """Validate incoming signal format."""
        required_fields = ['symbol', 'action']
        return all(field in signal_data for field in required_fields)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Mock autopilot for testing
    class MockAutoPilot:
        def __init__(self):
            self.is_running = True
            self.positions = {}
            
        async def add_signal(self, signal):
            print(f"Received signal: {signal.symbol} {signal.side.value}")
            
        def get_status(self):
            return {
                "is_running": True,
                "total_positions": 0,
                "open_positions": 0,
                "realized_profit": 0.0,
                "unrealized_profit": 0.0,
                "progress_to_target": 0.0
            }
    
    # Test webhook server
    async def test_server():
        autopilot = MockAutoPilot()
        server = WebhookServer(autopilot, port=8000)
        await server.start_server()
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_server())
    else:
        print("TradingView Webhook Integration Module")
        print("Run with 'python webhook.py test' to start test server")