"""
Real-time stock data API endpoints for the dashboard.
Provides live prices, AI predictions, and trading signals.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# Router for API endpoints
router = APIRouter(prefix="/api", tags=["Trading Data"])

# Global state (will be set by trading system)
trading_system = None
latest_prices = {}
latest_predictions = {}
active_positions = []

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@router.get("/stocks")
async def get_stocks():
    """Get current stock data with prices and changes."""
    try:
        stocks_data = []
        
        # Get data from trading system if available
        if trading_system and trading_system.data_feed_manager:
            for feed in trading_system.data_feed_manager.feeds:
                if feed.is_running and hasattr(feed, 'buffers'):
                    for symbol, buffer in feed.buffers.items():
                        if len(buffer.data) > 0:
                            latest = buffer.data[-1]
                            prev = buffer.data[-2] if len(buffer.data) > 1 else latest
                            
                            change = ((latest.price - prev.price) / prev.price) * 100
                            
                            stocks_data.append({
                                'symbol': symbol,
                                'price': round(latest.price, 2),
                                'change': round(change, 2),
                                'volume': int(latest.volume),
                                'bid': round(latest.bid, 2),
                                'ask': round(latest.ask, 2),
                                'timestamp': latest.timestamp.isoformat()
                            })
        
        # If no data from feeds, return demo data
        if not stocks_data:
            symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']
            for symbol in symbols:
                stocks_data.append({
                    'symbol': symbol,
                    'price': 0.0,
                    'change': 0.0,
                    'volume': 0,
                    'bid': 0.0,
                    'ask': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'waiting_for_data'
                })
        
        return {'stocks': stocks_data, 'timestamp': datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Error getting stocks: {e}")
        return {'stocks': [], 'error': str(e)}

@router.get("/predictions")
async def get_predictions():
    """Get AI predictions for all symbols."""
    try:
        predictions = []
        
        if trading_system and trading_system.predictor:
            for feed in trading_system.data_feed_manager.feeds:
                if feed.is_running and hasattr(feed, 'buffers'):
                    for symbol, buffer in feed.buffers.items():
                        features = buffer.get_features()
                        if features:
                            # Get prediction from AI
                            try:
                                pred = trading_system.predictor.predict(features)
                                predictions.append({
                                    'symbol': symbol,
                                    'signal': 'BUY' if pred > 0.6 else 'SELL' if pred < 0.4 else 'HOLD',
                                    'confidence': round(abs(pred - 0.5) * 200, 1),  # 0-100%
                                    'prediction_value': round(pred, 3),
                                    'price': features.get('current_price', 0),
                                    'rsi': round(features.get('rsi', 50), 1),
                                    'macd': round(features.get('macd', 0), 4),
                                })
                            except Exception as e:
                                logger.error(f"Prediction error for {symbol}: {e}")
        
        return {'predictions': predictions, 'timestamp': datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return {'predictions': [], 'error': str(e)}

@router.get("/positions")
async def get_positions():
    """Get current trading positions."""
    try:
        positions = []
        
        if trading_system and trading_system.autopilot:
            for pos in trading_system.autopilot.active_positions.values():
                positions.append({
                    'symbol': pos.symbol,
                    'side': pos.side.value,
                    'entry_price': round(pos.entry_price, 2),
                    'current_price': round(pos.current_price, 2),
                    'quantity': pos.quantity,
                    'unrealized_pnl': round(pos.unrealized_pnl, 2),
                    'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
                })
        
        return {'positions': positions, 'count': len(positions)}
    
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {'positions': [], 'error': str(e)}

@router.get("/account")
async def get_account():
    """Get account balance and stats."""
    try:
        account_data = {
            'balance': 100.0,
            'equity': 100.0,
            'buying_power': 100.0,
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'target': 2000.0,
            'progress': 0.0,
        }
        
        if trading_system and trading_system.broker_manager:
            broker = trading_system.broker_manager.get_primary_broker()
            if broker and broker.is_connected:
                try:
                    acc = await broker.get_account()
                    if acc:
                        account_data['balance'] = float(acc.get('cash', 100))
                        account_data['equity'] = float(acc.get('equity', 100))
                        account_data['buying_power'] = float(acc.get('buying_power', 100))
                        account_data['pnl'] = account_data['equity'] - 100.0
                        account_data['pnl_percent'] = (account_data['pnl'] / 100.0) * 100
                        account_data['progress'] = (account_data['equity'] / 2000.0) * 100
                except Exception as e:
                    logger.error(f"Error fetching account: {e}")
        
        return account_data
    
    except Exception as e:
        logger.error(f"Error getting account: {e}")
        return {'error': str(e)}

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send updates every 2 seconds
            await asyncio.sleep(2)
            
            # Get latest data
            stocks = await get_stocks()
            predictions = await get_predictions()
            account = await get_account()
            
            # Broadcast to all connected clients
            await websocket.send_json({
                'type': 'update',
                'stocks': stocks.get('stocks', []),
                'predictions': predictions.get('predictions', []),
                'account': account,
                'timestamp': datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

def set_trading_system(system):
    """Set the trading system reference for API endpoints."""
    global trading_system
    trading_system = system
