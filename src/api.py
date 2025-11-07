"""
Real-time stock data API endpoints for the dashboard.
Provides live prices, AI predictions, trading signals, and algorithm management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
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
algorithms = {}  # Store user-created algorithms

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
        
        # If no data from feeds, fetch real-time data from Alpaca
        if not stocks_data:
            try:
                import os
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestBarRequest
                
                api_key = os.getenv('ALPACA_API_KEY')
                secret = os.getenv('ALPACA_API_SECRET')
                
                if api_key and secret:
                    data_client = StockHistoricalDataClient(api_key, secret)
                    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'GOOGL', 'AMZN', 'META']
                    
                    # Get latest bars (1-minute data)
                    request = StockLatestBarRequest(symbol_or_symbols=symbols)
                    bars = data_client.get_stock_latest_bar(request)
                    
                    for symbol in symbols:
                        if symbol in bars:
                            bar = bars[symbol]
                            
                            # Calculate change from open to close
                            change = ((bar.close - bar.open) / bar.open) * 100 if bar.open > 0 else 0.0
                            
                            stocks_data.append({
                                'symbol': symbol,
                                'price': round(bar.close, 2),
                                'change': round(change, 2),
                                'volume': int(bar.volume),
                                'high': round(bar.high, 2),
                                'low': round(bar.low, 2),
                                'timestamp': bar.timestamp.isoformat() if hasattr(bar, 'timestamp') else datetime.now().isoformat()
                            })
                else:
                    # No credentials - return demo data
                    for symbol in ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']:
                        stocks_data.append({
                            'symbol': symbol,
                            'price': 0.0,
                            'change': 0.0,
                            'volume': 0,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'no_credentials'
                        })
            except Exception as e:
                logger.error(f"Error fetching Alpaca data: {e}")
                # Return demo data on error
                for symbol in ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']:
                    stocks_data.append({
                        'symbol': symbol,
                        'price': 0.0,
                        'change': 0.0,
                        'volume': 0,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'error'
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
        
        # Try from trading system first
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
            
            if positions:
                return {'positions': positions, 'count': len(positions)}
        
        # Fallback: fetch directly from Alpaca
        try:
            import os
            from alpaca.trading.client import TradingClient
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret = os.getenv('ALPACA_API_SECRET')
            
            if api_key and secret:
                client = TradingClient(api_key, secret, paper=True)
                alpaca_positions = client.get_all_positions()
                
                for pos in alpaca_positions:
                    positions.append({
                        'symbol': pos.symbol,
                        'side': 'LONG' if float(pos.qty) > 0 else 'SHORT',
                        'entry_price': round(float(pos.avg_entry_price), 2),
                        'current_price': round(float(pos.current_price), 2),
                        'quantity': abs(float(pos.qty)),
                        'unrealized_pnl': round(float(pos.unrealized_pl), 2),
                        'entry_time': None,
                    })
        except Exception as e:
            logger.error(f"Error fetching positions from Alpaca: {e}")
        
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
        
        # Try to get from trading system first
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
                        return account_data
                except Exception as e:
                    logger.error(f"Error fetching account from broker: {e}")
        
        # Fallback: fetch directly from Alpaca
        try:
            import os
            from alpaca.trading.client import TradingClient
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret = os.getenv('ALPACA_API_SECRET')
            
            if api_key and secret:
                client = TradingClient(api_key, secret, paper=True)
                account = client.get_account()
                
                equity = float(account.equity)
                cash = float(account.cash)
                buying_power = float(account.buying_power)
                
                account_data = {
                    'balance': cash,
                    'equity': equity,
                    'buying_power': buying_power,
                    'pnl': equity - 100000.0,  # Starting paper trading balance
                    'pnl_percent': ((equity - 100000.0) / 100000.0) * 100,
                    'target': 2000.0,
                    'progress': (equity / 102000.0) * 100,  # Progress to $102k (2% gain on $100k)
                }
        except Exception as e:
            logger.error(f"Error fetching account from Alpaca: {e}")
        
        return account_data
    
    except Exception as e:
        logger.error(f"Error getting account: {e}")
        return {'error': str(e)}

@router.get("/universe")
async def get_stock_universe(
    search: Optional[str] = Query(None),
    exchange: Optional[str] = Query(None),
    limit: int = Query(100, le=1000)
):
    """Get all tradeable stocks with optional filtering."""
    try:
        from src.stock_universe import get_all_tradeable_stocks, search_stocks, filter_stocks
        
        stocks = await get_all_tradeable_stocks()
        
        # Apply search
        if search:
            stocks = search_stocks(search, stocks)
        
        # Apply filters
        if exchange:
            stocks = filter_stocks(stocks, exchange=exchange)
        
        # Limit results
        stocks = stocks[:limit]
        
        return {'stocks': stocks, 'count': len(stocks)}
    
    except Exception as e:
        logger.error(f"Error getting stock universe: {e}")
        return {'stocks': [], 'error': str(e)}

@router.get("/stock/{symbol}")
async def get_stock_detail(symbol: str):
    """Get detailed info for a specific stock."""
    try:
        stock_data = {
            'symbol': symbol,
            'name': '',
            'price': 0.0,
            'change': 0.0,
            'volume': 0,
            'market_cap': 0,
            'pe_ratio': 0,
            'high_52w': 0,
            'low_52w': 0,
        }
        
        # Get live price
        if trading_system and trading_system.data_feed_manager:
            for feed in trading_system.data_feed_manager.feeds:
                if feed.is_running and hasattr(feed, 'buffers'):
                    if symbol in feed.buffers:
                        buffer = feed.buffers[symbol]
                        if len(buffer.data) > 0:
                            latest = buffer.data[-1]
                            prev = buffer.data[-2] if len(buffer.data) > 1 else latest
                            stock_data['price'] = round(latest.price, 2)
                            stock_data['change'] = round(((latest.price - prev.price) / prev.price) * 100, 2)
                            stock_data['volume'] = int(latest.volume)
        
        return stock_data
    
    except Exception as e:
        logger.error(f"Error getting stock detail: {e}")
        return {'error': str(e)}

@router.post("/orders")
async def place_order(order: dict):
    """Place a trading order."""
    try:
        import os
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret = os.getenv('ALPACA_API_SECRET')
        
        if not api_key or not secret:
            return {'success': False, 'error': 'API credentials not configured'}
        
        client = TradingClient(api_key, secret, paper=True)
        
        # Parse order parameters
        symbol = order.get('symbol')
        side = OrderSide.BUY if order.get('side', '').upper() == 'BUY' else OrderSide.SELL
        qty = order.get('qty', 1)
        order_type = order.get('type', 'market')
        
        # Create order request
        if order_type == 'market':
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
        elif order_type == 'limit':
            limit_price = order.get('limit_price')
            if not limit_price:
                return {'success': False, 'error': 'Limit price required for limit orders'}
            
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
        else:
            return {'success': False, 'error': f'Unsupported order type: {order_type}'}
        
        # Submit order
        submitted_order = client.submit_order(order_request)
        
        logger.info(f"Order placed: {side.value} {qty} {symbol} @ {order_type}")
        
        return {
            'success': True,
            'order_id': submitted_order.id,
            'symbol': symbol,
            'side': side.value,
            'qty': qty,
            'type': order_type,
            'status': submitted_order.status
        }
    
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return {'success': False, 'error': str(e)}

@router.get("/algorithms")
async def get_algorithms():
    """Get all user-created algorithms."""
    try:
        algo_list = []
        for algo_id, algo in algorithms.items():
            algo_list.append({
                'id': algo_id,
                'name': algo['name'],
                'symbol': algo['symbol'],
                'status': algo['status'],
                'total_trades': algo.get('total_trades', 0),
                'win_rate': algo.get('win_rate', 0),
                'net_pnl': algo.get('net_pnl', 0),
                'efficiency': algo.get('efficiency', 0),
                'created_at': algo.get('created_at'),
            })
        
        return {'algorithms': algo_list, 'count': len(algo_list)}
    
    except Exception as e:
        logger.error(f"Error getting algorithms: {e}")
        return {'algorithms': [], 'error': str(e)}

@router.post("/algorithms")
async def create_algorithm(algo_config: dict):
    """Create a new trading algorithm."""
    try:
        algo_id = f"algo_{len(algorithms) + 1}"
        algorithms[algo_id] = {
            'id': algo_id,
            'name': algo_config.get('name', f'Algorithm {len(algorithms) + 1}'),
            'symbol': algo_config.get('symbol'),
            'status': 'training',
            'config': algo_config,
            'total_trades': 0,
            'win_rate': 0,
            'net_pnl': 0,
            'efficiency': 0,
            'created_at': datetime.now().isoformat(),
        }
        
        return {'success': True, 'algorithm': algorithms[algo_id]}
    
    except Exception as e:
        logger.error(f"Error creating algorithm: {e}")
        return {'success': False, 'error': str(e)}

@router.post("/algorithms/launch")
async def launch_algorithm(config: dict):
    """Launch a live trading algorithm."""
    try:
        import random
        
        algo_id = f"algo_{datetime.now().timestamp()}"
        
        algorithm = {
            'id': algo_id,
            'symbol': config.get('symbol'),
            'capital': config.get('capital'),
            'splits': config.get('splits'),
            'take_profit': config.get('take_profit'),
            'stop_loss': config.get('stop_loss'),
            'strategy': config.get('strategy'),
            'status': 'running',
            'active_positions': 0,
            'total_trades': 0,
            'win_rate': 0.0,
            'pnl': 0.0,
            'recent_trades': [],
            'started_at': datetime.now().isoformat(),
            'completed': False
        }
        
        algorithms[algo_id] = algorithm
        
        logger.info(f"Algorithm {algo_id} launched for {algorithm['symbol']} with ${algorithm['capital']} capital")
        
        return {'success': True, 'algorithm': algorithm}
    
    except Exception as e:
        logger.error(f"Error launching algorithm: {e}")
        return {'success': False, 'error': str(e)}

@router.get("/algorithms/{algo_id}/status")
async def get_algorithm_status(algo_id: str):
    """Get live status of running algorithm."""
    try:
        import random
        
        if algo_id not in algorithms:
            return {'error': 'Algorithm not found'}
        
        algo = algorithms[algo_id]
        
        # Simulate algorithm activity (replace with real trading logic)
        if algo['status'] == 'running':
            # Simulate position changes
            if random.random() > 0.7 and algo['active_positions'] < algo['splits']:
                algo['active_positions'] += 1
                algo['total_trades'] += 1
                
                # Simulate a trade
                trade_pnl = random.uniform(-5, 15)
                trade = {
                    'side': 'BUY' if random.random() > 0.5 else 'SELL',
                    'qty': 1,
                    'price': random.uniform(100, 200),
                    'pnl': trade_pnl,
                    'timestamp': datetime.now().isoformat()
                }
                
                algo['recent_trades'].insert(0, trade)
                algo['recent_trades'] = algo['recent_trades'][:10]  # Keep last 10
                algo['pnl'] += trade_pnl
                
                # Update win rate
                winning_trades = len([t for t in algo['recent_trades'] if t['pnl'] > 0])
                if algo['total_trades'] > 0:
                    algo['win_rate'] = (winning_trades / min(algo['total_trades'], 10)) * 100
            
            # Check if take profit or stop loss hit
            capital = algo['capital']
            current_value = capital + algo['pnl']
            tp_target = capital * (1 + algo['take_profit'] / 100)
            sl_target = capital * (1 - algo['stop_loss'] / 100)
            
            if current_value >= tp_target or current_value <= sl_target:
                algo['status'] = 'completed'
                algo['completed'] = True
        
        return {'status': algo}
    
    except Exception as e:
        logger.error(f"Error getting algorithm status: {e}")
        return {'error': str(e)}

@router.get("/algorithms/{algo_id}")
async def get_algorithm(algo_id: str):
    """Get specific algorithm details."""
    try:
        if algo_id not in algorithms:
            return {'error': 'Algorithm not found'}
        
        return {'algorithm': algorithms[algo_id]}
    
    except Exception as e:
        logger.error(f"Error getting algorithm: {e}")
        return {'error': str(e)}

@router.delete("/algorithms/{algo_id}")
async def delete_algorithm(algo_id: str):
    """Delete an algorithm."""
    try:
        if algo_id in algorithms:
            del algorithms[algo_id]
            return {'success': True}
        return {'error': 'Algorithm not found'}
    
    except Exception as e:
        logger.error(f"Error deleting algorithm: {e}")
        return {'success': False, 'error': str(e)}

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
