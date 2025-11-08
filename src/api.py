"""
Real-time stock data API endpoints for the dashboard.
Provides live prices, AI predictions, trading signals, and algorithm management.
"""

import asyncio
import logging
import os
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
        
        # Fetch crypto prices from Alpaca (24/7 crypto paper trading!)
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoLatestBarRequest
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret = os.getenv('ALPACA_API_SECRET')
            
            if api_key and secret:
                crypto_client = CryptoHistoricalDataClient(api_key, secret)
                crypto_symbols = ['BTC/USD', 'ETH/USD', 'DOGE/USD', 'AVAX/USD', 'LTC/USD']
                
                request = CryptoLatestBarRequest(symbol_or_symbols=crypto_symbols)
                bars = crypto_client.get_crypto_latest_bar(request)
                
                logger.info(f"Fetched {len(bars)} crypto bars from Alpaca")
                
                for symbol, bar in bars.items():
                    try:
                        stocks_data.append({
                            'symbol': symbol,
                            'price': round(float(bar.close), 2),
                            'change': 0.0,  # Calculate from 24h data if needed
                            'volume': int(bar.volume),
                            'high': round(float(bar.high), 2),
                            'low': round(float(bar.low), 2),
                            'timestamp': datetime.now().isoformat(),
                            'type': 'crypto'
                        })
                        logger.info(f"Added crypto: {symbol} @ ${bar.close}")
                    except Exception as e:
                        logger.warning(f"Error processing {symbol}: {e}")
            else:
                logger.warning("No Alpaca API credentials - skipping crypto fetch")
        except Exception as e:
            logger.error(f"Error fetching Alpaca crypto: {e}", exc_info=True)
        
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
                    # No credentials - return demo data with crypto
                    for symbol in ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
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
                # Return demo data on error with crypto
                for symbol in ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
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
    """Launch a REAL live trading algorithm with actual trades (stocks or crypto)."""
    try:
        algo_id = f"algo_{datetime.now().timestamp()}"
        symbol = config.get('symbol')
        
        # Determine if crypto or stock based on symbol format
        is_crypto = '/' in symbol  # Crypto symbols have format like BTC/USD
        
        if is_crypto:
            # Use Alpaca for crypto paper trading (24/7 markets, same API as stocks!)
            from alpaca.trading.client import TradingClient
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret = os.getenv('ALPACA_API_SECRET')
            
            if not api_key or not secret:
                return {'success': False, 'error': 'Alpaca API credentials not configured'}
            
            try:
                # Alpaca paper trading supports both stocks AND crypto!
                trading_client = TradingClient(api_key, secret, paper=True)
                account = trading_client.get_account()
                
                available_cash = float(account.cash)
                logger.info(f"Alpaca crypto paper trading - Available cash: ${available_cash:.2f}")
                
                if available_cash < config.get('capital'):
                    return {'success': False, 'error': f'Insufficient paper trading balance. Available: ${available_cash:.2f}, Required: ${config.get("capital"):.2f}'}
                
            except Exception as e:
                return {'success': False, 'error': f'Failed to connect to Alpaca: {str(e)}'}
            
            is_real_trading = True  # Real paper trading on Alpaca
            logger.info(f"REAL crypto paper trading enabled on Alpaca for {symbol}")
        else:
            # Use Alpaca for stock trading
            from alpaca.trading.client import TradingClient
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret = os.getenv('ALPACA_API_SECRET')
            
            if not api_key or not secret:
                return {'success': False, 'error': 'Alpaca API credentials not configured'}
            
            # Initialize real trading client
            trading_client = TradingClient(api_key, secret, paper=True)
            
            # Verify account has sufficient capital
            account = trading_client.get_account()
            available_cash = float(account.cash)
            
            if available_cash < config.get('capital'):
                return {'success': False, 'error': f'Insufficient capital. Available: ${available_cash:.2f}'}
            
            is_real_trading = True
            logger.info(f"Real Alpaca trading enabled for {symbol}")
        
        algorithm = {
            'id': algo_id,
            'symbol': symbol,
            'capital': config.get('capital'),
            'splits': config.get('splits'),
            'take_profit': config.get('take_profit'),
            'stop_loss': config.get('stop_loss'),
            'strategy': config.get('strategy'),
            'timeframe': config.get('timeframe', '5m'),
            'status': 'running',
            'active_positions': 0,
            'total_trades': 0,
            'win_rate': 0.0,
            'pnl': 0.0,
            'recent_trades': [],
            'all_trades': [],
            'alpaca_orders': [],  # Track real Alpaca order IDs
            'started_at': datetime.now().isoformat(),
            'completed': False,
            'real_trading': is_real_trading,  # Flag for real trading mode
            'is_crypto': is_crypto  # Flag to identify crypto vs stock
        }
        
        # Initialize position tracking for real trades
        algorithm['positions'] = []
        position_capital = config.get('capital') / config.get('splits')
        
        for i in range(config.get('splits')):
            algorithm['positions'].append({
                'id': i,
                'capital': position_capital,
                'status': 'ready',
                'trades_count': 0,
                'current_order_id': None,
                'entry_price': None,
                'quantity': None,
                'entry_time': None,  # Track when position was opened
                'close_failed_count': 0,  # Track failed close attempts
                'last_close_attempt': None  # Track last failed close time
            })
        
        algorithms[algo_id] = algorithm
        
        trading_mode = "REAL" if is_real_trading else "SIMULATED"
        market_type = "Crypto" if is_crypto else "Stock"
        logger.info(f"{trading_mode} {market_type} Algorithm {algo_id} launched for {algorithm['symbol']} with ${algorithm['capital']} capital")
        
        return {'success': True, 'algorithm': algorithm}
    
    except Exception as e:
        logger.error(f"Error launching algorithm: {e}")
        return {'success': False, 'error': str(e)}

@router.get("/algorithms/{algo_id}/status")
async def get_algorithm_status(algo_id: str):
    """Get live status and execute REAL trades (stocks AND crypto via Alpaca - 24/7!)."""
    try:
        import random
        
        if algo_id not in algorithms:
            return {'error': 'Algorithm not found'}
        
        algo = algorithms[algo_id]
        
        if algo['status'] != 'running':
            return {'status': algo}
        
        # Check if this is real trading mode
        is_real_trading = algo.get('real_trading', False)
        is_crypto = algo.get('is_crypto', False)
        
        if not is_real_trading:
            # Fallback to simulated trading if not in real mode
            return await get_simulated_algo_status(algo_id, algo)
        
        # REAL TRADING MODE - Alpaca supports both stocks AND crypto!
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret = os.getenv('ALPACA_API_SECRET')
            
            if not api_key or not secret:
                logger.error("No Alpaca credentials for real trading")
                return {'status': algo}
            
            # Alpaca paper trading works for both stocks AND crypto (24/7!)
            trading_client = TradingClient(api_key, secret, paper=True)
            can_trade = True
            
            # Get current market price
            if is_crypto:
                # For crypto, use crypto data client
                from alpaca.data.historical import CryptoHistoricalDataClient
                from alpaca.data.requests import CryptoLatestBarRequest
                
                data_client = CryptoHistoricalDataClient(api_key, secret)
                request = CryptoLatestBarRequest(symbol_or_symbols=[algo['symbol']])
                bars = data_client.get_crypto_latest_bar(request)
                current_price = float(bars[algo['symbol']].close) if algo['symbol'] in bars else None
            else:
                # For stocks, use stock data client
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestBarRequest
                
                data_client = StockHistoricalDataClient(api_key, secret)
                request = StockLatestBarRequest(symbol_or_symbols=[algo['symbol']])
                bars = data_client.get_stock_latest_bar(request)
                current_price = float(bars[algo['symbol']].close) if algo['symbol'] in bars else None
            
            if not current_price:
                logger.warning(f"Could not get price for {algo['symbol']}")
                return {'status': algo}
            
            # NO MORE SIMULATED TRADING - Only real trades!
            if not can_trade:
                return {'status': algo, 'error': 'API credentials required for real trading'}
            
            # Check each position and execute real trades
            for position in algo['positions']:
                # OPEN NEW POSITION
                if position['status'] == 'ready' and random.random() > 0.3:
                    try:
                        # Calculate quantity based on position capital
                        if is_crypto:
                            # For crypto, calculate based on asset price
                            qty = position['capital'] / current_price
                            qty = round(qty, 6)  # Crypto allows decimals
                        else:
                            # For stocks, integer shares only
                            qty = max(1, int(position['capital'] / current_price))
                        
                        # Alpaca handles both stocks AND crypto with same API!
                        order_request = MarketOrderRequest(
                            symbol=algo['symbol'],
                            qty=qty,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC  # GTC for crypto (24/7), DAY for stocks
                        )
                        
                        order = trading_client.submit_order(order_request)
                        order_id = order.id
                        
                        position['status'] = 'trading'
                        position['current_order_id'] = order_id
                        position['entry_price'] = current_price
                        position['quantity'] = qty
                        position['entry_time'] = datetime.now()  # Track entry time
                        position['trades_count'] += 1
                        
                        asset_type = "crypto" if is_crypto else "shares"
                        logger.info(f"REAL TRADE OPENED: {algo['symbol']} {qty} {asset_type} @ ${current_price:.2f} (Order: {order_id})")
                        
                    except Exception as e:
                        logger.error(f"Error opening position: {e}")
                
                # CLOSE EXISTING POSITION
                elif position['status'] == 'trading' and position['current_order_id']:
                    try:
                        # Check if too many failed close attempts (cooldown)
                        if position.get('close_failed_count', 0) >= 3:
                            if position.get('last_close_attempt'):
                                cooldown_seconds = (datetime.now() - position['last_close_attempt']).total_seconds()
                                if cooldown_seconds < 30:  # 30 second cooldown after 3 failures
                                    continue
                            # Reset counter after cooldown
                            position['close_failed_count'] = 0
                        
                        # Check minimum holding time (60 seconds to avoid wash trade detection)
                        if position.get('entry_time'):
                            holding_seconds = (datetime.now() - position['entry_time']).total_seconds()
                            if holding_seconds < 60:
                                continue  # Don't close yet, too soon
                        
                        # Verify the BUY order is filled before trying to SELL
                        try:
                            # Alpaca works same for stocks and crypto
                            buy_order = trading_client.get_order_by_id(position['current_order_id'])
                            if buy_order.status != 'filled':
                                logger.info(f"Position {position['id']} BUY order not filled yet (status: {buy_order.status})")
                                continue  # Wait for fill
                        except Exception as e:
                            logger.warning(f"Could not check order status: {e}")
                            # Continue anyway, might be filled
                        
                        # Get AI prediction to decide exit
                        if trading_system and trading_system.predictor:
                            # Use real AI model prediction
                            should_exit = random.random() > 0.5  # Placeholder - replace with actual prediction
                        else:
                            should_exit = random.random() > 0.5
                        
                        if should_exit:
                            # Alpaca handles both stocks AND crypto with same API!
                            order_request = MarketOrderRequest(
                                symbol=algo['symbol'],
                                qty=position['quantity'],
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.GTC  # GTC for crypto (24/7), DAY for stocks
                            )
                            
                            order = trading_client.submit_order(order_request)
                            order_id = order.id
                            
                            # Calculate real P&L
                            exit_price = current_price
                            entry_price = position['entry_price']
                            trade_pnl = (exit_price - entry_price) * position['quantity']
                            
                            algo['total_trades'] += 1
                            trade = {
                                'position_id': position['id'],
                                'side': 'SELL',
                                'qty': position['quantity'],
                                'entry_price': round(entry_price, 2),
                                'exit_price': round(exit_price, 2),
                                'pnl': round(trade_pnl, 2),
                                'order_id': order_id,
                                'timestamp': datetime.now().isoformat(),
                                'real_trade': True
                            }
                            
                            algo['recent_trades'].insert(0, trade)
                            algo['recent_trades'] = algo['recent_trades'][:10]
                            algo['all_trades'].append(trade)
                            algo['pnl'] += trade_pnl
                            algo['alpaca_orders'].append(order_id)
                            
                            # Reset position to ready
                            position['status'] = 'ready'
                            position['current_order_id'] = None
                            position['entry_price'] = None
                            position['quantity'] = None
                            position['entry_time'] = None
                            position['close_failed_count'] = 0
                            position['last_close_attempt'] = None
                            
                            logger.info(f"REAL TRADE CLOSED: {algo['symbol']} P&L: ${trade_pnl:.2f} (Order: {order.id})")
                    
                    except Exception as e:
                        logger.error(f"Error closing position: {e}")
                        # Track failed close attempts
                        position['close_failed_count'] = position.get('close_failed_count', 0) + 1
                        position['last_close_attempt'] = datetime.now()
            
            # Count active trading positions
            algo['active_positions'] = len([p for p in algo['positions'] if p['status'] == 'trading'])
            
            # Update win rate
            if algo['all_trades']:
                winning_trades = len([t for t in algo['all_trades'] if t['pnl'] > 0])
                algo['win_rate'] = (winning_trades / len(algo['all_trades'])) * 100
            
            # Check if OVERALL take profit or stop loss hit
            capital = algo['capital']
            current_value = capital + algo['pnl']
            tp_target = capital * (1 + algo['take_profit'] / 100)
            sl_target = capital * (1 - algo['stop_loss'] / 100)
            
            if current_value >= tp_target:
                algo['status'] = 'completed'
                algo['completed'] = True
                algo['exit_reason'] = 'Take Profit Reached'
                logger.info(f"REAL Algorithm {algo_id} completed: TP at ${current_value:.2f}")
            elif current_value <= sl_target:
                algo['status'] = 'completed'
                algo['completed'] = True
                algo['exit_reason'] = 'Stop Loss Hit'
                logger.info(f"REAL Algorithm {algo_id} completed: SL at ${current_value:.2f}")
        
        except Exception as e:
            logger.error(f"Error in real trading execution: {e}")
        
        return {'status': algo}
    
    except Exception as e:
        logger.error(f"Error getting algorithm status: {e}")
        return {'error': str(e)}


async def get_simulated_algo_status(algo_id: str, algo: dict):
    """Simulated trading for demo purposes."""
    import random
    
    if algo['status'] == 'running':
        # Initialize position tracking if not exists
        if 'positions' not in algo:
            algo['positions'] = []
            for i in range(algo['splits']):
                algo['positions'].append({
                    'id': i,
                    'capital': algo['capital'] / algo['splits'],
                    'status': 'ready',
                    'trades_count': 0
                })
        
        timeframe_speeds = {'1m': 0.9, '5m': 0.8, '15m': 0.6, '30m': 0.4, '1h': 0.25}
        trade_probability = timeframe_speeds.get(algo.get('timeframe', '5m'), 0.8)
        
        for position in algo['positions']:
            if position['status'] == 'ready' and random.random() > (1 - trade_probability):
                position['status'] = 'trading'
                position['trades_count'] += 1
                
            elif position['status'] == 'trading' and random.random() > 0.5:
                is_winning_trade = random.random() > 0.35
                position_size = position['capital']
                
                if is_winning_trade:
                    trade_pnl = position_size * random.uniform(0.01, 0.08)
                else:
                    trade_pnl = -position_size * random.uniform(0.005, 0.03)
                
                algo['total_trades'] += 1
                trade = {
                    'position_id': position['id'],
                    'side': 'BUY' if random.random() > 0.5 else 'SELL',
                    'qty': round(position_size / 150, 2),
                    'price': random.uniform(100, 200),
                    'pnl': round(trade_pnl, 2),
                    'timestamp': datetime.now().isoformat(),
                    'real_trade': False
                }
                
                algo['recent_trades'].insert(0, trade)
                algo['recent_trades'] = algo['recent_trades'][:10]
                algo['all_trades'].append(trade)
                algo['pnl'] += trade_pnl
                position['status'] = 'ready'
        
        algo['active_positions'] = len([p for p in algo['positions'] if p['status'] == 'trading'])
        
        if algo['all_trades']:
            winning_trades = len([t for t in algo['all_trades'] if t['pnl'] > 0])
            algo['win_rate'] = (winning_trades / len(algo['all_trades'])) * 100
        
        capital = algo['capital']
        current_value = capital + algo['pnl']
        tp_target = capital * (1 + algo['take_profit'] / 100)
        sl_target = capital * (1 - algo['stop_loss'] / 100)
        
        if current_value >= tp_target or current_value <= sl_target:
            algo['status'] = 'completed'
            algo['completed'] = True
    
    return {'status': algo}

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


@router.post("/admin/cancel-order/{order_id}")
async def cancel_order(order_id: str):
    """Emergency endpoint to cancel a stuck order."""
    try:
        if trading_system and trading_system.autopilot:
            broker = trading_system.autopilot.exchange
            success = await broker.cancel_order(order_id)
            
            if success:
                logger.info(f"✅ Cancelled order: {order_id}")
                return {"status": "success", "message": f"Cancelled order {order_id}"}
            else:
                logger.warning(f"⚠️ Failed to cancel order: {order_id}")
                return {"status": "error", "message": "Failed to cancel order"}
        else:
            return {"status": "error", "message": "Trading system not available"}
    except Exception as e:
        logger.error(f"Error canceling order: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/admin/orders")
async def list_orders():
    """List all open orders."""
    try:
        if trading_system and trading_system.autopilot:
            broker = trading_system.autopilot.exchange
            # For Alpaca
            if hasattr(broker, 'client'):
                orders = broker.client.get_orders()
                return {
                    "status": "success",
                    "orders": [
                        {
                            "id": str(order.id),
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "qty": float(order.qty) if order.qty else None,
                            "notional": float(order.notional) if order.notional else None,
                            "status": order.status.value,
                            "created_at": str(order.created_at)
                        }
                        for order in orders
                    ]
                }
        return {"status": "error", "message": "Trading system not available"}
    except Exception as e:
        logger.error(f"Error listing orders: {e}")
        return {"status": "error", "message": str(e)}
