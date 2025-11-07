"""
Real-time market data feed system for cryptocurrency and stock markets.
Handles websocket connections, data buffering, and feature extraction.
"""

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
import websocket
import ccxt.async_support as ccxt

from config.settings import FEATURE_WINDOW_SIZE


class MarketDataPoint:
    """Represents a single market data point."""
    def __init__(self, symbol: str, price: float, volume: float, 
                 bid: float = None, ask: float = None, timestamp: datetime = None):
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.bid = bid or price
        self.ask = ask or price
        self.timestamp = timestamp or datetime.now()
        self.spread = self.ask - self.bid


class DataBuffer:
    """Ring buffer for storing market data with feature calculation."""
    
    def __init__(self, symbol: str, max_size: int = FEATURE_WINDOW_SIZE):
        self.symbol = symbol
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        
    def add(self, data_point: MarketDataPoint):
        """Add a new data point to the buffer."""
        self.data.append(data_point)
        
    def get_features(self) -> Optional[Dict]:
        """Extract trading features from the current buffer."""
        if len(self.data) < 10:
            return None
            
        try:
            # Convert to arrays for calculation
            prices = np.array([d.price for d in self.data])
            volumes = np.array([d.volume for d in self.data])
            spreads = np.array([d.spread for d in self.data])
            timestamps = [d.timestamp for d in self.data]
            
            # Price-based features
            returns = np.diff(prices) / prices[:-1]
            
            features = {
                # Current state
                'current_price': float(prices[-1]),
                'current_volume': float(volumes[-1]),
                'current_spread': float(spreads[-1]),
                'timestamp': timestamps[-1],
                
                # Price momentum features
                'price_change_1': float((prices[-1] - prices[-2]) / prices[-2]) if len(prices) > 1 else 0.0,
                'price_change_5': float((prices[-1] - prices[-6]) / prices[-6]) if len(prices) > 5 else 0.0,
                'price_change_10': float((prices[-1] - prices[-11]) / prices[-11]) if len(prices) > 10 else 0.0,
                
                # Moving averages
                'sma_5': float(np.mean(prices[-5:])) if len(prices) >= 5 else float(prices[-1]),
                'sma_10': float(np.mean(prices[-10:])) if len(prices) >= 10 else float(prices[-1]),
                'sma_20': float(np.mean(prices[-20:])) if len(prices) >= 20 else float(prices[-1]),
                
                # Volatility features
                'price_volatility': float(np.std(returns[-10:])) if len(returns) >= 10 else 0.0,
                'volume_volatility': float(np.std(volumes[-10:])) if len(volumes) >= 10 else 0.0,
                
                # Volume features
                'volume_sma_5': float(np.mean(volumes[-5:])) if len(volumes) >= 5 else float(volumes[-1]),
                'volume_ratio': float(volumes[-1] / np.mean(volumes[-10:])) if len(volumes) >= 10 else 1.0,
                
                # Spread features
                'spread_mean': float(np.mean(spreads[-10:])) if len(spreads) >= 10 else float(spreads[-1]),
                'spread_ratio': float(spreads[-1] / np.mean(spreads[-10:])) if len(spreads) >= 10 else 1.0,
                
                # Technical indicators
                'rsi': self._calculate_rsi(prices),
                'macd': self._calculate_macd(prices),
                'bb_position': self._calculate_bollinger_position(prices),
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return None
            
    def _calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
        
    def _calculate_macd(self, prices: np.array) -> float:
        """Calculate MACD indicator."""
        if len(prices) < 26:
            return 0.0
            
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd_line = ema_12 - ema_26
        return float(macd_line)
        
    def _calculate_bollinger_position(self, prices: np.array, period: int = 20) -> float:
        """Calculate position within Bollinger Bands (0=lower, 0.5=middle, 1=upper)."""
        if len(prices) < period:
            return 0.5
            
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        if std == 0:
            return 0.5
            
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        current_price = prices[-1]
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return float(np.clip(position, 0, 1))
        
    def _ema(self, prices: np.array, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return float(np.mean(prices))
            
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
            
        return float(ema)


class WebSocketDataFeed:
    """Base class for websocket-based market data feeds."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.buffers = {symbol: DataBuffer(symbol) for symbol in symbols}
        self.callbacks: List[Callable] = []
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
    def add_callback(self, callback: Callable):
        """Add a callback function for new data."""
        self.callbacks.append(callback)
        
    async def start(self):
        """Start the data feed."""
        raise NotImplementedError
        
    async def stop(self):
        """Stop the data feed."""
        self.is_running = False
        
    async def get_latest_data(self, symbol: str = None) -> Optional[Dict]:
        """Get the latest data and features for a symbol."""
        if symbol:
            return self.buffers[symbol].get_features() if symbol in self.buffers else None
        else:
            # Return latest data for all symbols
            return {sym: buf.get_features() for sym, buf in self.buffers.items()}
            
    def _notify_callbacks(self, symbol: str, data: MarketDataPoint):
        """Notify all registered callbacks of new data."""
        for callback in self.callbacks:
            try:
                asyncio.create_task(callback(symbol, data))
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")


class BinanceWebSocketFeed(WebSocketDataFeed):
    """Binance websocket data feed for cryptocurrency markets."""
    
    def __init__(self, symbols: List[str]):
        super().__init__(symbols)
        self.ws = None
        
    async def start(self):
        """Start Binance websocket connection."""
        self.is_running = True
        
        # Convert symbols to Binance format (lowercase with no separator)
        binance_symbols = [symbol.replace('/', '').lower() for symbol in self.symbols]
        
        # Create stream URL
        streams = [f"{symbol}@ticker" for symbol in binance_symbols]
        stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
        
        self.logger.info(f"Connecting to Binance WebSocket: {len(self.symbols)} symbols")
        
        await self._connect(stream_url)
        
    async def _connect(self, url: str):
        """Connect to websocket and handle messages."""
        import websockets
        
        try:
            async with websockets.connect(url) as websocket:
                self.logger.info("Connected to Binance WebSocket")
                
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                    except Exception as e:
                        self.logger.error(f"WebSocket error: {e}")
                        break
                        
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            if self.is_running:
                # Attempt to reconnect
                await asyncio.sleep(5)
                await self._connect(url)
                
    async def _handle_message(self, message: str):
        """Handle incoming websocket message."""
        try:
            data = json.loads(message)
            
            if 'stream' in data:
                symbol_data = data['data']
                symbol = symbol_data['s']  # Symbol in Binance format
                
                # Convert back to standard format
                if 'USDT' in symbol:
                    standard_symbol = symbol.replace('USDT', '/USDT')
                elif 'BTC' in symbol:
                    standard_symbol = symbol.replace('BTC', '/BTC')
                else:
                    standard_symbol = symbol
                    
                # Create data point
                data_point = MarketDataPoint(
                    symbol=standard_symbol,
                    price=float(symbol_data['c']),  # Close price
                    volume=float(symbol_data['v']), # Volume
                    bid=float(symbol_data['b']),    # Bid price
                    ask=float(symbol_data['a']),    # Ask price
                    timestamp=datetime.fromtimestamp(symbol_data['E'] / 1000)
                )
                
                # Add to buffer
                if standard_symbol in self.buffers:
                    self.buffers[standard_symbol].add(data_point)
                    self._notify_callbacks(standard_symbol, data_point)
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")


class AlpacaWebSocketFeed(WebSocketDataFeed):
    """Alpaca websocket data feed for US stock markets."""
    
    def __init__(self, symbols: List[str], api_key: str, secret_key: str):
        super().__init__(symbols)
        self.api_key = api_key
        self.secret_key = secret_key
        
    async def start(self):
        """Start Alpaca websocket connection."""
        try:
            from alpaca.data.live import StockDataStream
            
            self.is_running = True
            
            # Create Alpaca stream
            self.stream = StockDataStream(self.api_key, self.secret_key)
            
            # Subscribe to quotes
            self.stream.subscribe_quotes(self._handle_quote, *self.symbols)
            
            self.logger.info(f"Starting Alpaca WebSocket for {len(self.symbols)} symbols")
            
            await self.stream._run_forever()
            
        except ImportError:
            self.logger.error("alpaca-py not installed. Install with: pip install alpaca-py")
        except Exception as e:
            self.logger.error(f"Alpaca connection error: {e}")
            
    async def _handle_quote(self, quote):
        """Handle incoming quote data."""
        try:
            data_point = MarketDataPoint(
                symbol=quote.symbol,
                price=(quote.bid_price + quote.ask_price) / 2,
                volume=quote.bid_size + quote.ask_size,
                bid=quote.bid_price,
                ask=quote.ask_price,
                timestamp=quote.timestamp
            )
            
            if quote.symbol in self.buffers:
                self.buffers[quote.symbol].add(data_point)
                self._notify_callbacks(quote.symbol, data_point)
                
        except Exception as e:
            self.logger.error(f"Error handling quote: {e}")


class DataFeedManager:
    """Manager for multiple data feeds with failover and aggregation."""
    
    def __init__(self):
        self.feeds: List[WebSocketDataFeed] = []
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
    def add_feed(self, feed: WebSocketDataFeed):
        """Add a data feed to the manager."""
        self.feeds.append(feed)
        
    async def start_all(self):
        """Start all registered data feeds."""
        self.is_running = True
        self.logger.info(f"Starting {len(self.feeds)} data feeds")
        
        tasks = [feed.start() for feed in self.feeds]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def stop_all(self):
        """Stop all data feeds."""
        self.is_running = False
        
        for feed in self.feeds:
            await feed.stop()
            
        self.logger.info("All data feeds stopped")
        
    async def get_latest_data(self, symbol: str = None) -> Optional[Dict]:
        """Get latest data from the first available feed."""
        for feed in self.feeds:
            if feed.is_running:
                data = await feed.get_latest_data(symbol)
                if data:
                    return data
        return None