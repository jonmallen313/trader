"""
Broker integrations for automated trading.
Supports both paper and live trading with multiple brokers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    import ccxt.async_support as ccxt
    from alpaca.trading.client import TradingClient
    from alpaca.data.live import StockDataStream
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except ImportError as e:
    print(f"Broker dependencies not installed: {e}")

from config.settings import PAPER_MODE


class OrderResult:
    """Standardized order result across brokers."""
    def __init__(self, order_id: str, symbol: str, side: str, size: float, 
                 fill_price: float, status: str, timestamp: datetime = None):
        self.id = order_id
        self.symbol = symbol
        self.side = side
        self.size = size
        self.fill_price = fill_price
        self.status = status
        self.timestamp = timestamp or datetime.now()


class BrokerInterface(ABC):
    """Abstract base class for broker integrations."""
    
    def __init__(self, paper_mode: bool = PAPER_MODE):
        self.paper_mode = paper_mode
        self.is_connected = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def connect(self):
        """Connect to the broker."""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the broker."""
        pass
        
    @abstractmethod
    async def get_balance(self) -> float:
        """Get account balance."""
        pass
        
    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass
        
    @abstractmethod
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = "market", price: float = None) -> OrderResult:
        """Place a trading order."""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
        
    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        pass


class BinanceBroker(BrokerInterface):
    """Binance cryptocurrency broker integration."""
    
    def __init__(self, api_key: str, secret_key: str, paper_mode: bool = PAPER_MODE):
        super().__init__(paper_mode)
        self.api_key = api_key
        self.secret_key = secret_key
        self.exchange = None
        
    async def connect(self):
        """Connect to Binance."""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
            })
            
            if self.paper_mode:
                self.exchange.set_sandbox_mode(True)
                self.logger.info("Connected to Binance Testnet")
            else:
                self.logger.info("Connected to Binance Live")
                
            await self.exchange.load_markets()
            self.is_connected = True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from Binance."""
        if self.exchange:
            await self.exchange.close()
            self.is_connected = False
            self.logger.info("Disconnected from Binance")
            
    async def get_balance(self) -> float:
        """Get USDT balance."""
        try:
            balance = await self.exchange.fetch_balance()
            return float(balance.get('USDT', {}).get('free', 0.0))
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0
            
    async def get_price(self, symbol: str) -> float:
        """Get current price."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            raise
            
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = "market", price: float = None) -> OrderResult:
        """Place order on Binance."""
        try:
            if order_type.lower() == "market":
                if side.lower() == "buy":
                    order = await self.exchange.create_market_buy_order(symbol, size)
                else:
                    order = await self.exchange.create_market_sell_order(symbol, size)
            else:  # limit order
                if side.lower() == "buy":
                    order = await self.exchange.create_limit_buy_order(symbol, size, price)
                else:
                    order = await self.exchange.create_limit_sell_order(symbol, size, price)
                    
            # Get fill price (may need to fetch order details)
            fill_price = order.get('price', price)
            if not fill_price and order.get('status') == 'closed':
                # Market order - fetch average fill price
                order_details = await self.exchange.fetch_order(order['id'], symbol)
                fill_price = order_details.get('average', price)
                
            return OrderResult(
                order_id=order['id'],
                symbol=symbol,
                side=side,
                size=size,
                fill_price=float(fill_price or 0),
                status=order.get('status', 'unknown'),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
            
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel order."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return False
            
    async def get_positions(self) -> List[Dict]:
        """Get positions (for futures)."""
        try:
            if hasattr(self.exchange, 'fetch_positions'):
                positions = await self.exchange.fetch_positions()
                return [p for p in positions if float(p.get('size', 0)) > 0]
            else:
                # For spot trading, get balances instead
                balance = await self.exchange.fetch_balance()
                positions = []
                for currency, data in balance.items():
                    if currency != 'USDT' and float(data.get('free', 0)) > 0:
                        positions.append({
                            'symbol': f"{currency}/USDT",
                            'size': float(data['free']),
                            'side': 'long'
                        })
                return positions
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return []


class AlpacaBroker(BrokerInterface):
    """Alpaca stock broker integration."""
    
    def __init__(self, api_key: str, secret_key: str, paper_mode: bool = PAPER_MODE):
        super().__init__(paper_mode)
        self.api_key = api_key
        self.secret_key = secret_key
        self.client = None
        
    async def connect(self):
        """Connect to Alpaca."""
        try:
            # New alpaca-py API uses paper=True/False parameter
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper_mode  # Use paper parameter instead of base_url
            )
            
            # Test connection
            account = self.client.get_account()
            self.is_connected = True
            
            mode = "Paper" if self.paper_mode else "Live"
            self.logger.info(f"Connected to Alpaca {mode} - Account: {account.account_number}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from Alpaca."""
        self.is_connected = False
        self.client = None
        self.logger.info("Disconnected from Alpaca")
        
    async def get_balance(self) -> float:
        """Get account buying power."""
        try:
            account = self.client.get_account()
            return float(account.buying_power)
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0
            
    async def get_price(self, symbol: str) -> float:
        """Get current price for stocks or crypto."""
        try:
            # Detect if crypto (contains '/' or 'USD' suffix)
            is_crypto = '/' in symbol or symbol.endswith('USD')
            
            if is_crypto:
                # Use Crypto API
                from alpaca.data.historical import CryptoHistoricalDataClient
                from alpaca.data.requests import CryptoLatestBarRequest
                
                data_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
                request = CryptoLatestBarRequest(symbol_or_symbols=[symbol])
                bars = data_client.get_crypto_latest_bar(request)
            else:
                # Use Stock API
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestBarRequest
                
                data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
                request = StockLatestBarRequest(symbol_or_symbols=[symbol])
                bars = data_client.get_stock_latest_bar(request)
            
            if symbol in bars:
                return float(bars[symbol].close)
            else:
                raise ValueError(f"No price data for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            raise
            
    async def place_order(self, symbol: str, side: str, size: float = None, 
                         order_type: str = "market", price: float = None, notional: float = None) -> OrderResult:
        """Place order on Alpaca. Use either size (shares) or notional (dollars)."""
        try:
            # Detect if crypto (different time_in_force required)
            is_crypto = '/' in symbol or symbol.endswith('USD')
            # Crypto paper trading: Use GTC (IOC gets canceled instantly in Alpaca paper trading)
            time_in_force = TimeInForce.GTC if is_crypto else TimeInForce.DAY
            
            # For crypto with notional, convert to fractional qty
            if is_crypto and notional and not size:
                # Get current price and calculate fractional crypto amount
                current_price = await self.get_price(symbol)
                size = notional / current_price  # Fractional crypto amount
                self.logger.info(f"Crypto: Converting ${notional:.2f} → {size:.8f} {symbol} @ ${current_price:.2f}")
                
            # Determine shares to trade
            if notional and not is_crypto:
                # Notional order - STOCKS ONLY (crypto needs qty)
                order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    notional=notional,  # Use dollar amount
                    side=order_side,
                    time_in_force=time_in_force
                )
                self.logger.info(f"Placing notional order: {symbol} {side} ${notional:.2f}")
                order = self.client.submit_order(order_request)
                
                # Calculate shares from filled order
                shares = float(order.qty) if order.qty else 0
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
                
            elif size:
                # Size-based order (stocks or CRYPTO fractional)
                # Crypto allows fractional shares, stocks need integers
                if not is_crypto:
                    shares = int(size) if size >= 1 else 1  # Stocks: minimum 1 share
                    if shares < 1:
                        raise ValueError(f"Order too small: {shares} shares")
                else:
                    shares = size  # Crypto: allow fractional
                    
                order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
                
                if order_type.lower() == "market":
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=shares,
                        side=order_side,
                        time_in_force=time_in_force
                    )
                    shares_str = f"{shares:.8f}" if is_crypto else f"{shares}"
                    self.logger.info(f"Placing {'crypto' if is_crypto else 'stock'} order: {symbol} {side} {shares_str} shares")
                else:
                    order_request = LimitOrderRequest(
                        symbol=symbol,
                        qty=shares,
                        side=order_side,
                        time_in_force=time_in_force,
                        limit_price=price
                    )
                    
                order = self.client.submit_order(order_request)
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
            else:
                raise ValueError("Must provide either size or notional")
            
            return OrderResult(
                order_id=str(order.id),
                symbol=symbol,
                side=side,
                size=float(shares),
                fill_price=fill_price,
                status=str(order.status),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error placing Alpaca order: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Alpaca order."""
        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return False
            
    async def get_positions(self) -> List[Dict]:
        """Get Alpaca positions."""
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'size': float(pos.qty),
                    'side': 'long' if float(pos.qty) > 0 else 'short',
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl)
                }
                for pos in positions
            ]
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return []


class MockBroker(BrokerInterface):
    """Mock broker for testing and paper trading without real APIs."""
    
    def __init__(self, initial_balance: float = 1000.0):
        super().__init__(paper_mode=True)
        self.balance = initial_balance
        self.positions = {}
        self.orders = {}
        self.next_order_id = 1
        self.prices = {}  # Cache for mock prices
        
    async def connect(self):
        """Connect (mock)."""
        self.is_connected = True
        self.logger.info(f"Connected to Mock Broker - Balance: ${self.balance}")
        
    async def disconnect(self):
        """Disconnect (mock)."""
        self.is_connected = False
        self.logger.info("Disconnected from Mock Broker")
        
    async def get_balance(self) -> float:
        """Get mock balance."""
        return self.balance
        
    async def get_price(self, symbol: str) -> float:
        """Get mock price (random walk)."""
        import random
        
        if symbol not in self.prices:
            if 'BTC' in symbol:
                self.prices[symbol] = 50000.0
            elif 'ETH' in symbol:
                self.prices[symbol] = 3000.0
            else:
                self.prices[symbol] = 100.0
                
        # Random walk
        change = random.uniform(-0.002, 0.002)  # ±0.2%
        self.prices[symbol] *= (1 + change)
        
        return self.prices[symbol]
        
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = "market", price: float = None) -> OrderResult:
        """Place mock order."""
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        
        current_price = await self.get_price(symbol)
        fill_price = price if order_type == "limit" else current_price
        
        # Calculate cost
        cost = size if symbol.endswith('USDT') or symbol.endswith('USD') else size * fill_price
        
        if side.lower() == "buy":
            if cost > self.balance:
                raise ValueError("Insufficient balance")
            self.balance -= cost
        else:
            # For sells, add to balance
            self.balance += cost
            
        self.logger.info(f"Mock order {order_id}: {side} {size} {symbol} @ {fill_price}")
        
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            fill_price=fill_price,
            status="filled",
            timestamp=datetime.now()
        )
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel mock order."""
        return True
        
    async def get_positions(self) -> List[Dict]:
        """Get mock positions."""
        return list(self.positions.values())


class BrokerManager:
    """Manages multiple broker connections with failover."""
    
    def __init__(self):
        self.brokers: List[BrokerInterface] = []
        self.primary_broker: Optional[BrokerInterface] = None
        self.logger = logging.getLogger(__name__)
        
    def add_broker(self, broker: BrokerInterface, is_primary: bool = False):
        """Add a broker to the manager."""
        self.brokers.append(broker)
        if is_primary or not self.primary_broker:
            self.primary_broker = broker
            
    async def connect_all(self):
        """Connect to all brokers."""
        for broker in self.brokers:
            try:
                await broker.connect()
            except Exception as e:
                self.logger.error(f"Failed to connect broker {broker.__class__.__name__}: {e}")
                
    async def disconnect_all(self):
        """Disconnect from all brokers."""
        for broker in self.brokers:
            try:
                await broker.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting {broker.__class__.__name__}: {e}")
                
    async def get_active_broker(self) -> Optional[BrokerInterface]:
        """Get the first connected broker."""
        if self.primary_broker and self.primary_broker.is_connected:
            return self.primary_broker
            
        for broker in self.brokers:
            if broker.is_connected:
                return broker
                
        return None
        
    async def place_order_with_failover(self, symbol: str, side: str, size: float, 
                                       order_type: str = "market", price: float = None) -> OrderResult:
        """Place order with automatic failover."""
        broker = await self.get_active_broker()
        if not broker:
            raise RuntimeError("No active brokers available")
            
        return await broker.place_order(symbol, side, size, order_type, price)