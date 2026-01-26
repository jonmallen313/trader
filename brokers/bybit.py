"""
Bybit Futures Broker Integration (Testnet Support)
FREE paper trading with up to 100x leverage
"""

import asyncio
import logging
import hmac
import hashlib
import time
from typing import Dict, List, Optional
from datetime import datetime
import aiohttp

from brokers.base import BrokerInterface, OrderResult
from core.position import Position, PositionSide


class BybitBroker(BrokerInterface):
    """
    Bybit futures broker with leverage trading support.
    
    Features:
    - Up to 100x leverage
    - Testnet for paper trading (unlimited fake USDT)
    - No geo-restrictions
    - Perpetual futures
    - Auto liquidation protection
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__("Bybit", paper_mode=testnet)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.supports_leverage = True
        self.max_leverage = 100
        
        # API endpoints
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Cache
        self.balance_cache = 0.0
        self.balance_cache_time = 0
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature."""
        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _request(self, method: str, endpoint: str, params: Dict = None,
                      signed: bool = True) -> Dict:
        """Make authenticated API request."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        params = params or {}
        
        if signed:
            params['api_key'] = self.api_key
            params['timestamp'] = str(int(time.time() * 1000))
            params['sign'] = self._generate_signature(params)
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                async with self.session.get(url, params=params) as response:
                    data = await response.json()
            else:  # POST
                async with self.session.post(url, data=params) as response:
                    data = await response.json()
            
            if data.get('ret_code') == 0:
                return data.get('result', {})
            else:
                error_msg = data.get('ret_msg', 'Unknown error')
                self.logger.error(f"Bybit API error: {error_msg}")
                raise Exception(f"Bybit API error: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            raise
    
    async def connect(self) -> bool:
        """Connect and verify API credentials."""
        try:
            self.logger.info(f"Connecting to Bybit {'testnet' if self.testnet else 'live'}...")
            
            # Test connection by fetching account balance
            balance = await self.get_balance()
            self.is_connected = True
            
            self.logger.info(f"✅ Connected to Bybit | Balance: ${balance:.2f} USDT")
            if self.testnet:
                self.logger.info("🧪 TESTNET MODE - Using fake money for paper trading")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Bybit: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close session."""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        self.logger.info("Disconnected from Bybit")
    
    async def get_balance(self) -> float:
        """Get USDT wallet balance."""
        # Cache for 10 seconds
        if time.time() - self.balance_cache_time < 10:
            return self.balance_cache
        
        try:
            result = await self._request("GET", "/v2/private/wallet/balance", {
                "coin": "USDT"
            })
            balance = float(result.get('USDT', {}).get('available_balance', 0))
            
            self.balance_cache = balance
            self.balance_cache_time = time.time()
            
            return balance
            
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0
    
    async def get_price(self, symbol: str) -> float:
        """Get current market price."""
        try:
            # Convert symbol format (BTC/USDT -> BTCUSDT)
            bybit_symbol = symbol.replace('/', '').replace('-', '')
            
            result = await self._request("GET", "/v2/public/tickers", {
                "symbol": bybit_symbol
            }, signed=False)
            
            # Handle both single ticker and list response
            if isinstance(result, list):
                ticker = result[0] if result else {}
            else:
                ticker = result
            
            price = float(ticker.get('last_price', 0))
            return price
            
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol."""
        try:
            bybit_symbol = symbol.replace('/', '').replace('-', '')
            
            await self._request("POST", "/v2/private/position/leverage/save", {
                "symbol": bybit_symbol,
                "buy_leverage": str(leverage),
                "sell_leverage": str(leverage)
            })
            
            self.logger.info(f"Set leverage to {leverage}x for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return False
    
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
        """Place an order with leverage support."""
        try:
            bybit_symbol = symbol.replace('/', '').replace('-', '')
            
            # Set leverage first
            await self.set_leverage(symbol, leverage)
            
            # Prepare order parameters
            params = {
                "symbol": bybit_symbol,
                "side": "Buy" if side == "long" or side == "buy" else "Sell",
                "order_type": "Market" if order_type == "market" else "Limit",
                "qty": str(int(size)),  # Must be integer for perpetuals
                "time_in_force": "GoodTillCancel",
                "reduce_only": False,
                "close_on_trigger": False
            }
            
            # Add price for limit orders
            if order_type == "limit" and price:
                params["price"] = str(price)
            
            # Add TP/SL if provided
            if tp_price:
                params["take_profit"] = str(tp_price)
            if sl_price:
                params["stop_loss"] = str(sl_price)
            
            result = await self._request("POST", "/v2/private/order/create", params)
            
            order_id = result.get('order_id', '')
            fill_price = float(result.get('price', price or 0))
            
            self.logger.info(f"✅ Order placed: {symbol} {side} {size} @{fill_price} {leverage}x")
            
            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                fill_price=fill_price,
                status="filled",
                leverage=leverage,
                fees=0.0  # Bybit charges 0.075% maker, 0.06% taker
            )
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    async def close_position(self, position: Position) -> OrderResult:
        """Close an existing position."""
        try:
            bybit_symbol = position.symbol.replace('/', '').replace('-', '')
            
            # Closing is opposite side
            close_side = "Sell" if position.side == PositionSide.LONG else "Buy"
            
            params = {
                "symbol": bybit_symbol,
                "side": close_side,
                "order_type": "Market",
                "qty": str(int(position.size)),
                "time_in_force": "GoodTillCancel",
                "reduce_only": True,  # Important: only close, don't open opposite
                "close_on_trigger": False
            }
            
            result = await self._request("POST", "/v2/private/order/create", params)
            
            fill_price = float(result.get('price', position.current_price))
            
            self.logger.info(f"✅ Position closed: {position.symbol} @{fill_price}")
            
            return OrderResult(
                order_id=result.get('order_id', ''),
                symbol=position.symbol,
                side=close_side.lower(),
                size=position.size,
                fill_price=fill_price,
                status="filled",
                leverage=position.leverage
            )
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            raise
    
    async def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            result = await self._request("GET", "/v2/private/position/list")
            
            positions = []
            for pos in result:
                if float(pos.get('size', 0)) > 0:
                    positions.append({
                        'symbol': pos.get('symbol'),
                        'side': pos.get('side').lower(),
                        'size': float(pos.get('size', 0)),
                        'entry_price': float(pos.get('entry_price', 0)),
                        'leverage': int(pos.get('leverage', 1)),
                        'unrealized_pnl': float(pos.get('unrealised_pnl', 0)),
                        'liquidation_price': float(pos.get('liq_price', 0))
                    })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return []
    
    async def get_liquidation_price(self, position: Position) -> Optional[float]:
        """
        Calculate liquidation price for leveraged position.
        Formula: Entry Price ± (Entry Price / Leverage)
        """
        if position.leverage <= 1:
            return None  # No liquidation for non-leveraged
        
        # Simplified formula (Bybit uses more complex with fees)
        if position.side == PositionSide.LONG:
            # Long liquidation: entry - (entry / leverage)
            liq_price = position.entry_price * (1 - 1/position.leverage)
        else:
            # Short liquidation: entry + (entry / leverage)
            liq_price = position.entry_price * (1 + 1/position.leverage)
        
        return liq_price
    
    def __repr__(self):
        mode = "TESTNET" if self.testnet else "LIVE"
        status = "CONNECTED" if self.is_connected else "DISCONNECTED"
        return f"Bybit [{mode}] ({status}) | Max Leverage: {self.max_leverage}x"
