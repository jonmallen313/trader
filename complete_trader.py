"""
COMPLETE AI TRADER - ACTUALLY TRADES
- Aggressive trading strategy that EXECUTES
- 1-second candlestick charts for active trades
- Real-time position tracking
- No more mock bullshit
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import json
from collections import deque
import sqlite3
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import aiohttp
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trader")

# Global state
state = {
    'balance': 100.0,
    'positions': [],
    'trades': [],
    'candles': {},  # {symbol: [candles]}
    'market_prices': {},
    'is_trading': True
}

connections: List[WebSocket] = []


class AggressiveTrader:
    """Trading engine that ACTUALLY TRADES."""
    
    def __init__(self):
        # Load persistent state
        self.db_path = Path('trader_state.db')
        self._init_database()
        
        self.balance = self._load_balance()
        self.positions = []
        self.trades = self._load_trades()
        self.running = False
        
        # Candle data for charts (1-second bars)
        self.candles = {
            'BTC/USD': deque(maxlen=300),   # 5 min of 1s candles
            'ETH/USD': deque(maxlen=300),
            'SOL/USD': deque(maxlen=300),
            'AVAX/USD': deque(maxlen=300),
            'DOGE/USD': deque(maxlen=300)
        }
        
        # Price tracking
        self.last_prices = {}
        self.price_history = {s: deque(maxlen=60) for s in self.candles.keys()}
        
        # Trading params
        self.symbols = list(self.candles.keys())
        self.position_size_pct = 0.10  # 10% per trade to allow more positions
        self.max_positions = 30  # Increased from 15 to maximize profit
        self.trade_interval = 2  # Trade every 2 seconds - very aggressive
        
    def _init_database(self):
        """Initialize SQLite database for persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Balance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS balance (
                id INTEGER PRIMARY KEY,
                amount REAL,
                updated_at TEXT
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                close_price REAL,
                shares REAL,
                value REAL,
                pnl REAL,
                reason TEXT,
                opened_at TEXT,
                closed_at TEXT,
                alpaca_order_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("💾 Database initialized")
    
    def _load_balance(self) -> float:
        """Load balance from database or start with $100."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT amount FROM balance ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        
        if row:
            balance = row[0]
            logger.info(f"💰 Loaded balance: ${balance:.2f}")
            return balance
        else:
            logger.info("💰 Starting with $100.00")
            return 100.0
    
    def _save_balance(self):
        """Save current balance to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO balance (amount, updated_at) VALUES (?, ?)',
                      (self.balance, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def _load_trades(self) -> list:
        """Load recent trades from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT symbol, side, entry_price, close_price, shares, value, pnl, reason, opened_at, closed_at
            FROM trades ORDER BY id DESC LIMIT 50
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        trades = []
        for row in rows:
            trades.append({
                'symbol': row[0],
                'side': row[1],
                'entry_price': row[2],
                'close_price': row[3],
                'shares': row[4],
                'value': row[5],
                'pnl': row[6],
                'reason': row[7],
                'opened_at': row[8],
                'closed_at': row[9]
            })
        
        logger.info(f"📊 Loaded {len(trades)} historical trades")
        return trades
    
    def _save_trade(self, trade: dict):
        """Save completed trade to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, side, entry_price, close_price, shares, value, pnl, reason, opened_at, closed_at, alpaca_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['symbol'],
            trade['side'],
            trade['entry_price'],
            trade.get('close_price'),
            trade['shares'],
            trade['value'],
            trade.get('pnl', 0),
            trade.get('reason', ''),
            trade['opened_at'],
            trade.get('closed_at'),
            trade.get('alpaca_order_id')
        ))
        conn.commit()
        conn.close()
    
    async def start(self):
        """Start aggressive trading."""
        logger.info("🚀 STARTING AGGRESSIVE TRADER")
        self.running = True
        
        logger.info("🔧 Starting 4 background tasks...")
        try:
            await asyncio.gather(
                self._price_feed(),
                self._candle_builder(),
                self._trading_engine(),
                self._position_monitor(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"💥 CRITICAL ERROR in trader.start(): {e}")
            logger.exception(e)
    
    async def _price_feed(self):
        """Stream REAL-TIME crypto prices from Kraken WebSocket (no geo-restrictions)."""
        import json
        
        logger.info("📡 Connecting to Kraken WebSocket for real-time data...")
        
        # Kraken WebSocket - works globally, no API key needed
        ws_url = 'wss://ws.kraken.com'
        
        # Map our symbols to Kraken pairs
        kraken_pairs = {
            'BTC/USD': 'XBT/USD',
            'ETH/USD': 'ETH/USD',
            'SOL/USD': 'SOL/USD',
            'AVAX/USD': 'AVAX/USD',
            'DOGE/USD': 'DOGE/USD'
        }
        
        pair_names = list(kraken_pairs.values())
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        logger.info("✅ Connected to Kraken WebSocket")
                        
                        # Subscribe to ticker for all pairs
                        subscribe_msg = {
                            "event": "subscribe",
                            "pair": pair_names,
                            "subscription": {"name": "ticker"}
                        }
                        await ws.send_str(json.dumps(subscribe_msg))
                        logger.info(f"📊 Subscribed to {len(pair_names)} pairs")
                        
                        tick_count = 0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                
                                # Ticker updates are arrays
                                if isinstance(data, list) and len(data) >= 4:
                                    ticker = data[1]
                                    pair_name = data[3]
                                    
                                    # Find our symbol
                                    our_symbol = None
                                    for symbol, kraken_pair in kraken_pairs.items():
                                        if kraken_pair == pair_name:
                                            our_symbol = symbol
                                            break
                                    
                                    if our_symbol and 'c' in ticker:
                                        # 'c' is last close price [price, volume]
                                        price = float(ticker['c'][0])
                                        
                                        self.last_prices[our_symbol] = price
                                        self.price_history[our_symbol].append({
                                            'price': price,
                                            'time': datetime.now().isoformat()
                                        })
                                        
                                        state['market_prices'][our_symbol] = {
                                            'price': price,
                                            'timestamp': datetime.now().isoformat()
                                        }
                                        
                                        tick_count += 1
                                        if tick_count % 50 == 0:
                                            logger.info(f"💰 {our_symbol}: ${price:,.2f} | {tick_count} ticks")
                                        
                                        await self._broadcast()
                            
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {ws.exception()}")
                                break
                
            except Exception as e:
                logger.error(f"💥 Kraken WebSocket error: {e}")
                logger.info("🔄 Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
    
    async def _candle_builder(self):
        """Build 1-second candlesticks."""
        logger.info("🕯️ Candle builder starting in 2 seconds...")
        await asyncio.sleep(2)  # Wait for initial prices
        
        candle_count = 0
        while self.running:
            try:
                for symbol in self.symbols:
                    if symbol not in self.price_history or len(self.price_history[symbol]) == 0:
                        continue
                    
                    # Get last second of prices
                    recent = list(self.price_history[symbol])[-10:]  # Last 10 ticks
                    if not recent:
                        continue
                    
                    prices = [p['price'] for p in recent]
                    
                    candle = {
                        'time': datetime.now().isoformat(),
                        'open': prices[0],
                        'high': max(prices),
                        'low': min(prices),
                        'close': prices[-1],
                        'volume': len(prices)
                    }
                    
                    self.candles[symbol].append(candle)
                    state['candles'][symbol] = list(self.candles[symbol])
                    
                    candle_count += 1
                    if candle_count % 25 == 0:  # Log every 25 candles
                        logger.info(f"🕯️ Built {candle_count} candles | {symbol}: ${prices[-1]:.2f}")
                
                await self._broadcast()
                await asyncio.sleep(1)  # Build every 1 second
                
            except Exception as e:
                logger.error(f"Candle builder error: {e}")
                await asyncio.sleep(1)
    
    async def _trading_engine(self):
        """AGGRESSIVE trading - actually executes."""
        logger.info("🎯 Trading engine starting in 5 seconds...")
        await asyncio.sleep(5)  # Wait for data
        
        last_trade_time = {s: datetime.now() - timedelta(seconds=60) for s in self.symbols}
        scan_count = 0
        
        while self.running:
            try:
                scan_count += 1
                
                # Check if we can trade
                if len(self.positions) >= self.max_positions:
                    if scan_count % 10 == 0:
                        logger.info(f"⏸️ Max positions reached ({self.max_positions})")
                    await asyncio.sleep(2)
                    continue
                
                if self.balance < 20:  # Need at least $20
                    logger.warning("Balance too low to trade")
                    await asyncio.sleep(10)
                    continue
                
                # Scan for momentum
                for symbol in self.symbols:
                    if len(self.positions) >= self.max_positions:
                        break
                    
                    # Cooldown check
                    if (datetime.now() - last_trade_time[symbol]).total_seconds() < self.trade_interval:
                        continue
                    
                    # Skip if already have position in this symbol
                    if any(p['symbol'] == symbol for p in self.positions):
                        continue
                    
                    # Get momentum signal
                    signal = self._get_signal(symbol)
                    
                    if signal:
                        logger.info(f"📡 SIGNAL DETECTED: {symbol} {signal['side'].upper()} | Confidence: {signal['confidence']:.0%}")
                        await self._execute_trade(symbol, signal)
                        last_trade_time[symbol] = datetime.now()
                    elif scan_count % 20 == 0:  # Log occasionally when no signal
                        history_len = len(self.price_history.get(symbol, []))
                        logger.info(f"🔍 Scanning {symbol} | History: {history_len} ticks | No signal")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Trading engine error: {e}")
                logger.exception(e)
                await asyncio.sleep(2)
    
    def _get_signal(self, symbol: str) -> dict:
        """Generate trading signal - HYPER AGGRESSIVE SCALPING."""
        if len(self.price_history[symbol]) < 10:  # Only need 10 ticks
            return None
        
        history = list(self.price_history[symbol])
        prices = [p['price'] for p in history]
        
        # Multiple timeframe analysis
        recent_5 = prices[-5:]
        recent_10 = prices[-10:]
        older_15 = prices[-15:-5]  # Adjusted window
        
        avg_5 = sum(recent_5) / len(recent_5)
        avg_10 = sum(recent_10) / len(recent_10)
        avg_15 = sum(older_15) / len(older_15)
        
        # Trend alignment: short MA > mid MA > long MA
        short_trend = (avg_5 - avg_10) / avg_10
        mid_trend = (avg_10 - avg_15) / avg_15
        
        # Calculate volatility
        price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = (sum(abs(c) for c in price_changes[-10:]) / 10) if len(price_changes) >= 10 else 0.001
        
        # HYPER AGGRESSIVE - trade on tiny microtrends
        if short_trend > 0.00008 and mid_trend > 0.00005:  # Bullish (1000x more sensitive)
            confidence = min(short_trend * 5000 + mid_trend * 3000, 0.95)
            logger.info(f"🎯 LONG SIGNAL: {symbol} | Short: {short_trend*100:.6f}% | Mid: {mid_trend*100:.6f}% | Conf: {max(confidence, 0.6)*100:.1f}%")
            
            # Always trade on any signal (removed volatility filter)
            return {
                'side': 'long',
                'confidence': max(confidence, 0.6),
                'entry_price': prices[-1]
            }
        
        elif short_trend < -0.00008 and mid_trend < -0.00005:  # Bearish (1000x more sensitive)
            confidence = min(abs(short_trend) * 5000 + abs(mid_trend) * 3000, 0.95)
            logger.info(f"🎯 SHORT SIGNAL: {symbol} | Short: {short_trend*100:.6f}% | Mid: {mid_trend*100:.6f}% | Conf: {max(confidence, 0.6)*100:.1f}%")
            
            return {
                'side': 'short',
                'confidence': max(confidence, 0.6),
                'entry_price': prices[-1]
            }
        
        return None
    
    async def _execute_trade(self, symbol: str, signal: dict):
        """Execute PAPER TRADE via Alpaca API."""
        try:
            position_value = self.balance * self.position_size_pct
            entry_price = signal['entry_price']
            
            # SCALPING TP/SL - very tight for tiny movements
            tp_pct = 0.005  # 0.5% TP
            sl_pct = 0.003  # 0.3% SL
            
            # Adjust based on confidence
            if signal['confidence'] > 0.7:
                tp_pct = 0.008   # 0.8% for high confidence
                sl_pct = 0.004   # 0.4% SL
            
            if signal['side'] == 'long':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
            
            # FRACTIONAL CRYPTO TRADING - Alpaca supports fractions
            shares = round(position_value / entry_price, 8)  # Up to 8 decimals
            
            if shares < 0.00000001:  # Minimum fractional amount
                logger.warning(f"⚠️ Position too small: {shares} shares of {symbol}")
                return
            
            # Submit PAPER ORDER to Alpaca
            api_key = os.getenv('ALPACA_API_KEY', '')
            api_secret = os.getenv('ALPACA_API_SECRET', '')
            alpaca_order_id = None
            
            if api_key and api_secret:
                try:
                    # Map Kraken symbols to Alpaca crypto format
                    alpaca_symbol = symbol.replace('/', '')  # BTC/USD -> BTCUSD
                    
                    async with aiohttp.ClientSession() as session:
                        url = 'https://paper-api.alpaca.markets/v2/orders'
                        headers = {
                            'APCA-API-KEY-ID': api_key,
                            'APCA-API-SECRET-KEY': api_secret,
                            'Content-Type': 'application/json'
                        }
                        order_data = {
                            'symbol': alpaca_symbol,
                            'qty': str(shares),  # String for fractional crypto
                            'side': 'buy' if signal['side'] == 'long' else 'sell',
                            'type': 'market',
                            'time_in_force': 'gtc'  # Good till cancelled for crypto
                        }
                        
                        logger.info(f"📤 Submitting to Alpaca Paper API: {alpaca_symbol} {signal['side'].upper()} {shares} shares @ ${entry_price:.2f}")
                        
                        async with session.post(url, headers=headers, json=order_data) as resp:
                            resp_text = await resp.text()
                            if resp.status == 200:
                                order = json.loads(resp_text)
                                alpaca_order_id = order['id']
                                logger.info(f"✅ ALPACA PAPER TRADE EXECUTED! Order ID: {alpaca_order_id}")
                            else:
                                logger.error(f"❌ Alpaca API rejected: {resp.status} - {resp_text}")
                                logger.warning(f"⚠️ Falling back to local simulation")
                except Exception as e:
                    logger.error(f"💥 Alpaca API error: {e}")
                    logger.warning(f"⚠️ Using local simulation")
            else:
                logger.warning("⚠️ No Alpaca keys found - using local paper trade simulation only")
            
            position = {
                'id': f"{symbol}_{int(datetime.now().timestamp())}",
                'symbol': symbol,
                'side': signal['side'],
                'entry_price': entry_price,
                'shares': shares,
                'value': position_value,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'opened_at': datetime.now().isoformat(),
                'confidence': signal['confidence'],
                'alpaca_order_id': alpaca_order_id
            }
            
            self.positions.append(position)
            state['positions'] = self.positions
            
            logger.info(f"🎯 OPENED {symbol} {signal['side'].upper()} | ${position_value:.2f} @ ${entry_price:.2f} | Conf: {signal['confidence']:.0%}")
            
            await self._broadcast()
            
        except Exception as e:
            logger.error(f"Execute trade error: {e}")
    
    async def _position_monitor(self):
        """Monitor and close positions."""
        logger.info("👁️ Position monitor starting in 10 seconds...")
        await asyncio.sleep(10)
        
        while self.running:
            try:
                for position in list(self.positions):
                    symbol = position['symbol']
                    current_price = self.last_prices.get(symbol)
                    
                    if not current_price:
                        continue
                    
                    # UPDATE LIVE P&L
                    position['current_price'] = current_price
                    if position['side'] == 'long':
                        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    position['pnl'] = position['value'] * pnl_pct
                    position['pnl_pct'] = pnl_pct * 100
                    
                    should_close = False
                    reason = ""
                    
                    # Check TP/SL
                    if position['side'] == 'long':
                        if current_price >= position['tp_price']:
                            should_close, reason = True, "TP"
                        elif current_price <= position['sl_price']:
                            should_close, reason = True, "SL"
                    else:
                        if current_price <= position['tp_price']:
                            should_close, reason = True, "TP"
                        elif current_price >= position['sl_price']:
                            should_close, reason = True, "SL"
                    
                    # Time limit (2 min max)
                    age = (datetime.now() - datetime.fromisoformat(position['opened_at'])).total_seconds()
                    if age > 120:
                        should_close, reason = True, "TIME"
                    
                    if should_close:
                        await self._close_position(position, current_price, reason)
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(1)
    
    async def _close_position(self, position, close_price, reason):
        """Close position and calculate P&L."""
        try:
            entry = position['entry_price']
            
            if position['side'] == 'long':
                pnl_pct = (close_price - entry) / entry
            else:
                pnl_pct = (entry - close_price) / entry
            
            pnl = position['value'] * pnl_pct
            
            self.balance += pnl
            state['balance'] = self.balance
            
            trade = {
                **position,
                'close_price': close_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'reason': reason,
                'closed_at': datetime.now().isoformat()
            }
            
            self.trades.insert(0, trade)  # Add to front
            state['trades'] = self.trades[:50]  # Keep last 50
            
            # Save to database
            self._save_trade(trade)
            self._save_balance()
            
            self.positions.remove(position)
            state['positions'] = self.positions
            
            emoji = "✅" if pnl > 0 else "❌"
            logger.info(f"{emoji} CLOSED {position['symbol']} {position['side']} | ${pnl:+.2f} | {reason}")
            
            await self._broadcast()
            
        except Exception as e:
            logger.error(f"Close error: {e}")
    
    async def _broadcast(self):
        """Send updates to all WebSocket clients."""
        if not connections:
            return
        
        message = json.dumps({'type': 'update', 'data': state})
        
        dead = []
        for ws in connections:
            try:
                await ws.send_text(message)
            except:
                dead.append(ws)
        
        for ws in dead:
            connections.remove(ws)


trader = AggressiveTrader()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    
    try:
        await websocket.send_text(json.dumps({'type': 'init', 'data': state}))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connections.remove(websocket)


@app.get("/health")
async def health():
    logger.info("🏥 Health check called")
    return {"status": "ok", "trading": trader.running, "positions": len(trader.positions)}


@app.on_event("startup")
async def startup():
    asyncio.create_task(trader.start())


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>willAIm</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.2.1"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #000; color: #0f0; padding: 20px; }
        .container { max-width: 1800px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 20px; border: 2px solid #0f0; padding: 20px; }
        .header h1 { font-size: 2em; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }
        .stat { border: 1px solid #0f0; padding: 15px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; }
        .charts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }
        .chart-box { border: 1px solid #0f0; padding: 10px; height: 250px; }
        .positions, .trades { border: 1px solid #0f0; padding: 15px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; border: 1px solid #0f0; text-align: left; }
        .positive { color: #0f0; }
        .negative { color: #f00; }
        .pulse { animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>will<span style="color: #0ff;">AI</span>m</h1>
            <div class="pulse" id="status">● TRADING ACTIVE</div>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div>BALANCE</div>
                <div class="stat-value" id="balance">$100.00</div>
            </div>
            <div class="stat">
                <div>P&L</div>
                <div class="stat-value" id="pnl">$0.00</div>
            </div>
            <div class="stat">
                <div>POSITIONS</div>
                <div class="stat-value" id="pos-count">0</div>
            </div>
            <div class="stat">
                <div>WIN RATE</div>
                <div class="stat-value" id="win-rate">0%</div>
            </div>
        </div>
        
        <div class="charts" id="charts"></div>
        
        <div class="positions">
            <h2>ACTIVE POSITIONS</h2>
            <div id="positions">No positions</div>
        </div>
        
        <div class="trades">
            <h2>RECENT TRADES (LAST 10)</h2>
            <div id="trades">No trades yet</div>
        </div>
    </div>
    
    <script>
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
        
        const charts = {};
        const chartCanvases = {};
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'init' || msg.type === 'update') {
                update(msg.data);
            }
        };
        
        function update(data) {
            // Stats
            document.getElementById('balance').textContent = `$${data.balance.toFixed(2)}`;
            const pnl = data.balance - 100;
            const pnlEl = document.getElementById('pnl');
            pnlEl.textContent = `$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`;
            pnlEl.className = `stat-value ${pnl >= 0 ? 'positive' : 'negative'}`;
            
            document.getElementById('pos-count').textContent = (data.positions || []).length;
            
            const trades = data.trades || [];
            const wins = trades.filter(t => t.pnl > 0).length;
            const winRate = trades.length > 0 ? (wins / trades.length * 100) : 0;
            document.getElementById('win-rate').textContent = `${winRate.toFixed(0)}%`;
            
            // Update charts for active positions
            updateCharts(data);
            
            // Positions table
            updatePositions(data.positions || []);
            
            // Trades table
            updateTrades(trades);
        }
        
        function updateCharts(data) {
            const chartsContainer = document.getElementById('charts');
            const activeSymbols = new Set((data.positions || []).map(p => p.symbol));
            
            // Create charts for active positions
            activeSymbols.forEach(symbol => {
                if (!charts[symbol]) {
                    const position = (data.positions || []).find(p => p.symbol === symbol);
                    const direction = position ? position.side.toUpperCase() : 'LONG';
                    const dirColor = position && position.side === 'long' ? '#0f0' : '#f0f';
                    const pnl = position ? (position.pnl || 0) : 0;
                    const panelColor = pnl >= 0 ? '#003300' : '#330000';  // Green background for profit, red for loss
                    
                    const box = document.createElement('div');
                    box.className = 'chart-box';
                    box.id = `chart-${symbol}`;
                    box.style.backgroundColor = panelColor;
                    box.innerHTML = `<h3>${symbol} <span style="color: ${dirColor}; border: 1px solid ${dirColor}; padding: 2px 8px; border-radius: 3px; font-size: 0.8em;">${direction}</span> <span style="color: ${pnl >= 0 ? '#0f0' : '#f00'}; font-size: 0.7em;">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</span></h3>`;
                    
                    const canvas = document.createElement('canvas');
                    box.appendChild(canvas);
                    chartsContainer.appendChild(box);
                    
                    chartCanvases[symbol] = canvas;
                    
                    const entryPrice = position ? position.entry_price : 0;
                    
                    charts[symbol] = new Chart(canvas, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: symbol,
                                data: [],
                                borderColor: '#0f0',
                                backgroundColor: 'rgba(0, 255, 0, 0.1)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { labels: { color: '#0f0' } },
                                annotation: {
                                    annotations: {
                                        entryLine: {
                                            type: 'line',
                                            yMin: entryPrice,
                                            yMax: entryPrice,
                                            borderColor: '#ff0',
                                            borderWidth: 2,
                                            borderDash: [5, 5],
                                            label: {
                                                content: `Entry: $${entryPrice.toFixed(2)}`,
                                                enabled: true,
                                                position: 'start',
                                                backgroundColor: 'rgba(255, 255, 0, 0.8)',
                                                color: '#000'
                                            }
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: { ticks: { color: '#0f0' }, grid: { color: '#003300' } },
                                y: { ticks: { color: '#0f0' }, grid: { color: '#003300' } }
                            }
                        }
                    });
                }
                
                // Update chart data and panel color based on P&L
                if (data.candles && data.candles[symbol]) {
                    const candles = data.candles[symbol].slice(-60); // Last 60 seconds
                    charts[symbol].data.labels = candles.map((c, i) => i);
                    charts[symbol].data.datasets[0].data = candles.map(c => c.close);
                    charts[symbol].update('none');
                    
                    // Update panel background based on current P&L
                    const position = (data.positions || []).find(p => p.symbol === symbol);
                    if (position) {
                        const pnl = position.pnl || 0;
                        const panelColor = pnl >= 0 ? '#003300' : '#330000';
                        const chartBox = document.getElementById(`chart-${symbol}`);
                        if (chartBox) {
                            chartBox.style.backgroundColor = panelColor;
                            // Update P&L in header
                            const dirColor = position.side === 'long' ? '#0f0' : '#f0f';
                            chartBox.querySelector('h3').innerHTML = `${symbol} <span style="color: ${dirColor}; border: 1px solid ${dirColor}; padding: 2px 8px; border-radius: 3px; font-size: 0.8em;">${position.side.toUpperCase()}</span> <span style="color: ${pnl >= 0 ? '#0f0' : '#f00'}; font-size: 0.7em;">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</span>`;
                        }
                    }
                }
            });
            
            // Remove charts for closed positions
            Object.keys(charts).forEach(symbol => {
                if (!activeSymbols.has(symbol)) {
                    charts[symbol].destroy();
                    delete charts[symbol];
                    const box = document.getElementById(`chart-${symbol}`);
                    if (box) box.remove();
                }
            });
        }
        
        function updatePositions(positions) {
            const container = document.getElementById('positions');
            if (positions.length === 0) {
                container.innerHTML = 'No positions';
                return;
            }
            
            container.innerHTML = `
                <table>
                    <tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Current</th><th>P&L</th><th>P&L %</th><th>TP</th><th>SL</th></tr>
                    ${positions.map(p => {
                        const pnl = p.pnl || 0;
                        const pnlPct = p.pnl_pct || 0;
                        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                        return `
                            <tr>
                                <td>${p.symbol}</td>
                                <td>${p.side.toUpperCase()}</td>
                                <td>$${p.entry_price.toFixed(2)}</td>
                                <td>$${(p.current_price || p.entry_price).toFixed(2)}</td>
                                <td class="${pnlClass}">$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</td>
                                <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%</td>
                                <td>$${p.tp_price.toFixed(2)}</td>
                                <td>$${p.sl_price.toFixed(2)}</td>
                            </tr>
                        `;
                    }).join('')}
                </table>
            `;
        }
        
        function updateTrades(trades) {
            const container = document.getElementById('trades');
            if (trades.length === 0) {
                container.innerHTML = 'No trades yet';
                return;
            }
            
            const recent = trades.slice(0, 10);  // Already reversed in backend
            container.innerHTML = `
                <table>
                    <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>P&L %</th><th>Reason</th></tr>
                    ${recent.map(t => {
                        const time = new Date(t.closed_at).toLocaleTimeString();
                        const pnl = t.pnl || 0;
                        const pnlPct = t.pnl_pct || 0;
                        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                        return `
                            <tr>
                                <td>${time}</td>
                                <td>${t.symbol}</td>
                                <td>${t.side.toUpperCase()}</td>
                                <td>$${t.entry_price.toFixed(2)}</td>
                                <td>$${t.close_price.toFixed(2)}</td>
                                <td class="${pnlClass}">$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</td>
                                <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%</td>
                                <td>${t.reason}</td>
                            </tr>
                        `;
                    }).join('')}
                </table>
            `;
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
