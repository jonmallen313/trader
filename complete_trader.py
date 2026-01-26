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
        self.balance = 100.0
        self.positions = []
        self.trades = []
        self.running = False
        
        # Candle data for charts (1-second bars)
        self.candles = {
            'AAPL': deque(maxlen=300),   # 5 min of 1s candles
            'TSLA': deque(maxlen=300),
            'NVDA': deque(maxlen=300),
            'MSFT': deque(maxlen=300),
            'GOOGL': deque(maxlen=300)
        }
        
        # Price tracking
        self.last_prices = {}
        self.price_history = {s: deque(maxlen=60) for s in self.candles.keys()}
        
        # Trading params
        self.symbols = list(self.candles.keys())
        self.position_size_pct = 0.10  # 10% per trade (more positions)
        self.max_positions = 15
        self.trade_interval = 5  # Trade every 5 seconds
        
    async def start(self):
        """Start aggressive trading."""
        logger.info("🚀 STARTING AGGRESSIVE TRADER")
        self.running = True
        
        await asyncio.gather(
            self._price_feed(),
            self._candle_builder(),
            self._trading_engine(),
            self._position_monitor(),
            return_exceptions=True
        )
    
    async def _price_feed(self):
        """Fetch REAL-TIME stock prices from Alpaca API."""
        # Get API keys from environment
        api_key = os.getenv('ALPACA_API_KEY', '')
        api_secret = os.getenv('ALPACA_API_SECRET', '')
        
        if not api_key or not api_secret:
            logger.error("❌ NO ALPACA API KEYS - Cannot fetch real market data")
            logger.error("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
            self.running = False
            return
        
        # Alpaca market data API (separate from trading API)
        data_url = 'https://data.alpaca.markets'
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        logger.info("📡 Connecting to Alpaca for REAL market data...")
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    for symbol in self.symbols:
                        # Get latest quote (bid/ask/last price)
                        url = f'{data_url}/v2/stocks/{symbol}/quotes/latest'
                        async with session.get(url, headers=headers) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                quote = data['quote']
                                # Use bid-ask midpoint for most accurate price
                                price = (float(quote['ap']) + float(quote['bp'])) / 2
                                
                                self.last_prices[symbol] = price
                                self.price_history[symbol].append({
                                    'price': price,
                                    'time': datetime.now().isoformat(),
                                    'bid': float(quote['bp']),
                                    'ask': float(quote['ap'])
                                })
                                
                                state['market_prices'][symbol] = {
                                    'price': price,
                                    'timestamp': datetime.now().isoformat()
                                }
                            else:
                                error = await resp.text()
                                logger.warning(f"Failed to fetch {symbol}: {resp.status} - {error}")
                
                await self._broadcast()
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Price feed error: {e}")
                await asyncio.sleep(5)
    
    async def _candle_builder(self):
        """Build 1-second candlesticks."""
        await asyncio.sleep(2)  # Wait for initial prices
        
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
                
                await self._broadcast()
                await asyncio.sleep(1)  # Build every 1 second
                
            except Exception as e:
                logger.error(f"Candle builder error: {e}")
                await asyncio.sleep(1)
    
    async def _trading_engine(self):
        """AGGRESSIVE trading - actually executes."""
        await asyncio.sleep(5)  # Wait for data
        
        last_trade_time = {s: datetime.now() - timedelta(seconds=60) for s in self.symbols}
        
        while self.running:
            try:
                # Check if we can trade
                if len(self.positions) >= self.max_positions:
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
                        await self._execute_trade(symbol, signal)
                        last_trade_time[symbol] = datetime.now()
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Trading engine error: {e}")
                await asyncio.sleep(2)
    
    def _get_signal(self, symbol: str) -> dict:
        """Generate trading signal - IMPROVED STRATEGY."""
        if len(self.price_history[symbol]) < 30:
            return None
        
        history = list(self.price_history[symbol])
        prices = [p['price'] for p in history]
        
        # Multiple timeframe analysis
        recent_5 = prices[-5:]
        recent_10 = prices[-10:]
        older_20 = prices[-30:-10]
        
        avg_5 = sum(recent_5) / len(recent_5)
        avg_10 = sum(recent_10) / len(recent_10)
        avg_20 = sum(older_20) / len(older_20)
        
        # Trend alignment: short MA > mid MA > long MA
        short_trend = (avg_5 - avg_10) / avg_10
        mid_trend = (avg_10 - avg_20) / avg_20
        
        # Calculate volatility
        price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = (sum(abs(c) for c in price_changes[-10:]) / 10) if len(price_changes) >= 10 else 0.001
        
        # Strong directional move: both trends aligned and strong
        if short_trend > 0.0008 and mid_trend > 0.0005:  # Bullish
            confidence = min(short_trend * 50 + mid_trend * 30, 0.95)
            
            # Filter: only trade if volatility is reasonable
            if volatility < 0.003 and confidence > 0.5:  # Low volatility + good signal
                return {
                    'side': 'long',
                    'confidence': confidence,
                    'entry_price': prices[-1]
                }
        
        elif short_trend < -0.0008 and mid_trend < -0.0005:  # Bearish
            confidence = min(abs(short_trend) * 50 + abs(mid_trend) * 30, 0.95)
            
            if volatility < 0.003 and confidence > 0.5:
                return {
                    'side': 'short',
                    'confidence': confidence,
                    'entry_price': prices[-1]
                }
        
        return None
    
    async def _execute_trade(self, symbol: str, signal: dict):
        """Execute PAPER TRADE via Alpaca API."""
        try:
            position_value = self.balance * self.position_size_pct
            entry_price = signal['entry_price']
            
            # BETTER TP/SL ratios - wider TP, tighter SL
            tp_pct = 0.015  # 1.5% TP (was 1%)
            sl_pct = 0.008  # 0.8% SL (was 0.5%)
            
            # Adjust based on confidence
            if signal['confidence'] > 0.7:
                tp_pct = 0.02   # 2% for high confidence
                sl_pct = 0.01   # 1% SL
            
            if signal['side'] == 'long':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
            
            shares = int(position_value / entry_price)
            
            # Submit PAPER ORDER to Alpaca
            api_key = os.getenv('ALPACA_API_KEY', '')
            api_secret = os.getenv('ALPACA_API_SECRET', '')
            alpaca_order_id = None
            
            if api_key and api_secret:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = 'https://paper-api.alpaca.markets/v2/orders'
                        headers = {
                            'APCA-API-KEY-ID': api_key,
                            'APCA-API-SECRET-KEY': api_secret,
                            'Content-Type': 'application/json'
                        }
                        order_data = {
                            'symbol': symbol,
                            'qty': shares,
                            'side': 'buy' if signal['side'] == 'long' else 'sell',
                            'type': 'market',
                            'time_in_force': 'day'
                        }
                        
                        async with session.post(url, headers=headers, json=order_data) as resp:
                            if resp.status == 200:
                                order = await resp.json()
                                alpaca_order_id = order['id']
                                logger.info(f"📄 PAPER ORDER via Alpaca: {alpaca_order_id}")
                            else:
                                logger.warning(f"⚠️ Alpaca order failed: {resp.status} - falling back to local simulation")
                except Exception as e:
                    logger.warning(f"⚠️ Alpaca API error: {e} - using local simulation")
            else:
                logger.info("📝 Local paper trade simulation (no Alpaca keys)")
            
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
        await asyncio.sleep(10)
        
        while self.running:
            try:
                for position in list(self.positions):
                    symbol = position['symbol']
                    current_price = self.last_prices.get(symbol)
                    
                    if not current_price:
                        continue
                    
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
                'reason': reason,
                'closed_at': datetime.now().isoformat()
            }
            
            self.trades.append(trade)
            state['trades'] = self.trades[-50:]  # Keep last 50
            
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
    return {"status": "ok"}


@app.on_event("startup")
async def startup():
    asyncio.create_task(trader.start())


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>willAIm</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
                    
                    const box = document.createElement('div');
                    box.className = 'chart-box';
                    box.id = `chart-${symbol}`;
                    box.innerHTML = `<h3>${symbol} <span style="color: ${dirColor}; border: 1px solid ${dirColor}; padding: 2px 8px; border-radius: 3px; font-size: 0.8em;">${direction}</span></h3>`;
                    
                    const canvas = document.createElement('canvas');
                    box.appendChild(canvas);
                    chartsContainer.appendChild(box);
                    
                    chartCanvases[symbol] = canvas;
                    
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
                                legend: { labels: { color: '#0f0' } }
                            },
                            scales: {
                                x: { ticks: { color: '#0f0' }, grid: { color: '#003300' } },
                                y: { ticks: { color: '#0f0' }, grid: { color: '#003300' } }
                            }
                        }
                    });
                }
                
                // Update chart data
                if (data.candles && data.candles[symbol]) {
                    const candles = data.candles[symbol].slice(-60); // Last 60 seconds
                    charts[symbol].data.labels = candles.map((c, i) => i);
                    charts[symbol].data.datasets[0].data = candles.map(c => c.close);
                    charts[symbol].update('none');
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
                    <tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Current</th><th>P&L</th><th>TP</th><th>SL</th></tr>
                    ${positions.map(p => {
                        const current = (window.lastPrices && window.lastPrices[p.symbol]) || p.entry_price;
                        const pnl = p.side === 'long' 
                            ? (current - p.entry_price) * p.shares
                            : (p.entry_price - current) * p.shares;
                        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                        return `
                            <tr>
                                <td>${p.symbol}</td>
                                <td>${p.side.toUpperCase()}</td>
                                <td>$${p.entry_price.toFixed(2)}</td>
                                <td>$${current.toFixed(2)}</td>
                                <td class="${pnlClass}">$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</td>
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
            
            const recent = trades.slice(-10).reverse();
            container.innerHTML = `
                <table>
                    <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr>
                    ${recent.map(t => {
                        const time = new Date(t.closed_at).toLocaleTimeString();
                        const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
                        return `
                            <tr>
                                <td>${time}</td>
                                <td>${t.symbol}</td>
                                <td>${t.side.toUpperCase()}</td>
                                <td>$${t.entry_price.toFixed(2)}</td>
                                <td>$${t.close_price.toFixed(2)}</td>
                                <td class="${pnlClass}">$${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}</td>
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
