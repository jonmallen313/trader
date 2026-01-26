"""
RAILWAY DEPLOYMENT - WORKING AI TRADER
- Live market data display
- Real trading logic (actually executes)
- High-tech dashboard
- Bybit integration
"""

import asyncio
import os
import logging
from datetime import datetime
from typing import Dict, List
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from brokers.bybit import BybitBroker
import ccxt.async_support as ccxt


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="AI Trader")

# Global state
trader_state = {
    'balance': 100.0,
    'capital': 100.0,
    'target': 2000.0,
    'positions': [],
    'trades': [],
    'market_data': {},
    'is_trading': False,
    'last_update': None
}

# WebSocket connections for live updates
active_connections: List[WebSocket] = []


class LiveTrader:
    """Trading engine that actually works."""
    
    def __init__(self):
        self.broker = None
        self.exchange = None
        self.running = False
        self.positions = []
        self.balance = 100.0
        
        # Trading params
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.position_size = 0.15  # 15% per trade
        self.max_positions = 5
        self.min_confidence = 0.55
        
    async def start(self):
        """Start trading."""
        logger.info("🚀 Starting Live Trader")
        
        # Setup
        await self._setup_broker()
        
        # Update state
        trader_state['is_trading'] = True
        trader_state['balance'] = self.balance
        
        # Start loops
        self.running = True
        await asyncio.gather(
            self._market_data_loop(),
            self._trading_loop(),
            self._monitor_loop()
        )
    
    async def _setup_broker(self):
        """Initialize Bybit connection."""
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        
        if not api_key:
            logger.warning("No Bybit keys - using mock mode")
            return
        
        try:
            self.broker = BybitBroker(api_key, api_secret, testnet=True)
            await self.broker.connect()
            
            self.exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'linear'}
            })
            self.exchange.set_sandbox_mode(True)
            
            logger.info("✅ Connected to Bybit")
        except Exception as e:
            logger.error(f"Broker setup failed: {e}")
    
    async def _market_data_loop(self):
        """Fetch live market data."""
        while self.running:
            try:
                if not self.exchange:
                    await asyncio.sleep(5)
                    continue
                
                for symbol in self.symbols:
                    ticker = await self.exchange.fetch_ticker(f"{symbol[:3]}/USDT")
                    
                    market_data = {
                        'symbol': symbol,
                        'price': ticker['last'],
                        'change_24h': ticker.get('percentage', 0),
                        'volume': ticker.get('quoteVolume', 0),
                        'high_24h': ticker.get('high', ticker['last']),
                        'low_24h': ticker.get('low', ticker['last']),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    trader_state['market_data'][symbol] = market_data
                
                trader_state['last_update'] = datetime.now().isoformat()
                
                # Broadcast to websockets
                await broadcast_update()
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Market data error: {e}")
                await asyncio.sleep(5)
    
    async def _trading_loop(self):
        """Main trading logic."""
        await asyncio.sleep(10)  # Wait for data
        
        while self.running:
            try:
                if not self.broker or not self.exchange:
                    await asyncio.sleep(10)
                    continue
                
                # Check if we can trade
                if len(self.positions) >= self.max_positions:
                    await asyncio.sleep(5)
                    continue
                
                if self.balance < trader_state['capital'] * 0.3:
                    logger.warning("Balance too low")
                    await asyncio.sleep(30)
                    continue
                
                # Simple momentum strategy (actually trades)
                for symbol in self.symbols:
                    if len(self.positions) >= self.max_positions:
                        break
                    
                    market = trader_state['market_data'].get(symbol)
                    if not market:
                        continue
                    
                    # Simple signal: positive momentum
                    change = market.get('change_24h', 0)
                    price = market['price']
                    
                    # Trade on strong moves
                    if abs(change) > 2.0:  # >2% move
                        confidence = min(abs(change) / 10.0, 0.9)
                        
                        if confidence >= self.min_confidence:
                            side = 'long' if change > 0 else 'short'
                            await self._execute_trade(symbol, side, price, confidence)
                
                await asyncio.sleep(10)  # Check every 10s
                
            except Exception as e:
                logger.error(f"Trading error: {e}")
                await asyncio.sleep(10)
    
    async def _execute_trade(self, symbol: str, side: str, price: float, confidence: float):
        """Execute a trade."""
        try:
            position_value = self.balance * self.position_size
            
            # Calculate leverage
            leverage = int(5 + (confidence - 0.55) * 30)  # 5x-15x
            leverage = min(max(leverage, 5), 15)
            
            # Calculate TP/SL
            tp_pct = 0.015  # 1.5%
            sl_pct = 0.01   # 1%
            
            if side == 'long':
                tp_price = price * (1 + tp_pct)
                sl_price = price * (1 - sl_pct)
            else:
                tp_price = price * (1 - tp_pct)
                sl_price = price * (1 + sl_pct)
            
            # Execute on broker
            await self.broker.set_leverage(symbol, leverage)
            
            order = await self.broker.place_order(
                symbol=symbol,
                side=side,
                amount=position_value,
                price=price,
                take_profit=tp_price,
                stop_loss=sl_price
            )
            
            # Track position
            position = {
                'id': order.get('id', f"{symbol}_{int(datetime.now().timestamp())}"),
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'size': position_value,
                'leverage': leverage,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'opened_at': datetime.now().isoformat(),
                'confidence': confidence
            }
            
            self.positions.append(position)
            trader_state['positions'].append(position)
            
            logger.info(f"🎯 OPENED {symbol} {side.upper()} @ ${price:,.2f} | {leverage}x | Conf: {confidence:.0%}")
            
            await broadcast_update()
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def _monitor_loop(self):
        """Monitor and close positions."""
        await asyncio.sleep(15)
        
        while self.running:
            try:
                for position in list(self.positions):
                    symbol = position['symbol']
                    market = trader_state['market_data'].get(symbol)
                    
                    if not market:
                        continue
                    
                    current_price = market['price']
                    
                    # Check TP/SL
                    should_close = False
                    reason = ""
                    
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
                    
                    # Time limit
                    opened = datetime.fromisoformat(position['opened_at'])
                    age = (datetime.now() - opened).total_seconds() / 60
                    if age > 30:
                        should_close, reason = True, "TIME"
                    
                    if should_close:
                        await self._close_position(position, current_price, reason)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _close_position(self, position, close_price, reason):
        """Close a position."""
        try:
            # Close on broker
            await self.broker.close_position(position['symbol'])
            
            # Calculate P&L
            entry = position['entry_price']
            if position['side'] == 'long':
                pnl_pct = (close_price - entry) / entry
            else:
                pnl_pct = (entry - close_price) / entry
            
            pnl = position['size'] * pnl_pct * position['leverage']
            
            # Update balance
            self.balance += pnl
            trader_state['balance'] = self.balance
            
            # Record trade
            trade = {
                **position,
                'close_price': close_price,
                'pnl': pnl,
                'reason': reason,
                'closed_at': datetime.now().isoformat()
            }
            
            trader_state['trades'].append(trade)
            self.positions.remove(position)
            trader_state['positions'].remove(position)
            
            emoji = "✅" if pnl > 0 else "❌"
            logger.info(f"{emoji} CLOSED {position['symbol']} {position['side']} | ${pnl:+.2f} | {reason}")
            
            await broadcast_update()
            
        except Exception as e:
            logger.error(f"Close error: {e}")


# Global trader instance
trader = LiveTrader()


async def broadcast_update():
    """Send updates to all connected websockets."""
    if not active_connections:
        return
    
    message = json.dumps({
        'type': 'update',
        'data': trader_state
    })
    
    dead_connections = []
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            dead_connections.append(connection)
    
    for conn in dead_connections:
        active_connections.remove(conn)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            'type': 'init',
            'data': trader_state
        }))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.get("/health")
async def health():
    """Health check for Railway."""
    return {"status": "ok", "trading": trader_state['is_trading']}


@app.get("/api/stats")
async def stats():
    """Get trading stats."""
    trades = trader_state['trades']
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = len(trades)
    
    return {
        'balance': trader_state['balance'],
        'capital': trader_state['capital'],
        'target': trader_state['target'],
        'win_rate': (wins / total * 100) if total > 0 else 0,
        'total_trades': total,
        'active_positions': len(trader_state['positions']),
        'pnl': trader_state['balance'] - trader_state['capital']
    }


@app.on_event("startup")
async def startup():
    """Start trading on app startup."""
    asyncio.create_task(trader.start())


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Trader - Live</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0e27;
            color: #fff;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-bar {
            display: flex;
            gap: 20px;
            align-items: center;
            font-size: 1.1em;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        
        .card-title {
            font-size: 0.9em;
            opacity: 0.7;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .card-value {
            font-size: 2.5em;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 8px;
        }
        
        .card-sub {
            font-size: 1em;
            opacity: 0.6;
        }
        
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        
        .section {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        }
        
        .section-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .market-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .market-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            padding: 20px;
        }
        
        .market-symbol {
            font-size: 1.3em;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .market-price {
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .market-change {
            font-size: 1.1em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        table th {
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }
        
        table td {
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .badge.long {
            background: #4ade80;
            color: #000;
        }
        
        .badge.short {
            background: #f87171;
            color: #fff;
        }
        
        .empty {
            text-align: center;
            padding: 40px;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI TRADER - LIVE</h1>
            <div class="status-bar">
                <span class="status-indicator"></span>
                <span id="status">Connecting...</span>
                <span>|</span>
                <span id="last-update">--:--:--</span>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="card-title">Balance</div>
                <div class="card-value" id="balance">$0.00</div>
                <div class="card-sub">Target: $<span id="target">2000</span></div>
            </div>
            
            <div class="card">
                <div class="card-title">Profit/Loss</div>
                <div class="card-value" id="pnl">$0.00</div>
                <div class="card-sub" id="pnl-pct">0.00%</div>
            </div>
            
            <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value" id="win-rate">0%</div>
                <div class="card-sub"><span id="total-trades">0</span> trades</div>
            </div>
            
            <div class="card">
                <div class="card-title">Active Positions</div>
                <div class="card-value" id="active">0</div>
                <div class="card-sub">Max: 5</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Live Market Data</div>
            <div class="market-grid" id="markets">
                <div class="empty">Loading markets...</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">💼 Active Positions</div>
            <div id="positions">
                <div class="empty">No active positions</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📜 Recent Trades</div>
            <div id="trades">
                <div class="empty">No trades yet</div>
            </div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            document.getElementById('status').textContent = 'Live Trading Active';
        };
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'init' || msg.type === 'update') {
                updateDashboard(msg.data);
            }
        };
        
        ws.onerror = () => {
            document.getElementById('status').textContent = 'Connection Error';
        };
        
        function updateDashboard(data) {
            // Stats
            document.getElementById('balance').textContent = `$${data.balance.toFixed(2)}`;
            document.getElementById('target').textContent = data.target.toFixed(0);
            
            const pnl = data.balance - data.capital;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            document.getElementById('pnl').textContent = `$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`;
            document.getElementById('pnl').className = `card-value ${pnlClass}`;
            
            const pnlPct = (pnl / data.capital) * 100;
            document.getElementById('pnl-pct').textContent = `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%`;
            
            // Win rate
            const trades = data.trades || [];
            const wins = trades.filter(t => t.pnl > 0).length;
            const winRate = trades.length > 0 ? (wins / trades.length * 100) : 0;
            document.getElementById('win-rate').textContent = `${winRate.toFixed(0)}%`;
            document.getElementById('total-trades').textContent = trades.length;
            document.getElementById('active').textContent = (data.positions || []).length;
            
            // Last update
            if (data.last_update) {
                const time = new Date(data.last_update).toLocaleTimeString();
                document.getElementById('last-update').textContent = time;
            }
            
            // Market data
            updateMarkets(data.market_data || {});
            
            // Positions
            updatePositions(data.positions || []);
            
            // Trades
            updateTrades(trades);
        }
        
        function updateMarkets(markets) {
            const container = document.getElementById('markets');
            const symbols = Object.keys(markets);
            
            if (symbols.length === 0) {
                container.innerHTML = '<div class="empty">Loading markets...</div>';
                return;
            }
            
            container.innerHTML = symbols.map(symbol => {
                const m = markets[symbol];
                const changeClass = m.change_24h >= 0 ? 'positive' : 'negative';
                return `
                    <div class="market-card">
                        <div class="market-symbol">${symbol.replace('USDT', '')}/USDT</div>
                        <div class="market-price">$${Number(m.price).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                        <div class="market-change ${changeClass}">${m.change_24h >= 0 ? '+' : ''}${Number(m.change_24h).toFixed(2)}%</div>
                    </div>
                `;
            }).join('');
        }
        
        function updatePositions(positions) {
            const container = document.getElementById('positions');
            
            if (positions.length === 0) {
                container.innerHTML = '<div class="empty">No active positions</div>';
                return;
            }
            
            container.innerHTML = `
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry</th>
                        <th>Size</th>
                        <th>Leverage</th>
                        <th>TP</th>
                        <th>SL</th>
                        <th>Confidence</th>
                    </tr>
                    ${positions.map(p => `
                        <tr>
                            <td>${p.symbol.replace('USDT', '')}</td>
                            <td><span class="badge ${p.side}">${p.side.toUpperCase()}</span></td>
                            <td>$${Number(p.entry_price).toFixed(2)}</td>
                            <td>$${Number(p.size).toFixed(2)}</td>
                            <td>${p.leverage}x</td>
                            <td>$${Number(p.tp_price).toFixed(2)}</td>
                            <td>$${Number(p.sl_price).toFixed(2)}</td>
                            <td>${(p.confidence * 100).toFixed(0)}%</td>
                        </tr>
                    `).join('')}
                </table>
            `;
        }
        
        function updateTrades(trades) {
            const container = document.getElementById('trades');
            
            if (trades.length === 0) {
                container.innerHTML = '<div class="empty">No trades yet</div>';
                return;
            }
            
            const recent = trades.slice(-10).reverse();
            
            container.innerHTML = `
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>P&L</th>
                        <th>Reason</th>
                    </tr>
                    ${recent.map(t => {
                        const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
                        const time = new Date(t.closed_at).toLocaleTimeString();
                        return `
                            <tr>
                                <td>${time}</td>
                                <td>${t.symbol.replace('USDT', '')}</td>
                                <td><span class="badge ${t.side}">${t.side.toUpperCase()}</span></td>
                                <td class="${pnlClass}">$${t.pnl >= 0 ? '+' : ''}${Number(t.pnl).toFixed(2)}</td>
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
