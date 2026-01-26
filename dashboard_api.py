"""
Web Dashboard API for AI Trader.
Real-time updates via Server-Sent Events.
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from datetime import datetime
from typing import List, Dict
import logging

app = FastAPI(title="AI Trader Dashboard")
logger = logging.getLogger(__name__)

# Global state (shared with trader)
trader_state = {
    'balance': 0,
    'capital': 0,
    'target': 0,
    'positions': [],
    'closed_trades': [],
    'is_running': False
}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard."""
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/api/stats")
async def get_stats():
    """Get current trading statistics."""
    closed = trader_state['closed_trades']
    wins = sum(1 for t in closed if t['pnl'] > 0)
    total = len(closed)
    total_pnl = sum(t['pnl'] for t in closed)
    
    return {
        'balance': trader_state['balance'],
        'capital': trader_state['capital'],
        'target': trader_state['target'],
        'pnl': total_pnl,
        'pnl_pct': (total_pnl / trader_state['capital'] * 100) if trader_state['capital'] > 0 else 0,
        'win_rate': (wins / total * 100) if total > 0 else 0,
        'total_trades': total,
        'active_positions': len(trader_state['positions']),
        'is_running': trader_state['is_running']
    }


@app.get("/api/positions")
async def get_positions():
    """Get active positions."""
    return {'positions': trader_state['positions']}


@app.get("/api/trades")
async def get_trades():
    """Get recent closed trades."""
    return {'trades': trader_state['closed_trades'][-20:]}  # Last 20 trades


@app.get("/stream")
async def stream_updates():
    """Server-Sent Events stream for real-time updates."""
    async def event_generator():
        while True:
            stats = await get_stats()
            yield f"data: {json.dumps(stats)}\n\n"
            await asyncio.sleep(2)  # Update every 2 seconds
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# HTML Dashboard Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Trader Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header .status {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .status.running {
            color: #4ade80;
        }
        
        .status.stopped {
            color: #f87171;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .stat-card .label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 8px;
        }
        
        .stat-card .value {
            font-size: 2.5em;
            font-weight: 700;
            line-height: 1;
        }
        
        .stat-card .sub {
            font-size: 1em;
            opacity: 0.7;
            margin-top: 8px;
        }
        
        .positive {
            color: #4ade80;
        }
        
        .negative {
            color: #f87171;
        }
        
        .section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .section h2 {
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        table th {
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            font-weight: 600;
        }
        
        table td {
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
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
        
        .badge.tp {
            background: #4ade80;
            color: #000;
        }
        
        .badge.sl {
            background: #f87171;
            color: #fff;
        }
        
        .badge.time {
            background: #fbbf24;
            color: #000;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .progress-fill {
            height: 100%;
            background: #4ade80;
            transition: width 0.3s ease;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI TRADER</h1>
            <div class="status running" id="status">
                <span class="live-indicator"></span>
                <span id="status-text">Initializing...</span>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Balance</div>
                <div class="value" id="balance">$0.00</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress" style="width: 0%"></div>
                </div>
                <div class="sub">Goal: $<span id="target">0</span></div>
            </div>
            
            <div class="stat-card">
                <div class="label">Profit/Loss</div>
                <div class="value" id="pnl">$0.00</div>
                <div class="sub" id="pnl-pct">0.00%</div>
            </div>
            
            <div class="stat-card">
                <div class="label">Win Rate</div>
                <div class="value" id="win-rate">0%</div>
                <div class="sub"><span id="total-trades">0</span> trades</div>
            </div>
            
            <div class="stat-card">
                <div class="label">Active Positions</div>
                <div class="value" id="active-positions">0</div>
                <div class="sub">Max: 5</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Active Positions</h2>
            <div id="positions-container">
                <div class="empty-state">No active positions</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📜 Recent Trades</h2>
            <div id="trades-container">
                <div class="empty-state">No trades yet</div>
            </div>
        </div>
    </div>
    
    <script>
        // Connect to real-time updates
        const eventSource = new EventSource('/stream');
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        eventSource.onerror = function() {
            document.getElementById('status-text').textContent = 'Disconnected';
            document.getElementById('status').className = 'status stopped';
        };
        
        function updateDashboard(data) {
            // Status
            if (data.is_running) {
                document.getElementById('status-text').textContent = 'Live Trading';
                document.getElementById('status').className = 'status running';
            } else {
                document.getElementById('status-text').textContent = 'Stopped';
                document.getElementById('status').className = 'status stopped';
            }
            
            // Stats
            document.getElementById('balance').textContent = `$${data.balance.toFixed(2)}`;
            document.getElementById('target').textContent = data.target.toFixed(0);
            
            const pnl = data.pnl;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            document.getElementById('pnl').textContent = `$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`;
            document.getElementById('pnl').className = `value ${pnlClass}`;
            document.getElementById('pnl-pct').textContent = `${pnl >= 0 ? '+' : ''}${data.pnl_pct.toFixed(2)}%`;
            
            document.getElementById('win-rate').textContent = `${data.win_rate.toFixed(0)}%`;
            document.getElementById('total-trades').textContent = data.total_trades;
            document.getElementById('active-positions').textContent = data.active_positions;
            
            // Progress
            const progress = Math.min((data.balance / data.target) * 100, 100);
            document.getElementById('progress').style.width = `${progress}%`;
        }
        
        // Load positions
        async function loadPositions() {
            const res = await fetch('/api/positions');
            const data = await res.json();
            
            const container = document.getElementById('positions-container');
            
            if (data.positions.length === 0) {
                container.innerHTML = '<div class="empty-state">No active positions</div>';
                return;
            }
            
            container.innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Entry</th>
                            <th>Size</th>
                            <th>Leverage</th>
                            <th>TP</th>
                            <th>SL</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.positions.map(p => `
                            <tr>
                                <td>${p.symbol}</td>
                                <td><span class="badge ${p.side}">${p.side.toUpperCase()}</span></td>
                                <td>$${p.entry_price.toFixed(2)}</td>
                                <td>$${p.size.toFixed(2)}</td>
                                <td>${p.leverage}x</td>
                                <td>$${p.tp_price.toFixed(2)}</td>
                                <td>$${p.sl_price.toFixed(2)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
        
        // Load trades
        async function loadTrades() {
            const res = await fetch('/api/trades');
            const data = await res.json();
            
            const container = document.getElementById('trades-container');
            
            if (data.trades.length === 0) {
                container.innerHTML = '<div class="empty-state">No trades yet</div>';
                return;
            }
            
            container.innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>P&L</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.trades.slice().reverse().map(t => {
                            const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
                            return `
                                <tr>
                                    <td>${new Date(t.position.timestamp).toLocaleTimeString()}</td>
                                    <td>${t.position.symbol}</td>
                                    <td><span class="badge ${t.position.side}">${t.position.side.toUpperCase()}</span></td>
                                    <td class="${pnlClass}">$${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}</td>
                                    <td><span class="badge ${t.reason.toLowerCase()}">${t.reason}</span></td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            `;
        }
        
        // Refresh positions and trades every 3 seconds
        setInterval(() => {
            loadPositions();
            loadTrades();
        }, 3000);
        
        // Initial load
        loadPositions();
        loadTrades();
    </script>
</body>
</html>
"""


def update_trader_state(trader):
    """Update global state from trader instance."""
    trader_state['balance'] = trader.balance
    trader_state['capital'] = trader.capital
    trader_state['target'] = trader.target
    trader_state['positions'] = [
        {
            'symbol': p.symbol,
            'side': p.side,
            'entry_price': p.entry_price,
            'size': p.size,
            'leverage': p.leverage,
            'tp_price': p.tp_price,
            'sl_price': p.sl_price,
            'timestamp': p.timestamp.isoformat()
        }
        for p in trader.positions
    ]
    trader_state['closed_trades'] = [
        {
            'position': {
                'symbol': t['position'].symbol,
                'side': t['position'].side,
                'timestamp': t['position'].timestamp.isoformat()
            },
            'pnl': t['pnl'],
            'reason': t['reason']
        }
        for t in trader.closed_trades
    ]
    trader_state['is_running'] = True
