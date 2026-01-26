"""
ULTRA SIMPLE CRYPTO TRADER
Just run it and watch it trade BTC/ETH with leverage.
No dashboard BS. Just pure trading.
"""

import os
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict
import logging
import random

# Minimal broker implementation
class SimpleBroker:
    """Dead simple broker - just executes trades"""
    
    def __init__(self, api_key: str, api_secret: str, initial_capital: float):
        self.api_key = api_key
        self.api_secret = api_secret
        self.balance = initial_capital
        self.positions: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.trade_count = 0
        
    def get_price(self, symbol: str) -> float:
        """Mock price for testing - replace with real Bybit API"""
        base_prices = {
            'BTCUSDT': 95000 + random.randint(-1000, 1000),
            'ETHUSDT': 3400 + random.randint(-100, 100),
            'SOLUSDT': 180 + random.randint(-5, 5),
        }
        return base_prices.get(symbol, 100.0)
    
    def open_position(self, symbol: str, direction: str, size: float, 
                      leverage: int, entry_price: float) -> Dict:
        """Open new leveraged position"""
        position = {
            'id': self.trade_count,
            'symbol': symbol,
            'direction': direction,  # 'long' or 'short'
            'size': size,
            'leverage': leverage,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'tp_price': entry_price * (1.02 if direction == 'long' else 0.98),
            'sl_price': entry_price * (0.99 if direction == 'long' else 1.01),
        }
        
        self.positions.append(position)
        self.balance -= size  # Reserve capital
        self.trade_count += 1
        
        return position
    
    def check_exits(self) -> List[Dict]:
        """Check if any positions hit TP/SL"""
        exits = []
        
        for pos in self.positions[:]:
            current_price = self.get_price(pos['symbol'])
            
            # Calculate P&L
            if pos['direction'] == 'long':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
            
            pnl = pos['size'] * pnl_pct * pos['leverage']
            
            # Check TP/SL
            should_exit = False
            reason = None
            
            if pos['direction'] == 'long':
                if current_price >= pos['tp_price']:
                    should_exit = True
                    reason = 'TP'
                elif current_price <= pos['sl_price']:
                    should_exit = True
                    reason = 'SL'
            else:
                if current_price <= pos['tp_price']:
                    should_exit = True
                    reason = 'TP'
                elif current_price >= pos['sl_price']:
                    should_exit = True
                    reason = 'SL'
            
            # Also exit after 5 minutes
            if (datetime.now() - pos['entry_time']).seconds > 300:
                should_exit = True
                reason = 'TIME'
            
            if should_exit:
                self.balance += pos['size'] + pnl
                pos['exit_price'] = current_price
                pos['exit_reason'] = reason
                pos['pnl'] = pnl
                pos['exit_time'] = datetime.now()
                
                self.positions.remove(pos)
                self.closed_trades.append(pos)
                exits.append(pos)
        
        return exits


class AITrader:
    """Simple AI that picks trades"""
    
    def __init__(self, broker: SimpleBroker):
        self.broker = broker
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
    def scan_opportunities(self) -> Optional[Dict]:
        """Look for trade opportunities"""
        
        # Don't open new positions if we're full
        if len(self.broker.positions) >= 3:
            return None
        
        # Don't trade if we're out of money
        if self.broker.balance < 20:
            return None
        
        # Random simple strategy - replace with real AI
        if random.random() > 0.3:  # 70% chance to find opportunity
            symbol = random.choice(self.symbols)
            direction = random.choice(['long', 'short'])
            confidence = random.uniform(0.6, 0.95)
            
            # Higher confidence = more leverage
            if confidence > 0.85:
                leverage = 20
            elif confidence > 0.75:
                leverage = 10
            else:
                leverage = 5
            
            # Position size: 20% of balance
            size = min(self.broker.balance * 0.2, self.broker.balance - 10)
            
            return {
                'symbol': symbol,
                'direction': direction,
                'size': round(size, 2),
                'leverage': leverage,
                'confidence': confidence
            }
        
        return None


async def trade_loop(capital: float):
    """Main trading loop"""
    
    # Initialize
    broker = SimpleBroker("test_key", "test_secret", capital)
    ai = AITrader(broker)
    
    print("=" * 60)
    print("🚀 AI AUTONOMOUS TRADER - LIVE MODE")
    print("=" * 60)
    print(f"💰 Starting Capital: ${capital:.2f}")
    print(f"🎯 Target: ${capital * 20:.2f} (20x)")
    print(f"📊 Symbols: BTC, ETH, SOL")
    print("=" * 60)
    print("\n⏳ Scanning markets...\n")
    
    start_time = datetime.now()
    last_stats_time = time.time()
    
    # Main loop
    while True:
        try:
            # Check exits first
            exits = broker.check_exits()
            for exit in exits:
                pnl_emoji = "✅" if exit['pnl'] > 0 else "❌"
                print(f"{pnl_emoji} CLOSED {exit['symbol']} {exit['direction']} "
                      f"| ${exit['pnl']:+.2f} | {exit['exit_reason']}")
            
            # Look for new opportunities
            opportunity = ai.scan_opportunities()
            if opportunity:
                price = broker.get_price(opportunity['symbol'])
                position = broker.open_position(
                    opportunity['symbol'],
                    opportunity['direction'],
                    opportunity['size'],
                    opportunity['leverage'],
                    price
                )
                
                print(f"🎯 OPENED {position['symbol']} {position['direction'].upper()} "
                      f"| ${position['size']:.2f} @ {position['leverage']}x "
                      f"| Entry: ${position['entry_price']:.2f}")
            
            # Print stats every 10 seconds
            if time.time() - last_stats_time > 10:
                total_trades = len(broker.closed_trades)
                wins = sum(1 for t in broker.closed_trades if t['pnl'] > 0)
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                total_pnl = sum(t['pnl'] for t in broker.closed_trades)
                
                print("\n" + "=" * 60)
                print(f"💰 Balance: ${broker.balance:.2f} | P&L: ${total_pnl:+.2f}")
                print(f"📊 Active: {len(broker.positions)} | "
                      f"Closed: {total_trades} | "
                      f"Win: {win_rate:.0f}%")
                print(f"⏱️  Runtime: {(datetime.now() - start_time).seconds // 60} min")
                print("=" * 60 + "\n")
                
                last_stats_time = time.time()
                
                # Check if we hit target
                current_total = broker.balance + sum(p['size'] for p in broker.positions)
                if current_total >= capital * 20:
                    print("\n🎉 TARGET REACHED! 20X COMPLETE!")
                    break
            
            await asyncio.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\n⏸️  Stopping...")
            break
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Starting: ${capital:.2f}")
    print(f"Ending: ${broker.balance:.2f}")
    total_pnl = sum(t['pnl'] for t in broker.closed_trades)
    print(f"Total P&L: ${total_pnl:+.2f}")
    print(f"Total Trades: {len(broker.closed_trades)}")
    wins = sum(1 for t in broker.closed_trades if t['pnl'] > 0)
    if len(broker.closed_trades) > 0:
        print(f"Win Rate: {wins / len(broker.closed_trades) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    print("\n")
    print("╔════════════════════════════════════════╗")
    print("║   AI AUTONOMOUS TRADER - DEMO MODE    ║")
    print("╚════════════════════════════════════════╝")
    print("\n")
    
    # Get capital
    try:
        capital_input = input("💵 Enter starting capital (default $100): ").strip()
        capital = float(capital_input) if capital_input else 100.0
    except ValueError:
        capital = 100.0
    
    # Run
    asyncio.run(trade_loop(capital))
