"""
Unified CLI controller for the AI Trading System.
Clean, professional command-line interface.
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.metrics import MetricsCalculator
from analytics.logger import TradeLogger
from brokers.bybit import BybitBroker
from core.events import event_bus, EventType


class TradingCLI:
    """Main CLI controller."""
    
    def __init__(self):
        self.parser = self.create_parser()
        self.trade_logger = TradeLogger()
    
    def create_parser(self):
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="🤖 AI Trading System - Professional Edition",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Start trading with Bybit testnet, 10x leverage
  python cli/trader.py start --capital 100 --target 2000 --leverage 10 --broker bybit
  
  # Check current metrics
  python cli/trader.py metrics
  
  # View recent trades
  python cli/trader.py trades --last 50
  
  # Start API server for analytics
  python cli/trader.py serve --port 8000

Get Bybit Testnet API Keys (FREE):
  1. Visit: https://testnet.bybit.com
  2. Sign up (email only, no verification)
  3. Go to API Management → Create API Key
  4. Save API Key and Secret
  5. Set environment variables:
     export BYBIT_API_KEY="your_key"
     export BYBIT_API_SECRET="your_secret"
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # START command
        start_parser = subparsers.add_parser('start', help='Start trading')
        start_parser.add_argument('--capital', type=float, default=100,
                                help='Starting capital in USD (default: 100)')
        start_parser.add_argument('--target', type=float, default=2000,
                                help='Target profit in USD (default: 2000)')
        start_parser.add_argument('--leverage', type=int, default=10,
                                help='Leverage multiplier (default: 10, max: 100)')
        start_parser.add_argument('--broker', choices=['bybit', 'alpaca', 'binance', 'mock'],
                                default='bybit', help='Broker to use (default: bybit)')
        start_parser.add_argument('--strategy', choices=['microtrend', 'momentum', 'manual'],
                                default='microtrend', help='Trading strategy (default: microtrend)')
        start_parser.add_argument('--testnet', action='store_true', default=True,
                                help='Use testnet/paper trading (default: True)')
        start_parser.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'ETH/USDT'],
                                help='Symbols to trade (default: BTC/USDT ETH/USDT)')
        start_parser.add_argument('--max-positions', type=int, default=10,
                                help='Max open positions (default: 10)')
        start_parser.add_argument('--risk-per-trade', type=float, default=0.02,
                                help='Risk per trade as fraction of capital (default: 0.02 = 2%%)')
        
        # METRICS command
        metrics_parser = subparsers.add_parser('metrics', help='Show performance metrics')
        metrics_parser.add_argument('--format', choices=['text', 'json'], default='text',
                                  help='Output format (default: text)')
        
        # TRADES command
        trades_parser = subparsers.add_parser('trades', help='Show trade history')
        trades_parser.add_argument('--last', type=int, default=20,
                                 help='Number of recent trades (default: 20)')
        trades_parser.add_argument('--symbol', type=str,
                                 help='Filter by symbol')
        trades_parser.add_argument('--wins-only', action='store_true',
                                 help='Show only winning trades')
        
        # SERVE command
        serve_parser = subparsers.add_parser('serve', help='Start analytics API server')
        serve_parser.add_argument('--port', type=int, default=8000,
                                help='Server port (default: 8000)')
        serve_parser.add_argument('--host', type=str, default='0.0.0.0',
                                help='Server host (default: 0.0.0.0)')
        
        # STOP command
        subparsers.add_parser('stop', help='Stop trading (emergency stop)')
        
        # CONFIG command
        config_parser = subparsers.add_parser('config', help='Show/edit configuration')
        config_parser.add_argument('--show', action='store_true',
                                 help='Show current config')
        config_parser.add_argument('--edit', action='store_true',
                                 help='Open config in editor')
        
        return parser
    
    async def run(self):
        """Main entry point."""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        # Route to command handler
        if args.command == 'start':
            await self.cmd_start(args)
        elif args.command == 'metrics':
            await self.cmd_metrics(args)
        elif args.command == 'trades':
            await self.cmd_trades(args)
        elif args.command == 'serve':
            await self.cmd_serve(args)
        elif args.command == 'stop':
            await self.cmd_stop(args)
        elif args.command == 'config':
            await self.cmd_config(args)
    
    async def cmd_start(self, args):
        """Start trading."""
        print("🤖 AI Trading System - Starting...\n")
        print(f"💰 Capital: ${args.capital:,.2f}")
        print(f"🎯 Target: ${args.target:,.2f} ({args.target/args.capital:.1f}x)")
        print(f"📊 Leverage: {args.leverage}x")
        print(f"🏦 Broker: {args.broker} ({'TESTNET' if args.testnet else 'LIVE'})")
        print(f"🎲 Strategy: {args.strategy}")
        print(f"📈 Symbols: {', '.join(args.symbols)}")
        print(f"⚠️  Risk per trade: {args.risk_per_trade*100:.1f}%")
        print()
        
        # Validate broker credentials
        if args.broker == 'bybit':
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET')
            
            if not api_key or not api_secret:
                print("❌ Error: Bybit API credentials not found!")
                print("\nGet FREE testnet credentials:")
                print("1. Visit: https://testnet.bybit.com")
                print("2. Sign up and create API key")
                print("3. Set environment variables:")
                print("   export BYBIT_API_KEY='your_key'")
                print("   export BYBIT_API_SECRET='your_secret'")
                print("\nOr use mock broker:")
                print("   python cli/trader.py start --broker mock")
                return
            
            # Test connection
            print("🔌 Connecting to Bybit testnet...")
            broker = BybitBroker(api_key, api_secret, testnet=args.testnet)
            
            if await broker.connect():
                balance = await broker.get_balance()
                print(f"✅ Connected! Balance: ${balance:,.2f} USDT\n")
                await broker.disconnect()
            else:
                print("❌ Connection failed! Check your API credentials.")
                return
        
        print("⚡ System ready! Starting trading engine...")
        print("⏸️  Press Ctrl+C to stop\n")
        print("=" * 60)
        
        # TODO: Actually start the trading engine here
        # For now, just show it's ready
        print("\n⚠️  Trading engine not yet connected to new architecture")
        print("📋 Next step: Integrate with core/engine.py")
    
    async def cmd_metrics(self, args):
        """Show performance metrics."""
        print("📊 Performance Metrics\n")
        
        summary = self.trade_logger.get_summary()
        
        if summary['total_trades'] == 0:
            print("No trades yet. Start trading first!")
            return
        
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Wins: {summary['wins']} | Losses: {summary['losses']}")
        print(f"Win Rate: {summary['win_rate']*100:.1f}%")
        print(f"Total P&L: ${summary['total_pnl']:+,.2f}")
        print(f"Avg P&L: ${summary['average_pnl']:+,.2f}")
        print(f"Avg Duration: {summary['average_duration']:.0f}s")
    
    async def cmd_trades(self, args):
        """Show trade history."""
        print(f"📜 Last {args.last} Trades\n")
        
        trades = self.trade_logger.query_trades(symbol=args.symbol)
        
        if not trades:
            print("No trades found.")
            return
        
        # Filter wins only if requested
        if args.wins_only:
            trades = [t for t in trades if t.win]
        
        # Show last N
        trades = trades[-args.last:]
        
        print(f"{'Time':<10} {'Symbol':<10} {'Side':<6} {'P&L':<12} {'%':<8} {'Duration':<10} {'Result'}")
        print("-" * 80)
        
        for trade in trades:
            timestamp = datetime.fromisoformat(trade.timestamp).strftime('%H:%M:%S')
            result = "✅ WIN" if trade.win else "❌ LOSS"
            duration = f"{trade.duration:.0f}s"
            
            print(f"{timestamp:<10} {trade.symbol:<10} {trade.side:<6} "
                  f"${trade.realized_pnl:+8.2f} {trade.pnl_pct:+6.2f}% "
                  f"{duration:<10} {result}")
    
    async def cmd_serve(self, args):
        """Start analytics API server."""
        print(f"🌐 Starting analytics API server on {args.host}:{args.port}")
        print("📊 Endpoints:")
        print(f"  http://{args.host}:{args.port}/metrics")
        print(f"  http://{args.host}:{args.port}/trades")
        print(f"  http://{args.host}:{args.port}/positions")
        print()
        
        # TODO: Actually start FastAPI server
        print("⚠️  API server not yet implemented")
        print("📋 Next step: Implement api/analytics.py endpoints")
    
    async def cmd_stop(self, args):
        """Emergency stop."""
        print("🛑 EMERGENCY STOP - Closing all positions...")
        
        # TODO: Send stop event to trading engine
        await event_bus.publish(Event(EventType.EMERGENCY_STOP))
        
        print("✅ Stop signal sent")
    
    async def cmd_config(self, args):
        """Show/edit configuration."""
        if args.show:
            print("⚙️  Configuration\n")
            print("TODO: Load from config files")
        elif args.edit:
            print("Opening config editor...")
            print("TODO: Open editor")
        else:
            print("Use --show or --edit")


def main():
    """Entry point."""
    cli = TradingCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
