#!/usr/bin/env python3
"""
Simple AI Trading - Just set capital and go!
AI controls EVERYTHING else.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from core.autopilot import FullAutoPilot, AutoPilotConfig


def setup_logging():
    """Setup clean logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Clean output, no timestamps
        handlers=[logging.StreamHandler(sys.stdout)]
    )


async def main():
    """Super simple entry point."""
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║          🤖 AI AUTONOMOUS TRADER                          ║
║                                                           ║
║  YOU control:  Capital amount                            ║
║  AI controls:  Everything else                           ║
║                                                           ║
║  ✓ Symbol selection      ✓ Entry/exit timing            ║
║  ✓ Leverage calculation  ✓ TP/SL levels                 ║
║  ✓ Position sizing       ✓ Risk management              ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Get capital from user
    print("How much capital do you want to trade with?")
    print("(Use testnet fake money - get it from https://testnet.bybit.com)")
    print()
    
    try:
        capital_input = input("💰 Enter capital (default: $100): ").strip()
        capital = float(capital_input) if capital_input else 100.0
    except ValueError:
        print("Invalid amount, using $100")
        capital = 100.0
    
    # Auto-calculate target (20x)
    target = capital * 20
    
    print(f"\n✅ Capital: ${capital:,.2f}")
    print(f"🎯 Target: ${target:,.2f} (20x)")
    print()
    
    # Confirm
    response = input("Start AI autopilot? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        return
    
    print("\n🚀 Starting AI autopilot...\n")
    
    # Setup logging
    setup_logging()
    
    # Create autopilot config
    config = AutoPilotConfig(
        initial_capital=capital,
        target_profit=target,
        max_positions=5,  # AI manages 5 positions at once
        broker="bybit",
        testnet=True
    )
    
    # Start autopilot
    autopilot = FullAutoPilot(config)
    
    try:
        await autopilot.start()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping autopilot...")
        autopilot.is_running = False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for API keys
    if not os.getenv('BYBIT_API_KEY') or not os.getenv('BYBIT_API_SECRET'):
        print("""
❌ Bybit API credentials not found!

Get FREE testnet credentials:
  1. Visit: https://testnet.bybit.com
  2. Sign up (email only, instant)
  3. Go to: API Management → Create New Key
  4. Enable "Read-Write" permissions
  5. Copy API Key and Secret

Then set environment variables:

Windows (PowerShell):
  $env:BYBIT_API_KEY="your_key"
  $env:BYBIT_API_SECRET="your_secret"

Linux/Mac:
  export BYBIT_API_KEY="your_key"
  export BYBIT_API_SECRET="your_secret"

Then run again: python trade.py
        """)
        sys.exit(1)
    
    # Run
    asyncio.run(main())
