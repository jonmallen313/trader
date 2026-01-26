"""
COMPLETE AI TRADER
- Real AI model (XGBoost microtrend detection)
- Real market data (live crypto prices)
- Real web dashboard (beautiful UI with live updates)
- Actually works
"""

import asyncio
import uvicorn
import threading
from trade_live import RealAITrader
from dashboard_api import app, update_trader_state


async def run_trader(capital: float):
    """Run the trader in background."""
    trader = RealAITrader(capital)
    
    # Update dashboard state every 2 seconds
    async def update_loop():
        while True:
            update_trader_state(trader)
            await asyncio.sleep(2)
    
    # Run trader and update loop
    await asyncio.gather(
        trader.start(),
        update_loop(),
        return_exceptions=True
    )


def run_dashboard():
    """Run the dashboard server."""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="error"  # Quiet
    )


def main():
    """Main entry point."""
    print("╔" + "═" * 60 + "╗")
    print("║" + "  🤖 AI TRADER - COMPLETE SYSTEM  ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    print("✅ Real AI Model (XGBoost)")
    print("✅ Real Market Data (Live Prices)")
    print("✅ Real Web Dashboard (http://127.0.0.1:8000)")
    print()
    
    capital = input("💵 Enter starting capital (default $100): ").strip()
    capital = float(capital) if capital else 100.0
    
    print()
    print("🚀 Starting...")
    print(f"📊 Dashboard: http://127.0.0.1:8000")
    print("⌨️  Press Ctrl+C to stop")
    print()
    
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Give dashboard time to start
    import time
    time.sleep(2)
    
    # Start trader
    try:
        asyncio.run(run_trader(capital))
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")


if __name__ == "__main__":
    main()
