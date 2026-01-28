"""
COMPLETE AI TRADER
- Real AI model (XGBoost microtrend detection)
- Real market data (live crypto prices)
- Real web dashboard (beautiful UI with live updates)
- ALWAYS RUNNING background trader
"""

import asyncio
import uvicorn
import threading
from background_trader import BackgroundAITrader
from dashboard_api import app, update_trader_state


async def run_trader(capital: float):
    """Run the background trader."""
    trader = BackgroundAITrader(capital)
    await trader.start()


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
    print("║" + "  🤖 AI TRADER - ALWAYS RUNNING  ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    print("✅ Real AI Model (XGBoost)")
    print("✅ Real Market Data (Live Prices)")
    print("✅ 15 Concurrent Positions (Maximum Profit)")
    print("✅ ALWAYS Trading (Background Service)")
    print("✅ Real Web Dashboard (http://127.0.0.1:8000)")
    print()
    
    capital = input("💵 Enter starting capital (default $100): ").strip()
    capital = float(capital) if capital else 100.0
    
    print()
    print("🚀 Starting background trader...")
    print(f"📊 Dashboard: http://127.0.0.1:8000")
    print("⌨️  Press Ctrl+C to stop")
    print()
    print("⚠️  The trader will run CONTINUOUSLY in the background")
    print("⚠️  You can close the webpage and it will keep trading")
    print()
    
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Give dashboard time to start
    import time
    time.sleep(2)
    
    # Start trader (runs forever)
    try:
        asyncio.run(run_trader(capital))
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")


if __name__ == "__main__":
    main()
