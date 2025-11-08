"""
Quick utility to cancel the stuck order blocking trades.
"""
import os
from alpaca.trading.client import TradingClient

# Get API keys from environment
api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_SECRET_KEY")

if not api_key or not api_secret:
    print("❌ Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
    exit(1)

# Initialize Alpaca client (paper trading)
client = TradingClient(api_key, api_secret, paper=True)

# The stuck order ID from the error logs
stuck_order_id = "1fc9808a-2be0-4061-b117-7439fb33159c"

try:
    print(f"🔍 Attempting to cancel order: {stuck_order_id}")
    client.cancel_order_by_id(stuck_order_id)
    print(f"✅ Successfully cancelled order: {stuck_order_id}")
except Exception as e:
    print(f"⚠️ Error canceling order (may already be filled/cancelled): {e}")
    
# List all open orders to verify
try:
    print("\n📋 Checking remaining open orders...")
    orders = client.get_orders()
    if orders:
        print(f"Found {len(orders)} open orders:")
        for order in orders:
            print(f"  - {order.id}: {order.symbol} {order.side} {order.qty} @ {order.status}")
    else:
        print("✅ No open orders remaining")
except Exception as e:
    print(f"Error listing orders: {e}")
