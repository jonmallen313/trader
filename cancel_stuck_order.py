"""
Quick utility to cancel the stuck order blocking trades.
"""
import os
import requests

# Get API keys from environment or use Railway values
api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_API_SECRET")

if not api_key or not api_secret:
    print("❌ Error: ALPACA_API_KEY and ALPACA_API_SECRET must be set")
    print("💡 You can find these variables in your Railway project settings")
    exit(1)

# Use Alpaca REST API directly (no library needed)
ALPACA_BASE = "https://paper-api.alpaca.markets"
headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": api_secret
}

# The stuck order ID from the error logs
stuck_order_id = "1fc9808a-2be0-4061-b117-7439fb33159c"

try:
    print(f"🔍 Attempting to cancel order: {stuck_order_id}")
    response = requests.delete(
        f"{ALPACA_BASE}/v2/orders/{stuck_order_id}",
        headers=headers
    )
    
    if response.status_code == 204:
        print(f"✅ Successfully cancelled order: {stuck_order_id}")
    else:
        print(f"⚠️ Response: {response.status_code} - {response.text}")
except Exception as e:
    print(f"⚠️ Error canceling order: {e}")
    
# List all open orders to verify
try:
    print("\n📋 Checking remaining open orders...")
    response = requests.get(f"{ALPACA_BASE}/v2/orders", headers=headers)
    
    if response.status_code == 200:
        orders = response.json()
        if orders:
            print(f"Found {len(orders)} open orders:")
            for order in orders:
                print(f"  - {order['id']}: {order['symbol']} {order['side']} {order.get('qty', 'N/A')} @ {order['status']}")
        else:
            print("✅ No open orders remaining")
    else:
        print(f"Error listing orders: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error listing orders: {e}")
