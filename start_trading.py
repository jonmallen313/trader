"""
Simple script to start the trading system on Railway.
Run this after deployment to initialize the trading system.
"""
import requests
import sys

def start_trading(railway_url):
    """Start the trading system via the /start endpoint."""
    if not railway_url.startswith('http'):
        railway_url = f'https://{railway_url}'
    
    print(f"🚀 Starting trading system at {railway_url}...")
    
    try:
        response = requests.post(f"{railway_url}/start")
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Response: {result}")
        
        # Check status
        print("\n📊 Checking status...")
        status_response = requests.get(f"{railway_url}/status")
        status = status_response.json()
        print(f"Status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python start_trading.py <railway-url>")
        print("Example: python start_trading.py your-app.up.railway.app")
        sys.exit(1)
    
    railway_url = sys.argv[1]
    success = start_trading(railway_url)
    sys.exit(0 if success else 1)
