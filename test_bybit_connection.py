"""
Quick test script to verify Bybit testnet connection.
Run this to make sure your API credentials work.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from brokers.bybit import BybitBroker


async def test_bybit_connection():
    """Test Bybit testnet connection and display account info."""
    
    print("🧪 Bybit Testnet Connection Test\n")
    print("=" * 60)
    
    # Get credentials
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ ERROR: Bybit API credentials not found!\n")
        print("Please set environment variables:")
        print("  export BYBIT_API_KEY='your_key'")
        print("  export BYBIT_API_SECRET='your_secret'")
        print("\nGet FREE testnet credentials:")
        print("  https://testnet.bybit.com")
        return False
    
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"API Secret: {'*' * 32}")
    print()
    
    # Create broker
    print("🔌 Connecting to Bybit testnet...")
    broker = BybitBroker(api_key, api_secret, testnet=True)
    
    try:
        # Test connection
        if not await broker.connect():
            print("❌ Connection failed!")
            return False
        
        print("✅ Connection successful!\n")
        
        # Get balance
        print("💰 Account Information:")
        print("-" * 60)
        balance = await broker.get_balance()
        print(f"Available Balance: ${balance:,.2f} USDT")
        print(f"Max Leverage: {broker.max_leverage}x")
        print()
        
        # Test market data
        print("📊 Market Data Test:")
        print("-" * 60)
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        for symbol in symbols:
            price = await broker.get_price(symbol)
            print(f"{symbol:<12} ${price:,.2f}")
        
        print()
        print("=" * 60)
        print("✅ All tests passed!")
        print()
        print("You're ready to start trading:")
        print("  python cli/trader.py start --capital 100 --leverage 10")
        
        await broker.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify API key and secret are correct")
        print("2. Check API key has 'Read-Write' permissions")
        print("3. Make sure you're using TESTNET credentials")
        print("4. Visit: https://testnet.bybit.com/app/user/api-management")
        
        if broker.session:
            await broker.disconnect()
        
        return False


if __name__ == "__main__":
    success = asyncio.run(test_bybit_connection())
    sys.exit(0 if success else 1)
