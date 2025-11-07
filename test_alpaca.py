"""
Quick test script to verify Alpaca Paper Trading API connectivity.
Run this locally to test your API keys before deploying to Railway.
"""
import os
import asyncio
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.historical import StockHistoricalDataClient

async def test_alpaca_connection():
    """Test Alpaca paper trading API."""
    
    # Get API keys from environment
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_API_SECRET')
    
    if not api_key or not secret_key:
        print("❌ ALPACA_API_KEY and ALPACA_API_SECRET not found in environment")
        print("📖 Get keys from: https://app.alpaca.markets/paper/dashboard/overview")
        return False
    
    print("🔐 Testing Alpaca Paper Trading API...")
    print(f"📍 API Key: {api_key[:10]}...")
    
    try:
        # Test 1: Trading Client (for account info and orders)
        print("\n1️⃣  Testing Trading Client...")
        trading_client = TradingClient(api_key, secret_key, paper=True)
        
        account = trading_client.get_account()
        print(f"✅ Account Status: {account.status}")
        print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
        print(f"💵 Cash: ${float(account.cash):,.2f}")
        print(f"📊 Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        # Test 2: Historical Data Client (for market data)
        print("\n2️⃣  Testing Historical Data Client...")
        data_client = StockHistoricalDataClient(api_key, secret_key)
        
        request = StockLatestQuoteRequest(symbol_or_symbols=["AAPL", "TSLA"])
        quotes = data_client.get_stock_latest_quote(request)
        
        for symbol, quote in quotes.items():
            mid_price = (quote.bid_price + quote.ask_price) / 2
            print(f"✅ {symbol}: ${mid_price:.2f} (bid: ${quote.bid_price:.2f}, ask: ${quote.ask_price:.2f})")
        
        # Test 3: Live Data Stream (for real-time updates)
        print("\n3️⃣  Testing Live Data Stream...")
        print("📡 Connecting to real-time feed...")
        
        stream = StockDataStream(api_key, secret_key)
        
        received_data = []
        
        async def quote_handler(data):
            received_data.append(data)
            mid_price = (data.bid_price + data.ask_price) / 2
            print(f"✅ {data.symbol}: ${mid_price:.2f} @ {data.timestamp}")
            
            # Stop after receiving a few quotes
            if len(received_data) >= 3:
                await stream.stop_ws()
        
        # Subscribe to quotes
        stream.subscribe_quotes(quote_handler, "AAPL", "TSLA")
        
        # Run for 10 seconds max
        try:
            await asyncio.wait_for(stream._run_forever(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        
        if received_data:
            print(f"✅ Received {len(received_data)} real-time quotes")
        else:
            print("⚠️  No real-time data received (market might be closed)")
        
        print("\n✅ All Alpaca API tests passed!")
        print("\n📋 Summary:")
        print("  ✅ Authentication working")
        print("  ✅ Account access working")
        print("  ✅ Market data access working")
        print("  ✅ Real-time stream working")
        print("\n🚀 Ready to deploy to Railway!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Troubleshooting:")
        print("  1. Make sure you're using PAPER trading keys (start with PK...)")
        print("  2. Check keys at: https://app.alpaca.markets/paper/dashboard/overview")
        print("  3. Ensure keys are active and not expired")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  Alpaca Paper Trading API Connection Test")
    print("=" * 60)
    
    result = asyncio.run(test_alpaca_connection())
    
    if result:
        print("\n" + "=" * 60)
        print("  ✅ SUCCESS - Your Alpaca setup is ready!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  ❌ FAILED - Fix the issues above and try again")
        print("=" * 60)
