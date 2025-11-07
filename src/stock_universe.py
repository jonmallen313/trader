"""
Stock universe management - handles all tradeable stocks from Alpaca.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Cache for stock universe
_stock_cache = None
_cache_time = None

async def get_all_tradeable_stocks() -> List[Dict]:
    """Get all tradeable stocks from Alpaca."""
    global _stock_cache, _cache_time
    
    # Cache for 1 hour
    if _stock_cache and _cache_time:
        if (datetime.now() - _cache_time).seconds < 3600:
            return _stock_cache
    
    try:
        from alpaca.trading.client import TradingClient
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret = os.getenv('ALPACA_API_SECRET')
        
        if not api_key or not secret:
            logger.warning("No Alpaca credentials - returning sample stocks")
            return get_sample_stocks()
        
        client = TradingClient(api_key, secret, paper=True)
        
        # Get all assets (no status parameter)
        assets = client.get_all_assets()
        
        stocks = []
        for asset in assets:
            # Filter for active, tradable, fractionable stocks
            if (hasattr(asset, 'status') and asset.status == 'active' and 
                hasattr(asset, 'tradable') and asset.tradable and 
                hasattr(asset, 'fractionable') and asset.fractionable):
                stocks.append({
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'exchange': asset.exchange,
                    'tradable': asset.tradable,
                    'marginable': asset.marginable,
                    'shortable': asset.shortable,
                    'fractionable': asset.fractionable,
                })
        
        _stock_cache = stocks
        _cache_time = datetime.now()
        
        logger.info(f"Loaded {len(stocks)} tradeable stocks from Alpaca")
        return stocks
        
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
        return get_sample_stocks()

def get_sample_stocks() -> List[Dict]:
    """Get sample popular stocks for demo/testing."""
    return [
        # Tech Giants
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'AMD', 'name': 'Advanced Micro Devices', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        
        # Finance
        {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'BAC', 'name': 'Bank of America Corp.', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'GS', 'name': 'Goldman Sachs Group Inc.', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        
        # ETFs
        {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF Trust', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'DIA', 'name': 'SPDR Dow Jones Industrial Average ETF', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        
        # Popular Stocks
        {'symbol': 'DIS', 'name': 'Walt Disney Company', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'NKE', 'name': 'Nike Inc.', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'SBUX', 'name': 'Starbucks Corporation', 'exchange': 'NASDAQ', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'V', 'name': 'Visa Inc.', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
        {'symbol': 'MA', 'name': 'Mastercard Inc.', 'exchange': 'NYSE', 'tradable': True, 'marginable': True, 'shortable': True, 'fractionable': True},
    ]

def search_stocks(query: str, stocks: List[Dict]) -> List[Dict]:
    """Search stocks by symbol or name."""
    query = query.upper()
    return [
        s for s in stocks 
        if query in s['symbol'] or query in s['name'].upper()
    ]

def filter_stocks(stocks: List[Dict], 
                 exchange: Optional[str] = None,
                 marginable: Optional[bool] = None,
                 shortable: Optional[bool] = None) -> List[Dict]:
    """Filter stocks by criteria."""
    filtered = stocks
    
    if exchange:
        filtered = [s for s in filtered if s['exchange'] == exchange]
    if marginable is not None:
        filtered = [s for s in filtered if s['marginable'] == marginable]
    if shortable is not None:
        filtered = [s for s in filtered if s['shortable'] == shortable]
    
    return filtered
