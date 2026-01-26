#!/usr/bin/env python3
"""
Railway startup script - runs the new working trader.
"""
import os
import sys
from datetime import datetime

def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

if __name__ == "__main__":
    log("🚀 Starting AI Trader on Railway")
    log("📊 Live market data + Real trading logic")
    
    port = os.getenv('PORT', '8000')
    log(f"📍 Port: {port}")
    
    # Run the working app
    os.execvp(sys.executable, [sys.executable, 'railway_app.py'])
