#!/usr/bin/env python3
"""
Railway startup script that ensures health checks pass immediately.
Starts a lightweight health server first, then launches the main trading system.
"""
import os
import sys
import asyncio
import subprocess
import signal
from datetime import datetime

def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

async def main():
    """Start health check server and main application."""
    port = os.getenv('PORT', '8000')
    
    log("🚀 Starting AI Trading System on Railway")
    log(f"📍 Port: {port}")
    
    # Start health check server in background
    log("🏥 Starting health check server...")
    health_process = subprocess.Popen(
        [sys.executable, 'healthcheck_server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Give health server time to start
    await asyncio.sleep(3)
    log("✅ Health check server ready")
    
    # Start main trading application
    log("🤖 Starting AI Trading System...")
    main_process = subprocess.Popen(
        [sys.executable, 'main.py', 'paper', '--capital', '100', '--target', '2000'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Function to handle shutdown
    def shutdown(signum, frame):
        log("🛑 Shutting down...")
        health_process.terminate()
        main_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    # Stream output from both processes
    async def stream_output(process, prefix):
        """Stream process output with prefix."""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[{prefix}] {line.rstrip()}", flush=True)
        except Exception as e:
            log(f"Error streaming {prefix}: {e}")
    
    # Wait for both processes
    try:
        await asyncio.gather(
            stream_output(health_process, "HEALTH"),
            stream_output(main_process, "MAIN")
        )
    except KeyboardInterrupt:
        log("🛑 Interrupted")
        health_process.terminate()
        main_process.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log(f"❌ Startup error: {e}")
        sys.exit(1)
