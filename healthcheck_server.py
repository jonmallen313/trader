"""
Health check server that starts immediately, then boots trading system in background.
"""
import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AI Trading System")

# Add CORS for Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_time = datetime.now()
system_status = {
    "health": "healthy",
    "trading_system": "starting",
    "initialized": False
}

trading_task = None
auto_started = False

@app.get("/health")
async def health_check():
    """Health check endpoint - responds immediately, auto-starts trading in background."""
    global auto_started, trading_task
    
    # Auto-start trading system on first health check if not already started
    if not auto_started and not trading_task:
        logger.info("🚀 Auto-starting trading system after health check...")
        trading_task = asyncio.create_task(initialize_trading_system())
        auto_started = True
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": (datetime.now() - start_time).total_seconds(),
        "trading_system": system_status["trading_system"]
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "online",
        "name": "AI Trading System",
        "health_endpoint": "/health",
        "start_endpoint": "/start",
        "system_status": system_status
    }

@app.get("/status")
async def get_status():
    """Get system status."""
    return {
        "status": "running" if system_status["initialized"] else "ready",
        "uptime": (datetime.now() - start_time).total_seconds(),
        "health": system_status["health"],
        "trading_system": system_status["trading_system"]
    }

@app.post("/start")
async def start_trading():
    """Start the trading system."""
    global trading_task
    
    if trading_task and not trading_task.done():
        return {"status": "already_running", "message": "Trading system is already running"}
    
    if system_status["initialized"]:
        return {"status": "already_initialized", "message": "Trading system already initialized"}
    
    # Start trading system in background
    trading_task = asyncio.create_task(initialize_trading_system())
    
    return {
        "status": "starting",
        "message": "Trading system initialization started"
    }

async def initialize_trading_system():
    """Initialize the trading system in the background."""
    try:
        logger.info("🤖 Initializing AI Trading System...")
        system_status["trading_system"] = "loading"
        
        # Import and start the main trading system
        logger.info("📦 Importing main module...")
        from main import AITradingSystem
        
        logger.info("📦 Creating trading system instance...")
        trading_system = AITradingSystem(mode="paper")
        
        logger.info("🚀 Starting trading system...")
        system_status["trading_system"] = "starting"
        
        # Start system WITHOUT starting webhook server (we'll use existing FastAPI app)
        await trading_system.start_system(start_webhook=False)
        
        # Mount webhook routes to this app
        if hasattr(trading_system, 'webhook_server') and trading_system.webhook_server:
            logger.info("📡 Mounting webhook routes to health check server...")
            # Include the webhook router in our app
            app.include_router(trading_system.webhook_server.app.router, prefix="/webhook")
            logger.info("✅ Webhook routes mounted at /webhook/*")
        
        system_status["trading_system"] = "running"
        system_status["initialized"] = True
        logger.info("✅ AI Trading System fully initialized and running!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize trading system: {e}")
        import traceback
        logger.error(traceback.format_exc())
        system_status["trading_system"] = f"error: {str(e)}"
        system_status["health"] = "degraded"

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    logger.info(f"🚀 Starting health check server on 0.0.0.0:{port}")
    logger.info(f"📍 Health endpoint: http://0.0.0.0:{port}/health")
    logger.info(f"🎯 Trading start endpoint: http://0.0.0.0:{port}/start")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

