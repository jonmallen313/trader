"""
Combined health check and trading system launcher.
Starts health endpoint immediately, then initializes trading system in background.
"""
import os
import asyncio
import logging
from fastapi import FastAPI, BackgroundTasks
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
    "trading_system": "initializing",
    "initialized": False
}

@app.get("/health")
async def health_check():
    """Health check endpoint - responds immediately."""
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
        "system_status": system_status
    }

@app.get("/status")
async def get_status():
    """Get system status."""
    return {
        "status": "running" if system_status["initialized"] else "initializing",
        "uptime": (datetime.now() - start_time).total_seconds(),
        "health": system_status["health"],
        "trading_system": system_status["trading_system"]
    }

async def initialize_trading_system():
    """Initialize the trading system in the background."""
    try:
        logger.info("🤖 Initializing AI Trading System...")
        system_status["trading_system"] = "loading"
        
        # Import and start the main trading system
        from main import AITradingSystem
        
        logger.info("📦 Creating trading system instance...")
        trading_system = AITradingSystem(mode="paper")
        
        logger.info("🚀 Starting trading system...")
        system_status["trading_system"] = "starting"
        
        await trading_system.start_system()
        
        system_status["trading_system"] = "running"
        system_status["initialized"] = True
        logger.info("✅ AI Trading System fully initialized and running!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize trading system: {e}")
        system_status["trading_system"] = f"error: {str(e)}"
        system_status["health"] = "degraded"

@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    logger.info("🏥 Health check server started - ready for Railway health checks")
    # Start trading system initialization in background
    asyncio.create_task(initialize_trading_system())

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    logger.info(f"🚀 Starting server on 0.0.0.0:{port}")
    logger.info(f"📍 Health endpoint: http://0.0.0.0:{port}/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

