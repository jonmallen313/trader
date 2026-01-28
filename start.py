"""
Railway startup script - ensures health endpoint is available immediately
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING AI TRADER ON RAILWAY")
    logger.info("=" * 60)
    logger.info(f"📡 Port: {port}")
    logger.info(f"🌍 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'unknown')}")
    logger.info(f"📦 Service: {os.getenv('RAILWAY_SERVICE_NAME', 'unknown')}")
    logger.info("=" * 60)
    
    # Import and run the main app
    try:
        from complete_trader import app
        import uvicorn
        
        logger.info("✅ App imported successfully")
        logger.info(f"🏥 Health endpoint will be available at http://0.0.0.0:{port}/health")
        logger.info("🌐 Starting uvicorn server...")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"❌ STARTUP FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
