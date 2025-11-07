"""
Railway cloud deployment configuration for the AI Trading System.
Handles environment variables, port binding, and cloud-specific settings.
"""

import os
from pathlib import Path


class RailwayConfig:
    """Railway-specific configuration management."""
    
    def __init__(self):
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
        self.port = int(os.getenv('PORT', os.getenv('RAILWAY_PORT', 8000)))
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
    def setup_environment(self):
        """Setup Railway environment variables."""
        if self.is_railway:
            # Set default values for Railway deployment
            os.environ.setdefault('WEBHOOK_PORT', str(self.port))
            os.environ.setdefault('PAPER_MODE', 'True')
            os.environ.setdefault('LOG_LEVEL', 'INFO')
            
            # Use Railway's provided environment variables
            if not os.getenv('DATABASE_URL'):
                os.environ['DATABASE_URL'] = 'sqlite:///trader.db'
                
            # Create necessary directories
            self._create_directories()
            
    def _create_directories(self):
        """Create necessary directories for Railway deployment."""
        dirs = ['logs', 'models/saved', 'data/historical']
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def get_webhook_url(self):
        """Get the public webhook URL for Railway deployment."""
        if self.is_railway:
            # Railway provides RAILWAY_STATIC_URL or RAILWAY_PUBLIC_DOMAIN
            domain = os.getenv('RAILWAY_STATIC_URL') or os.getenv('RAILWAY_PUBLIC_DOMAIN')
            if domain:
                return f"https://{domain}/webhook/tradingview"
        return f"http://localhost:{self.port}/webhook/tradingview"
    
    def get_dashboard_url(self):
        """Get the public dashboard URL."""
        if self.is_railway:
            domain = os.getenv('RAILWAY_STATIC_URL') or os.getenv('RAILWAY_PUBLIC_DOMAIN')
            if domain:
                return f"https://{domain}"
        return f"http://localhost:{self.port}"


# Initialize Railway configuration
railway_config = RailwayConfig()
railway_config.setup_environment()