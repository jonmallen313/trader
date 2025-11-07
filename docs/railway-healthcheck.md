# Railway Deployment Checklist

## Health Check Configuration

✅ **Health Endpoint**: `/health`
✅ **Expected Response**: HTTP 200 with JSON `{"status": "healthy"}`
✅ **Timeout**: 300 seconds (5 minutes)
✅ **Port**: Uses `PORT` environment variable from Railway

## Important Railway Settings

In your Railway service settings, configure:

1. **Health Check Path**: `/health`
2. **Health Check Timeout**: 300 seconds (or set `RAILWAY_HEALTHCHECK_TIMEOUT_SEC`)
3. **Port**: Railway automatically sets the `PORT` variable - do NOT override it

## Required Environment Variables

### For Mock/Testing (No API Keys)
```bash
PAPER_MODE=true
TRADING_MODE=paper
# System will use MockBroker automatically
```

### For Paper Trading with Live Data
```bash
# Alpaca Paper Trading (recommended)
ALPACA_API_KEY=your_paper_key
ALPACA_API_SECRET=your_paper_secret
USE_EXCHANGE=alpaca
PAPER_MODE=true

# OR Binance Testnet
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
BINANCE_TESTNET=true
USE_EXCHANGE=binance
PAPER_MODE=true
```

### Optional Configuration
```bash
INITIAL_CAPITAL=100
GLOBAL_TAKE_PROFIT=2000
GLOBAL_STOP_LOSS=-50
LOG_LEVEL=INFO
```

## Troubleshooting Health Check Issues

### "Service Unavailable" Error
**Cause**: Application not listening on Railway's PORT variable
**Fix**: Ensure the app uses `os.getenv('PORT')` - this is already configured in `config/railway.py`

### "Status 400" Error  
**Cause**: Application rejecting requests from `healthcheck.railway.app`
**Fix**: CORS middleware added to accept all origins - already configured in `src/webhook.py`

### "Timeout" Error
**Cause**: Application takes too long to start
**Fix**: Increase timeout in Railway settings or set `RAILWAY_HEALTHCHECK_TIMEOUT_SEC=600`

## Deployment Commands

Railway automatically detects and runs:
```bash
# From Procfile
python main.py paper --capital 100 --target 2000
```

## Verification

After deployment, check:
1. **Logs**: Look for "✅ Webhook server started and ready for health checks"
2. **Health Check**: Visit `https://your-app.railway.app/health`
3. **Dashboard**: Visit `https://your-app.railway.app/`
4. **Status API**: Visit `https://your-app.railway.app/status`

## Common Log Messages

✅ **Success**:
```
Starting AI Trading System...
Initializing system components...
✅ Webhook server started and ready for health checks
🚀 AI Trading System running in paper mode
```

❌ **Issues**:
```
Broker setup failed: [error] - Using mock broker fallback
Data feed setup failed: [error] - Continuing without live data
```
These are warnings - the system will still work with MockBroker for testing.

## Network Configuration

- **Allowed Origins**: All origins (for Railway health checks)
- **Health Check Hostname**: `healthcheck.railway.app`
- **Binding**: `0.0.0.0` (all interfaces) on Railway
- **Port**: Dynamic, provided by Railway's `PORT` variable
