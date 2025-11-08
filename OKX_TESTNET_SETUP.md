# OKX Testnet Setup - REAL 24/7 Crypto Paper Trading

## Why OKX Testnet?
- **REAL paper trading** with actual testnet API (not fake simulation)
- **24/7 crypto markets** (BTC, ETH, SOL, DOGE, AVAX)
- **No geographic restrictions** (works from Ohio and everywhere else)
- **Free virtual funds** for testing
- **Professional platform** used by real traders

## Setup Steps

### 1. Create OKX Account
- Go to https://www.okx.com/
- Sign up for a free account
- Complete email verification

### 2. Switch to Demo Trading Mode
- Log in to OKX
- Click your profile icon (top right)
- Select "Demo Trading" to switch to testnet mode
- You'll get **virtual funds** automatically (typically $100,000+ USDT)

### 3. Create API Keys
- In Demo Trading mode, go to: https://www.okx.com/account/my-api
- Click "Create API Key"
- Settings:
  - **Name**: "AI Trader Testnet"
  - **Passphrase**: Create a strong passphrase (save it!)
  - **Permissions**: Trade + Read
  - **IP Whitelist**: Leave blank for testing
- Click "Confirm"
- **SAVE ALL THREE VALUES**:
  - API Key
  - Secret Key
  - Passphrase

### 4. Add to Railway Environment Variables
Go to your Railway project: https://railway.app/dashboard

Add these three environment variables:
```
OKX_TESTNET_API_KEY=your_api_key_here
OKX_TESTNET_API_SECRET=your_secret_key_here
OKX_TESTNET_PASSPHRASE=your_passphrase_here
```

### 5. Redeploy
Railway will auto-redeploy when you save the environment variables.

## Testing Your Setup

### Check Balance
Once deployed, your trading platform will:
1. Connect to OKX Testnet automatically
2. Display crypto prices from OKX (BTC/USDT, ETH/USDT, SOL/USDT, DOGE/USDT, AVAX/USDT)
3. Show your testnet balance when launching algorithms

### Launch Algorithm
1. Go to your platform: https://web-production-d85a77.up.railway.app
2. Click any crypto symbol
3. Configure algorithm (default $100 capital)
4. Click "Launch Algorithm"
5. Watch REAL orders execute on OKX Testnet

## Verification
- All trades are REAL API calls to OKX Testnet
- Orders create actual order IDs on the testnet
- Balance updates reflect real testnet transactions
- But uses virtual funds (no real money at risk)

## Switching to Live Trading
When ready for real money:
1. Switch OKX from Demo Trading back to Live Trading
2. Create new API keys in Live mode
3. Update Railway environment variables
4. Deploy

**DO NOT DO THIS until you've tested thoroughly and are ready to risk real capital!**

## Support
- OKX Demo Trading: https://www.okx.com/help/demo-trading
- OKX API Docs: https://www.okx.com/docs-v5/en/
- Our platform: https://web-production-d85a77.up.railway.app
