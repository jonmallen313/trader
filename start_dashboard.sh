#!/bin/bash
# AI Trader - With Dashboard (Linux/Mac)

echo "========================================"
echo "   AI TRADER - WITH DASHBOARD"
echo "========================================"
echo ""
echo "Starting trader with web dashboard..."
echo "Dashboard will be available at http://127.0.0.1:8000"
echo ""
echo "The trader runs in background even if you close the browser."
echo "Press Ctrl+C in this window to stop."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the trader with dashboard
python run.py
