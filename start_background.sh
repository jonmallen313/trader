#!/bin/bash
# AI Trader - Background Service (Linux/Mac)

echo "========================================"
echo "   AI TRADER - BACKGROUND SERVICE"
echo "========================================"
echo ""
echo "Starting trader in background mode..."
echo "The trader will run continuously."
echo ""
echo "Press Ctrl+C to stop the trader."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the background trader
python background_trader.py
