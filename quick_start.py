"""
Quick start script to run the AI trading system in paper mode.
This provides an easy entry point for testing the system.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import pandas
        import numpy
        import sklearn
        import fastapi
        import streamlit
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path("config/.env")
    env_example = Path("config/.env.example")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        with open(env_example) as src, open(env_file, "w") as dst:
            dst.write(src.read())
        print("✅ .env file created! Please edit it with your API keys.")


def create_logs_dir():
    """Create logs directory if it doesn't exist."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir()
        print("✅ Created logs directory")


def run_paper_trading():
    """Run the system in paper trading mode."""
    print("\n🚀 Starting AI Trading System in Paper Mode...")
    print("📊 Target: $100 → $2000 (20x multiplier)")
    print("⚠️  This is PAPER TRADING - no real money at risk")
    print("\nTo stop the system, press Ctrl+C\n")
    
    try:
        subprocess.run([sys.executable, "main.py", "paper", "--capital", "100", "--target", "2000"])
    except KeyboardInterrupt:
        print("\n🛑 Paper trading stopped by user")
    except Exception as e:
        print(f"❌ Error running paper trading: {e}")


def show_dashboard_instructions():
    """Show instructions for running the dashboard."""
    print("\n📊 MONITORING DASHBOARD")
    print("=" * 50)
    print("To view real-time trading performance, open a new terminal and run:")
    print("  streamlit run src/dashboard.py")
    print("\nThen open: http://localhost:8501")
    print("=" * 50)


def main():
    """Main quick start function."""
    print("🤖 AI Trading System - Quick Start")
    print("=" * 40)
    
    # Check and install dependencies
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install manually:")
            print("   pip install -r requirements.txt")
            return
    
    # Setup configuration
    create_env_file()
    create_logs_dir()
    
    # Show instructions
    print("\n✅ Setup complete!")
    show_dashboard_instructions()
    
    # Ask user if they want to start
    response = input("\nStart paper trading now? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        run_paper_trading()
    else:
        print("\n📚 MANUAL START COMMANDS:")
        print("  Paper trading:  python main.py paper")
        print("  Live trading:   python main.py live")
        print("  Backtesting:    python main.py backtest")
        print("  Dashboard:      streamlit run src/dashboard.py")
        print("\n📖 For more details, see README.md")


if __name__ == "__main__":
    main()