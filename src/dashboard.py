"""
Real-time monitoring dashboard for the AI trading system.
Built with Streamlit for interactive visualization of trading performance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import time
import requests
import json
from typing import Dict, List, Optional

# Set page config
st.set_page_config(
    page_title="AI Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


class TradingDashboard:
    """Real-time trading dashboard."""
    
    def __init__(self, webhook_url: str = "http://localhost:8000"):
        self.webhook_url = webhook_url
        self.refresh_interval = 5  # seconds
        
    def fetch_data(self, endpoint: str) -> Optional[Dict]:
        """Fetch data from the trading system API."""
        try:
            response = requests.get(f"{self.webhook_url}/{endpoint}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")
            return None
    
    def render_header(self):
        """Render dashboard header."""
        st.title("🤖 AI Trading System Dashboard")
        st.markdown("---")
        
        # System status
        status_data = self.fetch_data("status")
        if status_data:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                status_color = "🟢" if status_data["status"] == "running" else "🔴"
                st.metric("Status", f"{status_color} {status_data['status'].title()}")
            
            with col2:
                uptime_hours = status_data["uptime"] / 3600
                st.metric("Uptime", f"{uptime_hours:.1f}h")
            
            with col3:
                st.metric("Open Positions", status_data["open_positions"])
            
            with col4:
                progress = status_data["progress_to_target"]
                st.metric("Progress to Target", f"{progress:.1f}%")
                
            with col5:
                total_pnl = status_data["realized_pnl"] + status_data["unrealized_pnl"]
                color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Total P&L", f"${total_pnl:.2f}", delta=f"${status_data['unrealized_pnl']:.2f}", delta_color=color)
    
    def render_pnl_chart(self):
        """Render P&L performance chart."""
        st.subheader("📊 P&L Performance")
        
        # Get signal history to build P&L timeline
        history_data = self.fetch_data("signals/history?limit=100")
        
        if history_data and history_data.get("signals"):
            # Create mock P&L timeline (in real implementation, this would come from trade history)
            signals = history_data["signals"]
            
            # Generate sample P&L data
            timestamps = []
            cumulative_pnl = []
            current_pnl = 0
            
            for signal in signals:
                timestamp = signal.get("timestamp", datetime.now().isoformat())
                timestamps.append(pd.to_datetime(timestamp))
                
                # Mock P&L calculation (replace with actual trade results)
                pnl_change = np.random.uniform(-5, 10)  # Random P&L for demo
                current_pnl += pnl_change
                cumulative_pnl.append(current_pnl)
            
            if timestamps and cumulative_pnl:
                df = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Cumulative P&L': cumulative_pnl
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Timestamp'],
                    y=df['Cumulative P&L'],
                    mode='lines+markers',
                    name='P&L',
                    line=dict(color='#00ff88', width=2),
                    fill='tonexty' if df['Cumulative P&L'].iloc[-1] > 0 else None,
                    fillcolor='rgba(0, 255, 136, 0.1)'
                ))
                
                # Add target line
                target_line = 2000  # $2000 target
                fig.add_hline(y=target_line, line_dash="dash", line_color="gold", 
                            annotation_text=f"Target: ${target_line}")
                
                fig.update_layout(
                    title="Cumulative P&L Over Time",
                    xaxis_title="Time",
                    yaxis_title="P&L ($)",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No P&L data available yet. Start trading to see performance metrics.")
    
    def render_positions_table(self):
        """Render current positions table."""
        st.subheader("💼 Current Positions")
        
        positions_data = self.fetch_data("positions")
        
        if positions_data and positions_data.get("positions"):
            positions = positions_data["positions"]
            
            df = pd.DataFrame(positions)
            
            # Format the dataframe for display
            if not df.empty:
                df['side'] = df['side'].apply(lambda x: "🟢 LONG" if x.lower() == "long" else "🔴 SHORT")
                df['entry_price'] = df['entry_price'].apply(lambda x: f"${x:.4f}")
                df['tp_price'] = df['tp_price'].apply(lambda x: f"${x:.4f}")
                df['sl_price'] = df['sl_price'].apply(lambda x: f"${x:.4f}")
                df['current_pnl'] = df['current_pnl'].apply(lambda x: f"${x:.2f}")
                df['opened_at'] = pd.to_datetime(df['opened_at']).dt.strftime('%H:%M:%S')
                
                # Reorder columns
                display_cols = ['symbol', 'side', 'size', 'entry_price', 'tp_price', 'sl_price', 'current_pnl', 'opened_at']
                df_display = df[display_cols]
                
                # Rename columns for display
                df_display.columns = ['Symbol', 'Side', 'Size', 'Entry Price', 'TP Price', 'SL Price', 'Unrealized P&L', 'Opened']
                
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("No open positions")
        else:
            st.info("No positions data available")
    
    def render_signal_history(self):
        """Render recent signal history."""
        st.subheader("📡 Recent Signals")
        
        history_data = self.fetch_data("signals/history?limit=10")
        
        if history_data and history_data.get("signals"):
            signals = history_data["signals"]
            
            for signal in reversed(signals):  # Show newest first
                signal_data = signal.get("signal", {})
                timestamp = signal.get("timestamp", "")
                
                with st.expander(f"{signal_data.get('symbol', 'N/A')} - {signal_data.get('action', 'N/A')} @ {timestamp[:19]}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Symbol", signal_data.get('symbol', 'N/A'))
                    with col2:
                        action = signal_data.get('action', 'N/A')
                        color = "🟢" if action in ['BUY', 'LONG'] else "🔴"
                        st.metric("Action", f"{color} {action}")
                    with col3:
                        st.metric("TP %", f"{signal_data.get('tp_pct', 0)*100:.1f}%")
                    with col4:
                        st.metric("SL %", f"{signal_data.get('sl_pct', 0)*100:.1f}%")
                    
                    if signal_data.get('price'):
                        st.write(f"**Price:** ${signal_data['price']:.4f}")
                    if signal_data.get('strategy'):
                        st.write(f"**Strategy:** {signal_data['strategy']}")
        else:
            st.info("No recent signals")
    
    def render_controls(self):
        """Render system controls in sidebar."""
        st.sidebar.title("🎛️ Controls")
        
        # System control buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                response = requests.post(f"{self.webhook_url}/control/start")
                if response.status_code == 200:
                    st.success("System started!")
                else:
                    st.error("Failed to start system")
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                response = requests.post(f"{self.webhook_url}/control/stop")
                if response.status_code == 200:
                    st.success("System stopped!")
                else:
                    st.error("Failed to stop system")
        
        st.sidebar.markdown("---")
        
        # Manual signal form
        st.sidebar.subheader("📝 Manual Signal")
        
        with st.sidebar.form("manual_signal"):
            symbol = st.text_input("Symbol", value="BTCUSDT")
            action = st.selectbox("Action", ["BUY", "SELL"])
            tp_pct = st.number_input("TP %", min_value=0.1, max_value=10.0, value=2.0, step=0.1) / 100
            sl_pct = st.number_input("SL %", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100
            
            submitted = st.form_submit_button("Send Signal")
            
            if submitted:
                signal_data = {
                    "symbol": symbol,
                    "action": action,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "confidence": 1.0,
                    "strategy": "Manual"
                }
                
                try:
                    response = requests.post(
                        f"{self.webhook_url}/webhook/manual",
                        json=signal_data,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        st.success("Signal sent successfully!")
                    else:
                        st.error(f"Failed to send signal: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error sending signal: {e}")
        
        st.sidebar.markdown("---")
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        
        self.refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=30,
            value=self.refresh_interval,
            step=1
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        
        if auto_refresh:
            time.sleep(self.refresh_interval)
            st.experimental_rerun()
    
    def render_metrics_overview(self):
        """Render key trading metrics."""
        st.subheader("📈 Trading Metrics")
        
        # Mock metrics (replace with actual calculation from trade history)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Win Rate", "65.2%", delta="2.1%")
        
        with col2:
            st.metric("Average Win", "$12.35", delta="$1.20")
        
        with col3:
            st.metric("Average Loss", "$6.80", delta="-$0.50")
        
        with col4:
            st.metric("Sharpe Ratio", "1.85", delta="0.15")
        
        # Risk metrics
        st.markdown("### Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Drawdown", "-3.2%")
        
        with col2:
            st.metric("Daily VaR (95%)", "$15.60")
        
        with col3:
            st.metric("Exposure", "45%")
    
    def render_symbol_performance(self):
        """Render performance by symbol."""
        st.subheader("🎯 Symbol Performance")
        
        # Mock data for symbol performance
        symbols_data = {
            'BTCUSDT': {'trades': 45, 'pnl': 125.30, 'win_rate': 0.67},
            'ETHUSDT': {'trades': 32, 'pnl': 89.15, 'win_rate': 0.63},
            'ADAUSDT': {'trades': 28, 'pnl': -12.45, 'win_rate': 0.54},
            'DOTUSDT': {'trades': 21, 'pnl': 56.78, 'win_rate': 0.71}
        }
        
        df = pd.DataFrame.from_dict(symbols_data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Symbol'}, inplace=True)
        
        # Create performance chart
        fig = px.bar(
            df, 
            x='Symbol', 
            y='pnl', 
            color='win_rate',
            title="P&L by Symbol",
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_main_dashboard(self):
        """Render the main dashboard."""
        # Header with key metrics
        self.render_header()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_pnl_chart()
            self.render_positions_table()
        
        with col2:
            self.render_signal_history()
        
        # Additional metrics
        st.markdown("---")
        self.render_metrics_overview()
        
        # Symbol performance
        self.render_symbol_performance()
        
        # Controls in sidebar
        self.render_controls()


# Main app
def main():
    """Main Streamlit app."""
    try:
        import numpy as np
    except ImportError:
        st.error("NumPy is required but not installed. Please install with: pip install numpy")
        return
    
    # Initialize dashboard
    dashboard = TradingDashboard()
    
    # Check API connection
    health_check = dashboard.fetch_data("health")
    
    if health_check:
        st.sidebar.success("✅ Connected to Trading System")
        dashboard.render_main_dashboard()
    else:
        st.error("❌ Cannot connect to trading system")
        st.info("Make sure the trading system is running on http://localhost:8000")
        
        # Show connection help
        with st.expander("Connection Help"):
            st.markdown("""
            **To connect to your trading system:**
            
            1. Make sure your trading system is running
            2. The webhook server should be accessible at `http://localhost:8000`
            3. Check firewall settings
            4. Verify the system is not in maintenance mode
            
            **Test the connection manually:**
            ```bash
            curl http://localhost:8000/health
            ```
            """)


if __name__ == "__main__":
    main()