
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import src.inference.predict_v3
importlib.reload(src.inference.predict_v3)
from src.inference.predict_v3 import PredictorV3
from src.inference.paper_trader import PaperTrader
from src.inference.feedback_manager import FeedbackManager
from src.utils.enhanced_logger import EnhancedLogger, TradeLogger

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardAggressive")

# Page Config
st.set_page_config(
    page_title="Aethelgard Vanguard (15m)",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import yfinance as yf

# --- CACHED RESOURCES ---
@st.cache_resource
def get_predictors_aggressive_v3():
    """Load models once (Cache Busted v3 - Lazy Import)"""
    import importlib
    import src.inference.predict_v3
    importlib.reload(src.inference.predict_v3)
    from src.inference.predict_v3 import PredictorV3
    
    p15 = PredictorV3("models_v3/model_v3_0_15m.pth")
    return {"15m": p15}

def fetch_live_data(symbol, timeframe, limit=100):
    try:
        yf_symbol = symbol.replace("/", "-") 
        if "USDT" in yf_symbol: yf_symbol = yf_symbol.replace("USDT", "USD")

        interval = "15m"
        period = "60d"

        df = yf.download(yf_symbol, period=period, interval=interval, progress=False)

        if df is None or len(df) == 0:
            return None

        # Standardization
        df = df.reset_index()
        # Handle multi-index columns (tuples) from yfinance
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]

        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})

        if 'timestamp' not in df.columns:
            return None

        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- PAGES ---
def render_diagnostics_page():
    st.header("⚡ Vanguard Diagnostics")
    log_file = "logs/predictions/predictions_15m.csv"
    
    if os.path.exists(log_file):
        try:
            try:
                df = pd.read_csv(log_file)
            except pd.errors.EmptyDataError:
                st.warning("Log file is currently empty.")
                return

            if df is not None and not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp', ascending=False)
                
                # Stats
                st.subheader("15m Model Performance (Last 100)")
                c1, c2, c3, c4 = st.columns(4)
                recent = df.head(100)
                
                avg_conf = recent['confidence'].mean()
                max_conf = recent['confidence'].max()
                
                c1.metric("Avg Confidence", f"{avg_conf:.1%}")
                c2.metric("Max Confidence", f"{max_conf:.1%}")
                c3.metric("Bullish", len(recent[recent['predicted_direction'] == 'UP']))
                c4.metric("Bearish", len(recent[recent['predicted_direction'] == 'DOWN']))
                
                # Chart
                st.subheader("Confidence Trend")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent['date'], y=recent['confidence'], mode='lines+markers', name='Confidence'))
                fig.add_hline(y=0.47, line_dash="dash", line_color="red", annotation_text="Limit")
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.subheader("Live Logs")
                st.dataframe(recent.head(50), use_container_width=True)
            else:
                st.warning("Invalid log format")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("No logs found.")

# --- MAIN APP ---
def main():
    st.sidebar.title("⚔️ Vanguard")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Diagnostics"])
    
    if page == "Diagnostics":
        render_diagnostics_page()
        return

    st.title("⚔️ Aethelgard: Vanguard (15m Aggressive)")
    st.markdown("### ⚡ High Frequency Scalping Mode")
    
    # Sidebar
    st.sidebar.header("Vanguard Control")
    auto_refresh = st.sidebar.checkbox("Auto-Refresh (60s)", value=True)
    symbol = st.sidebar.text_input("Symbol", "ETH/USDT")
    
    # HARDCODED AGGRESSIVE SETTINGS
    selected_tf = "15m"
    strategy_mode = "Simple (Driver Only)" 
    
    st.sidebar.info(f"Strategy: {strategy_mode}")
    st.sidebar.info(f"Driver: {selected_tf}")

    # Initialize Components
    try:
        predictors = get_predictors_aggressive_v3()
        model = predictors['15m']
    except Exception as e:
        st.error(f"Failed to load 15m model. {e}")
        return

    # USE SEPARATE PORTFOLIO
    trader = PaperTrader(state_file="data/paper_portfolio_aggressive.json", confidence_threshold=0.1)
    feedback = FeedbackManager()
    enhanced_logger = EnhancedLogger()
    # Separate trade log
    trade_logger = TradeLogger(log_file="logs/trades_aggressive.csv")

    # Container for Live Updates
    main_display = st.empty()
    
    # Loop
    if auto_refresh:
        time_to_wait = 60 # 60 seconds
        count = st.empty()
        
    # Main container
    with main_display.container():
        # Get Data
        df = fetch_live_data(symbol, "15m")
        
        if df is not None:
            current_price = df['close'].iloc[-1]
            last_timestamp = df['timestamp'].iloc[-1]
            
            # --- 1. RUN PREDICTION ---
            try:
                # Predict (Standard V3 Call)
                pred = model.predict(df, safety_lock=False) # remove old aggressive overrides, rely on PaperTrader dynamic logic
                
                # --- DISPLAY ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Price", f"${current_price:,.2f}")
                with col2:
                    direction = pred['predicted_direction']
                    color = "green" if direction == "UP" else "red" if direction == "DOWN" else "gray"
                    st.metric("Vanguard Signal", direction, delta_color="normal" if direction == "FLAT" else "inverse")
                with col3:
                    st.metric("Confidence", f"{pred['confidence']*100:.1f}%")

                # --- TRADING LOGIC (AGGRESSIVE) ---
                ts_int = int(time.time())
                
                # Get Stats for Portfolio
                p_stats = trader.get_stats()
                
                portfolio_before = {
                    'balance': trader.state['balance_usdt'],
                    'position': trader.state['position_eth'],
                    'equity': p_stats['current_equity']
                }

                # Execute Trade
                current_volume = df['volume'].iloc[-1]
                quote_volume = current_volume * current_price
                
                # Extract Probs if available (Model Predictor should return them)
                # Need to verify PredictorV3 returns probs. If not, we might need to update PredictorV3 too.
                # Assuming PredictorV3.predict returns 'probs' key if we updated it?
                # Actually we updated TrainerV3.inference, but did we update PredictorV3 class in src/inference/predict_v3.py?
                # We need to check PredictorV3.predict first.
                probs = pred.get('probs', None)

                action = trader.update(
                    str(direction), 
                    float(pred['confidence']), 
                    current_price, 
                    ts_int,
                    recent_prices=df['close'].values,
                    volume=quote_volume,
                    recent_candles=df, # Pass DF for Regime Detection
                    probs=probs
                )
                
                # Log Prediction
                enhanced_logger.log_prediction(
                    "15m", pred, current_price, 
                    gate_info={'gate1_passed':True, 'gate2_passed':True, 'gate3_applied':False},
                    trade_info={'signal': direction, 'executed': action is not None},
                    symbol=symbol,
                    model_version="v3.0 Aggressive"
                )
                
                if action:
                    st.success(f"⚔️ VANGUARD ACTION: {action} at ${current_price:.2f}")
                    
                    # Log to CSV
                    p_stats_after = trader.get_stats()
                    portfolio_after = {
                        'balance': trader.state['balance_usdt'],
                        'position': trader.state['position_eth'],
                        'equity': p_stats_after['current_equity']
                    }
                    trade_size = abs(portfolio_after['position'] - portfolio_before['position'])
                    
                    trade_logger.log_trade(
                        action=action, price=current_price, size=trade_size,
                        confidence=pred['confidence'], timeframe="15m",
                        predicted_direction=str(direction),
                        portfolio_before=portfolio_before, portfolio_after=portfolio_after
                    )

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                import traceback
                st.write(traceback.format_exc())

        # --- PORTFOLIO ---
        st.divider()
        st.subheader("🎒 Vanguard Portfolio (Paper)")
        
        portfolio = trader.get_stats()
        p_cols = st.columns(4)
        # Fix keys: PaperTrader.get_stats returns current_equity, pnl_percent, open_position
        balance = trader.state['balance_usdt'] # Direct access for balance
        p_cols[0].metric("Balance (USDT)", f"${balance:,.2f}")
        p_cols[1].metric("Position (ETH)", f"{portfolio['open_position']:.4f}")
        p_cols[2].metric("Equity (Total)", f"${portfolio['current_equity']:,.2f}")
        
        pnl_color = "normal"
        if portfolio['pnl_percent'] > 0: pnl_color = "normal"
        p_cols[3].metric("PnL %", f"{portfolio['pnl_percent']:.2f}%")

        # --- CHARTS ---
        tab1, tab2, tab3 = st.tabs(["Price Action", "Portfolio History", "Trade Logs"])
        
        with tab1:
            # Simple Close Price Chart using internal Streamlit
            if df is not None:
                st.line_chart(df.set_index('timestamp')['close'])

        with tab2:
            hist = trader.state['equity_history']
            if len(hist) > 0:
                hist_df = pd.DataFrame(hist)
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], unit='s')
                
                # Filter for Today
                today_start = pd.Timestamp.now().normalize()
                hist_df = hist_df[hist_df['timestamp'] >= today_start]
                
                st.line_chart(hist_df.set_index('timestamp')['equity'])
            else:
                st.write("No history yet.")

        with tab3:
            if os.path.exists("logs/trades_aggressive.csv"):
                try:
                    try:
                        df_trades = pd.read_csv("logs/trades_aggressive.csv")
                    except pd.errors.EmptyDataError:
                        st.info("Trade log is empty.")
                        df_trades = pd.DataFrame()
                        
                    if not df_trades.empty:
                        st.dataframe(df_trades.sort_values("timestamp", ascending=False))
                    else:
                        st.write("Log file exists but is empty.")
                    csv = df_trades.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Logs", csv, "trades_aggressive.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error reading logs: {e}")
            else:
                st.info("No trades yet in using aggressive strategy.")

    # Auto Refresh Logic
    if auto_refresh:
        for i in range(time_to_wait, 0, -1):
            count.caption(f"Refreshing in {i}s...")
            time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
