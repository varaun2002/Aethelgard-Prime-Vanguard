import streamlit as st

# Page Config (Must be first)
st.set_page_config(
    page_title="Aethelgard Prime v1.0",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import plotly.graph_objects as go
import ccxt
import torch

import sys
import os
# Add project root to path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal Imports
from src.inference.predict_v3 import PredictorV3
from src.inference.paper_trader import PaperTrader
from src.inference.feedback_manager import FeedbackManager
from src.utils.enhanced_logger import EnhancedLogger, TradeLogger
from src.data.external_history import ExternalHistoryFetcher # For sentiment display
from src.utils.logger import setup_logger

logger = setup_logger("Dashboard")


# Internal Imports

import yfinance as yf

# --- CACHED RESOURCES ---
@st.cache_resource
def get_predictors():
    """Load models once"""
    p15 = PredictorV3("models_v3/model_v3_0_15m.pth")
    p1h = PredictorV3("models_v3/model_v3_0_1h.pth")
    p4h = PredictorV3("models_v3/model_v3_0_4h.pth")
    return {"15m": p15, "1h": p1h, "4h": p4h}

def fetch_live_data(symbol, timeframe, limit=100):
    """
    Fetch live data using yfinance.
    Map timeframe: 15m->15m, 1h->1h, 4h->1h (resampled) or similar.
    YF intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    try:
        # Map symbol -> YF Ticker
        yf_symbol = symbol.replace("/", "-") # ETH/USDT -> ETH-USDT (or ETH-USD)
        if "USDT" in yf_symbol: yf_symbol = yf_symbol.replace("USDT", "USD") # YF prefers USD usually

        # Map Timeframe - fetch more data to ensure enough for feature engineering
        # Feature engineering needs ~200+ rows for proper calculation (180 for normalization + buffer)
        interval_map = {
            "15m": ("15m", "60d"),   # 15min candles, 60 days → ~5600 rows
            "1h": ("1h", "730d"),    # 1h candles, 730 days (2 years) → ~17000 rows
            "4h": ("1h", "180d")     # Will resample 1h to 4h, 180 days → ~4000 rows
        }

        interval, period = interval_map.get(timeframe, ("1h", "30d"))

        df = yf.download(yf_symbol, period=period, interval=interval, progress=False)

        if df is None or len(df) == 0:
            return None

        # Standardization
        df = df.reset_index()
        # Handle multi-index columns (tuples) from yfinance
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]

        # Rename 'date' or 'datetime' -> 'timestamp'
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})

        # Ensure we have a timestamp column
        if 'timestamp' not in df.columns:
            return None

        # Convert timestamp to proper format if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Resample for 4h if needed
        if timeframe == "4h":
            # Simple resampling
            df = df.set_index('timestamp')
            df_4h = df.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            df = df_4h.reset_index()

        # Ensure enough data
        if len(df) < 200:
            logger.warning(f"Only {len(df)} rows for {timeframe}, may not be enough for features")

        return df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# --- PAGES ---
def render_diagnostics_page():
    st.header("🧠 Model Diagnostics & Predictions")
    
    # Load Logs
    log_files = {
        "15m Model": "logs/predictions/predictions_15m.csv",
        "1h Model": "logs/predictions/predictions_1h.csv",
        "4h Model": "logs/predictions/predictions_4h.csv"
    }
    
    tabs = st.tabs(list(log_files.keys()))
    
    for i, (model_name, file_path) in enumerate(log_files.items()):
        with tabs[i]:
            if os.path.exists(file_path):
                try:
                    # Safe read
                    try:
                        df = pd.read_csv(file_path)
                    except pd.errors.EmptyDataError:
                         st.warning(f"Log file {model_name} is empty (initializing...)")
                         continue
                         
                    if df is not None and not df.empty and 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.sort_values('timestamp', ascending=False)
                        
                        # 1. Stats
                        st.markdown(f"### {model_name} Performance (Last 100)")
                        cols = st.columns(4)
                        recent = df.head(100)
                        
                        avg_conf = recent['confidence'].mean()
                        max_conf = recent['confidence'].max()
                        up_signals = len(recent[recent['predicted_direction'] == 'UP'])
                        down_signals = len(recent[recent['predicted_direction'] == 'DOWN'])
                        
                        cols[0].metric("Avg Confidence", f"{avg_conf:.1%}")
                        cols[1].metric("Max Confidence", f"{max_conf:.1%}")
                        cols[2].metric("Bullish Signals", up_signals)
                        cols[3].metric("Bearish Signals", down_signals)
                        
                        # 2. Charts
                        st.markdown("### Confidence Trend")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=recent['date'], y=recent['confidence'], mode='lines+markers', name='Confidence'))
                        fig.add_hline(y=0.47, line_dash="dash", line_color="red", annotation_text="Threshold")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 3. Data Table
                        st.markdown("### Prediction Log")
                        st.dataframe(recent.head(50), use_container_width=True)
                    else:
                         st.warning("Log file format incorrect (missing timestamp)")
                         st.dataframe(df)

                except Exception as e:
                    st.error(f"Error reading log: {e}")
            else:
                st.warning(f"No log file found for {model_name}")

# --- MAIN APP ---
def main():
    st.sidebar.title("🔮 Aethelgard Prime")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Model Diagnostics"])
    
    if page == "Model Diagnostics":
        render_diagnostics_page()
        return

    st.title("🔮 Aethelgard Prime v1.0")
    
    # Sidebar
    st.sidebar.header("Configuration") # Corrected typo from "Configuration")l Panel"
    auto_refresh = st.sidebar.checkbox("Auto-Refresh (60s)", value=True)
    symbol = st.sidebar.text_input("Symbol", "ETH/USDT")
    
    st.sidebar.markdown("### ⚙️ Strategy Settings")
    selected_tf = st.sidebar.selectbox("Driver Timeframe", ["15m", "1h", "4h"], index=1) # Default 1h
    strategy_mode = st.sidebar.selectbox("Strategy Mode", ["Simple (Driver Only)", "Consensus (15m + 1h)"], index=1) # Default Consensus

    # Initialize Components
    try:
        predictors = get_predictors()
    except Exception as e:
        st.error(f"Failed to load models. Make sure 'models_v3' has .pth files. {e}")
        return

    trader = PaperTrader()
    feedback = FeedbackManager()
    enhanced_logger = EnhancedLogger()
    trade_logger = TradeLogger()

    # Container for Live Updates
    main_display = st.empty()
    
    # Loop
    if auto_refresh:
        time_to_wait = 60 # 60 seconds
        count = st.empty()
        
    # Main container
    with main_display.container():
        # 1. Fetch Data & Predictions
        results = {}
        current_price = 0.0
        df_chart = None # Use driver TF for chart
        
        # Helper for Consensus
        signals = {'15m': 'NEUTRAL', '1h': 'NEUTRAL', '4h': 'NEUTRAL'}
        confidences = {'15m': 0.0, '1h': 0.0, '4h': 0.0}

        for tf, predictor in predictors.items():
            df = fetch_live_data(symbol, tf)
            if df is not None:
                try:
                    # Keep df for driver TF for charting
                    if tf == selected_tf:
                        df_chart = df

                    current_price = df['close'].iloc[-1]
                    res = predictor.predict(df)
                    results[tf] = res
                    
                    # Store signal for consensus
                    signals[tf] = res['predicted_direction']
                    confidences[tf] = res['confidence']

                    # LOGGING & FEEDBACK
                    if res:
                        feedback.log_prediction(tf, res, current_price)
                        gate_info = res.get('gate_info', {'gate1_passed': True, 'gate2_passed': True, 'gate3_applied': False})
                        enhanced_logger.log_prediction(tf, res, current_price, gate_info, None)
                                
                except Exception as e:
                    st.error(f"Error {tf}: {e}")

        # 2. TRADE LOGIC (v3.2)
        trade_signal = "HOLD"
        trade_conf = 0.0
        trade_executed = False
        
        # Determined by Strategy
        if strategy_mode.startswith("Simple"):
            # Driver Only
            trade_signal = signals.get(selected_tf, "HOLD")
            trade_conf = confidences.get(selected_tf, 0.0)
        else:
            # Consensus: Require 15m AND 1h to match for Entry
            s15 = signals.get('15m', 'HOLD')
            s1h = signals.get('1h', 'HOLD')
            
            # Entry Logic
            if s15 == "UP" and s1h == "UP":
                trade_signal = "UP"
                trade_conf = min(confidences['15m'], confidences['1h']) # Conservative confidence
            elif s15 == "DOWN" or s1h == "DOWN":
                # Exit if EITHER is bearish (Safety First)
                trade_signal = "DOWN"
                trade_conf = max(confidences['15m'], confidences['1h'])
            else:
                trade_signal = "HOLD" # Mixed signals = Sit on hands

        # Execute on Trader
        # We pass the calculated consensus signal to the trader
        # The Trader handles Position Sizing, Fees, and Trailing Stops internally
        portfolio_before = {
            'balance': trader.state['balance_usdt'],
            'position': trader.state['position_eth'],
            'equity': trader.get_portfolio_value(current_price)
        }
        
        action = trader.update(
            trade_signal,
            trade_conf,
            current_price,
            int(time.time())
        )
        
        if action:
            portfolio_after = {
                'balance': trader.state['balance_usdt'],
                'position': trader.state['position_eth'],
                'equity': trader.get_portfolio_value(current_price)
            }
            color = "green" if "BUY" in action else "red"
            st.toast(f"🤖 Action: {action} @ {current_price:.2f}", icon="🤖")
            
            # Log Trade
            trade_size = abs(portfolio_after['position'] - portfolio_before['position'])
            trade_logger.log_trade(
                action=action, price=current_price, size=trade_size,
                confidence=trade_conf, timeframe=selected_tf,
                predicted_direction=trade_signal,
                portfolio_before=portfolio_before, portfolio_after=portfolio_after
            )
            # Re-fetch state for UI
            trade_executed = True

        # 3. Check Feedback
        if current_price > 0:
            feedback.check_accuracy(current_price, int(time.time()))

        # --- RENDER UI (v3.2) ---
        
        # Row A: Market & Position Status
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        with col_a1:
            st.metric("Live Price (" + symbol + ")", f"${current_price:,.2f}")
        with col_a2:
            stats = trader.get_stats()
            st.metric("Sim Equity", f"${stats['current_equity']:,.2f}", f"{stats['pnl_percent']:.2f}%")
        with col_a3:
            st.metric("Position", f"{stats['open_position']:.4f} ETH")
        with col_a4:
            # LIVE PnL Calculation
            if stats['open_position'] > 0:
                entry = trader.state.get('avg_entry_price', current_price)
                if entry > 0:
                    pnl_unrealized = (current_price - entry) / entry * 100
                    pnl_val = (current_price - entry) * stats['open_position']
                    color = "normal"
                    st.metric("Open PnL", f"${pnl_val:.2f}", f"{pnl_unrealized:.2f}%")
                else:
                    st.metric("Open PnL", "$0.00", "0.00%")
            else:
                st.metric("Open PnL", "Flat", "0.00%")

        # Row B: The "Master Chart" (Candlesticks + Markers)
        if df_chart is not None and len(df_chart) > 0:
            st.markdown("### 🕯️ Market View")
            fig = go.Figure(data=[go.Candlestick(x=df_chart['timestamp'],
                            open=df_chart['open'],
                            high=df_chart['high'],
                            low=df_chart['low'],
                            close=df_chart['close'],
                            name=symbol)])
            
            # Add Buy/Sell Markers from History
            trades = trader.state.get('trades', [])
            if trades:
                buys = [t for t in trades if t['action'] == 'BUY']
                sells = [t for t in trades if 'SELL' in t['action']] # TP_SELL, SL_SELL, SELL
                
                # Convert timestamps to datetime for plotting matches
                # Note: This is an approximation if dates don't exactly align, but good enough for visual
                if buys:
                    buy_dates = [t['date'] for t in buys]
                    buy_prices = [t['price'] for t in buys]
                    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', 
                                            marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy'))
                if sells:
                    sell_dates = [t['date'] for t in sells]
                    sell_prices = [t['price'] for t in sells]
                    fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', 
                                            marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell'))
            
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0),
                              xaxis_rangeslider_visible=False, # Hide slider to save space
                              yaxis=dict(autorange=True, fixedrange=False)) # Dynamic Y
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Row C: Model Consensus View
        st.subheader("🤖 AI Consensus Matrix")
        cols = st.columns(3)
        timeframes = ["15m", "1h", "4h"]
        
        for i, tf in enumerate(timeframes):
            if tf in results and results[tf]:
                res = results[tf]
                with cols[i]:
                    # Highlight if this is the Driver
                    header = f"{tf} Model"
                    if tf == selected_tf: header += " (DRIVER)"
                    st.markdown(f"**{header}**")
                    
                    direction = res['predicted_direction']
                    color = "gray"
                    if direction == "UP": color = "green"
                    elif direction == "DOWN": color = "red"
                    
                    st.markdown(f"## :{color}[{direction}]")
                    st.progress(res['confidence'], text=f"Conf: {res['confidence']:.1%}")
                    
                    if res.get('caution'):
                        st.warning("⚠️ Uncertainty")
            else:
                with cols[i]:
                    st.write(f"{tf}: ...")
        
        st.markdown("---")
        
        # Row D: Performance Trends (Rolling Accuracy & Equity)
        st.subheader("📈 Live Performance Tracking")
        
        tab1, tab2, tab3 = st.tabs(["Rolling Accuracy", "Equity Curve", "Trade Logs"])
        
        with tab1:
            # Use the legacy log file because that's where validation happens
            if os.path.exists("data/prediction_log.csv"):
                try:
                    df_acc = pd.read_csv("data/prediction_log.csv")
                    if len(df_acc) > 0 and 'outcome' in df_acc.columns:
                        # Filter validated
                        validated = df_acc[df_acc['outcome'].isin(['GOOD', 'BAD', 'PERFECT', 'WRONG'])].copy()
                        if len(validated) > 10:
                            validated['is_correct'] = validated['outcome'].isin(['GOOD', 'PERFECT']).astype(int)
                            validated['rolling_acc'] = validated['is_correct'].rolling(window=20).mean()
                            
                            fig_acc = go.Figure()
                            fig_acc.add_trace(go.Scatter(y=validated['rolling_acc'], mode='lines', name='Rolling Accuracy (20)'))
                            fig_acc.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Random")
                            fig_acc.add_hline(y=0.6, line_dash="dash", line_color="green", annotation_text="Target")
                            fig_acc.update_layout(title="Rolling Accuracy (Last 20 validated)", height=300, 
                                                  yaxis=dict(range=[0, 1.05])) # Fixed 0-100% scale
                            st.plotly_chart(fig_acc, use_container_width=True)
                        else:
                            st.info(f"Need more validated predictions ({len(validated)}/10).")
                except pd.errors.EmptyDataError:
                    st.warning("Prediction log is empty")
                except Exception as e:
                    st.error(f"Chart Error: {e}")
            else:
                st.warning("data/prediction_log.csv not found")
                    
        with tab2:
            hist = trader.state['equity_history']
            if hist:
                hist_df = pd.DataFrame(hist)
                if not hist_df.empty:
                    # Use Plotly for proper scaling (st.line_chart can be stiff)
                    fig_eq = go.Figure()
                    
                    # Ensure we have a datetime column
                    if 'date' not in hist_df.columns:
                        hist_df['date'] = pd.to_datetime(hist_df['timestamp'], unit='s')
                    else:
                        hist_df['date'] = pd.to_datetime(hist_df['date'])

                    # Filter for Today Only (User Request)
                    today_start = pd.Timestamp.now().normalize()
                    hist_df = hist_df[hist_df['date'] >= today_start]
                    
                    fig_eq.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['equity'], 
                                              mode='lines', name='Equity',
                                              line=dict(color='blue', width=2)))
                    fig_eq.update_layout(title="Portfolio Equity (Today)", 
                                       xaxis_title="Time", 
                                       yaxis_title="USDT Value", 
                                       height=300,
                                       yaxis=dict(autorange=True)) # Ensure dynamic scaling
                    st.plotly_chart(fig_eq, use_container_width=True)
            else:
                st.write("No equity history yet.")

        with tab3:
            if os.path.exists("logs/trades.csv"):
                try:
                    df_trades = pd.read_csv("logs/trades.csv")
                    if df_trades is not None and not df_trades.empty:
                        st.dataframe(df_trades.sort_values("timestamp", ascending=False))
                        
                        csv = df_trades.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Trade Logs",
                            csv,
                            "trade_logs.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.info("No trades logged yet.")
                except pd.errors.EmptyDataError:
                    st.info("Trade log is empty.")
                except Exception as e:
                    st.error(f"Error reading trade logs: {e}")
            else:
                st.info("No trade logs found (logs/trades.csv).")

        # Row D: System Health
        # (kept previous logic for analytics if needed, or simplified)
        
        today = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Last Update: {today}")

    # AUTO REFRESH LOGIC
    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
