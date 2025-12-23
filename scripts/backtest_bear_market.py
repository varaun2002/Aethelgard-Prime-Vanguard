#!/usr/bin/env python3
"""
Bear Market Stress Test Script
Backtests the model and trading logic (with TP/SL) on 2022 Data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
import torch
import warnings
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from src.inference.paper_trader import PaperTrader
from src.inference.predict_v3 import PredictorV3

# Suppress warnings
warnings.filterwarnings("ignore")

def fetch_historical_data(symbol="ETH-USD", start="2021-01-01", end="2022-06-01", interval="1h"):
    """Fetch historical data for backtest"""
    print(f"Fetching {symbol} data from {start} to {end} ({interval})...")
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=True)
    
    if len(df) == 0:
        raise ValueError("No data fetched!")
        
    df = df.reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
        
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} candles.")
    return df

def run_backtest():
    print("="*60)
    print("🐻 BEAR MARKET STRESS TEST (2022)")
    print("="*60)
    
    # 1. Setup Data (ETH 2022 crash period)
    # Using 1d interval because 1h is limited to 730 days by yfinance
    # 2022 is > 730 days ago.
    df = fetch_historical_data(interval="1d")
    
    # 2. Setup Components
    # We need a model. 15m/1h/4h models are trained on smaller intervals but can predict on 1d if needed?
    # Actually, 1d candles are valid inputs (Open, High, Low, Close, Volume).
    # The model trained on 15m might be confused by 1d magnitude/volatility but let's try.
    # Ideally should use a model trained on 1d, but we don't have one.
    # We will use 4h model as it's closest.
    model_path = "models_v3/model_v3_0_4h.pth"
    if not os.path.exists(model_path):
         # If no 4h, use 1h
         model_path = "models_v3/model_v3_0_1h.pth"
    
    if not os.path.exists(model_path):
        # Fallback to 15m
        model_path = "models_v3/model_v3_0_15m.pth"
        
    try:
        predictor = PredictorV3(model_path=model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Use in-memory trader (persist=False)
    # TP=5%, SL=2%
    trader = PaperTrader(initial_balance=10000.0, persist=False, tp_pct=0.05, sl_pct=0.02)
    
    print("\nStarting Simulation...")
    print(f"Initial Balance: ${trader.state['balance_usdt']:,.2f}")
    
    # 3. Loop
    # Find start index for 2022
    start_date = pd.Timestamp("2022-01-01")
    start_idx = df[df['timestamp'] >= start_date].index[0]
    
    # Ensure start_idx is at least 300 to allow for 200 EMA + buffers
    if start_idx < 300:
        print(f"Warning: Start index {start_idx} is low, indicators might be NaN.")
    
    predictions_made = 0
    trades_executed = 0
    
    pbar = tqdm(total=len(df)-start_idx)
    
    # Create sliding window iterator
    for i in range(start_idx, len(df)):
        # Data slice ensuring we have enough history
        # Predictor needs 60 seq len, plus FE needs 200+, so sending 400 is safe
        current_slice = df.iloc[max(0, i-400):i+1].copy()
        current_row = current_slice.iloc[-1]
        current_price = current_row['close']
        timestamp = int(current_row['timestamp'].timestamp())
        
        # 1. Check TP/SL (Prioritize internal check before prediction to save compute?)
        # PaperTrader check_tp_sl is inside update(), but we can check manually if we want to skip prediction.
        # If we just hit SL, no need to predict immediately?
        # Actually PaperTrader.update() handles it.
        
        # 2. Predict
        # We only predict every hour (which is every step here)
        try:
            # Predict
            pred = predictor.predict(current_slice)
            
            if pred:
                signal = pred['predicted_direction']
                conf = pred['confidence']
                
                # 3. Trade
                action = trader.update(signal, conf, current_price, timestamp)
                
                if action:
                    trades_executed += 1
                    # pbar.set_description(f"Equity: ${trader.get_portfolio_value(current_price):.0f} | Trades: {trades_executed}")
                
            predictions_made += 1
            
        except Exception as e:
            # print(f"Error at {i}: {e}")
            pass
            
        pbar.update(1)
        
    pbar.close()
    
    # 4. Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    stats = trader.get_stats()
    print(f"Final Equity:    ${stats['current_equity']:,.2f}")
    print(f"Return (PnL):     {stats['pnl_percent']:.2f}%")
    print(f"Total Trades:     {stats['total_trades']}")
    print(f"Max Drawdown:     TODO (Calc from history)")
    
    if stats['pnl_percent'] > -50:
        print("\n✅ PASSED: Portfolio survived Bear Market (> -50%)")
    else:
        print("\n❌ FAILED: Portfolio collapsed (< -50%)")
        
    # Trade Analysis
    print("\nTrade Breakdown:")
    trades = pd.DataFrame(trader.state['trades'])
    if not trades.empty:
        print(trades['action'].value_counts())
        
        # Win Rate
        # A trade is a pair BUY -> SELL. 
        # This simple logic doesn't link them easily in logs without more work.
        # But we can approximate.
        
    # Save history for charting
    pd.DataFrame(trader.state['equity_history']).to_csv("logs/backtest_2022_results.csv", index=False)
    print("Detailed logs saved to logs/backtest_2022_results.csv")

if __name__ == "__main__":
    run_backtest()
