#!/usr/bin/env python3
"""
Rigorous Backtest Script
Tests the current Binary/Aggressive model on the last 59 days of 15m data.
Uses valid yfinance limits.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
import warnings
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from src.inference.predict_v3 import PredictorV3

# Suppress warnings
warnings.filterwarnings("ignore")

def fetch_data(symbol="BTC-USD", period="59d", interval="15m"):
    print(f"Fetching {symbol} data for {period} ({interval})...")
    df = yf.download(symbol, period=period, interval=interval, progress=True)
    
    if len(df) == 0:
        raise ValueError("No data fetched!")
        
    df = df.reset_index()
    # Normalize columns
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure TZ-naive
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    print(f"Loaded {len(df)} candles.")
    return df

def run_simulation(df, predictor):
    print("\nRunning Aggressive SWEET SPOT Simulation...")
    
    initial_balance = 10000.0
    balance = initial_balance
    position = 0.0 # Amount of BTC
    entry_price = 0.0
    
    trades = []
    equity_curve = []
    
    # Fees
    FEE_PCT = 0.001 # 0.1% per trade
    
    # Window settings
    seq_len = 60
    # Start after we have enough data (seq_len + buffer)
    start_idx = 300 
    
    for i in tqdm(range(start_idx, len(df))):
        current_idx = i
        candle = df.iloc[current_idx]
        current_price = candle['close']
        timestamp = candle['timestamp']
        
        # Prepare data slice
        # Predictor handles feature engineering internally but usually expects a chunk
        # Let's pass a decent chunk
        slice_df = df.iloc[max(0, current_idx-400):current_idx+1].copy()
        
        try:
            # PREDICT
            # Aggressive Settings: safety_lock=False, uncertainty=0.28
            # The model is now hardcoded for Binary outcomes mostly, but we pass flags anyway
            pred = predictor.predict(slice_df, safety_lock=False, uncertainty_threshold=0.28, force_threshold=0.1)
            
            if not pred:
                continue
                
            signal = pred['predicted_direction']    # UP / DOWN
            conf = pred['confidence']               # Binary confidence
            
            # --- TRADING LOGIC ---
            # Simple Flip-Flop Strategy: Always be in the market direction
            # If we are LONG and signal is DOWN -> FLIP SHORT
            # If we are SHORT and signal is UP -> FLIP LONG
            # If FLAT (shouldn't happen now), HOLD
            
            action = None
            
            # 1. Check if we need to close/flip
            if position > 0: # We are LONG
                if signal == "DOWN":
                    # CLOSE LONG
                    pnl_val = (current_price - entry_price) * position
                    balance += (position * current_price)
                    balance -= (position * current_price * FEE_PCT) # Fee
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'SELL_CLOSE',
                        'price': current_price,
                        'pnl': pnl_val,
                        'pnl_pct': (current_price - entry_price)/entry_price
                    })
                    position = 0.0
                    action = "FLIP_SHORT"

            elif position < 0: # We are SHORT
                if signal == "UP":
                    # CLOSE SHORT
                    pos_size_abs = abs(position)
                    pnl_val = (entry_price - current_price) * pos_size_abs
                    # Return margin
                    margin = pos_size_abs * entry_price
                    balance += margin + pnl_val
                    balance -= (margin * FEE_PCT) # Fee
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'BUY_CLOSE',
                        'price': current_price,
                        'pnl': pnl_val,
                        'pnl_pct': (entry_price - current_price)/entry_price
                    })
                    position = 0.0
                    action = "FLIP_LONG"
            
            else:
                # No position
                action = "OPEN_" + signal
                
            # 2. Open new position if we flipped or were empty
            if action:
                trade_amt = balance * 0.98 # Use 98% of balance
                
                if signal == "UP" and position == 0:
                    # OPEN LONG
                    position = trade_amt / current_price
                    entry_price = current_price
                    balance -= trade_amt # Deduct cash
                    
                elif signal == "DOWN" and position == 0:
                    # OPEN SHORT
                    # simplistic: we hold negative BTC
                    position = -(trade_amt / current_price)
                    entry_price = current_price
                    balance -= trade_amt # Margin used
        
        except Exception as e:
            # print(f"Error: {e}")
            pass
            
        # Track Equity
        curr_equity = balance
        if position > 0:
            curr_equity += (position * current_price)
        elif position < 0:
            pos_abs = abs(position)
            unrealized = (entry_price - current_price) * pos_abs
            curr_equity += (pos_abs * entry_price) + unrealized
            
        equity_curve.append({'timestamp': timestamp, 'equity': curr_equity})

    return trades, equity_curve

def analyze_results(trades, equity):
    if not trades:
        print("No trades executed.")
        return
        
    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity)
    
    initial = 10000.0
    final = df_equity.iloc[-1]['equity']
    total_ret = ((final - initial) / initial) * 100
    
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] <= 0]
    
    win_rate = len(wins) / len(df_trades) * 100
    avg_win = wins['pnl_pct'].mean() * 100 if not wins.empty else 0
    avg_loss = losses['pnl_pct'].mean() * 100 if not losses.empty else 0
    
    print("\n" + "="*40)
    print(f"📊 BACKTEST RESULTS (59 Days - 15m)")
    print("="*40)
    print(f"Final Equity:   ${final:,.2f}")
    print(f"Total Return:   {total_ret:+.2f}%")
    print(f"Total Trades:   {len(df_trades)}")
    print(f"Win Rate:       {win_rate:.1f}%")
    print(f"Avg Win:        {avg_win:+.2f}%")
    print(f"Avg Loss:       {avg_loss:+.2f}%")
    print("="*40)
    
    # Save
    df_trades.to_csv("logs/backtest_trades.csv", index=False)
    print("Detailed trades saved to logs/backtest_trades.csv")

if __name__ == "__main__":
    # Load Model
    model_path = "models_v3/model_v3_0_15m.pth"
    if not os.path.exists(model_path):
        print("Model not found!")
        sys.exit(1)
        
    predictor = PredictorV3(model_path=model_path)
    
    # Fetch Data
    df = fetch_data()
    
    # Run
    trades, equity = run_simulation(df, predictor)
    
    # Analyze
    analyze_results(trades, equity)
    
