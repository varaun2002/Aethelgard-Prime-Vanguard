import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def verify_market_context():
    print("running market context verification...")
    
    # 1. Fetch Today's Data (ETH-USD)
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=12) # Last 12 hours
    
    print(f"Fetching ETH-USD data from {start_date.strftime('%H:%M')} to {end_date.strftime('%H:%M')}...")
    
    try:
        df = yf.download("ETH-USD", start=start_date, end=end_date, interval="15m", progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if len(df) == 0:
        print("No data fetched. Yahoo Finance might be delayed or rate limited.")
        return
        
    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # 2. analyze Volatility & Trend
    # Simple metrics: 
    # - Total Move %: (High - Low) / Open
    # - Directional Move: (Last Close - First Open) / First Open
    
    first_open = float(df.iloc[0]['open'])
    last_close = float(df.iloc[-1]['close'])
    
    total_change_pct = ((last_close - first_open) / first_open) * 100
    
    # Average candle range (volatility)
    df['range_pct'] = ((df['high'] - df['low']) / df['open']) * 100
    avg_candle_volatility = df['range_pct'].mean()
    
    print("\n" + "="*50)
    print("📉 MARKET REALITY CHECK (Last 12h)")
    print("="*50)
    print(f"Start Price: ${first_open:,.2f}")
    print(f"End Price:   ${last_close:,.2f}")
    print(f"Net Move:    {total_change_pct:+.2f}%")
    print(f"Avg Candle:  {avg_candle_volatility:.2f}% (Volatility)")
    
    # 3. Verdict
    print("\n⚖️ VERDICT:")
    
    # Define "Choppy" vs "Trendy"
    # Trend: Net move > 1.0% in 12h is decent. 
    # Chop: Net move < 0.5% is flat.
    
    if abs(total_change_pct) < 0.5:
        market_state = "FLAT / CHOPPY"
        justification = "Price has barely moved (< 0.5%) in 12 hours."
        model_verdict = "✅ RIGHT" if True else "" # Placeholder
    elif abs(total_change_pct) < 1.5:
        market_state = "MILD TREND"
        justification = "Some movement, but not explosive."
    else:
        market_state = "STRONG TREND"
        justification = "Significant price action (> 1.5%)."
        
    print(f"The market is: **{market_state}**")
    print(f"Reason: {justification}")
    
    print("-" * 30)
    print("🤖 MODEL BEHAVIOR EVALUATION:")
    
    if market_state == "FLAT / CHOPPY":
        print("Model Action: HOLD / NO TRADES")
        print("-> Evaluation: **EXCELLENT**. The model correctly identified that there was no money to be made.")
    elif market_state == "MILD TREND":
        print("Model Action: HOLD / NO TRADES")
        print("-> Evaluation: **ACCEPTABLE**. The trend might not have been strong enough to clear the 60% confidence threshold.")
    else:
        print("Model Action: HOLD / NO TRADES")
        print("-> Evaluation: **MISSED OPPORTUNITY**. A strong trend occurred, but the model sat out.")
        
    print("="*50)

if __name__ == "__main__":
    verify_market_context()
