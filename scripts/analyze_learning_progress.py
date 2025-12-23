import pandas as pd
import numpy as np
from datetime import datetime

def analyze_progress(filepath):
    print(f"Analyzing {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if df.empty or 'outcome' not in df.columns:
        print("No data or missing 'outcome' column.")
        return

    # Convert timestamps
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('datetime')

    # Filter for validated predictions only
    validated = df[df['outcome'].isin(['CORRECT', 'WRONG'])].copy()
    
    if validated.empty:
        print("No validated predictions found yet.")
        return

    # 1. Overall Stats
    total = len(validated)
    correct = len(validated[validated['outcome'] == 'CORRECT'])
    accuracy = (correct / total) * 100
    
    print(f"\n--- Overall Performance ---")
    print(f"Total Validated: {total}")
    print(f"Overall Accuracy: {accuracy:.2f}%")

    # 2. Daily Trends (Learning Curve)
    validated['date_str'] = validated['datetime'].dt.date
    daily_stats = validated.groupby('date_str').agg({
        'outcome': lambda x: (x == 'CORRECT').sum() / len(x) * 100,
        'confidence': 'mean',
        'uncertainty': 'mean',
        'predicted_direction': lambda x: (x == 'FLAT').sum() / len(x) * 100
    }).rename(columns={'outcome': 'Accuracy', 'current_price': 'Price', 'predicted_direction': 'Pct_FLAT'})
    
    print(f"\n--- Daily Trends (Last 5 Days) ---")
    print(daily_stats.tail(5))

    # 3. Pre/Post Retraining Analysis (assuming retraining happened recently, e.g., last 24h)
    # Let's verify if there's a shift in Regime detection or Confidence
    print(f"\n--- Regime Adaptation ---")
    regime_counts = df['regime'].value_counts(normalize=True) * 100
    print(regime_counts)

    # 4. "Sniper Mode" Verification
    # Check accuracy when confidence > 0.6 (High Confidence)
    high_conf = validated[validated['confidence'] > 0.6]
    if not high_conf.empty:
        hc_acc = (high_conf['outcome'] == 'CORRECT').sum() / len(high_conf) * 100
        print(f"\n--- Sniper Mode (Conf > 0.6) ---")
        print(f"High Confidence Accuracy: {hc_acc:.2f}% (Samples: {len(high_conf)})")
    
    # 5. Recent Performance (Last 100 vs First 100)
    if len(validated) > 200:
        first_100_acc = (validated.head(100)['outcome'] == 'CORRECT').sum() 
        last_100_acc = (validated.tail(100)['outcome'] == 'CORRECT').sum()
        print(f"\n--- Progress Comparison ---")
        print(f"First 100 Validated Accuracy: {first_100_acc}%")
        print(f"Last 100 Validated Accuracy:  {last_100_acc}%")
        delta = last_100_acc - first_100_acc
        print(f"Improvement: {delta:+.2f}%")

if __name__ == "__main__":
    analyze_progress("logs/predictions/predictions_15m.csv")
