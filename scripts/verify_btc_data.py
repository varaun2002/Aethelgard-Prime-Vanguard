import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.external_history import ExternalHistoryFetcher

def verify_btc_data():
    print("Initializing Fetcher...")
    fetcher = ExternalHistoryFetcher()
    
    # Create dummy dataframe to enrich
    # Last 10 hours of timestamps
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='1h').astype(int) // 10**9
    df = pd.DataFrame({'timestamp': timestamps})
    
    print("Enriching Data...")
    df_enriched = fetcher.enrich_training_data(df)
    
    print("\n--- BTC Features Check ---")
    btc_cols = ['ext_btc_ret1h', 'ext_btc_ret4h', 'ext_btc_vol', 'ext_btc_corr', 'ext_btc_div']
    
    for col in btc_cols:
        if col in df_enriched.columns:
            vals = df_enriched[col].values
            non_zeros = np.count_nonzero(vals)
            print(f"{col}: {non_zeros}/{len(vals)} non-zero values. Sample: {vals[:3]}")
            if non_zeros == 0:
                print(f"⚠️  WARNING: {col} IS ALL ZEROS!")
        else:
            print(f"❌ MISSING COLUMN: {col}")

if __name__ == "__main__":
    verify_btc_data()
