import pandas as pd
import os
import sys

# Define log files
log_files = [
    "logs/predictions/predictions_15m.csv",
    "logs/predictions/predictions_1h.csv",
    "logs/predictions/predictions_4h.csv",
    "logs/predictions/predictions_all.csv"
]

def repair_log(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    print(f"Repairing {filepath}...")
    try:
        df = pd.read_csv(filepath)
        if df.empty: return

        # Identify corrupted rows:
        # 1. Actual Price > 10,000 (ETH is ~3k)
        # OR 2. Outcome is 'MISSED'/'WRONG'/'CORRECT' but we want to re-validate everything to be safe?
        # Let's target the BTC contamination specifically.
        
        # Check if 'actual_price' exists
        if 'actual_price' not in df.columns:
            return

        mask = df['actual_price'] > 10000
        count = mask.sum()
        
        if count > 0:
            print(f"Found {count} corrupted rows (BTC prices). Resetting...")
            
            # Reset values
            df.loc[mask, 'actual_price'] = 0.0
            df.loc[mask, 'actual_direction'] = ""
            df.loc[mask, 'price_diff'] = 0.0
            df.loc[mask, 'price_status'] = "PENDING"
            
            # Reset outcome/direction_status
            if 'direction_status' in df.columns:
                df.loc[mask, 'direction_status'] = "PENDING"
            if 'outcome' in df.columns:
                df.loc[mask, 'outcome'] = "PENDING"
                
            df.to_csv(filepath, index=False)
            print("✅ Fix saved.")
        else:
            print("No corruption found.")

    except Exception as e:
        print(f"Error repairing {filepath}: {e}")

if __name__ == "__main__":
    for f in log_files:
        repair_log(f)
