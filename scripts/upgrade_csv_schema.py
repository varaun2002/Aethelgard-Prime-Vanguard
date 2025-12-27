import pandas as pd
import os

log_files = [
    "logs/predictions/predictions_15m.csv",
    "logs/predictions/predictions_1h.csv",
    "logs/predictions/predictions_4h.csv",
    "logs/predictions/predictions_all.csv"
]

def upgrade_schema(filepath):
    if not os.path.exists(filepath): return
    
    print(f"Checking schema for {filepath}...")
    try:
        df = pd.read_csv(filepath)
        changed = False
        
        # Renaissance mapping
        rename_map = {
            'outcome': 'direction_status',
            'price_error_pct': 'price_diff'
        }
        
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                print(f"  Renaming {old} -> {new}")
                df.rename(columns={old: new}, inplace=True)
                changed = True
                
        # Add new columns if missing
        if 'price_status' not in df.columns:
             print("  Adding price_status")
             df['price_status'] = 'PENDING'
             changed = True
             
        if changed:
            df.to_csv(filepath, index=False)
            print("  ✅ Accession complete (Start -> Fix).")
        else:
            print("  Schema already matches.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    for f in log_files:
        upgrade_schema(f)
