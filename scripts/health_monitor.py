import time
import os
import pandas as pd
from datetime import datetime
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_health():
    print(f"\n🏥 SYSTEM HEALTH CHECK - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # 1. Prediction Validator (Auto-Run)
    try:
        from scripts.validate_live_predictions import LogValidator
        validator = LogValidator()
        validator.validate_all()
    except Exception as e:
        print(f"⚠️ Validation Error: {e}")

    # 2. Check Log Freshness
    log_file = "logs/predictions/predictions_15m.csv"
    if os.path.exists(log_file):
        try:
            mtime = os.path.getmtime(log_file)
            age = time.time() - mtime
            status = "✅ ACTIVE" if age < 120 else "⚠️  STALLED"
            print(f"Log File (15m): {status} (Updated {age:.1f}s ago)")
            
            # Read last few rows
            try:
                df = pd.read_csv(log_file)
                if not df.empty:
                    last_row = df.iloc[-1]
                    print(f"Latest Prediction: {last_row['timestamp']} | {last_row['predicted_direction']} ({last_row['confidence']:.2f})")
                    
                    # Check Pending vs Validated
                    pending = len(df[df['outcome'] == 'PENDING'])
                    validated = len(df[df['outcome'].isin(['CORRECT', 'WRONG', 'MISSED', 'GOOD', 'BAD'])])
                    print(f"Stats: {validated} Validated | {pending} Pending")
                    
                    if validated > 0:
                        acc = len(df[df['outcome'].isin(['CORRECT', 'GOOD', 'PERFECT'])]) / validated
                        print(f"Current Accuracy: {acc:.1%}")
                else:
                    print("Log file is empty (Waiting for first prediction...)")
            except Exception as e:
                print(f"Error reading log: {e}")
        except Exception as e:
            print(f"Error checking file: {e}")
    else:
        print("❌ Log file not found! (Is dashboard running?)")

    # 2. Check Trades
    trade_file = "logs/trades_aggressive.csv"
    if os.path.exists(trade_file):
        try:
            df_t = pd.read_csv(trade_file)
            if not df_t.empty:
                print(f"Trades Executed: {len(df_t)}")
                last_trade = df_t.iloc[-1]
                print(f"Last Trade: {last_trade['action']} @ ${last_trade['price']:.2f} ({last_trade['date']})")
                print(f"PnL: {last_trade.get('pnl_pct', 0):.2f}%")
            else:
                print("No trades execution yet.")
        except:
            print("Error reading trade logs.")
    else:
        print("No trade log found yet.")
        
    print("-" * 60)

if __name__ == "__main__":
    print("Health Monitor started. Press Ctrl+C to stop.")
    try:
        while True:
            check_health()
            time.sleep(60) # Check every minute
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
