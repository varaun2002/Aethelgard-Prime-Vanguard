import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.enhanced_logger import EnhancedLogger

class LogValidator:
    def __init__(self, symbol="ETH-USD"):
        self.symbol = symbol
        self.logger = EnhancedLogger()
        self.intervals = {
            '15m': 15,
            '1h': 60,
            '4h': 240
        }
    
    def validate_all(self):
        """Checking all timeframe logs for pending predictions that have matured."""
        print(f"🔍 Validating predictions for {self.symbol}...")
        
        try:
            df_price = yf.download(self.symbol, period="5d", interval="15m", progress=False)
            if df_price.empty:
                print("❌ Failed to fetch price data.")
                return
                
            # Flatten columns if MultiIndex (yfinance v0.2+)
            if isinstance(df_price.columns, pd.MultiIndex):
                df_price.columns = df_price.columns.get_level_values(0)
            
            # Ensure index is timezone naive or compatible
            df_price.index = df_price.index.tz_localize(None)
            
        except Exception as e:
            print(f"❌ Error fetching price data: {e}")
            return

        updates_count = 0
        
        for timeframe, minutes in self.intervals.items():
            try:
                log_file = self.logger.log_files.get(timeframe)
                if not log_file or not os.path.exists(log_file):
                    continue
                    
                try:
                    df_log = pd.read_csv(log_file)
                except pd.errors.EmptyDataError:
                    continue
                    
                if 'direction_status' not in df_log.columns:
                    continue
                    
                # Filter PENDING
                df_log = df_log.dropna(subset=['timestamp'])
                if df_log.empty:
                    continue
                    
                pending_mask = df_log['direction_status'] == 'PENDING'
                if not pending_mask.any():
                    continue
                
                # Iterate pending rows and collect updates
                batch_updates = []
                
                for idx, row in df_log[pending_mask].iterrows():
                    try:
                        # Robust timestamp conversion
                        try:
                            # Handle cases where timestamp might be float string
                            pred_ts = int(float(row['timestamp']))
                        except:
                            continue
                            
                        target_ts = datetime.fromtimestamp(pred_ts) + timedelta(minutes=minutes)
                        
                        # Check if target time is in the past (with 2 min buffer for data delay)
                        if target_ts < datetime.now() - timedelta(minutes=2):
                            
                            target_lookup_ts = datetime.fromtimestamp(pred_ts)
                            
                            # Find price at exact lookup timestamp (candle start)
                            # We want the CLOSE of that candle.
                            
                            # Find nearest index
                            # Using nearest match within tolerance
                            idx_loc = df_price.index.get_indexer([target_lookup_ts], method='nearest')[0]
                            match_ts = df_price.index[idx_loc]
                            
                            # Check time diff (tolerance 5 mins)
                            diff = abs((match_ts - target_lookup_ts).total_seconds())
                            if diff > 300: 
                                # Too far, maybe gap in data
                                continue
                                
                            actual_close = float(df_price.iloc[idx_loc]['Close'])
                            
                            # Validate
                            predicted_dir = row['predicted_direction']
                            predicted_price = row.get('predicted_price', 0)
                            current_price_at_pred = row.get('current_price', 0)
                            
                            if pd.isna(current_price_at_pred) or current_price_at_pred == 0:
                                continue
                                
                            pct_change = (actual_close - current_price_at_pred) / current_price_at_pred
                            
                            # 0.1% threshold for minimal direction
                            if pct_change > 0.001:
                                actual_dir = "UP"
                            elif pct_change < -0.001:
                                actual_dir = "DOWN"
                            else:
                                actual_dir = "FLAT"
                                
                            # Outcome Logic
                            direction_status = "WRONG"
                            if predicted_dir == actual_dir:
                                direction_status = "CORRECT"
                            elif predicted_dir == "FLAT":
                                # If predicted FLAT
                                if actual_dir == "FLAT":
                                    direction_status = "CORRECT" # Correctly predicted noise
                                else:
                                    direction_status = "MISSED" # Market moved but we stayed flat (Safe but missed opp)
                            else:
                                # Predicted UP/DOWN but got something else
                                direction_status = "WRONG"
                            
                            # Price Status Logic
                            # Hitting predicted price or just direction?
                            # Let's say if Price Diff % < 0.5% it is accurate
                            price_diff_val = actual_close - predicted_price
                            price_status = "MISS"
                            if abs(pct_change) < 0.005: 
                                price_status = "HIT"

                            # Add to batch
                            batch_updates.append({
                                'timestamp': pred_ts,
                                'actual_price': actual_close,
                                'actual_direction': actual_dir,
                                'direction_status': direction_status,
                                'price_status': price_status,
                                'price_diff': price_diff_val
                            })
                                
                    except Exception as e:
                        continue
                                
                # Apply Batch
                if batch_updates:
                    self.logger.update_validation_batch(timeframe, batch_updates)
                    updates_count += len(batch_updates)
                    print(f"✅ Processed {len(batch_updates)} updates for {timeframe}.")
                            
            except Exception as e:
                print(f"Error processing {timeframe}: {e}")
                
        if updates_count > 0:
            print(f"✨ Total predictions validated: {updates_count}")
        else:
             pass

if __name__ == "__main__":
    validator = LogValidator()
    validator.validate_all()
