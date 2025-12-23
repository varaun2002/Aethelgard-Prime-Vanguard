import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
from src.utils.enhanced_logger import EnhancedLogger

class LogValidator:
    def __init__(self, symbol="BTC-USD"):
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
        
        # 1. Fetch recent data (enough to cover the last 5 days of 15m candles)
        # Using 15m interval allows us to validate 15m, 1h (4 * 15m), and 4h (16 * 15m) roughly.
        # But for accuracy, 1h and 4h alignment might vary. 
        # yfinance 15m data is good.
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
                    
                if 'outcome' not in df_log.columns:
                    continue
                    
                # Filter PENDING
                df_log = df_log.dropna(subset=['timestamp'])
                if df_log.empty:
                    continue
                    
                pending_mask = df_log['outcome'] == 'PENDING'
                if not pending_mask.any():
                    continue
                
                # Iterate pending rows
                for idx, row in df_log[pending_mask].iterrows():
                    try:
                        pred_ts = int(row['timestamp'])
                    except ValueError:
                        continue
                        
                    target_ts = datetime.fromtimestamp(pred_ts) + timedelta(minutes=minutes)
                    
                    # Check if target time is in the past (with 2 min buffer for data delay)
                    if target_ts < datetime.now() - timedelta(minutes=2):
                        # Find closest candle
                        # We want the candle whose CLOSE matches the target time?
                        # Or the candle that *starts* at target_ts?
                        # Usually, if we predict 10:00 -> 10:15. We want Close at 10:15.
                        # yfinance timestamps are usually start of interval.
                        # So the candle starting at 10:00 closes at 10:15.
                        # So we look for index = pred_ts (approx).
                        # Let's verify closest match.
                        
                        target_lookup_ts = datetime.fromtimestamp(pred_ts)
                        
                        # Find price at exact lookup timestamp (candle start)
                        # We want the CLOSE of that candle.
                        
                        # Find nearest index
                        try:
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
                            
                            # Determine actual direction relative to Entry Price (current_price)
                            # Or relative to previous close? 
                            # The model predicts direction from Current Price.
                            
                            if current_price_at_pred == 0:
                                continue # Cannot validate
                                
                            pct_change = (actual_close - current_price_at_pred) / current_price_at_pred
                            
                            if abs(pct_change) < 0.001: # 0.1% threshold for FLAT?
                                actual_dir = "FLAT" # or use model threshold? 
                                # The model defines direction classes.
                                # But for outcome, we just want to know if it went UP or DOWN.
                                # Let's use simple logic:
                                pass
                            
                            if pct_change > 0.001:
                                actual_dir = "UP"
                            elif pct_change < -0.001:
                                actual_dir = "DOWN"
                            else:
                                actual_dir = "FLAT"
                                
                            # Outcome
                            outcome = "WRONG"
                            if predicted_dir == actual_dir:
                                outcome = "CORRECT"
                            elif predicted_dir == "FLAT":
                                # If predicted FLAT, and it was FLAT -> Correct
                                # If predicted FLAT, effectively saved fees.
                                if actual_dir == "FLAT":
                                    outcome = "CORRECT"
                                else:
                                    outcome = "WRONG" # Missed move? Or 'SAFE'?
                                    # Let's call it WRONG for now, or SAFE.
                                    # Actually, if we predict FLAT, we are neutral.
                                    # If market moved, we missed it.
                                    outcome = "MISSED"
                            
                            # Update (Using EnhancedLogger logic)
                            self.logger.update_validation(
                                timeframe=timeframe,
                                timestamp=pred_ts,
                                actual_price=actual_close,
                                actual_direction=actual_dir,
                                outcome=outcome
                            )
                            updates_count += 1
                            
                        except Exception as e:
                            # print(f"Error validating row {idx}: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error processing {timeframe}: {e}")
                
        if updates_count > 0:
            print(f"✅ Updated {updates_count} predictions.")
        else:
            # print("No pending predictions matured.")
            pass

if __name__ == "__main__":
    validator = LogValidator()
    validator.validate_all()
