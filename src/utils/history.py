import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
from src.utils.logger import setup_logger

logger = setup_logger("HistoryManager")

class HistoryManager:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.pred_log_paths = {
            "15m": os.path.join(self.data_dir, "prediction_log_15m.csv"),
            "1h": os.path.join(self.data_dir, "prediction_log_1h.csv"),
            "4h": os.path.join(self.data_dir, "prediction_log_4h.csv")
        }
        self.train_log_path = os.path.join(self.data_dir, "training_log.csv")
        
        self._init_logs()

    def _init_logs(self):
        for tf, path in self.pred_log_paths.items():
            if not os.path.exists(path):
                df = pd.DataFrame(columns=[
                    "timestamp", "symbol", "timeframe", "current_price", 
                    "predicted_price", "predicted_return", "predicted_direction", "confidence", 
                    "flat_prob", "regime",
                    "direction_status", "price_status", "actual_price", "actual_direction", "price_diff", "model_version"
                ])
                df.to_csv(path, index=False)
            
        if not os.path.exists(self.train_log_path):
            df = pd.DataFrame(columns=["timestamp", "symbol", "timeframe", "loss", "accuracy", "duration_seconds"])
            df.to_csv(self.train_log_path, index=False)

    def _parse_timeframe(self, tf_str):
        if tf_str.endswith('m'):
            return timedelta(minutes=int(tf_str[:-1]))
        elif tf_str.endswith('h'):
            return timedelta(hours=int(tf_str[:-1]))
        elif tf_str.endswith('d'):
            return timedelta(days=int(tf_str[:-1]))
        else:
            return timedelta(hours=1)

    def _floor_time(self, dt, timeframe):
        """Rounds down datetime to the nearest timeframe interval."""
        delta = self._parse_timeframe(timeframe)
        
        # Convert to timestamp, floor, convert back
        # This works well for minutes/hours
        ts = dt.timestamp()
        seconds = delta.total_seconds()
        floored_ts = (ts // seconds) * seconds
        return datetime.fromtimestamp(floored_ts)


    def _initialize_log(self, path):
        """
        Creates a new log file with headers if it doesn't exist.
        """
        if not os.path.exists(path):
            df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'timeframe', 
                'current_price', 'predicted_price', 'predicted_return',
                'predicted_direction', 'confidence', 'flat_prob', 'regime',
                'direction_status', 'price_status', 
                'actual_price', 'actual_direction', 'price_diff',
                'model_version'
            ])
            df.to_csv(path, index=False)

    def log_prediction(self, symbol, timeframe, current_price, pred_price, pred_return, pred_dir, confidence, flat_prob, regime, model_version="2.1.0"):
        """
        Logs a new prediction.
        """
        log_path = self.pred_log_paths.get(timeframe)
        if not log_path:
            logger.error(f"Invalid timeframe for logging: {timeframe}")
            return

        # Ensure log exists
        self._initialize_log(log_path)
        
        # Load existing
        df = pd.read_csv(log_path)
        
        # Timestamp (Local - Pittsburgh)
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S')
        
        new_row = {
            'timestamp': now,
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'predicted_price': pred_price,
            'predicted_return': pred_return,
            'predicted_direction': pred_dir,
            'confidence': confidence,
            'flat_prob': flat_prob,
            'regime': regime,
            'direction_status': 'PENDING',
            'price_status': 'PENDING',
            'actual_price': None,
            'actual_direction': None,
            'price_diff': None,
            'model_version': model_version
        }
        
        # Append
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Enforce column order
        cols = [
            'timestamp', 'symbol', 'timeframe', 
            'current_price', 'predicted_price', 'predicted_return',
            'predicted_direction', 'confidence', 'flat_prob', 'regime',
            'direction_status', 'price_status', 
            'actual_price', 'actual_direction', 'price_diff',
            'model_version'
        ]
        df = df[cols]
        
        df.to_csv(log_path, index=False)
        logger.info(f"Logged prediction for {symbol} {timeframe} (v{model_version})")

    def log_training(self, symbol, timeframe, loss, accuracy, duration):
        try:
            et_tz = pytz.timezone('America/New_York')
            new_row = {
                "timestamp": datetime.now(et_tz).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "timeframe": timeframe,
                "loss": loss,
                "accuracy": accuracy,
                "duration_seconds": duration
            }
            df = pd.DataFrame([new_row])
            
            # Write header if file doesn't exist
            header = not os.path.exists(self.train_log_path)
            df.to_csv(self.train_log_path, mode='a', header=header, index=False)
            logger.info(f"Logged training for {symbol}")
        except Exception as e:
            logger.error(f"Failed to log training: {e}")

    def get_prediction_history(self, timeframe="1h", limit=50):
        log_path = self.pred_log_paths.get(timeframe)
        if log_path and os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                if 'timestamp' in df.columns:
                    return df.tail(limit).sort_values("timestamp", ascending=False)
            except Exception as e:
                logger.error(f"Error reading prediction history: {e}")
        return pd.DataFrame()

    def get_training_history(self, limit=50):
        if os.path.exists(self.train_log_path):
            try:
                df = pd.read_csv(self.train_log_path)
                # Check for required columns
                if 'timestamp' not in df.columns:
                    # Try to handle headerless case or just return empty
                    # If columns match headerless structure: timestamp,symbol,timeframe,loss,accuracy,duration
                    if len(df.columns) == 6:
                        # Assume corrupted header, reset names
                        df.columns = ["timestamp", "symbol", "timeframe", "loss", "accuracy", "duration_seconds"]
                    else:
                        logger.warning("Training log has missing columns")
                        return pd.DataFrame()
                        
                return df.tail(limit).sort_values("timestamp", ascending=False)
            except Exception as e:
                logger.error(f"Error reading training history: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def update_actuals(self, symbol, df, timeframe="1h", force_update=False):
        """
        Updates 'actual_price' and 'error' for past predictions using the latest data.
        """
        log_path = self.pred_log_paths.get(timeframe)
        if not log_path or not os.path.exists(log_path):
            return

        try:
            log_df = pd.read_csv(log_path)
            updated = False
            
            # Ensure timestamps are datetime
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
            
            # Timezone handling: Logs are in Local (Eastern), Data is in UTC (Naive)
            import pytz
            eastern = pytz.timezone('US/Eastern')
            
            # Filter for rows that need update and match symbol
            if force_update:
                mask = (log_df['symbol'] == symbol)
            else:
                mask = (log_df['symbol'] == symbol) & (log_df['actual_price'].isna())
            indices = log_df[mask].index
            
            logger.info(f"DEBUG: update_actuals found {len(indices)} rows to update for {symbol} {timeframe}")
            
            for idx in indices:
                row = log_df.loc[idx]
                
                # Parse timeframe
                delta = self._parse_timeframe(row['timeframe'])
                
                # Calculate Target Time (UTC)
                # 1. Localize log time to Eastern
                local_time = row['timestamp']
                if pd.isna(local_time): continue
                
                local_aware = eastern.localize(local_time)
                # 2. Convert to UTC
                utc_aware = local_aware.astimezone(pytz.UTC)
                # 3. Make naive to match df
                utc_naive = utc_aware.replace(tzinfo=None)
                
                # Align prediction time to grid
                # Manual floor to timeframe seconds
                ts = utc_naive.timestamp()
                seconds = delta.total_seconds()
                floored_ts = (ts // seconds) * seconds
                aligned_time = datetime.fromtimestamp(floored_ts, tz=pytz.UTC).replace(tzinfo=None)
                
                target_time = aligned_time + delta
                
                # Find match in df
                # Check if 'timestamp' column exists, otherwise use index
                if 'timestamp' in df.columns:
                    candidates = df[df['timestamp'] == target_time]
                else:
                    candidates = df[df.index == target_time]
                
                if not candidates.empty:
                    match = candidates.iloc[0]
                    
                    # Check if candle is closed
                    candle_close_time = target_time + delta
                    now_utc = datetime.utcnow()
                    
                    if now_utc > candle_close_time:
                        actual = match['close']
                        log_df.at[idx, 'actual_price'] = actual
                        
                        # Calculate new fields
                        current_price = row['current_price']
                        pred_price = row['predicted_price']
                        pred_dir = row['predicted_direction']
                        
                        # Actual Direction with Flat Threshold
                        if pd.notna(current_price) and current_price > 0:
                            actual_return = (actual - current_price) / current_price
                            
                            # Use same threshold as training (0.25%)
                            FLAT_THRESHOLD = 0.0025 
                            
                            if abs(actual_return) < FLAT_THRESHOLD:
                                actual_dir = "FLAT"
                            else:
                                actual_dir = "UP" if actual > current_price else "DOWN"
                                
                            log_df.at[idx, 'actual_direction'] = actual_dir
                            
                            # Direction Status (Classification Head)
                            dir_status = "CORRECT" if actual_dir == pred_dir else "INCORRECT"
                            log_df.at[idx, 'direction_status'] = dir_status
                            
                            # Price Status (Regression Head Direction)
                            pred_price_dir = "UP" if pred_price > current_price else "DOWN"
                            # Regression head doesn't usually predict exact same price, so we stick to binary or match flat logic?
                            # Let's keep regression binary for now, or maybe applied same flat logic?
                            # Actually, sticking to binary UP/DOWN for regression is safer comparison unless we add flat there too.
                            # But to be fair:
                            actual_price_dir_binary = "UP" if actual > current_price else "DOWN"
                            price_status = "CORRECT" if actual_price_dir_binary == pred_price_dir else "INCORRECT"
                            log_df.at[idx, 'price_status'] = price_status
                            
                            # Price Diff
                            diff = actual - pred_price
                            log_df.at[idx, 'price_diff'] = diff
                        
                        updated = True

            if updated:
                log_df.to_csv(log_path, index=False)
                logger.info(f"Updated actuals for {symbol} ({timeframe})")

        except Exception as e:
            logger.error(f"Failed to update actuals: {e}")
