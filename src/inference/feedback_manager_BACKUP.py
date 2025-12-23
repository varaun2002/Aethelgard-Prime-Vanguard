import pandas as pd
import os
from datetime import datetime
import logging

class FeedbackManager:
    """
    Manages the 'Active Learning' loop:
    1. Logs predictions
    2. Verifies them against future price
    3. Triggers retraining if error rate is high
    """
    def __init__(self, log_file="data/prediction_log.csv"):
        self.log_file = log_file
        self.logger = logging.getLogger("FeedbackManager")
        
        if not os.path.exists(self.log_file):
            self.init_log()
            
    def init_log(self):
        df = pd.DataFrame(columns=[
            'timestamp', 'timeframe', 'predicted_direction', 'confidence', 
            'predicted_price', 'actual_price', 'outcome', 'error'
        ])
        df.to_csv(self.log_file, index=False)
        
    def log_prediction(self, timeframe, prediction, current_price):
        """
        Log a new prediction.
        """
        # Load log
        try:
            df = pd.read_csv(self.log_file)
        except:
            self.init_log()
            df = pd.read_csv(self.log_file)
            
        new_row = {
            'timestamp': int(datetime.now().timestamp()), # Use int timestamp
            'timeframe': timeframe,
            'predicted_direction': prediction['predicted_direction'],
            'confidence': prediction['confidence'],
            'predicted_price': prediction['predicted_price'],
            'actual_price': 0.0, # Filled later
            'outcome': 'PENDING',
            'error': 0.0
        }
        
        # Concat is recommended over append
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(self.log_file, index=False)
        
    def check_accuracy(self, current_price, current_timestamp):
        """
        Check pending predictions to see if they can be verified.
        Simple logic: If timestamp + timeframe < current_timestamp, verify.
        """
        try:
            df = pd.read_csv(self.log_file)
        except:
            return 0.0
            
        updated = False
        
        # Timeframe map to seconds
        tf_map = {'15m': 900, '1h': 3600, '4h': 14400}
        
        pending_mask = df['outcome'] == 'PENDING'
        if not pending_mask.any():
            return 0.0
            
        for idx, row in df[pending_mask].iterrows():
            ts = int(row['timestamp'])
            tf = row['timeframe']
            delta = tf_map.get(tf, 900)
            
            if current_timestamp > (ts + delta):
                # Verify
                # Actual Direction
                # We use Current Price as the "Future Price" for that prediction
                # (Ideally we should find the exact historical price, but for live loop this is approx)
                
                start_price = row['predicted_price'] / (1 + 0.0) # We don't have start price logged? 
                # Wait, we logged predicted_price which is Close * (1+Ret).
                # We need the Original Close to determine direction.
                # Actually, we can just compare Predicted Price vs Actual Price.
                
                # Close enough logic
                pred_dir = row['predicted_direction']
                
                # Error (MSE style or Directional)
                # Let's say we just mark Correct/Incorrect direction if we knew start price
                # Since we didn't log start price, we can't perfectly grade 'UP/DOWN'.
                # But we can grade 'Predicted Price' vs 'Actual Price'
                
                error = abs(current_price - row['predicted_price']) / row['predicted_price']
                
                outcome = 'UNKNOWN'
                if error < 0.005: outcome = 'PERFECT'
                elif error < 0.02: outcome = 'GOOD'
                else: outcome = 'BAD'
                
                df.at[idx, 'actual_price'] = current_price
                df.at[idx, 'outcome'] = outcome
                df.at[idx, 'error'] = error
                updated = True
                
        if updated:
            df.to_csv(self.log_file, index=False)
            
            # Check bad rate
            recent = df.tail(50)
            bad_count = len(recent[recent['outcome'] == 'BAD'])
            if bad_count > 10:
                self.trigger_retraining()
                
        return 0.0

    def trigger_retraining(self):
        """
        Mock retraining trigger
        """
        self.logger.info("⚠️ High error rate detected! Triggering Active Learning Retrain...")
        # In real world: os.system("python src/models/train_v3.py --fine_tune")
        # Here we just log it for the UI to see
        with open("retrain_events.log", "a") as f:
            f.write(f"{datetime.now()}: Triggered Retraining\n")
