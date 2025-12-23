import pandas as pd
import os
from datetime import datetime
import logging

class FeedbackManager:
    """
    Manages the 'Active Learning' loop:
    1. Logs predictions
    2. Verifies them against future price using DIRECTIONAL accuracy
    3. Triggers retraining if error rate is high

    FIXED: Now uses directional accuracy instead of price accuracy
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
            'timestamp': int(datetime.now().timestamp()),
            'timeframe': timeframe,
            'predicted_direction': prediction['predicted_direction'],
            'confidence': prediction['confidence'],
            'predicted_price': prediction['predicted_price'],
            'actual_price': 0.0,  # Filled later
            'outcome': 'PENDING',
            'error': 0.0
        }

        # Concat is recommended over append
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(self.log_file, index=False)

    def calculate_actual_direction(self, predicted_price, actual_price, threshold_pct=0.5):
        """
        Calculate the actual direction based on price movement.

        Args:
            predicted_price: Starting price (what the model saw)
            actual_price: Price after the timeframe elapsed
            threshold_pct: Percentage threshold for FLAT (default 0.5%)

        Returns:
            str: 'UP', 'DOWN', or 'FLAT'
        """
        if actual_price == 0 or predicted_price == 0:
            return 'UNKNOWN'

        # Calculate percentage change
        pct_change = ((actual_price - predicted_price) / predicted_price) * 100

        # Determine direction
        if abs(pct_change) < threshold_pct:
            return 'FLAT'
        elif pct_change > 0:
            return 'UP'
        else:
            return 'DOWN'

    def check_accuracy(self, current_price, current_timestamp):
        """
        Check pending predictions to see if they can be verified.

        FIXED: Now uses directional accuracy instead of price accuracy.
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
                # Calculate actual direction
                predicted_dir = row['predicted_direction']
                actual_dir = self.calculate_actual_direction(
                    row['predicted_price'],
                    current_price
                )

                # Compare directions
                direction_match = (predicted_dir == actual_dir)

                # Calculate price error (for logging purposes)
                price_error = abs(current_price - row['predicted_price']) / row['predicted_price']

                # Determine outcome based on DIRECTION, not price
                if direction_match:
                    # Direction was correct
                    if price_error < 0.01:
                        outcome = 'PERFECT'  # Direction correct AND price within 1%
                    else:
                        outcome = 'GOOD'     # Direction correct (price may be off)
                else:
                    # Direction was wrong
                    outcome = 'BAD'

                # Update the row
                df.at[idx, 'actual_price'] = current_price
                df.at[idx, 'outcome'] = outcome
                df.at[idx, 'error'] = price_error
                updated = True

                # Log the validation
                self.logger.info(
                    f"Validated {tf} prediction: {predicted_dir} vs {actual_dir} "
                    f"({outcome}) - Price error: {price_error*100:.2f}%"
                )

        if updated:
            df.to_csv(self.log_file, index=False)

            # Check bad rate (only count recent predictions)
            recent = df.tail(50)
            recent_validated = recent[recent['outcome'].isin(['GOOD', 'BAD', 'PERFECT'])]

            if len(recent_validated) > 0:
                bad_count = len(recent_validated[recent_validated['outcome'] == 'BAD'])
                bad_rate = bad_count / len(recent_validated)

                self.logger.info(
                    f"Recent accuracy: {(1-bad_rate)*100:.1f}% "
                    f"({len(recent_validated)-bad_count}/{len(recent_validated)} correct)"
                )

                # Trigger retraining if >40% of recent predictions are bad
                if bad_rate > 0.40 and len(recent_validated) >= 20:
                    self.trigger_retraining()

        return 0.0

    def trigger_retraining(self):
        """
        Trigger retraining when accuracy degrades below threshold.
        """
        self.logger.warning("⚠️ High error rate detected! Triggering Active Learning Retrain...")

        with open("logs/retrain_events.log", "a") as f:
            f.write(f"{datetime.now()}: Triggered Retraining - Directional accuracy below 60%\n")

        # In production: os.system("python src/models/train_v3.py --fine_tune")
