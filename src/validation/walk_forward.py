import pandas as pd
import numpy as np
from datetime import timedelta

class WalkForwardValidator:
    def __init__(self, train_months=6, val_months=1, test_months=1):
        """
        train_months: months to train on
        val_months: months to validate hyperparameters
        test_months: months to test (out-of-sample)
        """
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
    
    def split_data(self, df, timestamp_col='timestamp'):
        """
        Create walk-forward splits
        Returns: list of (train_df, val_df, test_df) tuples
        """
        df = df.sort_values(timestamp_col)
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
             # Try to convert if it's int/float (ms or s) or string
             # Assuming ms if huge numbers, but let's be safe.
             # Codebase usually uses ints in ms.
             # If numbers > 3000000000, likely ms.
             if df[timestamp_col].dtype in [np.int64, np.float64, int, float]:
                 if df[timestamp_col].max() > 3000000000: # Definitely ms
                     df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
                 else:
                     df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
             else:
                 df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        total_months = self.train_months + self.val_months + self.test_months
        start_date = df[timestamp_col].min()
        end_date = df[timestamp_col].max()
        
        splits = []
        current_date = start_date
        
        while current_date + timedelta(days=30*total_months) <= end_date:
            train_end = current_date + timedelta(days=30*self.train_months)
            val_end = train_end + timedelta(days=30*self.val_months)
            test_end = val_end + timedelta(days=30*self.test_months)
            
            train = df[(df[timestamp_col] >= current_date) & 
                      (df[timestamp_col] < train_end)]
            val = df[(df[timestamp_col] >= train_end) & 
                    (df[timestamp_col] < val_end)]
            test = df[(df[timestamp_col] >= val_end) & 
                     (df[timestamp_col] < test_end)]
            
            # Ensure sufficient data in each split
            if len(train) > 1000 and len(val) > 100 and len(test) > 100:
                splits.append((train, val, test))
            
            # Roll forward by 1 month
            current_date += timedelta(days=30)
        
        return splits
    
    def calculate_metrics_by_regime(self, predictions, actuals, regimes):
        """
        Calculate accuracy separately for each market regime
        """
        metrics = {}
        for regime in ['Trending UP', 'Trending DOWN', 'Range Quiet', 'Range Noisy']:
            # Need to map regime strings to whatever format 'regimes' is in
            # If regimes is encoded integers, we need a map.
            # Assuming regimes is passed as list/array of strings or mapped before calling.
            # If regimes is integers (0-3), we map:
            # 0: Range Quiet, 1: Range Noisy, 2: Trend UP, 3: Trend DOWN
            
            if isinstance(regimes[0], (int, np.integer)):
                 regime_map = {'Range Quiet': 0, 'Range Noisy': 1, 'Trending UP': 2, 'Trending DOWN': 3}
                 target_val = regime_map.get(regime)
                 mask = regimes == target_val
            else:
                 mask = regimes == regime
                 
            if mask.sum() > 10:
                acc = (predictions[mask] == actuals[mask]).mean()
                metrics[regime] = acc
        return metrics
