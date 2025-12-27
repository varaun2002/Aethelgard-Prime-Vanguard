from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

class TimeSeriesCrossValidator:
    """
    Time-series aware cross-validation
    Never trains on future data
    """
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size # Fraction of total data for test in each fold? 
                                   # Or implied by n_splits in TimeSeriesSplit?
                                   # standard TimeSeriesSplit doesn't take test_size exactly like this.
                                   # We'll implement custom rolling window.
    
    def split(self, df):
        """
        Generate train/test splits
        """
        n = len(df)
        # Custom logic matching plan:
        # Each split moves forward in time?
        # Plan says:
        # train_end = n - (self.n_splits - i) * test_n
        
        test_n = int(n * self.test_size)
        
        splits = []
        for i in range(self.n_splits):
            # Each split moves forward in time
            # i=0: Test is earliest segment? No, usually valid is last.
            
            # Plan logic:
            # i=0: train_end = n - (5)*test_n. test_end = train_end + test_n.
            # i=4: train_end = n - (1)*test_n. test_end = n.
            
            folds_left = self.n_splits - i
            
            train_end = n - (folds_left * test_n)
            test_end = train_end + test_n
            
            # Ensure indices are valid
            if train_end < 100: continue
            
            train_idx = range(0, train_end)
            test_idx = range(train_end, min(test_end, n))
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def validate_model(self, model, df, X_cols, y_col):
        """
        Perform cross-validation and return metrics.
        Assumes 'model' has fit/predict interface (sklearn style)
        or can be adapted.
        """
        scores = []
        
        for train_idx, test_idx in self.split(df):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            
            # Train
            model.fit(train_data[X_cols], train_data[y_col])
            
            # Test
            predictions = model.predict(test_data[X_cols])
            actuals = test_data[y_col]
            
            # Calculate directional accuracy
            acc = (predictions == actuals).mean()
            scores.append(acc)
        
        return {
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'min_accuracy': np.min(scores),
            'max_accuracy': np.max(scores),
            'all_folds': scores
        }
