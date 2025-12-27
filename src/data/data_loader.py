import numpy as np
import pandas as pd
from src.data.external_data import ExternalDataFetcher
from src.features.feature_engineering import FeatureEngineer

class DataLoader:
    def __init__(self):
        self.external_fetcher = ExternalDataFetcher()
        self.feature_engineer = FeatureEngineer()
    
    def extract_technical_features(self, eth_data):
        """
        Extract 50 technical features using FeatureEngineer
        """
        # Convert list of lists (OHLCV) to DataFrame if needed
        if isinstance(eth_data, list):
            df = pd.DataFrame(eth_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if len(df) > 0 and isinstance(df.iloc[0]['timestamp'], (int, float)):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df = eth_data.copy()
            
        # Add features
        df = self.feature_engineer.add_features(df)
        cols = self.feature_engineer.get_feature_columns()
        
        if len(df) == 0:
            return np.zeros(len(cols), dtype=np.float32)

        # Get latest row features
        last_row = df.iloc[-1][cols].values.astype(np.float32)
        
        # Fill NaNs
        last_row = np.nan_to_num(last_row)
             
        return last_row

    def prepare_features(self, eth_data, timestamp=None):
        """
        Combine technical and external features
        
        Returns:
            numpy array of shape (70,) - 50 technical + 20 external
        """
        
        # Extract technical features (50 features)
        technical_features = self.extract_technical_features(eth_data)
        
        # Get external features (20 features)
        external_features = self.external_fetcher.get_all_features(eth_data, timestamp)
        
        # Combine: [technical (50) + external (20)] = 70 total
        combined_features = np.concatenate([technical_features, external_features])
        
        # Ensure finite
        combined_features = np.nan_to_num(combined_features)
        
        expected_features = len(self.feature_engineer.get_feature_columns()) + 20
        assert len(combined_features) == expected_features, f"Expected {expected_features} features, got {len(combined_features)}"
        
        return combined_features

    def validate_data_quality(self, df):
        """
        Check incoming data for quality issues.
        Returns: (is_valid, issues_list)
        """
        issues = []
        
        # Check for price gaps
        if 'close' in df.columns:
            pct_changes = df['close'].pct_change().abs()
            extreme_moves = pct_changes > 0.30  # 30% moves
            if extreme_moves.any():
                issues.append(f"Found {extreme_moves.sum()} extreme price moves (>30%)")
        
        # Check for zero volume
        if 'volume' in df.columns:
            zero_vol = (df['volume'] == 0).sum()
            if zero_vol > 0:
                issues.append(f"Found {zero_vol} candles with zero volume")
        
        # Check for timestamp gaps
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            expected_gap = df_sorted['timestamp'].diff().mode().iloc[0]
            actual_gaps = df_sorted['timestamp'].diff()
            missing = (actual_gaps > expected_gap * 1.5).sum()
            if missing > 0:
                issues.append(f"Found {missing} potential missing candles")
        
        return len(issues) == 0, issues
