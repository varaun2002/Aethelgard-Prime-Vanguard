import pandas as pd
import numpy as np
import pandas_ta as ta
from src.utils.logger import setup_logger

logger = setup_logger("FeatureEngineerV2")

class FeatureEngineerV2:
    def __init__(self):
        pass

    def add_features(self, df):
        """
        Adds robust technical indicators and normalizes features.
        """
        if df is None or df.empty:
            return df

        df = df.copy()
        
        # Drop duplicates if timestamp exists
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'])
        
        df = df.reset_index(drop=True)
        
        # Ensure numeric columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # --- 1. Volatility Indicators ---
        # ATR (Average True Range)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_1'] = ta.atr(df['high'], df['low'], df['close'], length=1)
        df['atr_3'] = ta.atr(df['high'], df['low'], df['close'], length=3)
        df['atr_60'] = ta.atr(df['high'], df['low'], df['close'], length=60) # v2.2: For Volatility Gate
        
        # ATR Rank (Percentile of current ATR vs last 100)
        # Fix: Shift rolling window
        df['atr_rank'] = df['atr'].rolling(window=100).rank(pct=True).shift(1)
        
        # Bollinger Bands Width
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            # pandas_ta returns columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
            # But sometimes names vary. Let's find them.
            upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
            lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
            mid_col = [c for c in bb.columns if c.startswith('BBM')][0]
            
            df['bb_width'] = (bb[upper_col] - bb[lower_col]) / bb[mid_col]
        
        # Historical Volatility (Log Returns Std Dev)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        # Fix: Shift rolling window
        df['hist_vol'] = df['log_ret'].rolling(window=20).std().shift(1)
        
        # --- 2. Momentum & Trend Indicators (New v2.3) ---
        # RSI & RSI Slope
        df['rsi'] = ta.rsi(df['close'], length=14)
        if df['rsi'] is not None and not df['rsi'].isnull().all():
            df['rsi_slope'] = df['rsi'].diff(3)
            
            # [NEW v2.4] RSI Divergence
            # Fix: Shift PCA/correlation calc
            price_mom = df['close'].pct_change(5)
            rsi_mom = df['rsi'].pct_change(5)
            df['rsi_divergence'] = (price_mom - (rsi_mom / 100)).rolling(3).mean().shift(1)
        else:
            df['rsi'] = 0
            df['rsi_slope'] = 0
            df['rsi_divergence'] = 0
        
        # ADX (Average Directional Index)
        adx = ta.adx(df['high'], df['low'], df['close'])
        if adx is not None:
            df['adx'] = adx['ADX_14']
            df['dmp'] = adx['DMP_14']
            df['dmn'] = adx['DMN_14']
            
        # Choppiness Index (CHOP)
        df['chop'] = ta.chop(df['high'], df['low'], df['close'])
        
        # Aroon
        aroon = ta.aroon(df['high'], df['low'])
        if aroon is not None:
            df['aroon_up'] = aroon['AROONU_14']
            df['aroon_down'] = aroon['AROOND_14']
            df['aroon_osc'] = aroon['AROONOSC_14']
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            
        # --- 3. Candle Features ---
        df['body_size'] = abs(df['close'] - df['open'])
        df['wick_upper'] = df['high'] - np.maximum(df['open'], df['close'])
        df['wick_lower'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_perc'] = df['body_size'] / (df['high'] - df['low'] + 1e-9)
        
        # [NEW v2.4] HL Ratio (Volatility Microstructure)
        df['hl_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
        
        # --- 4. Volume Indicators ---
        # Volume Z-Score
        # Fix: Shift rolling stats
        vol_mean = df['volume'].rolling(window=20).mean().shift(1)
        vol_std = df['volume'].rolling(window=20).std().shift(1)
        df['vol_z'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)
        
        # Relative Volume (RVOL)
        df['rvol'] = df['volume'] / (vol_mean + 1e-9)
        
        df['vol_delta'] = df['volume'].pct_change()
        # Handle infs from 0 volume
        df['vol_delta'] = df['vol_delta'].replace([np.inf, -np.inf], 0)
        
        # [NEW v2.4] Cumulative Volume Delta (CVD)
        df['candle_direction'] = np.where(df['close'] > df['open'], 1, -1)
        df['signed_volume'] = df['candle_direction'] * df['volume']
        df['cvd'] = df['signed_volume'].rolling(20).sum()
        df['cvd_norm_raw'] = df['cvd'] / (df['volume'].rolling(20).sum() + 1e-9)
        
        # --- 5. Market Structure ---
        # Rolling Mean/Std
        # Fix: Shift rolling stats
        df['roll_mean_20'] = df['close'].rolling(window=20).mean().shift(1)
        df['roll_std_20'] = df['close'].rolling(window=20).std().shift(1)
        
        # Trend Angle (Slope of last 5 closes)
        df['trend_slope'] = (df['close'] - df['close'].shift(5)) / 5
        
        # [NEW v2.4] Distance from Support/Resistance (52 period High/Low)
        # Fix: Shift rolling window (using past 52 periods, not including current)
        df['dist_52_high'] = (df['high'].rolling(52).max().shift(1) - df['close']) / (df['close'] + 1e-9)
        df['dist_52_low'] = (df['close'] - df['low'].rolling(52).min().shift(1)) / (df['close'] + 1e-9)
        
        # --- 6. Normalization (Z-Score) ---
        features_to_norm = [
            'atr', 'atr_1', 'atr_3', 'bb_width', 'hist_vol', 
            'rsi', 'rsi_slope', 'adx', 'chop', 'aroon_osc', # v2.3 Added
            'macd', 'macd_hist', 'stoch_k', 'stoch_d', 
            'body_size', 'wick_upper', 'wick_lower', 
            'vol_z', 'vol_delta', 'trend_slope', 'rvol',
            # [NEW v2.4]
            'rsi_divergence', 'hl_ratio', 'cvd_norm_raw', 'dist_52_high', 'dist_52_low'
        ]
        
        for col in features_to_norm:
            if col in df.columns:
                # v2.2: Increased window to 180 for stability
                # Fix: Look-Ahead Bias - Shift stats by 1
                mean = df[col].rolling(window=180).mean().shift(1)
                std = df[col].rolling(window=180).std().shift(1)
                
                # Forward-fill the first NaN from shift to avoid initial data loss equivalent to window size
                mean = mean.ffill().fillna(0)
                std = std.ffill().fillna(1e-9)
                
                df[f'{col}_norm'] = (df[col] - mean) / (std + 1e-9)
                
                # Fill NaNs in normalized columns with 0 (mean)
                df[f'{col}_norm'] = df[f'{col}_norm'].fillna(0)
        
        # Drop NaNs created by rolling windows (only for non-normalized essential cols if any)
        df.dropna(inplace=True)
        
        return df

    def get_feature_columns(self):
        """
        Returns the list of feature column names used for training.
        """
        return [
            # 1. Volatility (6)
            'atr_norm', 'atr_rank', 'bb_width_norm', 'hist_vol_norm', 'vol_z_norm', 'rvol_norm',
            
            # 2. Momentum (8)
            'rsi_norm', 'rsi_slope_norm', 'rsi_divergence_norm', 
            'adx_norm', 'chop_norm', 'aroon_osc_norm', 'macd_norm', 'macd_hist_norm',
            
            # 3. Oscillators (2)
            'stoch_k_norm', 'stoch_d_norm',
            
            # 4. Price/Candle (6)
            'log_ret', 'body_perc', 'hl_ratio_norm', 'trend_slope_norm',
            'dist_52_high_norm', 'dist_52_low_norm',
            
            # 5. Volume/Flow (4)
            'vol_delta_norm', 'cvd_norm_raw_norm', 
            'wick_upper', 'wick_lower',
            
            # 6. Raw Backups (4) - to reach 30
            'rsi', 'adx', 'atr', 'bb_width'
        ]

# Backward Compatibility
FeatureEngineer = FeatureEngineerV2
