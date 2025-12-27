import pandas as pd
import pandas_ta as ta
import numpy as np

class RegimeDetector:
    """
    Real-time market regime detection
    """
    def __init__(self, window=100):
        self.window = window
        
    def detect_regime(self, df):
        """
        Detect regime for the latest candle in df
        
        Returns:
            regime (str): 'trend_up', 'trend_down', 'ranging', 'high_vol'
            details (dict): specific metrics
        """
        if len(df) < 50:
            return "ranging", {} # Not enough data
            
        # Work on a copy of recent data
        df = df.tail(self.window).copy()
        
        # Calculate Indicators if missing
        if 'adx' not in df.columns:
            adx = ta.adx(df['high'], df['low'], df['close'])
            if adx is not None:
                df['adx'] = adx['ADX_14']
                df['dmp'] = adx['DMP_14']
                df['dmn'] = adx['DMN_14']
        
        if 'ema20' not in df.columns:
            df['ema20'] = ta.ema(df['close'], length=20)
            
        if 'atr' not in df.columns:
             df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
             
        # Get latest values
        current = df.iloc[-1]
        
        adx = current.get('adx', 0)
        dmp = current.get('dmp', 0)
        dmn = current.get('dmn', 0)
        close = current['close']
        ema20 = current.get('ema20', close)
        atr = current.get('atr', 0)
        
        # Volatility check (using ATR percentile of last 100)
        # We need historical ATR for percentile
        if 'atr' in df.columns:
             atr_series = df['atr']
             atr_rank = atr_series.rank(pct=True).iloc[-1]
        else:
             atr_rank = 0.5
             
        regime = 'ranging'
        
        if atr_rank > 0.8:
            regime = 'high_vol'
        elif adx > 25:
            if dmp > dmn and close > ema20:
                regime = 'trend_up'
            elif dmn > dmp and close < ema20:
                regime = 'trend_down'
            else:
                regime = 'ranging' # Strong trend but direction unclear or close not confirming
        else:
            regime = 'ranging'
            
        return regime, {
            'adx': adx,
            'atr_rank': atr_rank,
            'dmp': dmp,
            'dmn': dmn
        }
