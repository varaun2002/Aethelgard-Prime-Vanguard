import numpy as np
import pandas as pd

class VolatilityAwareStops:
    """
    Adjust stop-loss and take-profit based on market volatility
    """
    def __init__(self):
        self.default_stop = 0.02  # 2%
        self.default_tp = 0.05    # 5%
    
    def calculate_atr_multiple(self, recent_prices, period=14):
        """
        Calculate ATR as multiple of price.
        Expects pandas Series or DataFrame.
        """
        if isinstance(recent_prices, (list, np.ndarray)):
            recent_prices = pd.Series(recent_prices)
            
        highs = recent_prices.rolling(2).max()
        lows = recent_prices.rolling(2).min()
        closes = recent_prices.shift(1)
        
        tr1 = highs - lows
        tr2 = abs(highs - closes)
        tr3 = abs(lows - closes)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Handle NaN at start
        tr = tr.fillna(0)
        
        atr = tr.rolling(period).mean().iloc[-1]
        
        current_price = recent_prices.iloc[-1]
        if current_price == 0: return 0.01
        
        atr_pct = atr / current_price
        return atr_pct
    
    def get_stops(self, recent_prices, confidence):
        """
        Calculate dynamic stop-loss and take-profit
        """
        # print(f"DEBUG: get_stops called with {len(recent_prices)}")
        atr_pct = self.calculate_atr_multiple(recent_prices)
        
        # Volatility adjustment (1-3x multiplier)
        vol_multiplier = np.clip(atr_pct / 0.015, 1.0, 3.0)
        
        # Confidence adjustment (0.5-1.5x multiplier)
        conf_multiplier = 0.5 + confidence
        
        # Calculate stops
        stop_loss = self.default_stop * vol_multiplier * conf_multiplier
        take_profit = self.default_tp * vol_multiplier * conf_multiplier
        
        # Clamp to reasonable ranges
        stop_loss = np.clip(stop_loss, 0.01, 0.05)     # 1-5%
        take_profit = np.clip(take_profit, 0.03, 0.15) # 3-15%
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr_pct': atr_pct,
            'vol_regime': 'high' if atr_pct > 0.025 else 'low'
        }
        

class DrawdownProtection:
    """
    Circuit breaker for consecutive losses or max drawdown
    """
    def __init__(self, max_drawdown_pct=0.10, pause_duration_hours=24):
        self.max_drawdown_pct = max_drawdown_pct
        self.pause_duration = pause_duration_hours * 3600 # seconds
        self.circuit_breaker_active = False
        self.breaker_trigger_time = 0
    
    def check_drawdown(self, history, timestamp):
        """
        Check if drawdown exceeds threshold
        history: list of dicts with 'equity'
        """
        if self.circuit_breaker_active:
             if timestamp - self.breaker_trigger_time > self.pause_duration:
                 self.circuit_breaker_active = False
                 return False # Resume
             return True # Still paused
             
        if not history:
             return False
             
        # Calculate DD
        equities = [h['equity'] for h in history]
        peak = np.max(equities)
        current = equities[-1]
        
        dd = (peak - current) / peak
        
        if dd > self.max_drawdown_pct:
            self.circuit_breaker_active = True
            self.breaker_trigger_time = timestamp
            return True # Triggered
            
        return False

