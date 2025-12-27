import numpy as np
import pandas as pd

class CorrelationHedger:
    """
    Hedge ETH long positions using short BTC futures based on correlation
    """
    def __init__(self, hedge_ratio=0.5):
        self.hedge_ratio = hedge_ratio # Hedge 50% of delta
        self.btc_eth_correlation = 0.8
        
    def update_correlation(self, eth_prices, btc_prices):
        """
        Update rolling correlation
        """
        if len(eth_prices) != len(btc_prices):
            return
            
        if len(eth_prices) < 30:
            return
            
        eth_ret = np.diff(eth_prices) / eth_prices[:-1]
        btc_ret = np.diff(btc_prices) / btc_prices[:-1]
        
        # Calculate correlation
        corr = np.corrcoef(eth_ret, btc_ret)[0, 1]
        
        self.btc_eth_correlation = corr
        
    def calculate_hedge_size(self, eth_position_value, market_regime):
        """
        Calculate short BTC size needed to hedge
        """
        # Only hedge in uncertain regimes or downtrends if we are stuck long
        target_hedge_ratio = 0.0
        
        if market_regime == 'Trending DOWN':
            target_hedge_ratio = 1.0 # Full hedge
        elif market_regime == 'Range Noisy':
            target_hedge_ratio = 0.5
        elif market_regime == 'Trending UP':
            target_hedge_ratio = 0.0 # No hedge
            
        if target_hedge_ratio == 0:
            return 0.0
            
        # Beta adjustment
        # If ETH is 2x more volatile than BTC, we need less BTC to hedge dollar for dollar?
        # Beta = Cov(eth, btc) / Var(btc)
        # Simplified: Use correlation * relative_volatility
        # Assume Beta ~ 1.2
        beta = 1.2 
        
        hedge_value = eth_position_value * target_hedge_ratio * beta
        
        return hedge_value
