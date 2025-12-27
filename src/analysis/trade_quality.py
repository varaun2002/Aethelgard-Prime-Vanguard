class TradeQualityScorer:
    """
    Score trade setup quality (0-100)
    """
    def calculate_score(self, spread_bps, volume_24h, volatility_rank):
        score = 100
        
        # 1. Spread Penalty
        if spread_bps > 10: # > 0.1%
             score -= (spread_bps - 10) * 2
             
        # 2. Volume Penalty (Low liquidity)
        if volume_24h < 10000000: # < 10M
             score -= 20
             
        # 3. Volatility Penalty (Too low or too high?)
        # Too low = no movement. Too high = unpredictable.
        if volatility_rank < 0.2:
             score -= 10
        elif volatility_rank > 0.9:
             score -= 10
             
        return max(0, score)
