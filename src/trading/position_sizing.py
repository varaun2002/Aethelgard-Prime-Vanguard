import numpy as np

class KellyPositionSizer:
    """
    Size positions based on edge and uncertainty
    """
    def __init__(self, max_position=0.95, kelly_fraction=0.25):
        """
        max_position: Maximum fraction of capital per trade
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        """
        self.max_position = max_position
        self.kelly_fraction = kelly_fraction
    
    def calculate_position_size(self, confidence, historical_win_rate, 
                               avg_win, avg_loss, current_capital):
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            confidence: Model confidence (0.5-1.0)
            historical_win_rate: Recent win rate
            avg_win: Average winning trade %
            avg_loss: Average losing trade %
            current_capital: Available capital
        
        Returns:
            position_size: Dollar amount to trade
        """
        # Edge = probability of winning * confidence
        # Use simple Kelly first:
        # f = (bp - q) / b
        
        # Cold Start: If no history, use safe default (e.g. 5% of capital)
        if avg_loss == 0 or historical_win_rate == 0:
            return current_capital * 0.05
        
        odds = abs(avg_win / avg_loss)
        win_prob = historical_win_rate
        
        loss_prob = 1 - win_prob
        
        kelly = (odds * win_prob - loss_prob) / odds
        
        if kelly <= 0:
            return 0
            
        # Apply Kelly fraction for safety
        kelly_fraction = kelly * self.kelly_fraction
        
        # Confidence adjustment: Bet more when confident
        # Base on deviation from neutral 0.33 (3-class) rather than 0.5
        conf_multiplier = min(2.0, max(0.5, (confidence - 0.33) / 0.1))
        
        final_fraction = kelly_fraction * conf_multiplier
        
        # Apply limits
        final_fraction = np.clip(final_fraction, 0, self.max_position)
        
        position_size = current_capital * final_fraction
        
        return position_size
    
    def should_trade(self, confidence, min_edge=0.0):
        """
        Determine if edge is sufficient to trade.
        For 3-class model, baseline is 0.33. 
        But PaperTrader already filters by threshold, so we can be lenient here.
        """
        return True # Rely on PaperTrader threshold
