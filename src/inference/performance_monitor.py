import numpy as np

class PerformanceMonitor:
    """
    Shut down trading if performance degrades
    """
    def __init__(self, min_accuracy=0.55, window=100):
        self.min_accuracy = min_accuracy
        self.window = window
        self.predictions = []
        self.actuals = []
    
    def add_result(self, prediction, actual):
        self.predictions.append(prediction)
        self.actuals.append(actual)
        
        # Keep only recent window
        if len(self.predictions) > self.window:
            self.predictions.pop(0)
            self.actuals.pop(0)
    
    def should_continue_trading(self):
        """
        Check if recent accuracy meets minimum threshold
        """
        if len(self.predictions) < 20: # Minimum samples before judgment
            return True  # Not enough data yet
        
        # Calculate matches
        matches = []
        for p, a in zip(self.predictions, self.actuals):
            # If model predicted FLAT (no trade), we might ignore it for accuracy?
            # Or if it predicted UP/DOWN and actual was FLAT, is that wrong?
            # Let's say:
            # If Pred == Actual: 1
            # If Pred != Actual: 0
            
            # Strict accuracy
            is_correct = (p == a)
            matches.append(is_correct)
            
        recent_accuracy = np.mean(matches)
        
        if recent_accuracy < self.min_accuracy:
            # print(f"⛔ TRADING HALTED: Accuracy {recent_accuracy:.1%} < {self.min_accuracy:.1%}")
            return False
        
        return True
