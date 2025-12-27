import numpy as np
from scipy import stats

class DriftDetector:
    """
    Detect feature or concept drift
    """
    def __init__(self, reference_data=None, p_value_threshold=0.05):
        self.reference_data = reference_data # List/Array of past successful predictions' confidence?
        self.p_value_threshold = p_value_threshold
        self.current_window = []
        
        # If no reference, assume uniform distribution of confidence for now (bad assumption but placeholder)
        if reference_data is None:
             self.reference_data = [0.6, 0.7, 0.8, 0.9] * 10 
             
    def add_prediction(self, confidence):
        self.current_window.append(confidence)
        if len(self.current_window) > 1000:
            self.current_window.pop(0)
            
    def check_drift(self):
        """
        Check if current confidence distribution matches reference (training) distribution
        """
        if len(self.current_window) < 50:
            return False, 1.0
            
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(self.reference_data, self.current_window)
        
        is_drift = p_value < self.p_value_threshold
        
        return is_drift, p_value
