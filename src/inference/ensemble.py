import numpy as np
import torch

class EnsemblePredictor:
    """
    Combine predictions from multiple models
    """
    def __init__(self, models):
        """
        models: list of dict {'model': model_obj, 'weight': 1.0}
        """
        self.models = models
        
    def predict(self, x, vol_features=None):
        """
        Weighted voting
        """
        votes = {
            'UP': 0.0,
            'DOWN': 0.0,
            'FLAT': 0.0
        }
        
        total_weight = 0.0
        
        for item in self.models:
            model = item['model']
            weight = item['weight']
            
            with torch.no_grad():
                out = model(x, vol_features) # Ensure signature matches
                logits = out['direction_logits']
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # [batch=1]
                
                # Probs array: [Noisy?, UP, DOWN]? 
                # Need to check model output mapping.
                # Usually: 0=FLAT, 1=UP, 2=DOWN or similar.
                # Checking TrainV3 / TrainerV3 labels.
                # y_dir mapping: 0: HOLD/FLAT, 1: BUY/UP, 2: SELL/DOWN.
                # Assuming index 0=FLAT, 1=UP, 2=DOWN.
                
                votes['FLAT'] += probs[0] * weight
                votes['UP'] += probs[1] * weight
                votes['DOWN'] += probs[2] * weight
                
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for k in votes:
                votes[k] /= total_weight
                
        # Determine winner
        winner = max(votes, key=votes.get)
        confidence = votes[winner]
        
        return winner, confidence, votes
