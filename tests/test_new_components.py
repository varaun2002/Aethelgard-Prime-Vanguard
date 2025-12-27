import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.regime_trainer import RegimeSpecificTrainer
from src.validation.cross_validator import TimeSeriesCrossValidator
from src.trading.position_sizing import KellyPositionSizer
from src.trading.risk_management import VolatilityAwareStops, DrawdownProtection
from src.inference.regime_detector import RegimeDetector
from src.inference.ensemble import EnsemblePredictor

class TestNewComponents(unittest.TestCase):
    def test_instantiation(self):
        print("\nTesting Instantiation...")
        try:
            rt = RegimeSpecificTrainer()
            ts_cv = TimeSeriesCrossValidator()
            kelly = KellyPositionSizer()
            vol_stops = VolatilityAwareStops()
            dd_prot = DrawdownProtection()
            reg_det = RegimeDetector()
            # ensemble needs model list
            ens = EnsemblePredictor([])
            
            print("✓ All components instantiated successfully")
        except Exception as e:
            self.fail(f"Instantiation failed: {e}")
            
    def test_kelly(self):
        print("\nTesting Kelly...")
        kelly = KellyPositionSizer(max_position=0.95, kelly_fraction=0.25)
        # Conf 0.7, WinRate 0.55, AvgWin 0.05, AvgLoss 0.02
        # Odds = 2.5
        # Kelly = 2.5*0.55 - 0.45 / 2.5 = (1.375 - 0.45)/2.5 = 0.925/2.5 = 0.37
        # Fraction = 0.25 * 0.37 = 0.09
        # Multiplier (0.7-0.5)/0.1 = 2 -> 2x? Logic: min(2, max(0.5, 2)) = 2.
        # Total = 0.18 roughly.
        
        size = kelly.calculate_position_size(0.7, 0.55, 0.05, 0.02, 10000)
        print(f"Kelly Size (Conf=0.7): ${size:.2f}")
        self.assertTrue(size > 0)
        
    def test_vol_stops(self):
        print("\nTesting Vol Stops...")
        stops = VolatilityAwareStops()
        # Mock prices
        prices = pd.Series([100 + np.sin(i)*2 for i in range(100)])
        res = stops.get_stops(prices, 0.8)
        print(f"Stops: {res}")
        self.assertTrue(res['stop_loss'] > 0)
        self.assertTrue(res['take_profit'] > 0)

if __name__ == '__main__':
    unittest.main()
