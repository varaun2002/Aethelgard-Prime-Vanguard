"""
Validation script for V3.0 external factors
"""

import numpy as np
import torch
import sys
import os

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.data.external_data import ExternalDataFetcher
from src.models.model import CryptoModelV3
from src.data.data_loader import DataLoader
import ccxt

def validate_external_fetcher():
    """Test external data fetcher"""
    print("\n1. Testing ExternalDataFetcher...")
    
    fetcher = ExternalDataFetcher()
    exchange = ccxt.binance()
    try:
        eth_data = exchange.fetch_ohlcv('ETH/USDT', '15m', limit=100)
    except Exception as e:
        print(f"   ! Warning: Could not fetch from Binance ({e}). Using dummy data.")
        eth_data = [[1600000000000 + i*900000, 100+i, 101+i, 99+i, 100+i, 1000] for i in range(100)]
    
    features = fetcher.get_all_features(eth_data)
    
    assert len(features) == 20, f"Expected 20 features, got {len(features)}"
    assert np.all(np.isfinite(features)), "Features contain NaN or Inf"
    
    print(f"   ✓ Generated {len(features)} external features")
    print(f"   ✓ All features are finite")
    
    return True

def validate_model_input():
    """Test model accepts 70 features"""
    print("\n2. Testing Model Input...")
    
    model = CryptoModelV3(input_dim=70, hidden_dim=128) if 'hidden_dim' in CryptoModelV3.__init__.__code__.co_varnames else CryptoModelV3(input_dim=70)
    
    # Create dummy input: [Batch=1, Seq=60, Dim=70]
    # Model expects sequence [Batch, Seq, Dim]
    dummy_input = torch.randn(1, 60, 70)
    
    try:
        output = model(dummy_input) # specific to model input structure
        # Output is dict
        print(f"   ✓ Model accepts 70-feature input")
        print(f"   ✓ Output keys: {output.keys()}")
        return True
    except Exception as e:
        print(f"   ✗ Model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_feature_combination():
    """Test combining technical + external features via DataLoader"""
    print("\n3. Testing Feature Combination (DataLoader)...")
    
    loader = DataLoader()
    
    # Create dummy ETH data
    eth_data = [[1600000000000 + i*900000, 100+i, 101+i, 99+i, 100+i, 1000] for i in range(100)]
    
    combined = loader.prepare_features(eth_data)
    
    assert len(combined) == 70, f"Expected 70 features, got {len(combined)}"
    print(f"   ✓ Combined features: {len(combined)}")
    print(f"   ✓ Technical: 0-49, External: 50-69")
    
    return True

def run_all_validations():
    """Run all validation tests"""
    print("="*60)
    print("V3.0 VALIDATION TESTS")
    print("="*60)
    
    results = []
    
    try:
        results.append(validate_external_fetcher())
    except Exception as e:
        print(f"Test 1 failed with exception: {e}")
        results.append(False)
        
    try:
         results.append(validate_model_input())
    except Exception as e:
        print(f"Test 2 failed with exception: {e}")
        results.append(False)
        
    try:
        results.append(validate_feature_combination())
    except Exception as e:
        print(f"Test 3 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)
    
    print("\n" + "="*60)
    if all(results):
        print("✓ ALL TESTS PASSED - V3.0 Ready!")
    else:
        print("✗ SOME TESTS FAILED - Fix errors above")
    print("="*60)

if __name__ == "__main__":
    run_all_validations()
