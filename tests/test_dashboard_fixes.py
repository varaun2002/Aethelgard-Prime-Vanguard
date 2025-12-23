#!/usr/bin/env python3
"""Test script to verify dashboard data fetching and prediction fixes"""

import pandas as pd
import yfinance as yf
from src.inference.predict_v3 import PredictorV3

def test_data_fetching():
    """Test yfinance data fetching with new parameters"""
    print("Testing data fetching...")

    symbol = "ETH-USD"

    # Test 15m timeframe
    print("\n1. Testing 15m timeframe...")
    df_15m = yf.download(symbol, period="30d", interval="15m", progress=False)
    df_15m = df_15m.reset_index()
    df_15m.columns = [c.lower() if isinstance(c, str) else c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df_15m.columns]

    if 'date' in df_15m.columns:
        df_15m = df_15m.rename(columns={'date': 'timestamp'})
    if 'datetime' in df_15m.columns:
        df_15m = df_15m.rename(columns={'datetime': 'timestamp'})

    print(f"   Fetched {len(df_15m)} rows")
    print(f"   Columns: {df_15m.columns.tolist()}")
    print(f"   Has enough data (200+): {len(df_15m) >= 200}")

    # Test 1h timeframe
    print("\n2. Testing 1h timeframe...")
    df_1h = yf.download(symbol, period="60d", interval="1h", progress=False)
    df_1h = df_1h.reset_index()
    df_1h.columns = [c.lower() if isinstance(c, str) else c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df_1h.columns]

    if 'date' in df_1h.columns:
        df_1h = df_1h.rename(columns={'date': 'timestamp'})
    if 'datetime' in df_1h.columns:
        df_1h = df_1h.rename(columns={'datetime': 'timestamp'})

    print(f"   Fetched {len(df_1h)} rows")
    print(f"   Has enough data (200+): {len(df_1h) >= 200}")

    return df_15m, df_1h

def test_predictions():
    """Test predictions with fetched data"""
    print("\n\nTesting predictions...")

    df_15m, df_1h = test_data_fetching()

    # Test 15m prediction
    print("\n3. Testing 15m prediction...")
    try:
        predictor_15m = PredictorV3("models_v3/model_v3_0_15m.pth")
        result = predictor_15m.predict(df_15m)

        if result:
            print("   ✅ Prediction successful!")
            print(f"   Direction: {result['predicted_direction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Uncertainty: {result['uncertainty']:.2%}")
            print(f"   Regime: {result['regime']}")
        else:
            print("   ⚠️  Prediction returned None")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 1h prediction
    print("\n4. Testing 1h prediction...")
    try:
        predictor_1h = PredictorV3("models_v3/model_v3_0_1h.pth")
        result = predictor_1h.predict(df_1h)

        if result:
            print("   ✅ Prediction successful!")
            print(f"   Direction: {result['predicted_direction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Uncertainty: {result['uncertainty']:.2%}")
            print(f"   Regime: {result['regime']}")
        else:
            print("   ⚠️  Prediction returned None")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("Testing Dashboard Fixes")
    print("="*60)

    test_predictions()

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)
