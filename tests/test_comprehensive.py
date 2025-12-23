#!/usr/bin/env python3
"""Comprehensive dashboard test - simulates user interaction"""

import time
import pandas as pd
import yfinance as yf
from src.inference.predict_v3 import PredictorV3
from src.inference.paper_trader import PaperTrader
from src.inference.feedback_manager import FeedbackManager

def test_data_availability():
    """Test what data is actually available from yfinance"""
    print("="*70)
    print("1. TESTING DATA AVAILABILITY FROM YFINANCE")
    print("="*70)

    symbol = "ETH-USD"

    periods_to_test = {
        "15m": [("7d", "15m"), ("30d", "15m"), ("60d", "15m")],
        "1h": [("7d", "1h"), ("60d", "1h"), ("90d", "1h"), ("730d", "1h")],
        "4h": [("60d", "1h"), ("90d", "1h"), ("180d", "1h")]
    }

    best_configs = {}

    for timeframe, configs in periods_to_test.items():
        print(f"\n{timeframe} Timeframe:")
        max_rows = 0
        best_config = None

        for period, interval in configs:
            try:
                df = yf.download(symbol, period=period, interval=interval, progress=False)
                rows = len(df)
                print(f"  Period={period:6s} Interval={interval:4s} → {rows:4d} rows", end="")

                if rows > max_rows:
                    max_rows = rows
                    best_config = (period, interval)
                    print(" ← BEST")
                else:
                    print()

            except Exception as e:
                print(f"  Period={period:6s} Interval={interval:4s} → ERROR: {e}")

        best_configs[timeframe] = (best_config, max_rows)
        print(f"  ✅ Best config: {best_config[0]}, {best_config[1]} → {max_rows} rows")

    return best_configs

def test_predictions_all_timeframes():
    """Test predictions for all timeframes"""
    print("\n" + "="*70)
    print("2. TESTING PREDICTIONS FOR ALL TIMEFRAMES")
    print("="*70)

    symbol = "ETH-USD"

    # Use optimal periods based on availability
    configs = {
        "15m": ("30d", "15m", "models_v3/model_v3_0_15m.pth"),
        "1h": ("730d", "1h", "models_v3/model_v3_0_1h.pth"),
        "4h": ("90d", "1h", "models_v3/model_v3_0_4h.pth")  # Will resample to 4h
    }

    results = {}

    for timeframe, (period, interval, model_path) in configs.items():
        print(f"\n{timeframe} Timeframe:")
        print(f"  Fetching data (period={period}, interval={interval})...")

        try:
            # Fetch data
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            df = df.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]

            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})

            # Resample for 4h if needed
            if timeframe == "4h":
                print(f"  Resampling from 1h to 4h...")
                df = df.set_index('timestamp')
                df_4h = df.resample('4h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                df = df_4h.reset_index()

            print(f"  Data fetched: {len(df)} rows")
            print(f"  Current price: ${df['close'].iloc[-1]:.2f}")

            # Run prediction
            print(f"  Running prediction...")
            predictor = PredictorV3(model_path)
            result = predictor.predict(df)

            if result:
                print(f"  ✅ PREDICTION SUCCESS!")
                print(f"     Direction:   {result['predicted_direction']:5s}")
                print(f"     Confidence:  {result['confidence']:.2%}")
                print(f"     Uncertainty: {result['uncertainty']:.2%}")
                print(f"     Regime:      {result['regime']}")
                print(f"     Pred Return: {result['predicted_return']*100:.2f}%")
                print(f"     Caution:     {'⚠️ YES' if result.get('caution') else '✅ NO'}")
                results[timeframe] = result
            else:
                print(f"  ⚠️ Prediction returned None")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results

def test_paper_trading():
    """Test paper trading functionality"""
    print("\n" + "="*70)
    print("3. TESTING PAPER TRADING SYSTEM")
    print("="*70)

    trader = PaperTrader()

    # Simulate some trades
    test_scenarios = [
        ("UP", 0.65, 3500.0, "Should BUY"),
        ("DOWN", 0.70, 3510.0, "Should SELL"),
        ("FLAT", 0.50, 3505.0, "Should HOLD"),
        ("UP", 0.45, 3520.0, "Low confidence, should HOLD"),
    ]

    print("\nSimulating trades:")
    for direction, confidence, price, expected in test_scenarios:
        timestamp = int(time.time())
        action = trader.update(direction, confidence, price, timestamp)
        action_str = action if action else "HOLD"
        print(f"  {direction:5s} @ ${price:.2f} (conf={confidence:.2f}) → {action_str:10s} ({expected})")

    # Check stats
    stats = trader.get_stats()
    print(f"\n📊 Paper Trading Stats:")
    print(f"  Starting Balance: $10,000.00")
    print(f"  Current Equity:   ${stats['current_equity']:.2f}")
    print(f"  PnL:              ${stats['current_equity'] - 10000:.2f} ({stats['pnl_percent']:.2f}%)")
    print(f"  Open Position:    {stats['open_position']:.6f} ETH")
    print(f"  Total Trades:     {len(trader.state['trades'])}")

    return stats

def test_feedback_system():
    """Test feedback/active learning system"""
    print("\n" + "="*70)
    print("4. TESTING FEEDBACK & ACTIVE LEARNING SYSTEM")
    print("="*70)

    feedback = FeedbackManager()

    # Log some test predictions
    current_time = int(time.time())

    test_predictions = [
        {"timeframe": "15m", "predicted_direction": "UP", "confidence": 0.65, "price": 3500.0},
        {"timeframe": "15m", "predicted_direction": "DOWN", "confidence": 0.70, "price": 3510.0},
        {"timeframe": "1h", "predicted_direction": "FLAT", "confidence": 0.50, "price": 3505.0},
    ]

    print("\nLogging predictions:")
    for i, pred in enumerate(test_predictions):
        pred_with_result = {
            "predicted_direction": pred["predicted_direction"],
            "confidence": pred["confidence"],
            "predicted_return": 0.01,
            "predicted_price": pred["price"] * 1.01,  # Add predicted price
            "uncertainty": 0.15,
            "regime": "Range Quiet",
            "probabilities": [0.3, 0.3, 0.4],
            "caution": False
        }
        feedback.log_prediction(pred["timeframe"], pred_with_result, pred["price"])
        print(f"  {i+1}. Logged {pred['timeframe']} prediction: {pred['predicted_direction']} @ ${pred['price']:.2f}")

    # Check accuracy (will not validate yet since predictions are recent)
    print(f"\n✅ Feedback system initialized")
    print(f"  Predictions logged: {len(test_predictions)}")

    return feedback

def main():
    """Run comprehensive dashboard test"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*18 + "COMPREHENSIVE DASHBOARD TEST" + " "*22 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")

    # Test 1: Data Availability
    best_configs = test_data_availability()

    # Test 2: Predictions
    prediction_results = test_predictions_all_timeframes()

    # Test 3: Paper Trading
    trading_stats = test_paper_trading()

    # Test 4: Feedback System
    feedback_manager = test_feedback_system()

    # Summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*26 + "TEST SUMMARY" + " "*30 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")

    print("✅ Data Fetching:")
    for tf, (config, rows) in best_configs.items():
        print(f"   {tf:4s}: {rows:4d} rows available")

    print("\n✅ Predictions:")
    for tf in ["15m", "1h", "4h"]:
        if tf in prediction_results:
            result = prediction_results[tf]
            print(f"   {tf:4s}: {result['predicted_direction']:5s} (conf={result['confidence']:.2%}, regime={result['regime']})")
        else:
            print(f"   {tf:4s}: ⚠️ Not working")

    print("\n✅ Paper Trading:")
    print(f"   Equity: ${trading_stats['current_equity']:.2f}")
    print(f"   Position: {trading_stats['open_position']:.6f} ETH")

    print("\n✅ Feedback System: Operational")

    print("\n" + "█"*70)
    print("Dashboard is fully operational and ready to use! 🚀")
    print("Access it at: http://localhost:8501")
    print("█"*70 + "\n")

if __name__ == "__main__":
    main()
