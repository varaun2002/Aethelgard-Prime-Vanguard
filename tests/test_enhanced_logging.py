#!/usr/bin/env python3
"""
Test the enhanced logging system with sample predictions
"""

from src.utils.enhanced_logger import EnhancedLogger, TradeLogger, SystemLogger
import time
import os
import pandas as pd

print("="*80)
print("TESTING ENHANCED LOGGING SYSTEM")
print("="*80)

# Create loggers
enhanced_logger = EnhancedLogger(log_dir="logs/predictions_test")
trade_logger = TradeLogger(log_file="logs/trades_test.csv")
system_logger = SystemLogger(log_file="logs/system_events_test.log")

print("\n1. Testing Prediction Logging...")
print("-" * 80)

# Simulate predictions for each timeframe
test_predictions = {
    '15m': {
        'predicted_direction': 'UP',
        'confidence': 0.65,
        'uncertainty': 0.15,
        'predicted_price': 2850.0,
        'predicted_return': 0.018,
        'regime': 'Trending UP',
        'probabilities': [0.20, 0.15, 0.65],
        'caution': False
    },
    '1h': {
        'predicted_direction': 'UP',
        'confidence': 0.58,
        'uncertainty': 0.18,
        'predicted_price': 2855.0,
        'predicted_return': 0.020,
        'regime': 'Range Quiet',
        'probabilities': [0.25, 0.17, 0.58],
        'caution': False
    },
    '4h': {
        'predicted_direction': 'FLAT',
        'confidence': 0.42,
        'uncertainty': 0.25,
        'predicted_price': 2840.0,
        'predicted_return': 0.005,
        'regime': 'Range Noisy',
        'probabilities': [0.42, 0.30, 0.28],
        'caution': True
    }
}

current_price = 2845.0

for timeframe, prediction in test_predictions.items():
    gate_info = {
        'gate1_passed': True,
        'gate2_passed': prediction['confidence'] > 0.45,
        'gate3_applied': False
    }

    trade_info = {
        'signal': 'HOLD',
        'executed': False
    }

    # Only 15m would potentially trade
    if timeframe == '15m' and prediction['predicted_direction'] == 'UP':
        trade_info = {
            'signal': 'BUY',
            'executed': True
        }

    enhanced_logger.log_prediction(
        timeframe=timeframe,
        prediction=prediction,
        current_price=current_price,
        gate_info=gate_info,
        trade_info=trade_info
    )

    print(f"✓ Logged {timeframe:4s} prediction: {prediction['predicted_direction']:5s} "
          f"(Conf: {prediction['confidence']:.1%}, Unc: {prediction['uncertainty']:.1%})")

print("\n2. Testing Trade Logging...")
print("-" * 80)

# Simulate a trade execution
portfolio_before = {
    'balance': 10000.0,
    'position': 0.0,
    'equity': 10000.0
}

portfolio_after = {
    'balance': 500.0,
    'position': 3.3392,  # Bought ~$9,500 worth at $2,845
    'equity': 10000.0
}

trade_logger.log_trade(
    action='BUY',
    price=2845.0,
    size=3.3392,
    confidence=0.65,
    timeframe='15m',
    predicted_direction='UP',
    portfolio_before=portfolio_before,
    portfolio_after=portfolio_after
)

print(f"✓ Logged BUY trade: 3.3392 ETH at $2,845.00")
print(f"  Portfolio: ${portfolio_before['balance']:,.2f} → ${portfolio_after['balance']:,.2f}")
print(f"  Position:  {portfolio_before['position']:.4f} ETH → {portfolio_after['position']:.4f} ETH")

print("\n3. Testing System Event Logging...")
print("-" * 80)

system_logger.log_event('INFO', 'Enhanced logging system initialized successfully')
system_logger.log_event('WARNING', 'High uncertainty detected', {'uncertainty': 0.28, 'threshold': 0.25})
system_logger.log_event('RETRAIN', 'Accuracy dropped below threshold', {'accuracy': 0.38, 'threshold': 0.40})

print("✓ Logged 3 system events (INFO, WARNING, RETRAIN)")

print("\n4. Verifying Log Files Created...")
print("-" * 80)

expected_files = [
    'logs/predictions_test/predictions_15m.csv',
    'logs/predictions_test/predictions_1h.csv',
    'logs/predictions_test/predictions_4h.csv',
    'logs/predictions_test/predictions_all.csv',
    'logs/trades_test.csv',
    'logs/system_events_test.log'
]

all_exist = True
for filepath in expected_files:
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    if not exists:
        all_exist = False

print("\n5. Checking Log Contents...")
print("-" * 80)

# Check prediction logs
for timeframe in ['15m', '1h', '4h', 'all']:
    filepath = f'logs/predictions_test/predictions_{timeframe}.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"✓ {timeframe:4s} log: {len(df)} prediction(s), {len(df.columns)} columns")

        # Verify all expected columns exist
        expected_cols = ['timestamp', 'date', 'timeframe', 'predicted_direction',
                        'confidence', 'uncertainty', 'gate1_passed', 'gate2_passed']
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            print(f"  ⚠️  Missing columns: {missing}")

# Check trade log
if os.path.exists('logs/trades_test.csv'):
    df = pd.read_csv('logs/trades_test.csv')
    print(f"✓ Trade log: {len(df)} trade(s), {len(df.columns)} columns")

# Check system log
if os.path.exists('logs/system_events_test.log'):
    with open('logs/system_events_test.log', 'r') as f:
        lines = f.readlines()
    print(f"✓ System log: {len(lines)} event(s)")

print("\n6. Sample Log Entry...")
print("-" * 80)

# Show one prediction log entry
df = pd.read_csv('logs/predictions_test/predictions_15m.csv')
if len(df) > 0:
    latest = df.iloc[-1]
    print(f"Latest 15m prediction:")
    print(f"  Timestamp:   {latest['date']}")
    print(f"  Direction:   {latest['predicted_direction']}")
    print(f"  Confidence:  {latest['confidence']:.1%}")
    print(f"  Uncertainty: {latest['uncertainty']:.1%}")
    print(f"  Regime:      {latest['regime']}")
    print(f"  Gate 1:      {'PASS' if latest['gate1_passed'] else 'FAIL'}")
    print(f"  Gate 2:      {'PASS' if latest['gate2_passed'] else 'FAIL'}")
    print(f"  Gate 3:      {'APPLIED' if latest['gate3_applied'] else 'NOT APPLIED'}")
    print(f"  Trade:       {latest['trade_signal']}")

print("\n" + "="*80)
if all_exist:
    print("✅ ALL TESTS PASSED - Enhanced logging system is working!")
else:
    print("⚠️  SOME FILES MISSING - Check errors above")
print("="*80)

print("\nLog files created in:")
print("  - logs/predictions_test/  (separate CSVs per timeframe)")
print("  - logs/trades_test.csv    (trade executions)")
print("  - logs/system_events_test.log (system events)")
print("\nYou can delete these test logs with: rm -rf logs/predictions_test logs/trades_test.csv logs/system_events_test.log")
