#!/usr/bin/env python3
"""
Test the fixed validation logic with a sample prediction
"""

from src.inference.feedback_manager import FeedbackManager
import time

print("="*80)
print("TESTING FIXED VALIDATION LOGIC")
print("="*80)

# Create manager
manager = FeedbackManager()

# Simulate a prediction
test_prediction = {
    'predicted_direction': 'UP',
    'confidence': 0.65,
    'predicted_price': 2750.0,  # Predicted $2,750
    'predicted_return': 0.02,
    'uncertainty': 0.15,
    'regime': 'Range Quiet',
    'probabilities': [0.2, 0.15, 0.65],
    'caution': False
}

current_price = 2750.0  # Current price when prediction made

print("\nTest Scenario:")
print(f"  Predicted Direction: {test_prediction['predicted_direction']}")
print(f"  Predicted Price:     ${test_prediction['predicted_price']:.2f}")
print(f"  Current Price:       ${current_price:.2f}")
print(f"  Confidence:          {test_prediction['confidence']:.1%}")

# Log the prediction
print("\n1. Logging prediction...")
manager.log_prediction('15m', test_prediction, current_price)
print("   ✓ Prediction logged")

# Simulate time passing and price moving
print("\n2. Simulating 15 minutes passing...")
time.sleep(1)  # Just 1 second for testing

# Simulate price going UP to $2,820 (2.5% gain)
future_price = 2820.0
future_time = int(time.time()) + 900  # 15 minutes later

print(f"   Actual price after 15m: ${future_price:.2f}")
print(f"   Price change: +${future_price - current_price:.2f} (+{((future_price - current_price) / current_price) * 100:.2f}%)")

# Check accuracy (this will validate the prediction)
print("\n3. Running validation...")
manager.check_accuracy(future_price, future_time)

# Load the log and check the result
import pandas as pd
df = pd.read_csv('prediction_log.csv')
latest = df.iloc[-1]

print(f"\n4. Validation Result:")
print(f"   Predicted Direction: {latest['predicted_direction']}")
print(f"   Actual Direction:    {'UP' if future_price > current_price else 'DOWN'}")
print(f"   Outcome:             {latest['outcome']}")
print(f"   Price Error:         {latest['error']*100:.2f}%")

# Determine if correct
if latest['outcome'] in ['GOOD', 'PERFECT']:
    print("\n✅ SUCCESS: Validation correctly marked this as GOOD!")
    print("   Even though price error is 2.5%, direction was correct (UP → UP)")
elif latest['outcome'] == 'BAD':
    print("\n❌ FAILED: Still marked as BAD (validation not working)")
elif latest['outcome'] == 'PENDING':
    print("\n⏸️  PENDING: Validation hasn't run yet (normal if <15min passed)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
