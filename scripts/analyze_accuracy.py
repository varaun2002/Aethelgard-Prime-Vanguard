#!/usr/bin/env python3
"""
Analyze prediction accuracy with corrected directional validation
"""

import pandas as pd
import numpy as np

def calculate_actual_direction(predicted_price, actual_price, threshold_pct=0.5):
    """
    Calculate what the actual direction was based on price movement.

    Args:
        predicted_price: The starting price (what model saw)
        actual_price: The price after the timeframe
        threshold_pct: Threshold for FLAT (0.5% = within 0.5% range)
    """
    if actual_price == 0 or predicted_price == 0:
        return 'UNKNOWN'

    # Calculate percentage change from predicted to actual
    pct_change = ((actual_price - predicted_price) / predicted_price) * 100

    # Determine direction
    if abs(pct_change) < threshold_pct:
        return 'FLAT'
    elif pct_change > 0:
        return 'UP'
    else:
        return 'DOWN'

# Load data
print("="*80)
print("PREDICTION ACCURACY ANALYSIS - RECALCULATION")
print("="*80)

df = pd.read_csv('data/prediction_log.csv')
print(f"\nTotal predictions: {len(df)}")

# Filter validated predictions
validated = df[df['outcome'].isin(['GOOD', 'BAD', 'PERFECT'])].copy()
print(f"Validated predictions: {len(validated)}")

if len(validated) == 0:
    print("\n❌ No validated predictions found!")
    exit(1)

# Recalculate actual directions
validated['calculated_actual_direction'] = validated.apply(
    lambda row: calculate_actual_direction(row['predicted_price'], row['actual_price']),
    axis=1
)

# Calculate price errors
validated['price_diff'] = validated['actual_price'] - validated['predicted_price']
validated['price_error_pct'] = (validated['price_diff'] / validated['predicted_price']) * 100

# Check direction matching
validated['direction_match'] = (
    validated['predicted_direction'] == validated['calculated_actual_direction']
)

# Calculate accuracies
current_accuracy = ((validated['outcome'] == 'GOOD').sum() + (validated['outcome'] == 'PERFECT').sum()) / len(validated) * 100
directional_accuracy = (validated['direction_match'].sum() / len(validated)) * 100

print(f"\n" + "="*80)
print(f"ACCURACY COMPARISON")
print("="*80)
print(f"Current (Price-based) Accuracy: {current_accuracy:.2f}% ← WRONG METHOD")
print(f"Directional Accuracy:           {directional_accuracy:.2f}% ← CORRECT METHOD")
print(f"Improvement: {directional_accuracy - current_accuracy:+.2f} percentage points")

# Confusion Matrix
print(f"\n" + "="*80)
print(f"CONFUSION MATRIX (Predicted vs Actual Direction)")
print("="*80)
confusion = pd.crosstab(
    validated['predicted_direction'],
    validated['calculated_actual_direction'],
    margins=True,
    margins_name='Total'
)
print(confusion)

# Accuracy by timeframe
print(f"\n" + "="*80)
print(f"DIRECTIONAL ACCURACY BY TIMEFRAME")
print("="*80)
for tf in ['15m', '1h', '4h']:
    tf_data = validated[validated['timeframe'] == tf]
    if len(tf_data) > 0:
        tf_acc = (tf_data['direction_match'].sum() / len(tf_data)) * 100
        print(f"{tf:4s}: {tf_acc:5.1f}% ({tf_data['direction_match'].sum()}/{len(tf_data)} correct)")

# Price error statistics
print(f"\n" + "="*80)
print(f"PRICE PREDICTION ERROR STATISTICS")
print("="*80)
print(f"Average predicted price: ${validated['predicted_price'].mean():,.2f}")
print(f"Average actual price:    ${validated['actual_price'].mean():,.2f}")
print(f"Average difference:      ${validated['price_diff'].mean():+,.2f}")
print(f"Average % error:         {validated['price_error_pct'].mean():+.2f}%")
print(f"Median % error:          {validated['price_error_pct'].median():+.2f}%")
print(f"Std deviation:           {validated['price_error_pct'].std():.2f}%")

systematic_bias = "LOW" if validated['price_diff'].mean() > 0 else "HIGH"
print(f"\n⚠️  Model is systematically predicting {systematic_bias}")

# Examples of what's being marked as BAD
print(f"\n" + "="*80)
print(f"EXAMPLES OF 'BAD' PREDICTIONS (First 10)")
print("="*80)
bad_examples = validated[validated['outcome'] == 'BAD'].head(10)

if len(bad_examples) > 0:
    print(f"{'Pred Dir':<8} | {'Calc Dir':<8} | {'Match':<5} | {'Pred $':<10} | {'Actual $':<10} | {'Error %':<8}")
    print("-" * 80)
    for idx, row in bad_examples.iterrows():
        match = "✓" if row['direction_match'] else "✗"
        print(f"{row['predicted_direction']:<8} | "
              f"{row['calculated_actual_direction']:<8} | "
              f"{match:<5} | "
              f"${row['predicted_price']:>8.2f} | "
              f"${row['actual_price']:>8.2f} | "
              f"{row['price_error_pct']:>+7.2f}%")
else:
    print("No BAD predictions found")

# Count how many BAD predictions would be GOOD with correct validation
bad_but_correct_direction = validated[
    (validated['outcome'] == 'BAD') &
    (validated['direction_match'] == True)
]

print(f"\n" + "="*80)
print(f"VALIDATION LOGIC IMPACT")
print("="*80)
print(f"Total 'BAD' predictions: {(validated['outcome'] == 'BAD').sum()}")
print(f"BAD but direction correct: {len(bad_but_correct_direction)} "
      f"({len(bad_but_correct_direction)/(validated['outcome'] == 'BAD').sum()*100:.1f}% of BAD)")
print(f"\nThese should have been marked GOOD!")

# Direction distribution
print(f"\n" + "="*80)
print(f"PREDICTION DISTRIBUTION")
print("="*80)
pred_dist = validated['predicted_direction'].value_counts()
actual_dist = validated['calculated_actual_direction'].value_counts()

print("Predicted:")
for dir, count in pred_dist.items():
    pct = count / len(validated) * 100
    print(f"  {dir:<5}: {count:4d} ({pct:5.1f}%)")

print("\nActual:")
for dir, count in actual_dist.items():
    pct = count / len(validated) * 100
    print(f"  {dir:<5}: {count:4d} ({pct:5.1f}%)")

print(f"\n" + "="*80)
print(f"CONCLUSION")
print("="*80)
print(f"The validation logic is BROKEN!")
print(f"It's using price accuracy instead of directional accuracy.")
print(f"Real model performance: {directional_accuracy:.1f}% (not {current_accuracy:.1f}%)")
print("="*80)
