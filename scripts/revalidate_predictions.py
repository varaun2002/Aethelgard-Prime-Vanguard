#!/usr/bin/env python3
"""
Revalidate all predictions using the FIXED directional accuracy logic.
This will update prediction_log.csv with correct outcomes.
"""

import pandas as pd
import sys

def calculate_actual_direction(predicted_price, actual_price, threshold_pct=0.5):
    """Calculate actual direction from price movement."""
    if actual_price == 0 or predicted_price == 0:
        return 'UNKNOWN'

    pct_change = ((actual_price - predicted_price) / predicted_price) * 100

    if abs(pct_change) < threshold_pct:
        return 'FLAT'
    elif pct_change > 0:
        return 'UP'
    else:
        return 'DOWN'

print("="*80)
print("REVALIDATING PREDICTIONS WITH CORRECTED LOGIC")
print("="*80)

# Load predictions
df = pd.read_csv('data/prediction_log.csv')
print(f"\nTotal predictions: {len(df)}")

# Find already validated predictions
validated_mask = (df['actual_price'] > 0) & (df['outcome'] != 'PENDING')
print(f"Previously validated: {validated_mask.sum()}")

if validated_mask.sum() == 0:
    print("\n✓ No predictions to revalidate.")
    sys.exit(0)

# Revalidate each one
updates = 0
for idx, row in df[validated_mask].iterrows():
    # Calculate actual direction
    actual_dir = calculate_actual_direction(row['predicted_price'], row['actual_price'])
    predicted_dir = row['predicted_direction']

    # Check if directions match
    direction_match = (predicted_dir == actual_dir)

    # Calculate price error
    price_error = abs(row['actual_price'] - row['predicted_price']) / row['predicted_price']

    # Determine outcome
    if direction_match:
        if price_error < 0.01:
            new_outcome = 'PERFECT'
        else:
            new_outcome = 'GOOD'
    else:
        new_outcome = 'BAD'

    # Update if changed
    if df.at[idx, 'outcome'] != new_outcome:
        df.at[idx, 'outcome'] = new_outcome
        updates += 1

print(f"\nUpdated {updates} predictions")

# Save
df.to_csv('data/prediction_log.csv', index=False)
print(f"✓ Saved to data/prediction_log.csv")

# Calculate new accuracy
revalidated = df[df['outcome'].isin(['GOOD', 'BAD', 'PERFECT'])]
if len(revalidated) > 0:
    good_count = ((revalidated['outcome'] == 'GOOD') | (revalidated['outcome'] == 'PERFECT')).sum()
    accuracy = (good_count / len(revalidated)) * 100

    print(f"\n" + "="*80)
    print(f"NEW ACCURACY AFTER FIX")
    print("="*80)
    print(f"Correct:  {good_count}/{len(revalidated)} predictions")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nBreakdown:")
    print(f"  GOOD:    {(revalidated['outcome'] == 'GOOD').sum()}")
    print(f"  PERFECT: {(revalidated['outcome'] == 'PERFECT').sum()}")
    print(f"  BAD:     {(revalidated['outcome'] == 'BAD').sum()}")

    # By timeframe
    print(f"\nAccuracy by Timeframe:")
    for tf in ['15m', '1h', '4h']:
        tf_data = revalidated[revalidated['timeframe'] == tf]
        if len(tf_data) > 0:
            tf_good = ((tf_data['outcome'] == 'GOOD') | (tf_data['outcome'] == 'PERFECT')).sum()
            tf_acc = (tf_good / len(tf_data)) * 100
            print(f"  {tf:4s}: {tf_acc:5.1f}% ({tf_good}/{len(tf_data)})")

print("\n" + "="*80)
print("✓ REVALIDATION COMPLETE")
print("="*80)
