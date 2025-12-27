# Changelog v3.3 - Optimized Validation Release

## New Features
- **Automated Log Validation**: Integrated `LogValidator` into `health_monitor.py`. Predictions are now validated automatically every minute.
- **Optimized Confidence Threshold**: Raised from 0.12 (Hyper-Aggressive) to **0.24** to reduce noise and drawdown.
- **Robust Dashboard**: Fixed crashing issues caused by missing keys in portfolio state.

## Bug Fixes
- **Probability Swap**: Fixed critical bug where DOWN probabilities were swapped with UP.
- **Portfolio Reset**: Fixed JSON schema mismatch during portfolio resets.

## Performance
- **Phase 6.2**: Currently running long-duration validation.
- **Threshold**: 0.24
- **Stop Loss**: 3% (Hard), 1.5% (Trailing)
