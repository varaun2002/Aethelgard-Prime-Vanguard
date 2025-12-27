# Changelog v3.2 - Hyper-Aggressive & Stability Patch
**Date**: 2025-12-25
**Version**: 3.2.0

## ⚡ Feature Updates

### 1. Hyper-Aggressive Mode Activated
**Goal**: Capture "minor changes" and trade on low-probability signals.
*   **Threshold Adjustment**:
    *   `src/inference/predict_v3.py`: Lowered "Range Quiet" and "Range Noisy" prediction thresholds from **0.25** to **0.12**.
    *   `src/inference/paper_trader.py`: Lowered execution threshold for 'ranging' regimes to **0.12** (12%).
*   **Impact**: The bot will now attempt to trade even if the directional probability is as low as 12%, significantly increasing activity during flat markets.

## 🔧 Critical Bug Fixes (Priority Patch List)

### 1. Feature Engineering: `wick_lower` Calculation
*   **File**: `src/features/feature_engineering.py`
*   **Fix**: Corrected formula from `min(open, close)` to `min(open, close) - low`.
*   **Impact**: Features now correctly value the lower shadow size instead of the price level.

### 2. Training Logic: Dead Code Removal
*   **File**: `src/models/train_v3.py`
*   **Fix**: Removed unreachable duplicate return statement in `resample_training_data`.
*   **Impact**: Code cleanliness.

### 3. Data Loader: Feature Count Assertion
*   **File**: `src/data/data_loader.py`
*   **Fix**: Updated assertion to dynamic count + 20 external features (Total 50).
*   **Impact**: Prevents `AssertionError` crashes during data loading.

### 4. Inference: Duplicate JSON Key
*   **File**: `src/inference/predict_v3.py`
*   **Fix**: Removed duplicate `'uncertainty'` key in return dictionary.
*   **Impact**: Ensures valid JSON output for downstream consumers.

### 5. Paper Trader: Contradictory Sell Logic
*   **File**: `src/inference/paper_trader.py`
*   **Fix**: 
    *   Removed impossible `signal == "UP"` condition inside `signal == "DOWN"` block.
    *   Added explicit logic to SELL if signal is "FLAT" with high confidence (> 0.65).
*   **Impact**: Correctly processes sell signals and avoids holding during strong neutral signals.

### 6. Config: Feature Dimensions
*   **File**: `config/config_v3.yaml`
*   **Fix**: Updated `external_features` to 20, `total_features` to 50, and `input_dim` to 50.
*   **Impact**: Aligns configuration with actual model architecture.

### 7. Inference: Input Validation
*   **File**: `src/inference/predict_v3.py`
*   **Fix**: Added rigorous dataframe checks (required columns, length, NaNs) at the start of `predict`.
*   **Impact**: Prevents silent failures or garbage predictions on bad input.

### 8. Data Loader: Quality Validation
*   **File**: `src/data/data_loader.py`
*   **Fix**: Added `validate_data_quality` method to check for price gaps, zero volume, and timestamp gaps.
*   **Impact**: Improves data integrity monitoring.

## 🧹 System Cleanup
*   **Deleted**: `debug_yf.py`, `logs/*.legacy.csv`, `logs/app.log.*`, `logs/regime_test*.log`, `data/regime_test_results.csv`, `data/api.txt`.
*   **Cleaned**: Recursive removal of `__pycache__` directories.

## 🚀 Deployment Status
*   **Dashboard**: Restarted on **http://localhost:8501** with file watcher disabled (`--server.fileWatcherType=none`).
*   **Status**: Active & Monitoring.
