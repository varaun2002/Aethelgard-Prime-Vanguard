# 🧠 Aethelgard Prime v1.0 - System Explainer

This document details the inner workings of the AI model, how it generates predictions, and how the simulation executes trades.

---

## 1. The AI Model (Brain)

The core is a **Hybrid Transformer-GRU** neural network designed to capture both long-term context and short-term volatility.

### Architecture (`src/models/model.py`)
It combines two powerful architectures:
1.  **Transformer Encoder**:
    *   **Role**: Analyzes the entire history (60 candles) to find global patterns and "regimes" (e.g., "This looks like the 2021 crash").
    *   **Mechanism**: Uses Self-Attention to weigh the importance of every past candle against every other candle.
2.  **GRU (Gated Recurrent Unit)**:
    *   **Role**: Focuses on the immediate sequence of events.
    *   **Mechanism**: Processes candles sequentially (t-1, t-2...) to capture momentum and immediate trend direction.
3.  **Cross-Attention Fusion**:
    *   **Innovation**: Instead of just adding the outputs, it uses an Attention mechanism where the "Volatile/Recent" state queries the "Long-Term/Stable" context.
    *   **Result**: The model knows *when* to trust the trend (Transformer) and when to react to immediate shock (GRU).

### Inputs (70 Features)
*   **50 Technical**: RSI, MACD, Bollinger Bands, Volume, Returns, etc.
*   **20 External**: SP500, Gold, DXY (Dollar Index), Interest Rates, Fear & Greed Index, Funding Rates.

### 🆕 Unified Data Pipeline (v3.1 Fix)
*   **Training vs Serving Parity**: We strictly ensure that the features used during training match the live inference features exactly.
    *   *Previous Bug Fix:* Fixed an issue where Training used zero-filled BTC data while Inference used real data, causing "Alien DNA" confusion.
    *   **Micro-Retraining**: The active model has been fine-tuned on the most recent 60 days of "Range Noisy" data to adapt to current market chop.

---

## 2. Prediction Logic (The "Three Gates")

The raw model output is not enough. We use a rigorous **"Three Gates"** filtering system (`src/inference/predict_v3.py`) to ensure safety.

### Step 1: Monte Carlo Dropout (Uncertainty Estimation)
Instead of predicting once, we run the model **10 times** slightly differently (randomly dropping connections).
*   If the 10 predictions vary wildly, **Uncertainty is HIGH**.
*   If they all agree, **Uncertainty is LOW**.

### Step 2: The Gates
A signal must pass these gates to become a "Trade":

*   **🚪 Gate 1: Epistemic Uncertainty (Confusion Check)**
    *   We calculate the variance of the 10 Monte Carlo runs.
    *   **Rule**: If Uncertainty > Dynamic Threshold (e.g., 0.20), the model is "confused".
    *   **Action**: Force output to **FLAT**. (Don't trade if you don't know).

*   **🚪 Gate 2: Aleatoric Confidence (Strength Check)**
    *   The model outputs a probability (e.g., "70% chance UP").
    *   **Rule**: If Confidence < **0.47** (Updated for Sniper Mode), the signal is too weak.
    *   **Action**: Force output to **FLAT**.

    *   The model also predicts the **Market Regime** (Trending vs. Ranging).
    *   **Rule**: If the Regime is "Strong Trend" but the specific candle is weak, we might "Force" a trade in the direction of the trend to avoid missing out.

---

## 3. Trading Simulation (Paper Trader v3.2)

The execution logic varies by bot profile.

### A. "Prime" (Consensus & Protect)
We run multiple instances of the model (15m, 1h, 4h) in parallel.
*   **Safety Rule**: Calculations for entry usually require the **15m (Short Term)** and **1h (Medium Term)** models to AGREE.
*   **Result**: If 1h says UP but 15m says DOWN, we wait. This filters out "noise".

### B. "Vanguard" (Aggressive Sniper)
Operates purely on the 15m timeframe for speed.
*   **Sniper Mode**: Uses a strict confidence floor (0.47).
    *   **Chop Filter**: If confidence is low, it strictly outputs **FLAT**.
    *   **Goal**: Trade less, trade better. Avoid the "Death by 1000 Cuts" of ranging markets.

### Execution Rules
1.  **Fees**: We simulate a **0.1% Fee** on every trade to ensure realistic profitability.
2.  **Check Exits First**:
    *   **Take Profit (TP)**: +5% gain (Moonbag).
    *   **Stop Loss (SL)**: -2% loss (Hard Floor).
    *   **Trailing Stop**: If profit drops **1.5% from its peak**, we sell to lock in gains.
3.  **Process Entry**:
    *   **BUY**: If Logic says UP and we hold Cash.
    *   **SELL**: If Logic says DOWN (or Mixed) and we hold Position.

---

## 4. Validation System (The Truth Engine)

We don't just "fire and forget". We verify every single prediction.

### Log Validator (`src/utils/log_validator.py`)
This background service runs continuously.
1.  **Tracks**: Every prediction ("I think price will go UP").
2.  **Verifies**: After the timeframe closes (e.g., 15 mins later), it looks at the *actual* price action.
3.  **Grades**:
    *   **CORRECT**: Direction matched (e.g., Predicted UP, Price went UP).
    *   **WRONG**: Direction failed.
4.  **Updates**: It retroactively updates the `logs/predictions/predictions_*.csv` files, replacing `PENDING` with the actual outcome.

---

## Summary Flow
1.  **Data**: Raw Market Data + Macro Indicators -> `src/data` (Unified Pipeline)
2.  **Brain**: Transformer + GRU -> Raw Logits (x3 Timeframes)
3.  **Safety**: Monte Carlo -> Uncertainty Gates -> Validated Signal (Sniper Mode Applied)
4.  **Consensus**: Do 15m and 1h agree? (Prime Only)
5.  **Action**: Check Trailing Stop -> Execute Trade (w/ Fees) -> Update Portfolio
6.  **Verify**: LogValidator grades the prediction after maturity.
7.  **Review**: Dashboard "Diagnostics" page shows raw probabilities and validation stats.
