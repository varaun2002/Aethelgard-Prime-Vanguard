# Aethelgard: Prime & Vanguard (v3.3 Optimized)

Aethelgard is an advanced cryptocurrency trading bot ecosystem powered by Hybrid Transformer-GRU models. It features two distinct trading strategies/configurations: **Prime** and **Vanguard**.

## 🚀 Recent Updates (v3.3)
*   **Optimized Vanguard Threshold**: Tuned to **0.24** to filter market noise while capturing verified trends.
*   **Automated Validation**: Integrated `HealthMonitor` now validates predictions against real market data every minute.
*   **Stability**: Fixed critical bugs including "Probability Swap" and dashboard state crashes.

## 🔮 Aethelgard Prime
**Dashboard**: `dashboard/dashboard.py`
**Strategy**: Standard / Balanced
**Timeframe**: Configurable (default 15m/1h)

Prime is the flagship configuration, designed for balanced trading with robust risk management. It utilizes the full power of the V3 model with a focus on high-confidence setups and confirmed trends.

## 🛡️ Aethelgard Vanguard
**Dashboard**: `dashboard/dashboard_15m_aggressive.py`
**Strategy**: Aggressive / Scalping
**Timeframe**: 15m

Vanguard is the aggressive counterpart, tuned for the 15-minute timeframe. It features:
- **Optimized Confidence**: 0.24 threshold (up from 0.12) to balance frequency and accuracy.
- **Trend Forcing**: A mechanism to override "FLAT" signals when the market regime is clearly trending.
- **Automated Health Monitoring**: Self-healing logs and real-time accuracy tracking.

---

## Technical Architecture (V3.3)

Both bots share the core V3.0 architecture:
*   **Model**: Hybrid Transformer-GRU with 70 input features.
*   **External Data**: Integrates 20+ external factors including:
    *   Binance Derivatives (Funding Rates, Open Interest)
    *   Market Sentiment (Fear & Greed)
    *   Macro Indicators (SPX, VIX)
    *   On-chain metrics
*   **Dual Scaling**: Separate preprocessing for technical and external features.

## Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Vanguard (Ready for Live Testing)**:
    ```bash
    # Start Dashboard
    streamlit run dashboard/dashboard_15m_aggressive.py
    
    # Start Health & Validation Monitor (in separate terminal)
    python3 scripts/health_monitor.py
    ```

## Project Structure

*   `src/models`: Model definitions and training scripts (Transformer-GRU).
*   `src/inference`: Prediction logic (`PredictorV3`) and paper trading engine (`PaperTrader`).
*   `dashboard`: Streamlit dashboards for Prime and Vanguard.
*   `data`: Historical data and active portfolio state.
*   `logs`: Transaction logs and prediction history.
*   `scripts`: Utilities for monitoring, validation, and data management.
