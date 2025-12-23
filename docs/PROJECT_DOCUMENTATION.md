# Crypto Predictor v3.0 - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Data Sources](#data-sources)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture](#model-architecture)
6. [Prediction Pipeline](#prediction-pipeline)
7. [Quality Control System](#quality-control-system)
8. [Paper Trading](#paper-trading)
9. [Active Learning](#active-learning)
10. [Dashboard](#dashboard)
11. [File Structure](#file-structure)
12. [Configuration](#configuration)
13. [How to Use](#how-to-use)

---

## Project Overview

**Crypto Predictor v3.0** is an advanced cryptocurrency price prediction system that uses deep learning to forecast Ethereum (ETH/USDT) price movements across multiple timeframes.

### Key Features
- **Multi-timeframe predictions**: 15-minute, 1-hour, and 4-hour intervals
- **Hybrid neural network**: Combines Transformer and GRU architectures
- **70-feature system**: 50 technical + 20 external market features
- **Uncertainty quantification**: Monte Carlo Dropout for confidence estimates
- **Three-gate quality control**: Filters low-quality predictions
- **Paper trading simulation**: Tests strategies without real money
- **Active learning**: Automatically retrains when accuracy degrades
- **Real-time dashboard**: Streamlit web interface for monitoring

### Performance Characteristics
- **Prediction latency**: ~11-12 seconds per timeframe
- **Average confidence**: 45-48% (marginal predictions in choppy markets)
- **Trade execution rate**: ~1-2% (very conservative)
- **Data requirements**: 200+ historical candles for proper feature engineering

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│              Streamlit Dashboard (dashboard.py)             │
│         Real-time predictions, charts, trading stats        │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION ENGINE                         │
│              PredictorV3 (predict_v3.py)                    │
│   - Monte Carlo Dropout inference                           │
│   - Three-gate quality control                              │
│   - Regime-aware thresholds                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                   NEURAL NETWORK                            │
│           CryptoModelV3 (model.py)                          │
│   - Hybrid Transformer-GRU architecture                     │
│   - Multi-task learning (regime, direction, return)         │
│   - Input: [batch, 60, 70] sequences                        │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                        │
│     FeatureEngineerV2 + ExternalDataFetcher                 │
│   - Technical indicators (50 features)                      │
│   - External market data (20 features)                      │
│   - Normalization and scaling                               │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                             │
│   - yfinance: ETH/USD price data (OHLCV)                    │
│   - CCXT: BTC/USDT data, funding rates                      │
│   - Alternative.me: Fear & Greed Index                      │
│   - yfinance: S&P 500, VIX (traditional markets)            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Data Collection
   └─> yfinance (ETH) → OHLCV candles
   └─> CCXT (BTC) → Correlation data
   └─> Alternative.me → Sentiment
   └─> yfinance (SPY, VIX) → Traditional markets

2. Feature Engineering
   └─> Technical: RSI, MACD, ATR, Bollinger, Volume, etc. (50)
   └─> External: Time, BTC, Sentiment, Stocks, Funding (20)
   └─> Normalization: Z-score over 180-period windows

3. Model Inference
   └─> Sequence: Last 60 timesteps × 70 features
   └─> Monte Carlo: 10 forward passes with dropout
   └─> Outputs: Direction probs, Return prediction, Regime

4. Quality Control
   └─> Gate 1: Check epistemic uncertainty
   └─> Gate 2: Require minimum 45% confidence
   └─> Gate 3: Force direction in trending markets

5. Actions
   └─> Log prediction to CSV
   └─> Update paper trading portfolio
   └─> Display on dashboard
   └─> Schedule accuracy validation
```

---

## Data Sources

### 1. Primary Price Data (yfinance)

**Source**: Yahoo Finance via `yfinance` library
**Symbol**: ETH-USD (Ethereum to US Dollar)
**Purpose**: Main prediction target

**Data Periods by Timeframe**:
- **15-minute**: 60 days → ~5,600 candles
- **1-hour**: 730 days (2 years) → ~17,000 candles
- **4-hour**: 180 days → ~4,000 1h candles → ~1,000 4h candles (resampled)

**Fields Retrieved**:
```python
{
    'timestamp': datetime,  # Candle open time
    'open': float,         # Opening price
    'high': float,         # Highest price
    'low': float,          # Lowest price
    'close': float,        # Closing price
    'volume': float        # Trading volume
}
```

**Why so much data?**
- Feature engineering requires 180-period rolling windows for normalization
- Indicator calculations need warm-up periods (e.g., 60-period ATR)
- Model needs 60 timesteps of historical context
- Total minimum: 180 + 60 = 240 candles

### 2. Bitcoin Data (CCXT)

**Source**: Binance via `ccxt` library
**Symbol**: BTC/USDT
**Purpose**: Market correlation and divergence analysis

**Fields Used**:
```python
btc_data = exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=100)
# Returns: [[timestamp, open, high, low, close, volume], ...]

Derived Features:
- btc_return_1h: % change over last 5 candles
- btc_return_4h: % change over last 17 candles
- btc_volatility: Std dev of returns (20 periods)
- eth_btc_correlation: Correlation coefficient (20 periods)
- btc_eth_divergence: Difference in returns
```

**Cache Duration**: 5 minutes (to reduce API calls)

### 3. Funding Rate Data (CCXT)

**Source**: Binance Perpetual Futures via `ccxt`
**Symbol**: ETH/USDT:USDT (perpetual contract)
**Purpose**: Derivatives market sentiment

**Field Retrieved**:
```python
funding = exchange.fetch_funding_rate('ETH/USDT:USDT')
# Returns: {'fundingRate': float}  # Typically -0.01% to +0.01%

Interpretation:
- Positive: Longs pay shorts (bullish sentiment)
- Negative: Shorts pay longs (bearish sentiment)
- Extreme values: Overleveraged market
```

**Cache Duration**: 1 hour (funding rates update every 8 hours)

### 4. Fear & Greed Index (Alternative.me API)

**Source**: https://api.alternative.me/fng/
**Purpose**: Market-wide sentiment indicator

**Data Format**:
```python
response = requests.get('https://api.alternative.me/fng/')
value = int(response.json()['data'][0]['value'])  # 0-100

Ranges:
- 0-25: Extreme Fear (potential buy signal)
- 25-45: Fear
- 45-55: Neutral
- 55-75: Greed
- 75-100: Extreme Greed (potential sell signal)
```

**Cache Duration**: 1 hour (updates daily)
**Fallback**: 50 (neutral) if API fails

### 5. Traditional Markets (yfinance)

**Purpose**: Macro market context and risk indicators

#### S&P 500 (SPY)
```python
spy_data = yf.download('SPY', period='5d', interval='1d')
spy_return = (spy_close[-1] - spy_close[-2]) / spy_close[-2]

Interpretation:
- Positive: Risk-on environment (may boost crypto)
- Negative: Risk-off (may suppress crypto)
```

#### VIX (Volatility Index)
```python
vix_data = yf.download('^VIX', period='5d', interval='1d')
vix_level = vix_data['Close'].iloc[-1]

Ranges:
- <15: Low volatility (complacent market)
- 15-20: Normal volatility
- 20-30: Elevated volatility
- >30: High volatility (fear in stocks)

Risk-On Indicator:
risk_on = 1 if (spy_return > 0 and vix_level < 20) else 0
```

**Cache Duration**: 1 hour (daily data, infrequent updates needed)

---

## Feature Engineering

### Technical Features (50 features)

Generated from OHLCV data using `pandas_ta` library.

#### 1. Volatility Indicators (8 features)

```python
# Average True Range - measures volatility
atr = ta.atr(high, low, close, length=14)           # Standard ATR
atr_1 = ta.atr(high, low, close, length=1)          # Instant volatility
atr_3 = ta.atr(high, low, close, length=3)          # Short-term volatility
atr_60 = ta.atr(high, low, close, length=60)        # Long-term volatility
atr_rank = atr.rolling(100).rank(pct=True)          # Percentile rank

# Bollinger Bands
bb = ta.bbands(close, length=20, std=2)
bb_width = (bb.upper - bb.lower) / bb.middle        # Normalized width

# Historical Volatility
log_ret = np.log(close / close.shift(1))            # Log returns
hist_vol = log_ret.rolling(20).std()                # Rolling std dev
```

**Purpose**: Identify volatile vs. calm periods. High volatility = higher uncertainty.

#### 2. Momentum & Trend Indicators (15 features)

```python
# RSI - Relative Strength Index (14-period)
rsi = ta.rsi(close, length=14)                      # 0-100 scale
rsi_slope = rsi.diff(3)                             # Rate of change
rsi_divergence = (price_mom - rsi_mom/100).rolling(3).mean()

# MACD - Moving Average Convergence Divergence
macd = ta.macd(close)
macd_line = macd['MACD_12_26_9']                    # Main line
macd_signal = macd['MACDs_12_26_9']                 # Signal line
macd_hist = macd['MACDh_12_26_9']                   # Histogram

# ADX - Average Directional Index
adx = ta.adx(high, low, close)
adx_value = adx['ADX_14']                           # Trend strength (0-100)
dmp = adx['DMP_14']                                 # Plus directional
dmn = adx['DMN_14']                                 # Minus directional

# Choppiness Index
chop = ta.chop(high, low, close)                    # 0-100 (higher=choppier)

# Aroon Indicator
aroon = ta.aroon(high, low)
aroon_up = aroon['AROONU_14']                       # Uptrend strength
aroon_down = aroon['AROOND_14']                     # Downtrend strength
aroon_osc = aroon['AROONOSC_14']                    # Oscillator

# Stochastic Oscillator
stoch = ta.stoch(high, low, close)
stoch_k = stoch['STOCHk_14_3_3']                    # %K line
stoch_d = stoch['STOCHd_14_3_3']                    # %D line (signal)
```

**Purpose**: Measure trend strength, momentum, and potential reversals.

#### 3. Candle Features (5 features)

```python
body_size = abs(close - open)                       # Candle body
wick_upper = high - max(open, close)                # Upper wick
wick_lower = min(open, close) - low                 # Lower wick
body_perc = body_size / (high - low + 1e-9)         # Body % of range
hl_ratio = (high - low) / (close + 1e-9)            # Range ratio
```

**Purpose**: Capture price action microstructure and candlestick patterns.

#### 4. Volume Indicators (6 features)

```python
# Volume Z-Score
vol_mean = volume.rolling(20).mean()
vol_std = volume.rolling(20).std()
vol_z = (volume - vol_mean) / (vol_std + 1e-9)      # Standardized

# Relative Volume
rvol = volume / (vol_mean + 1e-9)                   # Multiple of avg

# Volume Delta
vol_delta = volume.pct_change()                     # % change

# Cumulative Volume Delta
candle_direction = np.where(close > open, 1, -1)    # Up=1, Down=-1
signed_volume = candle_direction * volume
cvd = signed_volume.rolling(20).sum()               # Net buying pressure
cvd_norm = cvd / (volume.rolling(20).sum() + 1e-9)
```

**Purpose**: Measure buying/selling pressure and volume anomalies.

#### 5. Market Structure (6 features)

```python
# Rolling Statistics
roll_mean_20 = close.rolling(20).mean()             # SMA
roll_std_20 = close.rolling(20).std()               # Volatility

# Trend Slope
trend_slope = (close - close.shift(5)) / 5          # Linear slope

# Support/Resistance Distance
high_52 = high.rolling(52).max()
low_52 = low.rolling(52).min()
dist_52_high = (high_52 - close) / (close + 1e-9)   # % to resistance
dist_52_low = (close - low_52) / (close + 1e-9)     # % to support
```

**Purpose**: Identify market structure, trends, and key levels.

#### 6. Normalization (Applied to most features)

```python
# Z-score normalization over 180-period window
for feature in features_to_normalize:
    mean = df[feature].rolling(180).mean()
    std = df[feature].rolling(180).std()
    df[f'{feature}_norm'] = (df[feature] - mean) / (std + 1e-9)
```

**Why 180 periods?**
- Provides stable statistics
- Adapts to regime changes
- Long enough to avoid noise, short enough to be responsive

### External Features (20 features)

#### 1. Time Features (8 features)

```python
from datetime import datetime

hour = timestamp.hour                               # 0-23
day_of_week = timestamp.weekday()                   # 0-6 (Mon-Sun)

# Cyclical encoding (preserves circular nature)
hour_sin = np.sin(2 * np.pi * hour / 24)           # -1 to 1
hour_cos = np.cos(2 * np.pi * hour / 24)
day_sin = np.sin(2 * np.pi * day_of_week / 7)
day_cos = np.cos(2 * np.pi * day_of_week / 7)

# Trading sessions
is_us_trading = 1 if (9 <= hour < 16) else 0       # NYSE hours (EST)
is_asia_trading = 1 if (20 <= hour or hour < 4) else 0  # Asia hours (EST)
```

**Purpose**: Capture time-of-day and day-of-week patterns in crypto markets.

#### 2. BTC Market Context (5 features)

```python
# From Bitcoin data (see Data Sources section)
btc_return_1h                                       # Short-term BTC momentum
btc_return_4h                                       # Medium-term BTC momentum
btc_volatility                                      # BTC price volatility
eth_btc_correlation                                 # How much ETH follows BTC
btc_eth_divergence                                  # When ETH/BTC decouple
```

**Purpose**: ETH often follows BTC; correlation and divergence signal regime changes.

#### 3. Derivatives Market (1 feature)

```python
funding_rate                                        # Perpetual futures rate
```

**Purpose**: High positive funding = overleveraged longs (bearish). Negative = shorts squeezed.

#### 4. Sentiment (1 feature)

```python
fear_greed_normalized = fear_greed_value / 100      # 0.0 to 1.0
```

**Purpose**: Extreme fear/greed often precedes reversals (contrarian indicator).

#### 5. Traditional Markets (3 features)

```python
spy_return                                          # S&P 500 daily return
vix_level                                           # VIX value (volatility)
risk_on_indicator                                   # Combined signal
```

**Purpose**: Crypto increasingly correlates with stocks; risk-on/off affects all assets.

#### 6. Interaction Features (2 features)

```python
# Combining multiple signals
fear_funding = fear_greed_normalized * funding_rate  # Sentiment × leverage
btc_vol_vix = btc_volatility * vix_level            # Crypto × stock volatility
```

**Purpose**: Non-linear relationships captured through feature interactions.

---

## Model Architecture

### CryptoModelV3 (Hybrid Transformer-GRU)

**File**: `src/models/model.py`

#### Network Structure

```python
Input Shape: [batch_size, sequence_length, input_dim]
            [1, 60, 70]  # 60 timesteps, 70 features

┌─────────────────────────────────────────────────────────┐
│                   INPUT LAYER (70)                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              POSITIONAL ENCODING                        │
│  Adds learnable position embeddings to sequences        │
│  Shape: [60, 70] → [60, 70]                            │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌─────────────────┐           ┌─────────────────┐
│ TEMPORAL BRANCH │           │ CONTEXT BRANCH  │
│   (GRU Path)    │           │ (Transformer)   │
│                 │           │                 │
│ GRU Layer 1     │           │ Multi-Head      │
│   128 units     │           │ Attention       │
│ Dropout 0.2     │           │   4 heads       │
│                 │           │   128 dim       │
│ GRU Layer 2     │           │ Dropout 0.2     │
│   128 units     │           │                 │
│ Dropout 0.2     │           │ Feed Forward    │
│                 │           │   512 hidden    │
│ Output: [128]   │           │   128 output    │
│                 │           │ Dropout 0.2     │
│                 │           │                 │
│                 │           │ Output: [60,128]│
└─────────────────┘           └─────────────────┘
        ↓                               ↓
        └───────────────┬───────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│            CROSS-ATTENTION FUSION                       │
│  Query: GRU output [128]                                │
│  Key/Value: Transformer output [60, 128]                │
│  Output: Fused representation [128]                     │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
┌─────────────┐ ┌──────────────┐ ┌────────────┐
│ REGIME HEAD │ │DIRECTION HEAD│ │RETURN HEAD │
│ FC(128→64)  │ │ FC(128→64)   │ │FC(128→64)  │
│ ReLU        │ │ ReLU         │ │ReLU        │
│ Dropout 0.2 │ │ Dropout 0.2  │ │Dropout 0.2 │
│ FC(64→4)    │ │ FC(64→3)     │ │FC(64→1)    │
│ Softmax     │ │ Softmax      │ │Linear      │
│             │ │              │ │            │
│ 4 classes   │ │ 3 classes    │ │1 value     │
└─────────────┘ └──────────────┘ └────────────┘
```

#### Model Parameters

```python
input_dim = 70           # Feature count
hidden_dim = 128         # Hidden layer size
num_layers = 2           # GRU depth
num_heads = 4            # Attention heads
dropout = 0.2            # Regularization
```

#### Output Heads

**1. Regime Head (4 classes)**
```python
Classes:
0: Range Quiet    - Low volatility, no trend
1: Range Noisy    - High volatility, no trend
2: Trending UP    - Clear uptrend
3: Trending DOWN  - Clear downtrend

Output: [batch, 4] probability distribution
```

**2. Direction Head (3 classes)**
```python
Classes:
0: FLAT  - No significant move
1: DOWN  - Price decrease
2: UP    - Price increase

Output: [batch, 3] probability distribution
```

**3. Return Head (Regression)**
```python
Predicts: Percentage price change
Output: [batch, 1] continuous value (e.g., 0.02 = +2%)
```

#### Monte Carlo Dropout

```python
def mc_dropout_predict(self, x, n_samples=10):
    """
    Run multiple forward passes with dropout enabled
    to quantify model uncertainty.
    """
    self.train()  # Enable dropout

    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = self.forward(x)
            predictions.append(output)

    # Aggregate
    mean_prediction = torch.stack(predictions).mean(dim=0)
    std_prediction = torch.stack(predictions).std(dim=0)

    return {
        'direction_probs': mean_prediction['direction'],
        'direction_uncertainty': std_prediction['direction'].mean(),
        'predicted_return': mean_prediction['return'],
        # ...
    }
```

**Purpose**: Uncertainty estimates tell us when the model is confused vs. confident.

---

## Prediction Pipeline

### Step-by-Step Process

**File**: `src/inference/predict_v3.py`

#### Step 1: Data Preparation

```python
# Fetch recent OHLCV data
df = fetch_live_data(symbol="ETH/USDT", timeframe="15m", limit=300)

# Generate technical features
df = feature_engineer.add_features(df)  # 50 technical features

# Extract sequence
tech_seq = df[feature_columns].values[-60:]  # Last 60 timesteps
tech_seq = tech_scaler.transform(tech_seq)   # Normalize
```

#### Step 2: External Feature Integration

```python
# Get timestamps for sequence
timestamps = df['timestamp'].values[-60:]

# Fetch current external context
current_ext = external_fetcher.get_all_features(df, timestamp=None)
macro_features = current_ext[8:]  # BTC, sentiment, stocks (12 features)

# Build time-varying external sequence
ext_seq = []
for ts in timestamps:
    time_features = get_time_features(ts)  # 8 features
    combined = np.concatenate([time_features, macro_features])  # 20 features
    ext_seq.append(combined)

ext_seq = ext_scaler.transform(np.array(ext_seq))
```

#### Step 3: Feature Combination

```python
# Combine technical + external
combined_seq = np.concatenate([tech_seq, ext_seq], axis=1)
# Shape: [60, 70]

# Convert to tensor
X = torch.FloatTensor(combined_seq).unsqueeze(0)  # [1, 60, 70]
```

#### Step 4: Monte Carlo Inference

```python
mc_results = model.mc_dropout_predict(X, n_samples=10)

# Extract results
dir_probs = mc_results['direction_probs']  # [FLAT, DOWN, UP]
uncertainty = mc_results['direction_uncertainty']
predicted_return = mc_results['predicted_return']
regime_logits = mc_results['regime_logits']
```

#### Step 5: Three-Gate Quality Control

**Gate 1: Epistemic Uncertainty Check**
```python
# Get regime-specific threshold
thresholds = {
    "Trending UP": 0.30,
    "Trending DOWN": 0.30,
    "Range Quiet": 0.22,
    "Range Noisy": 0.18
}
threshold = thresholds[regime]

if uncertainty > threshold:
    predicted_direction = "FLAT"
    caution_flag = True
    # Model is too uncertain - reject prediction
```

**Gate 2: Minimum Confidence Check**
```python
MIN_CONFIDENCE = 0.45  # 45% minimum

pred_class = np.argmax(dir_probs)
confidence = dir_probs[pred_class]

if confidence < MIN_CONFIDENCE:
    predicted_direction = "FLAT"
    # Signal too weak - stay out
```

**Gate 3: Trend Forcing**
```python
# In trending markets, override FLAT if signal is decent
if regime in ["Trending UP", "Trending DOWN"]:
    if predicted_direction == "FLAT":
        # Check if UP or DOWN has reasonable backing
        best_dir = np.argmax(dir_probs[1:]) + 1  # Skip FLAT
        best_conf = dir_probs[best_dir]

        if best_conf > 0.40:  # Lower threshold for trends
            predicted_direction = direction_map[best_dir]
            confidence = best_conf
```

**Additional Safety: Range Noisy Override**
```python
if regime == "Range Noisy" and confidence < 0.80:
    predicted_direction = "FLAT"
    # Extremely conservative in choppy markets
```

#### Step 6: Output Assembly

```python
return {
    'predicted_price': current_price * (1 + predicted_return),
    'predicted_return': predicted_return,
    'predicted_direction': predicted_direction,  # UP/DOWN/FLAT
    'confidence': float(confidence),             # 0.0 to 1.0
    'uncertainty': float(uncertainty),           # 0.0 to 1.0
    'regime': regime_str,                        # Market regime
    'probabilities': dir_probs.tolist(),         # [FLAT, DOWN, UP]
    'caution': caution_flag                      # High uncertainty?
}
```

---

## Quality Control System

### Three-Gate Architecture

The system uses a cascading filter to prevent bad predictions:

```
Raw Model Output
      ↓
┌──────────────────────┐
│   GATE 1: EPISTEMIC  │  ← Is the model confused?
│  UNCERTAINTY CHECK   │
│  (Monte Carlo Std)   │
└──────────────────────┘
      ↓ [PASS]
┌──────────────────────┐
│   GATE 2: MINIMUM    │  ← Is the signal strong enough?
│  CONFIDENCE CHECK    │
│   (≥45% required)    │
└──────────────────────┘
      ↓ [PASS]
┌──────────────────────┐
│  GATE 3: TREND       │  ← Should we force a direction?
│  FORCING OVERRIDE    │
│ (Context awareness)  │
└──────────────────────┘
      ↓ [PASS]
Final Prediction
```

### Gate Details

#### Gate 1: Epistemic Uncertainty

**Purpose**: Detect when the model is confused (high disagreement between MC samples)

**Thresholds by Regime**:
- Trending markets: 0.30 (30%) - more tolerant
- Range Quiet: 0.22 (22%)
- Range Noisy: 0.18 (18%) - most conservative

**Logic**:
```python
if uncertainty > dynamic_threshold:
    return "FLAT"  # Model doesn't know - stay out
```

**Example**:
```
Uncertainty: 0.25
Regime: Range Noisy (threshold: 0.18)
Result: BLOCKED (0.25 > 0.18) → FLAT
```

#### Gate 2: Minimum Confidence

**Purpose**: Require a minimum probability for any prediction

**Threshold**: 45% (0.45)

**Logic**:
```python
if confidence < 0.45:
    return "FLAT"  # Too close to random (33.3%)
```

**Example**:
```
Probabilities: [FLAT: 0.40, DOWN: 0.44, UP: 0.16]
Best: DOWN with 44%
Result: BLOCKED (0.44 < 0.45) → FLAT
```

**Why 45%?**
- Random guess: 33.3% (1/3)
- 45% is significantly above random
- Prevents marginal predictions
- Currently filters ~21% of predictions

#### Gate 3: Trend Forcing

**Purpose**: Override FLAT in trending markets when directional signal is decent

**Threshold**: 40% confidence

**Logic**:
```python
if regime in ["Trending UP", "Trending DOWN"]:
    if predicted_direction == "FLAT":
        best_directional = max(prob_UP, prob_DOWN)
        if best_directional > 0.40:
            return best_directional_class
```

**Example**:
```
Regime: Trending UP
Probabilities: [FLAT: 0.50, DOWN: 0.12, UP: 0.38]
Gate 2 result: FLAT (50% confidence)
Gate 3 check: UP has 38% (< 40% threshold)
Result: Stay FLAT (signal not strong enough)
```

### Performance Metrics

Based on current logs:

```
Total Predictions: 155
Gate 2 Interventions: 33 (21.3%)
Average Confidence: 45.9%
Prediction Distribution:
  - FLAT: 34.4%
  - UP: 48.4%
  - DOWN: 17.2%
```

---

## Paper Trading

### PaperTrader System

**File**: `src/inference/paper_trader.py`

#### Configuration

```python
INITIAL_BALANCE = 10000.0       # Start with $10k USDT
MAX_POSITION_PCT = 0.95         # Use up to 95% of balance
CONFIDENCE_THRESHOLD = 0.60     # Only trade on 60%+ confidence
```

#### Trading Logic

```python
def update(predicted_direction, confidence, price, timestamp):
    """
    Execute paper trades based on predictions.
    """
    # Only trade if confident
    if confidence < CONFIDENCE_THRESHOLD:
        return None

    # BUY on UP signal
    if predicted_direction == "UP" and position_eth == 0:
        size = (balance_usdt * MAX_POSITION_PCT) / price
        balance_usdt -= size * price
        position_eth = size
        avg_entry_price = price
        log_trade("BUY", price, size, confidence)
        return "BUY"

    # SELL on DOWN/FLAT signal
    elif predicted_direction in ["DOWN", "FLAT"] and position_eth > 0:
        balance_usdt += position_eth * price
        log_trade("SELL", price, position_eth, confidence)
        position_eth = 0
        avg_entry_price = 0
        return "SELL"

    return None
```

#### State Tracking

```python
state = {
    "balance_usdt": 10027.14,
    "position_eth": 0.0,
    "avg_entry_price": 0.0,
    "equity_history": [
        {
            "timestamp": 1766008541,
            "date": "2025-12-17 16:55:41",
            "equity": 10000.0,
            "price": 2815.68
        },
        ...
    ],
    "trades": [
        {
            "timestamp": 1766008942,
            "date": "2025-12-17 17:02:22",
            "action": "BUY",
            "price": 3500.0,
            "size": 2.714,
            "confidence": 0.65,
            "new_balance": 500.0
        },
        ...
    ]
}
```

#### Performance Calculation

```python
def get_stats():
    """Calculate current portfolio statistics."""
    if position_eth > 0:
        # Long position
        current_equity = position_eth * current_price
    else:
        # All cash
        current_equity = balance_usdt

    pnl = current_equity - INITIAL_BALANCE
    pnl_percent = (pnl / INITIAL_BALANCE) * 100

    return {
        'current_equity': current_equity,
        'pnl_percent': pnl_percent,
        'open_position': position_eth
    }
```

#### Current Results

```
Starting Balance: $10,000.00
Current Equity:   $10,027.14
Profit/Loss:      +$27.14 (+0.27%)
Total Trades:     2
Trade Rate:       1.3% (2/155 predictions)

Reason for Low Activity:
- Confidence threshold (60%) filters most predictions
- Average confidence is only 45.9%
- System is correctly conservative in uncertain markets
```

---

## Active Learning

### FeedbackManager System

**File**: `src/inference/feedback_manager.py`

#### Workflow

```
1. Prediction Made
   └─> Log to CSV with timestamp

2. Wait for Timeframe to Elapse
   └─> 15m, 1h, or 4h depending on prediction

3. Fetch Actual Price
   └─> Compare to predicted direction

4. Evaluate Accuracy
   └─> GOOD: Direction correct
   └─> BAD: Direction wrong

5. Check Error Rate
   └─> If >10 BAD out of last 50 predictions:
       Trigger retraining
```

#### Prediction Logging

```python
def log_prediction(timeframe, prediction, current_price):
    """
    Log prediction to CSV for later validation.
    """
    row = {
        'timestamp': int(time.time()),
        'timeframe': timeframe,
        'predicted_direction': prediction['predicted_direction'],
        'confidence': prediction['confidence'],
        'predicted_price': prediction['predicted_price'],
        'actual_price': 0.0,            # Filled later
        'outcome': 'PENDING',            # GOOD/BAD after validation
        'error': 0.0                     # Price error %
    }

    # Append to CSV
    df = pd.read_csv('prediction_log.csv')
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv('prediction_log.csv', index=False)
```

#### Accuracy Validation

```python
def check_accuracy(current_price, current_time):
    """
    Validate old predictions against actual outcomes.
    """
    df = pd.read_csv('prediction_log.csv')

    # Find predictions ready for validation
    for idx, row in df.iterrows():
        if row['outcome'] != 'PENDING':
            continue

        # Check if enough time has passed
        timeframe_seconds = {
            '15m': 15 * 60,
            '1h': 60 * 60,
            '4h': 4 * 60 * 60
        }

        elapsed = current_time - row['timestamp']
        required = timeframe_seconds[row['timeframe']]

        if elapsed >= required:
            # Fetch actual price at that time
            actual_price = current_price  # Simplified

            # Determine outcome
            predicted_dir = row['predicted_direction']
            actual_dir = get_actual_direction(
                row['predicted_price'],
                actual_price
            )

            outcome = 'GOOD' if predicted_dir == actual_dir else 'BAD'
            error = abs(actual_price - row['predicted_price']) / actual_price

            # Update row
            df.at[idx, 'actual_price'] = actual_price
            df.at[idx, 'outcome'] = outcome
            df.at[idx, 'error'] = error

    df.to_csv('prediction_log.csv', index=False)

    # Check if retraining needed
    recent = df.tail(50)
    bad_count = (recent['outcome'] == 'BAD').sum()

    if bad_count > 10:
        trigger_retraining()
```

#### Retraining Trigger

```python
def trigger_retraining():
    """
    Initiate model retraining when accuracy degrades.
    """
    with open('retrain_events.log', 'a') as f:
        f.write(f"{datetime.now()}: Accuracy degraded. Retraining triggered.\n")

    # Launch training script
    os.system('python src/models/train_v3.py --epochs 10')
```

---

## Dashboard

### Streamlit Web Interface

**File**: `dashboard.py`

#### Layout

```
┌────────────────────────────────────────────────────────┐
│  🔮 Crypto Predictor v3.1 System                       │
├────────────────────────────────────────────────────────┤
│  Sidebar:                                              │
│  [x] Auto-Refresh (30s)                                │
│  Symbol: [ETH/USDT        ]                            │
└────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Live Price      Sim Equity      Active Position        │
│  $2,815.68       $10,027.14      0.0000 ETH             │
│                  +0.27%                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────┬─────────────┬─────────────┐
│   15m Model │   1h Model  │   4h Model  │
├─────────────┼─────────────┼─────────────┤
│     UP      │    FLAT     │    DOWN     │
│   (green)   │   (gray)    │   (red)     │
│             │             │             │
│ Confidence  │ Confidence  │ Confidence  │
│ ████ 48%    │ ████ 45%    │ ████ 47%    │
│             │             │             │
│ Uncertainty │ Uncertainty │ Uncertainty │
│ ██ 0.36%    │ ██ 0.44%    │ ██ 0.39%    │
│             │             │             │
│ Regime:     │ Regime:     │ Regime:     │
│ Range Noisy │ Range Quiet │ Range Noisy │
│             │             │             │
│ Return:     │ Return:     │ Return:     │
│ -2.45%      │ -2.60%      │ -2.52%      │
└─────────────┴─────────────┴─────────────┘

┌─────────────────────────────────────────────────────────┐
│  Paper Trading Simulation                               │
├─────────────────────────────────────────────────────────┤
│  [Equity Curve Chart]                                   │
│                                                          │
│  Recent Trades:                                          │
│  Date         Action  Price    Size   Balance            │
│  17:02:22     BUY     $3500    2.71   $500               │
│  17:02:22     SELL    $3510    2.71   $10027             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  System Health & Active Learning                        │
├─────────────────────────────────────────────────────────┤
│  ✅ System Stable (No retrain triggers)                 │
│  Last Update: 17:09:21                                   │
└─────────────────────────────────────────────────────────┘
```

#### Auto-Refresh Logic

```python
while True:
    with main_display.container():
        # Fetch data for all timeframes
        for timeframe in ["15m", "1h", "4h"]:
            df = fetch_live_data("ETH/USDT", timeframe)
            prediction = predictor.predict(df)

            # Update trading simulation
            if timeframe == "15m":
                action = trader.update(
                    prediction['predicted_direction'],
                    prediction['confidence'],
                    current_price,
                    timestamp
                )

            # Log prediction
            feedback.log_prediction(timeframe, prediction, price)

        # Render UI components
        render_metrics()
        render_predictions()
        render_trading_stats()
        render_system_health()

    if not auto_refresh:
        break

    time.sleep(30)  # Refresh every 30 seconds
```

#### Color Coding

```python
direction_colors = {
    "UP": "green",
    "DOWN": "red",
    "FLAT": "gray"
}

st.markdown(f"## :{color}[{direction}]")
```

---

## File Structure

```
crypto-predictor-v3/
│
├── config/
│   └── config_v3.yaml              # Model and training configuration
│
├── models_v3/                      # Trained models and scalers
│   ├── model_v3_0_15m.pth          # 15-minute model (4.9 MB)
│   ├── model_v3_0_1h.pth           # 1-hour model (4.9 MB)
│   ├── model_v3_0_4h.pth           # 4-hour model (4.9 MB)
│   ├── technical_scaler_15m.pkl    # Technical feature scaler
│   ├── technical_scaler_1h.pkl
│   ├── technical_scaler_4h.pkl
│   ├── external_scaler_15m.pkl     # External feature scaler
│   ├── external_scaler_1h.pkl
│   └── external_scaler_4h.pkl
│
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Combines technical + external
│   │   ├── external_data.py        # Live external data fetcher
│   │   └── external_history.py     # Historical external data
│   │
│   ├── features/
│   │   └── feature_engineering.py  # Technical indicator calculation
│   │
│   ├── models/
│   │   ├── model.py                # CryptoModelV3 architecture
│   │   └── train_v3.py             # Training script
│   │
│   ├── inference/
│   │   ├── predict_v3.py           # PredictorV3 engine
│   │   ├── paper_trader.py         # Paper trading simulation
│   │   └── feedback_manager.py     # Active learning system
│   │
│   └── utils/
│       ├── logger.py               # Logging utilities
│       ├── config.py               # Config loader
│       └── scheduler.py            # Automated scheduling
│
├── logs/
│   └── app.log                     # Application logs (44 KB)
│
├── prediction_log.csv              # Prediction history (155 rows)
├── paper_portfolio.json            # Trading state and history
├── retrain_events.log              # Retraining triggers (if any)
│
├── dashboard.py                    # Streamlit web interface
├── validate_v3.py                  # System validation script
├── test_dashboard_fixes.py         # Dashboard testing script
├── test_comprehensive.py           # Comprehensive test suite
│
├── requirements_v3.txt             # Python dependencies
└── PROJECT_DOCUMENTATION.md        # This file
```

---

## Configuration

### config_v3.yaml

```yaml
version: "3.0"

# Feature configuration
features:
  technical_features: 50
  external_features: 20
  total_features: 70

# Model architecture
model:
  architecture: "HybridTransformerGRU"
  input_dim: 70
  hidden_dim: 128
  num_layers: 2
  num_heads: 4
  dropout: 0.2

# Training parameters
training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
  sequence_length: 60

  # Loss weights
  regime_weight: 2.0
  direction_weight: 1.0
  return_weight: 1.5
  volatility_weight: 0.5

# Prediction parameters
prediction:
  mc_samples: 10
  min_confidence: 0.45
  trend_force_threshold: 0.40

  # Uncertainty thresholds by regime
  uncertainty_thresholds:
    trending_up: 0.30
    trending_down: 0.30
    range_quiet: 0.22
    range_noisy: 0.18

# Trading parameters
trading:
  initial_balance: 10000.0
  max_position_pct: 0.95
  confidence_threshold: 0.60

# Active learning
active_learning:
  validation_window: 50
  error_threshold: 10
  retrain_epochs: 10

# Timeframes
timeframes:
  - "15m"
  - "1h"
  - "4h"
```

---

## How to Use

### Installation

```bash
# Clone repository
cd ~/Desktop/stock_ai_project/crypto-predictor-v3

# Install dependencies
pip install -r requirements_v3.txt

# Verify installation
python validate_v3.py
```

### Running the Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard.py

# Access in browser
# http://localhost:8501
```

### Making Predictions Programmatically

```python
from src.inference.predict_v3 import PredictorV3
import yfinance as yf

# Load model
predictor = PredictorV3("models_v3/model_v3_0_15m.pth")

# Fetch data
df = yf.download("ETH-USD", period="60d", interval="15m", progress=False)

# Make prediction
result = predictor.predict(df)

print(f"Direction: {result['predicted_direction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Uncertainty: {result['uncertainty']:.2%}")
print(f"Regime: {result['regime']}")
```

### Training a New Model

```bash
# Full training (100 epochs)
python src/models/train_v3.py --timeframe 15m --epochs 100

# Quick test (10 epochs)
python src/models/train_v3.py --timeframe 15m --epochs 10 --test

# Micro-update (fine-tuning)
python src/models/train_v3.py --timeframe 15m --epochs 5 --micro
```

### Testing the System

```bash
# Run comprehensive tests
python test_comprehensive.py

# Test dashboard fixes
python test_dashboard_fixes.py

# Validate entire system
python validate_v3.py
```

### Viewing Logs

```bash
# Application logs
tail -f logs/app.log

# Prediction history
tail -100 prediction_log.csv

# Paper trading state
cat paper_portfolio.json | python -m json.tool
```

### Monitoring Performance

```bash
# Check prediction distribution
cat prediction_log.csv | awk -F',' '{print $3}' | sort | uniq -c

# Calculate average confidence
cat prediction_log.csv | awk -F',' 'NR>1 {sum+=$4; count++} END {print sum/count}'

# View recent predictions
tail -20 prediction_log.csv | column -t -s','
```

---

## Data Summary Table

| **Category** | **Feature/Source** | **Count** | **Update Frequency** | **Cache** | **Purpose** |
|--------------|-------------------|-----------|---------------------|-----------|-------------|
| **OHLCV** | ETH-USD (yfinance) | 6 fields | Per prediction | No | Primary target |
| **Technical** | Volatility indicators | 8 | Calculated | No | Market volatility |
| **Technical** | Momentum/Trend | 15 | Calculated | No | Trend detection |
| **Technical** | Candle patterns | 5 | Calculated | No | Price action |
| **Technical** | Volume indicators | 6 | Calculated | No | Buying pressure |
| **Technical** | Market structure | 6 | Calculated | No | Support/resistance |
| **Technical** | Normalized features | 10+ | Calculated | No | Z-score normalization |
| **External** | Time features | 8 | Real-time | No | Temporal patterns |
| **External** | BTC context | 5 | 5 minutes | Yes | Correlation |
| **External** | Funding rate | 1 | 1 hour | Yes | Derivatives sentiment |
| **External** | Fear & Greed | 1 | 1 hour | Yes | Market sentiment |
| **External** | Traditional markets | 3 | 1 hour | Yes | Macro context |
| **External** | Interaction features | 2 | Calculated | No | Non-linear effects |
| **TOTAL** | **All Features** | **70** | **Variable** | **Mixed** | **Prediction input** |

---

## Key Metrics Summary

### Current System State (2025-12-17)

| Metric | Value | Notes |
|--------|-------|-------|
| **Dashboard Status** | 🟢 Running | http://localhost:8501 |
| **Models Loaded** | 3/3 | 15m, 1h, 4h |
| **Total Predictions** | 155 | Since deployment |
| **Average Confidence** | 45.9% | Right at threshold |
| **Gate 2 Interventions** | 33 (21%) | Filtering weak signals |
| **Paper Trading Equity** | $10,027.14 | +0.27% gain |
| **Trade Execution Rate** | 1.3% | Very conservative |
| **Prediction Distribution** | UP: 48%, FLAT: 34%, DOWN: 17% | Slight bullish bias |
| **Market Regime** | Range Noisy/Quiet | Low predictability |

---

## Glossary

**OHLCV**: Open, High, Low, Close, Volume - standard price bar data
**ATR**: Average True Range - volatility measure
**RSI**: Relative Strength Index - momentum oscillator (0-100)
**MACD**: Moving Average Convergence Divergence - trend indicator
**ADX**: Average Directional Index - trend strength (0-100)
**Monte Carlo Dropout**: Running inference multiple times to estimate uncertainty
**Epistemic Uncertainty**: Model uncertainty due to lack of knowledge
**Regime**: Market state classification (trending vs. ranging)
**Paper Trading**: Simulated trading without real money
**Active Learning**: System that improves by learning from its mistakes
**Gate System**: Multi-stage filter for prediction quality control
**Funding Rate**: Perpetual futures payment rate (indicates leverage)
**Fear & Greed Index**: Sentiment indicator (0=extreme fear, 100=extreme greed)

---

## Credits & License

**Crypto Predictor v3.0**
Developed as an educational and research project.

**Technologies Used**:
- PyTorch (Deep Learning)
- Streamlit (Dashboard)
- yfinance (Market Data)
- CCXT (Crypto Exchange API)
- pandas-ta (Technical Analysis)
- NumPy, Pandas (Data Processing)

**Disclaimer**: This system is for educational purposes only. Not financial advice. Cryptocurrency trading involves substantial risk of loss.

---

**Last Updated**: December 17, 2025
**Version**: 3.0
**Status**: Production Ready ✅
