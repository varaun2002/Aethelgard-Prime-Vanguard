import torch
import numpy as np
import pandas as pd
import joblib
import re
import os
import sys
# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model import CryptoModelV3
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger("PredictorV3")

class PredictorV3:
    def __init__(self, model_path, input_dim=None, vol_dim=0, device=None):
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = DataLoader()
        self.feature_cols = self.data_loader.feature_engineer.get_feature_columns()
        
        # Dynamic Input Dim Calculation
        # Tech features (30) + External features (20) = 50
        if input_dim is None:
             self.input_dim = len(self.feature_cols) + 20 
        else:
             self.input_dim = input_dim
        
        # Determine timeframe from model path for scalers
        # Expected format: model_v3_0_{timeframe}.pth or model_{timeframe}.pt
        self.timeframe = "15m" # Default
        if "1h" in model_path: self.timeframe = "1h"
        elif "4h" in model_path: self.timeframe = "4h"
        
        # Load Scalers
        self.scaler_dir = os.path.dirname(model_path)
        try:
            self.tech_scaler = joblib.load(f"{self.scaler_dir}/technical_scaler_{self.timeframe}.pkl")
            self.ext_scaler = joblib.load(f"{self.scaler_dir}/external_scaler_{self.timeframe}.pkl")
            logger.info(f"Loaded scalers for {self.timeframe}")
        except Exception as e:
            logger.warning(f"Could not load scalers, prediction may be inaccurate: {e}")
            self.tech_scaler = None
            self.ext_scaler = None

        # Load Model
        self.model = CryptoModelV3(self.input_dim, vol_dim=vol_dim).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded v3.0 model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def predict(self, df, seq_len=60, confidence_threshold=0.6, uncertainty_threshold=0.2, force_threshold=0.4, safety_lock=True):
        """
        Generates a prediction using Monte Carlo Dropout.
        """
        # === INPUT VALIDATION ===
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return None
        
        # Check for sufficient data
        if len(df) < seq_len + 200:  # Need seq_len + warmup for indicators
            logger.warning(f"Insufficient data: {len(df)} rows, need {seq_len + 200}")
            return None
        
        # Check for NaN prices
        if df['close'].isna().any():
            logger.warning("NaN values in close prices, forward-filling")
            df['close'] = df['close'].ffill()
        # 1. Feature Engineering (Technical)
        # Using DataLoader's feature engineer logic but we need sequence
        fe = self.data_loader.feature_engineer
        df = fe.add_features(df)
        
        # 2. Prepare Sequence
        tech_data = df[self.feature_cols].values # (N, 50)
        
        if len(tech_data) < seq_len:
            logger.warning("Not enough data for sequence")
            return None
            
        recent_tech_seq = tech_data[-seq_len:].astype(np.float32) # (60, 50)
        recent_tech_seq = np.nan_to_num(recent_tech_seq)
        
        # Scale Technical
        if self.tech_scaler:
            recent_tech_seq = self.tech_scaler.transform(recent_tech_seq)
            
        # 3. External Features
        # We need (60, 20).
        # We fetch CURRENT external macro features and repeat them, 
        # but calculate TIME features per step.
        
        timestamps = pd.to_datetime(df['timestamp'].values[-seq_len:]).astype(int) // 10**9
        
        # Fetch current global external context (Macro)
        # We get "all features" for the LAST timestamp to capture Funding, Fear/Greed etc.
        # We ignore the returned Time features from this call and replace with sequence time.
        # We also ignore BTC features from this call? No, we use them as "Current Market State".
        
        # get_all_features returns (20,)
        # eth_data for correlation: pass recent technical slice?
        recent_eth_slice = df.iloc[-100:].copy() # context
        
        # We call external_fetcher to get "Current" state vector
        current_ext_vec = self.data_loader.external_fetcher.get_all_features(recent_eth_slice, timestamp=None)
        
        # Decompose: Indices 0-7 are Time. 8-19 are Macro/BTC.
        macro_part = current_ext_vec[8:] # (12,)
        
        ext_seq = []
        for ts in timestamps:
            t_feat = self.data_loader.external_fetcher.get_time_features(ts) # (8,)
            combined_row = np.concatenate([t_feat, macro_part]) # (20,)
            ext_seq.append(combined_row)
            
        ext_seq = np.array(ext_seq, dtype=np.float32) # (60, 20)
        
        # Scale External
        if self.ext_scaler:
            ext_seq = self.ext_scaler.transform(ext_seq)
            
        # Combine
        combined_seq = np.concatenate([recent_tech_seq, ext_seq], axis=1) # (60, 70)
        
        X = torch.FloatTensor(combined_seq).unsqueeze(0).to(self.device) # [1, seq_len, 70]
        
        # 3. Monte Carlo Inference
        # We run 10 passes to estimate uncertainty
        mc_results = self.model.mc_dropout_predict(X, n_samples=10)
        
        # 4. Parse Results
        # dir_probs: [Flat, Down, Up]
        dir_probs = mc_results['direction_probs'].cpu().numpy().squeeze() 
        if dir_probs.ndim == 0: 
             dir_probs = np.array([dir_probs])
        
        predicted_return = mc_results['predicted_return'].item() / 100 
        uncertainty = mc_results['direction_uncertainty'].item()
        
        # Classes: 0=FLAT, 1=DOWN, 2=UP
        pred_class = np.argmax(dir_probs)
        confidence = dir_probs[pred_class]
        
        direction_map = {0: "FLAT", 1: "DOWN", 2: "UP"}
        predicted_direction = direction_map[pred_class]

        # 5. Regime (Need this for dynamic threshold)
        with torch.no_grad():
            out = self.model(X)
            regime_logits = out['regime_logits']
            regime_class = torch.argmax(regime_logits, dim=1).item()
            
        regime_map = {0: "Range Quiet", 1: "Range Noisy", 2: "Trending UP", 3: "Trending DOWN"}
        regime_str = regime_map.get(regime_class, "Unknown")

        # 6. Safety Gates (Now just "Caution Flags")
        # We NEVER override to FLAT. We just flag if the move is risky.
        
        # Helper: Thresholds (Restored)
        def get_flat_threshold(regime):
            """
            Return uncertainty threshold for FLAT prediction based on market regime.
            Args:
                regime: Market regime classification
            Returns:
                float: Uncertainty threshold (0.18 to 0.30)
            """
            thresholds = {
                "Trending UP": 0.25,      # High threshold - lowered for sensitivity
                "Trending DOWN": 0.25,    # High threshold - lowered for sensitivity
                "Range Quiet": 0.12,      # Hyper Aggressive: Trade on 12% probability
                "Range Noisy": 0.12       # Hyper Aggressive: Trade on 12% probability
            }
            return thresholds.get(regime, 0.25)  # Default fallback
        
        caution_flag = False # Initialize early

        # Dynamic threshold based on regime (Baseline)
        dynamic_threshold = get_flat_threshold(regime_str)

        # Logic determining final threshold:
        # If safety_lock is ON, we strict adhere to dynamic_threshold (or tighter).
        # If safety_lock is OFF (Aggressive), we allow the passed 'uncertainty_threshold' 
        # to override if it is looser (higher) than dynamic.
        
        final_uncertainty_threshold = dynamic_threshold
        if not safety_lock:
             # Allow looser threshold if requested
             final_uncertainty_threshold = max(dynamic_threshold, uncertainty_threshold)

        gate1_passed = True
        gate2_passed = True
        gate3_applied = False # Not needed in binary mode
        
        # Gate 1: Uncertainty
        # Using baseline dynamic thresholds just for warning
        dynamic_threshold = get_flat_threshold(regime_str)
        
        if uncertainty > final_uncertainty_threshold:
            predicted_direction = "FLAT"
            caution_flag = True
            gate1_passed = False
            logger.info(f"Gate 1 (Uncertainty): Blocked ({uncertainty:.4f} > {final_uncertainty_threshold:.4f})")

        else:
            # Gate 2: Minimum Confidence (Aleatoric Strength)
            # Probability 0.33 = Random. 0.45 = Weak Edge. 0.60 = Strong.
            # User Request: Set floor to 0.47 to avoid fee erosion.
            # UPDATE: Allow override via confidence_threshold argument
            MIN_CONFIDENCE = min(0.47, confidence_threshold)

            # Check if default prediction passes quality bar
            if confidence < MIN_CONFIDENCE:
                logger.info(f"Gate 2 (Confidence): Weak signal ({confidence:.2f} < {MIN_CONFIDENCE}) -> FLAT")
                predicted_direction = "FLAT"
                gate2_passed = False

            # Gate 3: Trend Forcing (Context Awareness)
            # RE-ENABLED with STRICTER threshold (Sweet Spot)
            if regime_str in ["Trending UP", "Trending DOWN"] and predicted_direction == "FLAT":
                # Check underlying directional probabilities
                # dir_probs: [FLAT, DOWN, UP]
                best_dir_idx = np.argmax(dir_probs[1:]) + 1
                best_dir_conf = dir_probs[best_dir_idx]

                # SWEET SPOT: Increase threshold to 0.18 (was 0.1)
                # This ensures we only force if the signal is genuinely strong, not just noise.
                FORCE_THRESHOLD = 0.18

                if best_dir_conf > FORCE_THRESHOLD:
                    predicted_direction = direction_map[best_dir_idx]
                    confidence = best_dir_conf
                    gate3_applied = True
                    logger.info(f"Gate 3 (Trend Force): Overriding FLAT -> {predicted_direction} (Conf {confidence:.2f})")
                else:
                    logger.info(f"Gate 3 (Trend Force): Signal too weak to force ({best_dir_conf:.2f} < {FORCE_THRESHOLD})")

            # Legacy Safety: Range Noisy requires extreme confidence
            # Relaxed to 0.47 per user request (was 0.80) to allow trading in noise
            if safety_lock and regime_class == 1 and confidence < 0.47:
                 predicted_direction = "FLAT" # Strict clamp for noise


        return {
            'predicted_price': df['close'].iloc[-1] * (1 + predicted_return),
            'predicted_return': predicted_return,
            'predicted_direction': predicted_direction,
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'regime': regime_str,
            'probabilities': dir_probs.tolist(),
            'probs': dir_probs.tolist(), # Shortcut for Dashboard
            'caution': caution_flag,
            'gate_info': {
                'gate1_passed': gate1_passed,
                'gate2_passed': gate2_passed,
                'gate3_applied': gate3_applied
            }
        }

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--model_path", type=str, default="models_v3/model_v3_0_15m.pth")
    args = parser.parse_args()

    if args.test:
        # Check if model exists, if not create a dummy one for testing flow IF allowed
        # But we assume train_v3 --test ran first?
        # We'll try to load or warn.
        
        if not os.path.exists(args.model_path):
             print(f"Model not found at {args.model_path}. Please run train_v3.py --test first.")
        else:
            predictor = PredictorV3(model_path=args.model_path)
            
            # Dummy data
            dates = pd.date_range(start="2023-01-01", periods=1000, freq="15min")
            data = {
                'timestamp': dates.astype(np.int64) // 10**9 * 1000,
                'open': np.random.rand(1000) * 1000,
                'high': np.random.rand(1000) * 1000,
                'low': np.random.rand(1000) * 1000,
                'close': np.random.rand(1000) * 1000,
                'volume': np.random.rand(1000) * 1000,
                'log_ret': np.random.randn(1000) * 0.01
            }
            df = pd.DataFrame(data)
            
            print("Running prediction test...")
            try:
                result = predictor.predict(df)
                if result:
                    print("Prediction successful:")
                    print(result)
                else:
                    print("Prediction returned None (insufficient data?)")
            except Exception as e:
                print(f"Prediction failed: {e}")
