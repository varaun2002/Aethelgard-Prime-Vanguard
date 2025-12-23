import torch
import numpy as np
import pandas as pd
import joblib
import re
import os
from src.models.model import CryptoModelV3
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger("PredictorV3")

class PredictorV3:
    def __init__(self, model_path, input_dim=70, vol_dim=0, device=None):
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = DataLoader()
        self.feature_cols = self.data_loader.feature_engineer.get_feature_columns()
        
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
        self.model = CryptoModelV3(input_dim, vol_dim=vol_dim).to(self.device)
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
        # dir_probs: [1, 3] -> Squeeze to [3]
        dir_probs = mc_results['direction_probs'].cpu().numpy().squeeze() 
        if dir_probs.ndim == 0: # Handle edge case if single scalar
             dir_probs = np.array([dir_probs])
        
        predicted_return = mc_results['predicted_return'].item() / 100 # Unscale
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

        # 6. Dynamic Uncertainty Threshold
        # Trending markets are more predictable -> allow higher uncertainty (lower bar)
        # Range/Choppy markets are unpredictable -> require lower uncertainty (higher bar)
        def get_flat_threshold(regime):
            """
            Return uncertainty threshold for FLAT prediction based on market regime.
            
            Higher threshold = MORE directional trades (less conservative)
            Lower threshold = MORE FLAT predictions (more conservative)
            
            Args:
                regime: Market regime classification
                
            Returns:
                float: Uncertainty threshold (0.18 to 0.30)
            """
            thresholds = {
                "Trending UP": 0.30,      # High threshold - trends are predictable, allow trading
                "Trending DOWN": 0.30,    # High threshold - trends are predictable, allow trading
                "Range Quiet": 0.22,      # Medium threshold - moderate conservatism
                "Range Noisy": 0.18       # Lower threshold - noisy markets, be more conservative
            }
            return thresholds.get(regime, 0.25)  # Default fallback
            
        def _validate_thresholds():
            """Ensure thresholds are in valid range"""
            test_regimes = ["Trending UP", "Trending DOWN", "Range Quiet", "Range Noisy"]
            
            # Print validation table to log/console
            # Note: We use logger here since this runs inside predict method/class often
            logger.info("🔍 THRESHOLD VALIDATION")
            
            all_valid = True
            for regime in test_regimes:
                threshold = get_flat_threshold(regime)
                
                # Check range
                if threshold < 0.15:
                    logger.warning(f"❌ {regime:15s}: {threshold:.3f} (TOO LOW - should be 0.18-0.30)")
                    all_valid = False
                elif threshold > 0.35:
                    logger.warning(f"⚠️  {regime:15s}: {threshold:.3f} (TOO HIGH - should be 0.18-0.30)")
                    all_valid = False
                else:
                    logger.info(f"✅ {regime:15s}: {threshold:.3f}")
            
            if not all_valid:
                logger.warning("⚠️  WARNING: Some thresholds are outside valid range!")
            else:
                logger.info("✅ All thresholds validated successfully!")
            
            return all_valid
            
        # Run validation once
        _validate_thresholds()

        dynamic_threshold = get_flat_threshold(regime_str)

        caution_flag = False
        gate1_passed = True
        gate2_passed = True
        gate3_applied = False

        # Logic:
        # 1. Epistemic Safety Check: Is model confused?
        # Three Gates Logic (v2.4.2 Optimization)
        # Gate 1: Epistemic Uncertainty (Model Disagreement)
        if uncertainty > dynamic_threshold:
            predicted_direction = "FLAT"
            caution_flag = True
            gate1_passed = False
            logger.info(f"Gate 1 (Uncertainty): Blocked ({uncertainty:.4f} > {dynamic_threshold})")

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
            # If we are in a Trend, and we are FLAT (due to Gate 2 or Model Class 0),
            # we check if the 'next best' direction is strong enough to trade.
            if regime_str in ["Trending UP", "Trending DOWN"] and predicted_direction == "FLAT":
                # Check underlying directional probabilities
                # dir_probs: [FLAT, DOWN, UP]
                best_dir_idx = np.argmax(dir_probs[1:]) + 1
                best_dir_conf = dir_probs[best_dir_idx]

                # Only force if the direction has decent backing (e.g. 0.40 margin/prob)
                # We use a slightly lower bar for Trend Forcing because Context adds weight.
                FORCE_THRESHOLD = force_threshold

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
            'caution': caution_flag,
            'gate_info': {
                'gate1_passed': gate1_passed,
                'gate2_passed': gate2_passed,
                'gate3_applied': gate3_applied
            }
        }

if __name__ == "__main__":
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
