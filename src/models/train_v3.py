import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
from collections import Counter
import joblib
from sklearn.preprocessing import StandardScaler
from src.models.model import CryptoModelV3
from datetime import datetime
from src.data.data_loader import DataLoader
from src.data.external_history import ExternalHistoryFetcher
from src.utils.logger import setup_logger

logger = setup_logger("TrainerV3")

class TrainerV3:
    def __init__(self, model_dir="models_v3"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Loss Functions initialized later or used functional F
        
    def calculate_weighted_loss(self, regime_logits, direction_logits, return_pred, 
                              regime_target, direction_target, return_target, 
                              current_regime_idx=None):
        """
        [Fix #3] Weighted multi-task loss based on observed task performance.
        Refined for v2.4 to penalize errors in predictable regimes more heavily.
        """
        
        # 1. Base Task Losses
        regime_loss = F.cross_entropy(regime_logits, regime_target)
        direction_loss = F.cross_entropy(direction_logits, direction_target, label_smoothing=0.1)
        return_loss = F.huber_loss(return_pred, return_target, delta=0.5)
        
        # 2. Variable Task Weights
        # Regime: 2.0 (High reliability, we want to master this)
        # Direction: 1.0 (Noisy)
        # Return: 1.5 (Moderate)
        regime_weight = 2.0
        direction_weight = 1.0
        return_weight = 1.5
        
        # 3. Auxiliary Task: Volatility Prediction
        # Predicted volatility = abs(predicted return)
        # Actual volatility proxy = abs(actual return)
        volatility_loss = F.mse_loss(return_pred.abs(), return_target.abs())
        
        # 4. Regime-Adaptive Weighting
        # Penalize errors more during clean trends
        if current_regime_idx is not None:
            # Map index to regime name for clarity (assuming 0=Quiet, 1=Noisy, 2=Up, 3=Down)
            # We construct a weight tensor matching batch size
            weights = torch.ones_like(direction_target, dtype=torch.float)
            
            # Boost penalties for Trending (2, 3) and Range Quiet (0)
            # Reduce for Range Noisy (1)
            # Assuming tensor operation for batch efficiency:
            # Range Noisy (1) -> 0.7
            # Others -> 1.3
            
            # Create a mask for Range Noisy
            noisy_mask = (current_regime_idx == 1)
            weights[noisy_mask] = 0.7
            weights[~noisy_mask] = 1.3
            
            adaptive_factor = weights.mean() # Scalar approximate for logging
            
            # Apply element-wise weight to direction loss (requires reduction='none' in loss init usually, 
            # but here we are using F.cross_entropy with default reduction='mean'. 
            # To do adaptive weighting correctly per sample, we need unreduced loss.)
            
            # Re-calculating unreduced losses for adaptive weighting
            direction_loss_raw = F.cross_entropy(direction_logits, direction_target, label_smoothing=0.1, reduction='none')
            direction_loss_weighted = (direction_loss_raw * weights).mean()
            
            return_loss_raw = F.huber_loss(return_pred, return_target, delta=0.5, reduction='none')
            return_loss_weighted = (return_loss_raw * weights).mean()
            
        else:
            direction_loss_weighted = direction_loss
            return_loss_weighted = return_loss
            adaptive_factor = 1.0

        # Final Combined Loss
        total_loss = (
            regime_weight * regime_loss +
            direction_weight * direction_loss_weighted +
            return_weight * return_loss_weighted +
            0.5 * volatility_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'regime': regime_loss.item(),
            'direction': direction_loss.item(), # Log unweighted for comparison
            'return': return_loss.item(),
            'vol': volatility_loss.item()
        }
        
        return total_loss, loss_dict

    def resample_training_data(self, X, y_ret, y_dir, y_reg, oversample_factor=2.5):
        """
        [Fix #5] Rebalance training data to fix regime imbalance.
        """
        # Convert to numpy for indexing
        X_np = X.numpy()
        y_ret_np = y_ret.numpy()
        y_dir_np = y_dir.numpy()
        y_reg_np = y_reg.numpy()
        
        regime_counts = Counter(y_reg_np)
        logger.info(f"Original Regime Distribution: {regime_counts}")
        
        indices_by_regime = {
            r: np.where(y_reg_np == r)[0] for r in np.unique(y_reg_np)
        }
        
        resampled_indices = []
        
        # 0: Range Quiet, 1: Range Noisy, 2: Trend UP, 3: Trend DOWN
        for r, indices in indices_by_regime.items():
            if r == 2: # Trend UP - Oversample significantly (Best Perf)
                n = int(len(indices) * oversample_factor)
                sampled = np.random.choice(indices, size=n, replace=True)
            elif r == 3: # Trend DOWN - Oversample
                n = int(len(indices) * 2.0)
                sampled = np.random.choice(indices, size=n, replace=True)
            elif r == 0: # Range Quiet - Undersample (Worst Perf)
                n = int(len(indices) * 0.5)
                sampled = np.random.choice(indices, size=n, replace=False)
            else: # Range Noisy (1) - Keep as is
                sampled = indices
                
            resampled_indices.extend(sampled)
            
        # Shuffle
        np.random.shuffle(resampled_indices)
        
        # Create new tensors
        X_new = torch.FloatTensor(X_np[resampled_indices])
        y_ret_new = torch.FloatTensor(y_ret_np[resampled_indices])
        y_dir_new = torch.LongTensor(y_dir_np[resampled_indices])
        y_reg_new = torch.LongTensor(y_reg_np[resampled_indices])
        
        logger.info(f"Resampled Dataset Size: {len(X)} -> {len(X_new)}")
        return X_new, y_ret_new, y_dir_new, y_reg_new

        return X_new, y_ret_new, y_dir_new, y_reg_new

    def get_scalers(self, timeframe):
        """Load or create scalers"""
        scaler_dir = self.model_dir
        tech_path = f"{scaler_dir}/technical_scaler_{timeframe}.pkl"
        ext_path = f"{scaler_dir}/external_scaler_{timeframe}.pkl"
        
        if os.path.exists(tech_path) and os.path.exists(ext_path):
            technical_scaler = joblib.load(tech_path)
            external_scaler = joblib.load(ext_path)
            return technical_scaler, external_scaler
        else:
            return StandardScaler(), StandardScaler()
            
    def save_scalers(self, technical_scaler, external_scaler, timeframe):
        joblib.dump(technical_scaler, f"{self.model_dir}/technical_scaler_{timeframe}.pkl")
        joblib.dump(external_scaler, f"{self.model_dir}/external_scaler_{timeframe}.pkl")

    def create_sequences_v3(self, df, timeframe, seq_len=60):
        """
        Create 70-feature sequences using DataLoader
        """
        data_loader = DataLoader() # From src.data.data_loader
        technical_scaler, external_scaler = self.get_scalers(timeframe)
        
        # 1. Generate Technical Features (Offline batch possible?)
        # We use data_loader.feature_engineer directly for batch efficiency
        # instead of calling extract_technical_features row by row.
        
        tech_df = data_loader.feature_engineer.add_features(df.copy())
        tech_cols = data_loader.feature_engineer.get_feature_columns()
        tech_data = tech_df[tech_cols].values.astype(np.float32) # (N, 50)
        
        # Fill NaNs
        tech_data = np.nan_to_num(tech_data)
        
        # Targets
        returns = df['log_ret'].shift(-1).fillna(0).values * 100
        
        flat_threshold = 0.0025
        future_ret = df['log_ret'].shift(-1).fillna(0).values
        dirs = []
        for r in future_ret:
            if abs(r) < flat_threshold: dirs.append(0)
            elif r < 0: dirs.append(1)
            else: dirs.append(2)
        dirs = np.array(dirs)
        
        # Regimes (simplified logic matching original)
        regimes = np.zeros(len(df))
        if 'adx' in tech_df.columns:
            for i in range(len(tech_df)):
                adx = tech_df['adx'].iloc[i]
                chop = tech_df['chop'].iloc[i] if 'chop' in tech_df.columns else 50
                rsi = tech_df['rsi'].iloc[i] if 'rsi' in tech_df.columns else 50
                
                if adx < 20: regimes[i] = 0
                elif adx < 30 and chop > 50: regimes[i] = 1
                else:
                    if rsi > 50: regimes[i] = 2
                    else: regimes[i] = 3
        regimes = np.array(regimes)

        # 2. Generate External Features
        # v3.1 Upgrade: Use ExternalHistoryFetcher for real historical data
        logger.info("Generating external features (v3.1 with History)...")
        
        hist_fetcher = ExternalHistoryFetcher()
        spx, vix = hist_fetcher.fetch_stock_history()
        fng = hist_fetcher.fetch_fear_greed_history()
        
        # Pre-process indices for fast lookup
        if spx is not None:
            spx_idx = spx.index.tz_localize(None)
            vix_idx = vix.index.tz_localize(None)
            fng_idx = fng.index.tz_localize(None)
        
        external_data_list = []
        timestamps = pd.to_datetime(tech_df['timestamp']).astype(np.int64) // 10**9 # seconds
        
        for ts in timestamps:
             # A. Standard Time Features (8)
             time_feat = data_loader.external_fetcher.get_time_features(ts)
             
             # B. Historical External Features (12)
             # We need to reconstruct the exact 12-dim vector expected by the model
             # The model expects [BTC Features(5), Funding(1), F&G(1), Stock(3), Interaction(2)]
             
             # Note: We still lack granular historical BTC details (volatility/returns) aligned perfectly 
             # without re-fetching BTC history. 
             # Ideally we should compute BTC features from the main dataframe if it is BTC, 
             # but here we are training on ETH. 
             # For v3.1 we will rely on yfinance for Stock/Fng and keep BTC placeholders 
             # or try to approximate BTC from ETH (high correlation).
             
             # Let's use clean defaults for missing BTC data but fill the critical Sentiment/Stock data.
             
             # Fetch matched history
             fng_val, spx_ret, vix_val, risk_on = 0.5, 0.0, 20.0, 0.0
             
             if spx is not None and fng is not None:
                 dt = datetime.fromtimestamp(ts)
                 
                 # FNG
                 try:
                     loc = fng_idx.get_indexer([dt], method='pad')[0]
                     if loc != -1: fng_val = fng.iloc[loc].item() / 100.0
                 except: pass
                 
                 # Stock
                 try:
                     loc = spx_idx.get_indexer([dt], method='pad')[0]
                     if loc > 0:
                         curr = spx.iloc[loc].item()
                         prev = spx.iloc[loc-1].item()
                         spx_ret = (curr - prev)/prev
                         
                     loc_v = vix_idx.get_indexer([dt], method='pad')[0]
                     if loc_v != -1: vix_val = vix.iloc[loc_v].item()
                     risk_on = 1.0 if vix_val < 20 else 0.0
                 except: pass

             # Construct Vector
             # BTC Features (5): [ret1h, ret4h, vol, corr, div] -> Placeholders (zeros) for now
             btc_feat = np.zeros(5, dtype=np.float32) 
             
             # Funding (1): Placeholder
             funding = 0.0001
             
             # Interactions (2)
             fear_funding = fng_val * funding
             btc_vol_vix = 0.02 * (vix_val / 20.0) # approx btc vol
             
             # [BTC(5), Fund(1), Fng(1), Stock(3), Interact(2)] = 12
             others = np.concatenate([
                 btc_feat,
                 [funding],
                 [fng_val],
                 [spx_ret, vix_val, risk_on],
                 [fear_funding, btc_vol_vix]
             ])
             
             row = np.concatenate([time_feat, others])
             external_data_list.append(row)
             
        ext_data = np.array(external_data_list, dtype=np.float32)
        
        # 3. Scale and Save
        # Only fit if not fitted? Or re-fit every training?
        # Usually fit on Train set.
        technical_scaler.fit(tech_data)
        external_scaler.fit(ext_data)
        
        self.save_scalers(technical_scaler, external_scaler, timeframe)
        
        tech_scaled = technical_scaler.transform(tech_data)
        ext_scaled = external_scaler.transform(ext_data)
        
        combined_data = np.concatenate([tech_scaled, ext_scaled], axis=1) # (N, 70)
        
        X, y_ret, y_dir, y_reg = [], [], [], []
        
        for i in range(len(combined_data) - seq_len):
            X.append(combined_data[i:i+seq_len])
            y_ret.append(returns[i+seq_len])
            y_dir.append(dirs[i+seq_len])
            y_reg.append(regimes[i+seq_len])
            
        return (
            torch.FloatTensor(np.array(X)),
            torch.FloatTensor(np.array(y_ret)).unsqueeze(1),
            torch.LongTensor(np.array(y_dir)),
            torch.LongTensor(np.array(y_reg))
        )

    def train(self, symbol, timeframe, df=None, epochs=50, batch_size=64, learning_rate=0.001, micro_update=False):
        if df is None:
            logger.error("No DataFrame provided for training")
            return
            
        # 1. Features
        data_loader = DataLoader()
        fe = data_loader.feature_engineer
        df = fe.add_features(df)
        
        # 2. Sequences
        try:
            X, y_ret, y_dir, y_reg = self.create_sequences_v3(df, timeframe)
        except Exception as e:
            logger.error(f"Sequence creation failed: {e}")
            import traceback
            traceback.print_exc()
            return

        if len(X) < batch_size:
            logger.warning("Not enough data to train")
            return
            
        # [Fix #5] Resampling (Only if not micro-updating, or always? Ideally always for balance)
        # But for micro-updates on small rectent data, resampling might reduce diversity too much. 
        # Let's apply it mainly for full training.
        if not micro_update:
            try:
                X, y_ret, y_dir, y_reg = self.resample_training_data(X, y_ret, y_dir, y_reg)
            except Exception as e:
                logger.error(f"Resampling failed: {e}")

        dataset = TensorDataset(X, y_ret, y_dir, y_reg)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 3. Model Init
        input_dim = X.shape[2]
        vol_dim = 0 
        
        model = CryptoModelV3(input_dim=input_dim, vol_dim=vol_dim).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
        
        model_path = f"{self.model_dir}/model_v3_0_{timeframe}.pth"
        
        # Load or Init
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
            except Exception as e:
                logger.warning(f"Starting fresh model (Load failed: {e})")
        
        # 4. Training Loop
        model.train()
        
        if micro_update:
            for param in model.gru.parameters():
                param.requires_grad = False
            for param in model.transformer_encoder.parameters():
                param.requires_grad = False
            epochs = 5 
            
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_ret, batch_dir, batch_reg in dataloader:
                batch_X, batch_ret = batch_X.to(self.device), batch_ret.to(self.device)
                batch_dir, batch_reg = batch_dir.to(self.device), batch_reg.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                
                # [Fix #3] Weighted Loss Calculation
                loss, components = self.calculate_weighted_loss(
                    outputs['regime_logits'], 
                    outputs['direction_logits'], 
                    outputs['predicted_return'],
                    batch_reg, batch_dir, batch_ret,
                    current_regime_idx=batch_reg # Pass target regime for adaptive weighting during training
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
                
        # 5. Save
        torch.save(model.state_dict(), model_path)
        logger.info(f"v3.0 Model saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test", action="store_true", help="Run in test mode with dummy data")
    args = parser.parse_args()

    trainer = TrainerV3()
    
    if args.test:
        logger.info("Running in TEST mode")
        # Generate dummy data
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="15min")
        data = {
            'timestamp': dates.astype(np.int64) // 10**9 * 1000, # ms
            'open': np.random.rand(1000) * 100 + 1000,
            'high': np.random.rand(1000) * 100 + 1100,
            'low': np.random.rand(1000) * 100 + 900,
            'close': np.random.rand(1000) * 100 + 1000,
            'volume': np.random.rand(1000) * 1000,
            'log_ret': np.random.randn(1000) * 0.01
        }
        df = pd.DataFrame(data)
        # Ensure log_ret exists (FeatureEngineer might recalculate but we need it for targets)
        # FeatureEngineer calculates log_ret from close if missing, but we provide it.
        
        trainer.train(symbol="ETH/USDT", timeframe="15m", df=df, epochs=args.epochs, batch_size=args.batch_size)
    else:
        logger.info("Running in REAL mode (Placeholder)")
        # In a real scenario, we would load data here. 
        # For now, we print instructions.
        print("Please provide a dataframe to trainer.train() or run with --test")
