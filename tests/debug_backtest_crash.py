import sys
import os
sys.path.append(os.getcwd())

import torch
import pandas as pd
import numpy as np
from src.models.train_v3 import TrainerV3
from src.models.model import CryptoModelV3
from src.backtesting.backtester import Backtester
from src.utils.logger import setup_logger

logger = setup_logger("DebugBacktest")

def test_backtest_logic():
    logger.info("Starting Backtest Logic Test...")
    
    # 1. Dummy Data
    dates = pd.date_range(start="2021-01-01", periods=3000, freq="15min")
    df = pd.DataFrame({
        'timestamp': dates.astype(np.int64) // 10**9 * 1000,
        'open': np.random.rand(3000) * 100 + 1000,
        'high': np.random.rand(3000) * 100 + 1100,
        'low': np.random.rand(3000) * 100 + 900,
        'close': np.random.rand(3000) * 100 + 1000,
        'volume': np.random.rand(3000) * 1000,
        'log_ret': np.random.randn(3000) * 0.01
    })
    
    trainer = TrainerV3()
    
    # 2. Add Features (Standard)
    from src.data.data_loader import DataLoader
    dl = DataLoader()
    df = dl.feature_engineer.add_features(df)
    
    # 3. Create Sequences
    logger.info("Creating sequences...")
    X, y_ret, y_dir, y_reg = trainer.create_sequences_v3(df, "15m")
    input_dim = X.shape[2]
    logger.info(f"Sequences created. Dim: {input_dim}")
    
    # 4. Dummy Model Inference
    logger.info("Running Dummy Inference...")
    device = torch.device("cpu")
    model = CryptoModelV3(input_dim=input_dim, vol_dim=0).to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(X) # Forward pass
        dir_logits = outputs['direction_logits']
        probs = torch.softmax(dir_logits, dim=1)
        pred_indices = torch.argmax(probs, dim=1).numpy()
        confidences = torch.max(probs, dim=1).values.numpy()
        
    logger.info(f"Inference done. Predictions: {len(pred_indices)}")
    
    # 5. Prepare Backtest
    logger.info("Preparing Backtest Data...")
    # aligned_df = df.iloc[seq_len:].reset_index(drop=True)
    # Fix alignment: Feature engineering drops rows, so rely on prediction count
    aligned_df = df.iloc[-len(pred_indices):].reset_index(drop=True)
    mapping = {0: 'FLAT', 1: 'UP', 2: 'DOWN'}
    
    predictions_for_backtest = []
    for idx in range(len(pred_indices)):
         p_idx = pred_indices[idx]
         conf = confidences[idx]
         signal = mapping.get(p_idx, 'FLAT')
         predictions_for_backtest.append({'signal': signal, 'confidence': float(conf)})
         
    # 6. Run Backtest
    logger.info("Initializing Backtester...")
    backtester = Backtester(initial_balance=10000)
    
    logger.info("Running run_backtest...")
    try:
        equity_df = backtester.run_backtest(aligned_df, predictions_for_backtest)
        logger.info(f"Backtest done. Equity Rows: {len(equity_df)}")
        
        metrics = backtester.calculate_metrics(equity_df)
        logger.info(f"Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Backtest FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backtest_logic()
