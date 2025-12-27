import sys
import os
sys.path.append(os.getcwd())

import torch
import pandas as pd
import numpy as np
from src.models.train_v3 import TrainerV3
from src.utils.logger import setup_logger

logger = setup_logger("DebugSeq")

def test_sequence_creation():
    logger.info("Starting Sequence Creation Test...")
    
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
    logger.info(f"Features added. DF Shape: {df.shape}")
    
    # 3. Create Sequences (This is where it crashed?)
    logger.info("Calling create_sequences_v3...")
    try:
        X, y_ret, y_dir, y_reg = trainer.create_sequences_v3(df, "15m")
        logger.info("create_sequences_v3 SUCCESS")
        logger.info(f"X Shape: {X.shape}")
    except Exception as e:
        logger.error(f"create_sequences_v3 FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sequence_creation()
