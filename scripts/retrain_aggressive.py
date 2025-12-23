import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train_v3 import TrainerV3
from src.utils.logger import setup_logger

logger = setup_logger("RetrainAggressive")

def fetch_recent_data(days=60):
    logger.info(f"Fetching last {days} days of ETH-USD data...")
    # Fetch ample buffer to allow for indicator warmup and feature engineering
    # 60 days of 15m data ~ 5760 rows
    df = yf.download("ETH-USD", period=f"{days}d", interval="15m", progress=False)
    
    if len(df) == 0:
        logger.error("No data fetched!")
        return None
        
    df.reset_index(inplace=True)
    
    # Standardize columns
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    
    # Rename for compatibility
    if 'datetime' in df.columns: df.rename(columns={'datetime': 'timestamp'}, inplace=True)
    if 'date' in df.columns: df.rename(columns={'date': 'timestamp'}, inplace=True)
    
    # Ensure timestamp is int (ms) or s? 
    # train_v3 uses ms (lines 440) but create_sequences converts timestamp column to seconds (line 234)
    # yfinance returns timestamp as datetime64[ns]
    # We will let FeatureEngineer handle it, but it expects 'timestamp' column usually.
    
    # Convert timestamp to int64 (ms) to match expected format
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype('int64') // 10**6 # ns -> ms
    
    logger.info(f"Fetched {len(df)} candles.")
    return df

def main():
    logger.info("Starting Micro-Retraining Session...")
    
    # 1. Fetch Data
    df = fetch_recent_data(days=60)
    if df is None: return
    
    # 2. Initialize Trainer
    trainer = TrainerV3()
    
    # 3. Train (Micro Update)
    # micro_update=True will freeze deep layers and run for fewer epochs (5)
    logger.info("Running Training Loop (Micro-Update Mode)...")
    trainer.train(
        symbol="ETH/USDT", 
        timeframe="15m", 
        df=df, 
        epochs=5, 
        batch_size=32, 
        learning_rate=0.0005, # Slightly lower LR for fine-tuning
        micro_update=True
    )
    
    logger.info("Micro-Retraining Complete.")

if __name__ == "__main__":
    main()
