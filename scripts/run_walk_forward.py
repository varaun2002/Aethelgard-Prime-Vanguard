import os
import sys
import pandas as pd
import glob

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_v3 import TrainerV3
from src.utils.logger import setup_logger

logger = setup_logger("WalkForwardRun")

def main():
    # Find the data file
    data_dir = "data/historical"
    pattern = os.path.join(data_dir, "ETH_USDT_15m_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        logger.error("No historical 15m data found. Please run scripts/download_historical_data.py first.")
        return
        
    # Pick the largest file or just the first one
    data_file = sorted(files)[-1] 
    logger.info(f"Loading data from {data_file}...")
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort just in case
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} candles.")
    
    # Initialize Trainer
    trainer = TrainerV3()
    
    # Run Walk-Forward Validation
    # Train = 6 months, Val = 1 month, Test = 1 month
    # This moves forward month by month.
    logger.info("Starting Walk-Forward Validation...")
    results = trainer.train_walk_forward(
        df=df, 
        timeframe="15m", 
        train_months=6, 
        val_months=1, 
        test_months=1, 
        epochs=3 # Reduced epochs for speed during validation, keep 20-50 for final
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = "data/walk_forward_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    print("\nSummary:")
    print(results_df)
    print(f"\nAverage Accuracy: {results_df['accuracy'].mean():.2%}")

if __name__ == "__main__":
    main()
