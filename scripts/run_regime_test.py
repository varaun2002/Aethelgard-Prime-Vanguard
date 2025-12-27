import os
import sys
import pandas as pd
import glob

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_v3 import TrainerV3
from src.utils.logger import setup_logger

logger = setup_logger("RegimeTestRun")

def main():
    # Find the data file
    data_dir = "data/historical"
    pattern = os.path.join(data_dir, "ETH_USDT_15m_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        logger.error("No historical 15m data found.")
        return
        
    data_file = sorted(files)[-1] 
    logger.info(f"Loading data from {data_file}...")
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    trainer = TrainerV3()
    
    logger.info("Starting Targeted Regime Test (Folds 34, 35, 36)...")
    try:
        results = trainer.train_walk_forward(
            df=df, 
            timeframe="15m", 
            train_months=6, 
            val_months=1, 
            test_months=1, 
            epochs=3,
            limit_folds=[34, 35, 36]
        )
        
        results_df = pd.DataFrame(results)
        results_path = "data/regime_test_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Regime Test Results saved to {results_path}")
        print("\nRegime Test Summary:")
        print(results_df)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
