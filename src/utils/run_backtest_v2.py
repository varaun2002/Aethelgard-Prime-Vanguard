import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.backtest_v2 import BacktesterV2
from src.utils.logger import setup_logger
import pandas as pd

logger = setup_logger("BacktestRunner")

def run_all_backtests():
    timeframes = ["15m", "1h", "4h"]
    results = []
    
    for tf in timeframes:
        logger.info(f"Running backtest for {tf}...")
        bt = BacktesterV2("ETH/USDT", tf)
        # Use more data for real backtest
        res = bt.run(limit=2000) 
        
        if res:
            results.append({
                "Timeframe": tf,
                "Return": f"{res['return_pct']:.2f}%",
                "Sharpe": f"{res['sharpe_ratio']:.2f}",
                "Max DD": f"{res['max_drawdown']:.2f}%",
                "Win Rate": f"{res['win_rate']:.2f}%",
                "Profit Factor": f"{res['profit_factor']:.2f}",
                "Accuracy": f"{res['accuracy']:.2%}",
                "Trades": res['trades_count']
            })
        else:
            logger.error(f"Backtest failed for {tf}")

    # Print Report
    df_res = pd.DataFrame(results)
    print("\n" + "="*50)
    print("   BACKTEST RESULTS (Model v2.1)   ")
    print("="*50)
    print(df_res.to_markdown(index=False))
    print("="*50 + "\n")
    
    # Save to Markdown
    with open("backtest_results.md", "w") as f:
        f.write("# Rigorous Backtest Results (Model v2.1)\n\n")
        f.write(df_res.to_markdown(index=False))
        f.write("\n\n## Equity Curves\n")
        for tf in timeframes:
            f.write(f"### {tf}\n[Equity Curve {tf}](backtest_equity_{tf}.html)\n")

if __name__ == "__main__":
    run_all_backtests()
