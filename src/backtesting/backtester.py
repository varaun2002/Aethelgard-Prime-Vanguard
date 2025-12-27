import pandas as pd
import numpy as np
from src.inference.paper_trader import PaperTrader
import os
import shutil

class Backtester:
    """
    Run PaperTrader logic on historical data.
    """
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        
    def run_backtest(self, df, predictions, price_col='close'):
        """
        df: DataFrame with timestamps, prices, etc.
        predictions: List/Array of {'signal': X, 'confidence': Y} aligned with df
        """
        # Create a temp isolated trader
        temp_state_file = "data/backtest_temp.json"
        if os.path.exists(temp_state_file):
            os.remove(temp_state_file)
            
        trader = PaperTrader(
            initial_balance=self.initial_balance,
            state_file=temp_state_file,
            persist=False
        )
        
        # We need recent prices for volatility calc.
        # We can pass the slicing window.
        
        equity_curve = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            pred = predictions[i]
            
            signal = pred['signal']
            confidence = pred['confidence']
            
            current_price = row[price_col]
            
            ts_val = row['timestamp']
            if hasattr(ts_val, 'timestamp'):
                timestamp = int(ts_val.timestamp())
            else:
                timestamp = int(ts_val)
                # If milliseconds (huge number), convert to seconds
                if timestamp > 1e11: timestamp = timestamp // 1000
            
            # Recent prices (last 100 for regime detection)
            start_idx = max(0, i-100)
            recent_prices = df[price_col].iloc[start_idx:i+1].values
            recent_candles = df.iloc[start_idx:i+1]
            
            # Volume (if available)
            volume = row.get('volume', None)
            # Convert to Approx Quote Volume if needed (assuming Base Vol)
            if volume:
                volume = volume * current_price
            
            trader.update(
                signal=signal,
                confidence=confidence,
                probs=pred.get('probs'), # Pass probs
                current_price=current_price,
                timestamp=timestamp,
                recent_prices=recent_prices,
                volume=volume,
                recent_candles=recent_candles
            )
            
            stats = trader.get_stats()
            equity_curve.append({
                'timestamp': row['timestamp'],
                'equity': stats['current_equity'],
                'balance': trader.state['balance_usdt'],
                'position': trader.state['position_eth']
            })
            
        # Cleanup
        if os.path.exists(temp_state_file):
            os.remove(temp_state_file)
            
        return pd.DataFrame(equity_curve)

    def calculate_metrics(self, equity_curve_df):
        if equity_curve_df.empty:
            return {}
            
        initial = equity_curve_df['equity'].iloc[0]
        final = equity_curve_df['equity'].iloc[-1]
        
        # Returns
        total_return = (final - initial) / initial
        
        # Max Drawdown
        equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
        equity_curve_df['dd'] = (equity_curve_df['peak'] - equity_curve_df['equity']) / equity_curve_df['peak']
        max_dd = equity_curve_df['dd'].max()
        
        # Sharpe (Daily approximation)
        # Resample to daily? Or use per-candle returns?
        # Use candle returns * sqrt(candles_per_year)
        equity_curve_df['ret'] = equity_curve_df['equity'].pct_change()
        mean_ret = equity_curve_df['ret'].mean()
        std_ret = equity_curve_df['ret'].std()
        
        sharpe = 0
        if std_ret > 0:
            # 15m candles: 4 * 24 * 365 = 35040
            sharpe = (mean_ret / std_ret) * np.sqrt(35040)
            
        return {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'final_equity': final
        }
