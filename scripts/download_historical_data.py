import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def download_full_history(symbol='ETH/USDT', timeframe='15m', 
                          start_date='2020-01-01', exchange_name='binance'):
    """
    Download complete historical data from exchange
    """
    try:
        exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    except Exception as e:
        print(f"Failed to initialize exchange {exchange_name}: {e}")
        return None
    
    # Create data directory if it doesn't exist
    os.makedirs('data/historical', exist_ok=True)
    
    since = exchange.parse8601(f'{start_date}T00:00:00Z')
    all_ohlcv = []
    
    print(f"Downloading {symbol} {timeframe} data from {start_date}...")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Next millisecond after last candle
            
            print(f"Downloaded {len(all_ohlcv)} candles...", end='\r')
            
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
            # Stop if we've reached current time (minus 1 min buffer)
            if ohlcv[-1][0] >= exchange.milliseconds() - 60000:
                break
                
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save
    safe_symbol = symbol.replace('/', '_')
    filename = f"data/historical/{safe_symbol}_{timeframe}_{start_date}.csv"
    df.to_csv(filename, index=False)
    print(f"\nSaved {len(df)} candles to {filename}")
    
    return df

if __name__ == "__main__":
    # Download data for all timeframes
    for tf in ['15m', '1h', '4h']:
        download_full_history(timeframe=tf, start_date='2020-01-01')
