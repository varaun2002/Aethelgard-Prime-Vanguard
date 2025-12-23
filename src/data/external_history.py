import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging

class ExternalHistoryFetcher:
    """
    Fetches HISTORICAL external market factors for training.
    Replaces the zero-padding in train_v3.py.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_cache = {
            'spx': None,
            'vix': None,
            'fng': None
        }
    
    def fetch_stock_history(self, start_date=None):
        """
        Fetch hourly data for SPX and VIX from yfinance.
        If start_date is None, fetches last 730 days (2 years).
        """
        if self.data_cache['spx'] is not None:
            return self.data_cache['spx'], self.data_cache['vix']
            
        self.logger.info("Fetching historical SPX and VIX data...")
        try:
            # Fetch generic 2 years to cover most training needs
            # We use 1h interval to match our model granularity
            spx = yf.download('^GSPC', period='2y', interval='1h', progress=False)
            vix = yf.download('^VIX', period='2y', interval='1h', progress=False)
            
            # Standardization
            if isinstance(spx.columns, pd.MultiIndex):
                spx = spx.xs('Close', level=0, axis=1) if 'Close' in spx.columns else spx
            if isinstance(vix.columns, pd.MultiIndex):
                vix = vix.xs('Close', level=0, axis=1) if 'Close' in vix.columns else vix
                
            # Ensure timezone naive or consistently UTC
            # yfinance returns tz-aware usually. distinct handling needed?
            # We'll rely on index matching later.
            
            self.data_cache['spx'] = spx
            self.data_cache['vix'] = vix
            
            return spx, vix
        except Exception as e:
            self.logger.error(f"Failed to fetch stock history: {e}")
            return None, None

    def fetch_fear_greed_history(self):
        """
        Fetch full history of Fear & Greed Index
        """
        if self.data_cache['fng'] is not None:
            return self.data_cache['fng']
            
        self.logger.info("Fetching historical Fear & Greed Index...")
        try:
            url = "https://api.alternative.me/fng/?limit=0&format=json"
            r = requests.get(url)
            data = r.json()['data']
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['value'] = df['value'].astype(int)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Resample to hourly (ffill) since F&G is daily
            df_hourly = df['value'].resample('1h').ffill()
            
            self.data_cache['fng'] = df_hourly
            return df_hourly
        except Exception as e:
            self.logger.error(f"Failed to fetch F&G history: {e}")
            return None

    def get_external_data_for_timestamp(self, ts, spx_df, vix_df, fng_df):
        """
        Extract features for a specific timestamp (integer seconds).
        Returns specific features needed for the 20-dim vector.
        """
        target_time = pd.to_datetime(ts, unit='s')
        
        # 1. Fear & Greed (Normalized 0-1)
        # Use simple extraction
        try:
            # Find nearest index
            fng_idx = fng_df.index.get_indexer([target_time], method='nearest')[0]
            fng_val = fng_df.iloc[fng_idx] / 100.0
        except:
            fng_val = 0.5

        # 2. Stock Features (SPX Return, VIX, Risk On)
        try:
            # Round to nearest hour for matching
            # Data might be missing for weekends/nights in tradfi
            # We ffill from last available
            idx = spx_df.index.get_indexer([target_time], method='pad')[0]
            
            if idx > 0:
                curr_spx = spx_df.iloc[idx].item() # .item() to get scalar
                prev_spx = spx_df.iloc[idx-1].item()
                spx_ret = (curr_spx - prev_spx) / prev_spx
            else:
                spx_ret = 0.0
                
            idx_v = vix_df.index.get_indexer([target_time], method='pad')[0]
            vix_val = vix_df.iloc[idx_v].item()
            
            risk_on = 1.0 if vix_val < 20.0 else 0.0
            
        except:
            spx_ret = 0.0
            vix_val = 20.0
            risk_on = 1.0

        return fng_val, spx_ret, vix_val, risk_on

    def fetch_btc_history(self):
        """
        Fetch hourly BTC data from yfinance for feature calculation.
        """
        if 'btc' in self.data_cache and self.data_cache['btc'] is not None:
             return self.data_cache['btc']

        self.logger.info("Fetching historical BTC data...")
        try:
            # Fetch 2 years to match SPX
            btc = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
            
            if isinstance(btc.columns, pd.MultiIndex):
                btc = btc.xs('Close', level=0, axis=1) if 'Close' in btc.columns else btc
            
            # Keep just Close for now, maybe Volume later if needed
            if isinstance(btc, pd.DataFrame):
                btc = btc.iloc[:, 0]
                
            self.data_cache['btc'] = btc
            return btc
        except Exception as e:
            self.logger.error(f"Failed to fetch BTC history: {e}")
            return None

    def enrich_training_data(self, df):
        """
        Takes the training DataFrame (with timestamps) and adds columns for external features.
        Used before creating sequences.
        """
        # Pre-fetch everything
        spx, vix = self.fetch_stock_history()
        fng = self.fetch_fear_greed_history()
        btc = self.fetch_btc_history()
        
        if spx is None or fng is None or btc is None:
            self.logger.warning(f"Could not fetch complete history (SPX:{spx is not None} FNG:{fng is not None} BTC:{btc is not None}), filling gaps...")
            # We continue, assuming some caching or partial fills.

        self.logger.info("Enriching data with historical external factors (REAL DATA)...")
        
        fng_vals = []
        spx_rets = []
        vix_vals = []
        risk_ons = []
        
        # BTC Features
        btc_ret1h_list = []
        btc_ret4h_list = []
        btc_vol_list = []
        btc_corr_list = []
        btc_div_list = []
        
        # Pre-convert indices for speed
        if spx is not None: spx_idx = spx.index.tz_localize(None)
        if vix is not None: vix_idx = vix.index.tz_localize(None)
        if fng is not None: fng_idx = fng.index.tz_localize(None)
        if btc is not None: btc_idx = btc.index.tz_localize(None)
        
        # Helper for efficient rolling calculations would be better, but loop is safer for logic preservation
        # For BTC Rolling features, accessing by integer location in the full BTC series is best.
        
        for i, ts in enumerate(df['timestamp']):
            # Handle ms vs s
            ts_val = ts
            if ts_val > 1e11: ts_val = ts_val / 1000
            dt = datetime.fromtimestamp(ts_val)
            
            # --- FNG ---
            val = 0.5
            if fng is not None:
                try:
                    loc = fng_idx.get_indexer([dt], method='pad')[0]
                    if loc != -1: val = fng.iloc[loc] / 100.0
                except: val = 0.5
            fng_vals.append(val)
            
            # --- SPX/VIX ---
            r = 0.0
            v = 20.0
            if spx is not None and vix is not None:
                try:
                    loc = spx_idx.get_indexer([dt], method='pad')[0]
                    if loc > 0:
                        curr = spx.iloc[loc].item()
                        prev = spx.iloc[loc-1].item()
                        r = (curr - prev)/prev
                        
                    loc_v = vix_idx.get_indexer([dt], method='pad')[0]
                    if loc_v != -1: v = vix.iloc[loc_v].item()
                except: pass
            
            spx_rets.append(r)
            vix_vals.append(v)
            risk_ons.append(1.0 if v < 20 else 0.0)
            
            # --- BTC FEATURES ---
            # Defaults
            b_r1 = 0.0
            b_r4 = 0.0
            b_vol = 0.02
            b_corr = 0.7
            b_div = 0.0
            
            if btc is not None:
                try:
                    loc = btc_idx.get_indexer([dt], method='pad')[0]
                    if loc > 20: # Need history
                        curr_p = btc.iloc[loc]
                        
                        # Ret 1h (approx 1 index back if hourly)
                        # Ensure index delta matches time. But we pad-found, so indices are contiguous hours mostly.
                        p_1h = btc.iloc[loc-1]
                        if p_1h > 0: b_r1 = (curr_p - p_1h) / p_1h
                        
                        # Ret 4h
                        p_4h = btc.iloc[loc-4]
                        if p_4h > 0: b_r4 = (curr_p - p_4h) / p_4h
                        
                        # Volatility (20h std dev)
                        slice_vol = btc.iloc[loc-20:loc]
                        rets_vol = slice_vol.pct_change().dropna()
                        if len(rets_vol) > 5:
                            b_vol = rets_vol.std()
                            
                        # Correlation (ETH vs BTC 20h)
                        # We need ETH price history matching these BTC points.
                        # df['close'] has the ETH price history aligned with 'timestamp'.
                        # Window: [i-20 : i]
                        if i > 20:
                            eth_slice = df['close'].iloc[i-20:i].values
                            # We can't easily map exact BTC slice if data gaps differ, 
                            # but we can try to use the last 20 BTC points we fetched?
                            # Actually, `loc` is in btc_series.
                            btc_slice_prices = btc.iloc[loc-20:loc].values
                            
                            if len(eth_slice) == len(btc_slice_prices):
                                eth_rets = np.diff(eth_slice) / eth_slice[:-1]
                                btc_rets = np.diff(btc_slice_prices) / btc_slice_prices[:-1]
                                
                                if len(eth_rets) > 5 and len(btc_rets) > 5:
                                    # Handle different lengths if any nan
                                    min_len = min(len(eth_rets), len(btc_rets))
                                    corr = np.corrcoef(eth_rets[:min_len], btc_rets[:min_len])[0,1]
                                    if not np.isnan(corr):
                                        b_corr = corr
                        
                        # Divergence (BTC 1h - ETH 1h)
                        # We have b_r1. Need eth_r1.
                        # Using df data directly is safer than re-fetching
                        if i > 0:
                            eth_curr = df['close'].iloc[i]
                            eth_prev = df['close'].iloc[i-1]
                            if eth_prev > 0:
                                eth_r1 = (eth_curr - eth_prev) / eth_prev
                                b_div = b_r1 - eth_r1
                                    
                except Exception:
                    pass # Keep defaults
            
            btc_ret1h_list.append(b_r1)
            btc_ret4h_list.append(b_r4)
            btc_vol_list.append(b_vol)
            btc_corr_list.append(b_corr)
            btc_div_list.append(b_div)

        # Assign back
        df['ext_fng'] = fng_vals
        df['ext_spx_ret'] = spx_rets
        df['ext_vix'] = vix_vals
        df['ext_risk_on'] = risk_ons
        
        # BTC
        df['ext_btc_ret1h'] = btc_ret1h_list
        df['ext_btc_ret4h'] = btc_ret4h_list
        df['ext_btc_vol'] = btc_vol_list
        df['ext_btc_corr'] = btc_corr_list
        df['ext_btc_div'] = btc_div_list
        
        return df

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    fetcher = ExternalHistoryFetcher()
    fetcher.fetch_stock_history()
    fetcher.fetch_fear_greed_history()
    print("Test fetch complete.")
