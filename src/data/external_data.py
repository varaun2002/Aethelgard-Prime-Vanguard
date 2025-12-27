import ccxt
import yfinance as yf
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import time
import logging

class ExternalDataFetcher:
    """
    Fetches external market factors for crypto prediction
    
    Features provided:
    - Time-based: 8 features (hour, day, sessions, cyclical encoding)
    - BTC market: 5 features (returns, volatility, correlation, divergence)
    - Derivatives: 1 feature (funding rate)
    - Sentiment: 1 feature (fear & greed index)
    - Traditional markets: 3 features (SPX, VIX, risk-on indicator)
    - Interactions: 2 features (fear-funding, btc-vol-vix)
    
    Total: 20 external features
    """
    
    def __init__(self):
        try:
            self.exchange = ccxt.binance({'enableRateLimit': True})
        except:
            # Binance may be unavailable, but we have fallbacks
            self.exchange = None

        self.cache = {}
        self.cache_duration = {
            'btc_data': 300,        # 5 minutes
            'funding_rate': 3600,   # 1 hour
            'fear_greed': 3600,     # 1 hour
            'stock_data': 3600,     # 1 hour
        }
        self.logger = logging.getLogger(__name__)

        # Suppress ccxt verbose errors for known issues
        logging.getLogger('ccxt').setLevel(logging.WARNING)
    
    def get_time_features(self, timestamp=None):
        """
        Extract time-based features with cyclical encoding
        
        Returns 8 features:
        - hour, day_of_week, is_weekend (3)
        - is_asia_session, is_london_session, is_ny_session, is_us_market_open (4)
        - hour_sin, hour_cos, day_sin, day_cos (4) - but counted in original 8
        
        Actually returns: hour_sin, hour_cos, day_sin, day_cos, is_asia, is_london, is_ny, is_us_open
        """
        if timestamp is None:
            dt = datetime.now(pytz.UTC)
        else:
            dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        
        # Cyclical encoding for hour (0-23)
        hour_sin = np.sin(2 * np.pi * dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * dt.hour / 24)
        
        # Cyclical encoding for day (0-6)
        day_sin = np.sin(2 * np.pi * dt.weekday() / 7)
        day_cos = np.cos(2 * np.pi * dt.weekday() / 7)
        
        # Trading sessions (UTC times)
        is_asia = 1 if 0 <= dt.hour < 8 else 0
        is_london = 1 if 8 <= dt.hour < 16 else 0
        is_ny = 1 if 13 <= dt.hour < 21 else 0
        is_us_open = 1 if dt.weekday() < 5 and 13 <= dt.hour < 21 else 0
        
        return np.array([hour_sin, hour_cos, day_sin, day_cos, 
                        is_asia, is_london, is_ny, is_us_open])
    
    def _fetch_with_cache(self, cache_key, fetch_func, duration):
        """Generic caching wrapper"""
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < duration:
                return data
        
        try:
            data = fetch_func()
            self.cache[cache_key] = (time.time(), data)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching {cache_key}: {e}")
            return None
    
    def get_btc_features(self, eth_data, timeframe='15m'):
        """
        Calculate BTC-related features

        Returns 5 features:
        - btc_return_1h, btc_return_4h
        - btc_volatility
        - eth_btc_correlation
        - btc_eth_divergence
        """

        def fetch_btc():
            """Fetch BTC data from yfinance instead of Binance"""
            try:
                # Map timeframe to yfinance interval
                interval_map = {
                    '15m': '15m',
                    '1h': '1h',
                    '4h': '1h'  # yfinance doesn't have 4h, use 1h
                }
                interval = interval_map.get(timeframe, '15m')

                # Fetch BTC data from yfinance
                btc = yf.Ticker("BTC-USD")
                df = btc.history(period="7d", interval=interval)

                if len(df) < 20:
                    return None

                # Convert to OHLCV format for compatibility
                # Format: [[timestamp, open, high, low, close, volume], ...]
                btc_ohlcv = []
                for idx, row in df.iterrows():
                    btc_ohlcv.append([
                        int(idx.timestamp() * 1000),  # timestamp in ms
                        row['Open'],
                        row['High'],
                        row['Low'],
                        row['Close'],
                        row['Volume']
                    ])

                return btc_ohlcv[-100:]  # Return last 100 candles

            except Exception as e:
                self.logger.error(f"Error fetching BTC from yfinance: {e}")
                return None

        btc_data = self._fetch_with_cache('btc_data', fetch_btc,
                                          self.cache_duration['btc_data'])

        if btc_data is None or len(btc_data) < 20:
            # Return neutral defaults
            return np.array([0.0, 0.0, 0.02, 0.7, 0.0])
        
        # Calculate BTC metrics
        btc_prices = np.array([c[4] for c in btc_data])
        btc_return_1h = ((btc_prices[-1] - btc_prices[-5]) / btc_prices[-5]) if len(btc_prices) >= 5 else 0
        btc_return_4h = ((btc_prices[-1] - btc_prices[-17]) / btc_prices[-17]) if len(btc_prices) >= 17 else 0
        
        btc_returns = np.diff(btc_prices) / btc_prices[:-1]
        btc_volatility = np.std(btc_returns[-20:]) if len(btc_returns) >= 20 else 0.02
        
        # ETH metrics - handle both DataFrame and list formats
        import pandas as pd
        if isinstance(eth_data, pd.DataFrame):
            # DataFrame format
            eth_prices = eth_data['close'].values if 'close' in eth_data.columns else eth_data['Close'].values
        else:
            # OHLCV list format
            eth_prices = np.array([c[4] for c in eth_data])

        eth_returns = np.diff(eth_prices[-20:]) / eth_prices[-20:-1] if len(eth_prices) >= 20 else np.array([0])
        btc_returns_20 = btc_returns[-20:] if len(btc_returns) >= 20 else np.array([0])

        correlation = np.corrcoef(eth_returns, btc_returns_20)[0, 1] if len(eth_returns) >= 20 else 0.7

        eth_return_1h = ((eth_prices[-1] - eth_prices[-5]) / eth_prices[-5]) if len(eth_prices) >= 5 else 0
        divergence = btc_return_1h - eth_return_1h
        
        return np.array([btc_return_1h, btc_return_4h, btc_volatility, 
                        correlation, divergence])
    
    def get_funding_rate(self):
        """
        Fetch ETH perpetual funding rate

        Returns 1 feature: funding_rate

        Note: Funding rates are only available from derivatives exchanges.
        If unavailable, returns a neutral default (0.0001 = 0.01% = typical funding rate).
        """

        def fetch_funding():
            try:
                if self.exchange is None:
                    return None
                funding = self.exchange.fetch_funding_rate('ETH/USDT:USDT')
                return funding['fundingRate']
            except Exception as e:
                # Binance may be blocked or rate limiting
                # Return None to use default
                self.logger.debug(f"Funding rate unavailable: {e}")
                return None

        rate = self._fetch_with_cache('funding_rate', fetch_funding,
                                      self.cache_duration['funding_rate'])

        return rate if rate is not None else 0.0001
    
    def get_fear_greed_index(self):
        """
        Fetch crypto fear & greed index (0-100)
        
        Returns 1 feature: fear_greed_normalized (0-1)
        """
        
        def fetch_fear_greed():
            response = requests.get('https://api.alternative.me/fng/', timeout=5)
            return int(response.json()['data'][0]['value'])
        
        value = self._fetch_with_cache('fear_greed', fetch_fear_greed,
                                       self.cache_duration['fear_greed'])
        
        if value is None:
            value = 50  # Neutral
        
        return value / 100.0  # Normalize to 0-1
    
    def get_stock_market_features(self):
        """
        Fetch traditional market indicators
        
        Returns 3 features:
        - spx_return (S&P 500 1h return)
        - vix_level (VIX index level)
        - risk_on (1 if VIX < 20, else 0)
        """
        
        def fetch_stock_data():
            try:
                spx = yf.download('^GSPC', period='5d', interval='1h', progress=False)['Close']
                vix = yf.download('^VIX', period='5d', interval='1h', progress=False)['Close']
                
                # Handle MultiIndex if present
                if isinstance(spx, pd.DataFrame): spx = spx.iloc[:, 0]
                if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
                
                spx_return = ((spx.iloc[-1] - spx.iloc[-2]) / spx.iloc[-2]) if len(spx) >= 2 else 0
                vix_level = vix.iloc[-1] if len(vix) > 0 else 20.0
                
                return {
                    'spx_return': float(spx_return),
                    'vix_level': float(vix_level),
                    'risk_on': 1 if float(vix_level) < 20 else 0
                }
            except Exception as e:
                self.logger.error(f"YF Error: {e}")
                return None
        
        data = self._fetch_with_cache('stock_data', fetch_stock_data,
                                      self.cache_duration['stock_data'])
        
        if data is None:
            data = {'spx_return': 0.0, 'vix_level': 20.0, 'risk_on': 1}
        
        return np.array([data['spx_return'], data['vix_level'], data['risk_on']])
    
    def get_all_features(self, eth_data, timestamp=None):
        """
        Get all 20 external features in correct order
        
        Feature order:
        1-8:   Time features (8)
        9-13:  BTC features (5)
        14:    Funding rate (1)
        15:    Fear & greed (1)
        16-18: Stock market (3)
        19-20: Interaction features (2)
        
        Returns: numpy array of shape (20,)
        """
        
        # Time features (8)
        time_feat = self.get_time_features(timestamp)
        
        # BTC features (5)
        btc_feat = self.get_btc_features(eth_data)
        
        # Funding rate (1)
        funding = self.get_funding_rate()
        
        # Fear & greed (1)
        fear_greed = self.get_fear_greed_index()
        
        # Stock market (3)
        stock_feat = self.get_stock_market_features()
        
        # Interaction features (2)
        fear_funding_interaction = fear_greed * funding
        btc_vol_vix_interaction = btc_feat[2] * (stock_feat[1] / 20.0)  # btc_vol * normalized_vix
        
        # Combine (8 + 5 + 1 + 1 + 3 + 2 = 20 features)
        
        all_features = np.concatenate([
            time_feat,              # 8 features
            btc_feat,               # 5 features
            [funding],              # 1 feature
            [fear_greed],           # 1 feature
            stock_feat,             # 3 features
            [fear_funding_interaction, btc_vol_vix_interaction] # 2 features
        ])
        
        # Verify size
        if len(all_features) != 20:
             self.logger.warning(f"Feature vector size mismatch: {len(all_features)} != 20")
        
        return all_features.astype(np.float32)


# Validation function
def validate_external_features():
    """Test external data fetcher"""
    print("Testing ExternalDataFetcher...")
    
    fetcher = ExternalDataFetcher()
    exchange = ccxt.binance()
    
    # Get sample ETH data
    eth_data = exchange.fetch_ohlcv('ETH/USDT', '15m', limit=100)
    
    # Get features
    features = fetcher.get_all_features(eth_data)
    
    print(f"\n✓ Generated {len(features)} external features")
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Sample values: {features[:5]}")
    print(f"  All finite: {np.all(np.isfinite(features))}")
    
    return features

if __name__ == "__main__":
    validate_external_features()
