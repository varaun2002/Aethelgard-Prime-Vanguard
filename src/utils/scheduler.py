import time
import threading
import schedule
import os
from datetime import datetime
from src.models.train_v3 import TrainerV3
from src.models.predict_v3 import PredictorV3
from src.data.fetcher import DataFetcher
from src.utils.logger import setup_logger
from src.utils.history import HistoryManager

logger = setup_logger("Scheduler")

class AutoTrainer:
    def __init__(self, timeframes=None):
        self.running = False
        self.thread = None
        self.timeframes = timeframes if timeframes else ["15m", "1h", "4h"]
        self.symbol = "ETH/USDT"
        self.last_full_train = {
            "15m": 0,
            "1h": 0,
            "4h": 0
        }
        self.trainer = TrainerV3()
        
        # v2.3: Check for existing models to avoid immediate retrain
        for tf in self.timeframes:
            model_path = f"models_v3/model_{tf}.pt"
            if os.path.exists(model_path):
                # Get last modified time
                mtime = os.path.getmtime(model_path)
                # If model is recent (< 12 hours), set last_full_train to that time
                if (time.time() - mtime) < (12 * 3600):
                    self.last_full_train[tf] = mtime
                    logger.info(f"Found recent v2.3 model for {tf}, skipping immediate full train.")
        self.trainer = TrainerV3()

    def job(self, timeframe):
        logger.info(f"Starting job for {timeframe}")
        
        # 1. Fetch Data
        fetcher = DataFetcher()
        # We need enough data for features (180 window) + sequence (60)
        df = fetcher.fetch_ohlcv(self.symbol, timeframe, limit=1000)
        
        if df is None:
            logger.error(f"Failed to fetch data for {timeframe}")
            return
            
        # Get feature input dim from FeatureEngineer (since v3 needs it)
        # PredictorV3 initializes model inside, so we just instantiate
        # But wait, PredictorV3 needs input_dim and model_path.
        # Let's get input dim dynamically
        from src.data.features import FeatureEngineerV2
        input_dim = len(FeatureEngineerV2().get_feature_columns())

        # 2. Predict
        model_path = f"models_v3/model_{timeframe}.pt"
        if not os.path.exists(model_path):
             logger.warning(f"No model found for {timeframe} yet. Skipping prediction.")
             prediction = None
        else:
            try:
                predictor = PredictorV3(model_path=model_path, input_dim=input_dim)
                prediction = predictor.predict(df)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                prediction = None
        
        hm = HistoryManager()
        
        if prediction:
            logger.info(f"Prediction for {timeframe}: {prediction['predicted_direction']} (Conf: {prediction['confidence']:.2f}, Unc: {prediction['uncertainty']:.4f})")
            
            # Log Prediction
            # Note: HistoryManager might need updating to store uncertainty, 
            # but for now we can log it in 'flat_prob' or just rely on console/UI if HM schema is fixed.
            # Let's assume HM is flexible or we update it.
            # v2.3: We will pass uncertainty as metadata if possible, or just log standard fields.
            # To avoid breaking schema right now, we can map uncertainty to flat_prob (hack) or just log it.
            
            hm.log_prediction(
                symbol=self.symbol,
                timeframe=timeframe,
                current_price=df['close'].iloc[-1], # Use latest close from fetch
                pred_price=prediction['predicted_price'],
                pred_return=prediction['predicted_return'],
                pred_dir=prediction['predicted_direction'],
                confidence=prediction['confidence'],
                flat_prob=prediction['uncertainty'], # MAPPING UNCERTAINTY TO FLAT_PROB for storage
                regime=prediction['regime'],
                model_version="2.4.0"
            )
        else:
            logger.warning(f"Prediction skipped for {timeframe}")

        # 3. Update Actuals (Check previous predictions)
        # This also updates the actuals for the current timeframe based on the fetched df
        try:
            hm.update_actuals(self.symbol, df, timeframe)
        except Exception as e:
            logger.error(f"Failed to update actuals: {e}")
        
        # 4. Determine Training Mode (Full vs. Micro)
        now = time.time()
        last_train = self.last_full_train.get(timeframe, 0)
        # Full retrain every 12 hours (12 * 3600 seconds)
        is_full_train = (now - last_train) > (12 * 3600)
        
        # 5. Train (Full or Micro)
        if is_full_train:
            logger.info(f"Starting FULL training for {timeframe}")
            micro_update = False
            epochs = 50
            self.last_full_train[timeframe] = now
        else:
            # v2.3: Frequency Rules (Stochastic Micro-Updates)
            # Train every 5 candles for better stability
            
            should_micro_train = True
            count_file = f"data/.train_count_{timeframe}"
            count = 0
            if os.path.exists(count_file):
                try:
                    with open(count_file, 'r') as f:
                        count = int(f.read().strip())
                except:
                    count = 0
            
            count = (count + 1) % 5 # Train every 5th time
            with open(count_file, 'w') as f:
                f.write(str(count))
                
            if count != 0: 
                should_micro_train = False
                logger.info(f"Skipping micro-training for {timeframe} (Counter: {count}/5)")
            
            if should_micro_train:
                logger.info(f"Starting MICRO training for {timeframe}")
                micro_update = True
                epochs = 10 
            else:
                return 

        try:
            train_start_time = time.time()
            model = self.trainer.train(
                self.symbol.replace('/', '_'), 
                timeframe, 
                df=df,
                epochs=epochs, 
                micro_update=micro_update
            )
            loss = 0.0 # Logged internally

            train_duration = time.time() - train_start_time
            
            # Log Training
            hm.log_training(
                symbol=self.symbol,
                timeframe=timeframe,
                loss=loss,
                accuracy=0.0, # Placeholder, as we track loss primarily
                duration=train_duration
            )
            logger.info(f"{'Full' if not micro_update else 'Micro'}-training completed for {timeframe}")
        except Exception as e:
            logger.error(f"Training failed for {timeframe}: {e}")

    def start(self, interval_minutes=None):
        if self.running:
            return
        
        self.running = True
        
        def run_schedule():
            logger.info(f"Scheduler v2.4 started for {self.timeframes}")
            
            # Schedule jobs based on self.timeframes
            if "15m" in self.timeframes:
                schedule.every(15).minutes.do(self.job, "15m")
                self.job("15m") # Run immediately
            
            if "1h" in self.timeframes:
                schedule.every(1).hours.do(self.job, "1h")
                self.job("1h") # Run immediately
            
            if "4h" in self.timeframes:
                schedule.every(4).hours.do(self.job, "4h")
                self.job("4h") # Run immediately
            
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        
        self.thread = threading.Thread(target=run_schedule, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Scheduler stopped.")
