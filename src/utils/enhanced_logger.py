"""
Enhanced logging system for crypto predictor
Separate logs for each model with comprehensive information
"""

import pandas as pd
import os
from datetime import datetime
import json

class EnhancedLogger:
    """
    Comprehensive logging for all model predictions.
    Creates separate logs per timeframe with full prediction details.
    """

    def __init__(self, log_dir="logs/predictions"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Define log files for each timeframe
        self.log_files = {
            '15m': os.path.join(log_dir, 'predictions_15m.csv'),
            '1h': os.path.join(log_dir, 'predictions_1h.csv'),
            '4h': os.path.join(log_dir, 'predictions_4h.csv'),
            'all': os.path.join(log_dir, 'predictions_all.csv')
        }

        # Initialize log files with headers
        self.init_logs()

    def init_logs(self):
        """Initialize CSV log files with comprehensive headers"""
        columns = [
            'timestamp',
            'date',
            'timeframe',

            # Prediction outputs
            'predicted_direction',
            'confidence',
            'uncertainty',
            'predicted_price',
            'predicted_return',

            # Market context
            'current_price',
            'regime',

            # Probabilities
            'prob_flat',
            'prob_down',
            'prob_up',

            # Quality control
            'gate1_passed',  # Uncertainty check
            'gate2_passed',  # Confidence check
            'gate3_applied', # Trend forcing
            'caution_flag',

            # Model internals
            'mc_samples',
            'epistemic_uncertainty',

            # Validation (filled later)
            'actual_price',
            'actual_direction',
            'outcome',
            'price_error_pct',

            # Trading
            'trade_signal',
            'trade_executed'
        ]

        # Create each log file if it doesn't exist
        for timeframe, filepath in self.log_files.items():
            if not os.path.exists(filepath):
                df = pd.DataFrame(columns=columns)
                df.to_csv(filepath, index=False)

    def log_prediction(self, timeframe, prediction, current_price,
                      gate_info=None, trade_info=None):
        """
        Log a comprehensive prediction record.

        Args:
            timeframe: '15m', '1h', or '4h'
            prediction: Full prediction dict from PredictorV3
            current_price: Current market price
            gate_info: Dict with gate1_passed, gate2_passed, gate3_applied
            trade_info: Dict with trade_signal, trade_executed
        """
        timestamp = int(datetime.now().timestamp())

        # Extract probabilities
        probs = prediction.get('probabilities', [0, 0, 0])
        if len(probs) < 3:
            probs = [0, 0, 0]

        # Build comprehensive record
        record = {
            'timestamp': timestamp,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframe': timeframe,

            # Predictions
            'predicted_direction': prediction['predicted_direction'],
            'confidence': prediction['confidence'],
            'uncertainty': prediction['uncertainty'],
            'predicted_price': prediction['predicted_price'],
            'predicted_return': prediction['predicted_return'],

            # Context
            'current_price': current_price,
            'regime': prediction['regime'],

            # Probabilities
            'prob_flat': probs[0],
            'prob_down': probs[1],
            'prob_up': probs[2],

            # Quality control
            'gate1_passed': gate_info.get('gate1_passed', True) if gate_info else True,
            'gate2_passed': gate_info.get('gate2_passed', True) if gate_info else True,
            'gate3_applied': gate_info.get('gate3_applied', False) if gate_info else False,
            'caution_flag': prediction.get('caution', False),

            # Model internals
            'mc_samples': 10,  # Fixed in current implementation
            'epistemic_uncertainty': prediction['uncertainty'],

            # Validation (to be filled)
            'actual_price': 0.0,
            'actual_direction': '',
            'outcome': 'PENDING',
            'price_error_pct': 0.0,

            # Trading
            'trade_signal': trade_info.get('signal', 'HOLD') if trade_info else 'HOLD',
            'trade_executed': trade_info.get('executed', False) if trade_info else False
        }

        # Log to timeframe-specific file
        self._append_record(self.log_files[timeframe], record)

        # Also log to combined file
        self._append_record(self.log_files['all'], record)

    def _append_record(self, filepath, record):
        """Append a record to a CSV file"""
        try:
            df = pd.read_csv(filepath)
        except:
            self.init_logs()
            df = pd.read_csv(filepath)

        # Use pd.concat with explicit dtype casting to avoid FutureWarning
        new_df = pd.DataFrame([record])
        # Ensure columns match
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[df.columns]

        df = pd.concat([df, new_df], ignore_index=True, copy=False)
        df.to_csv(filepath, index=False)

    def update_validation(self, timeframe, timestamp, actual_price, actual_direction, outcome):
        """
        Update a prediction with validation results.

        Args:
            timeframe: Which log file to update
            timestamp: Timestamp of the prediction to update
            actual_price: Actual price after timeframe elapsed
            actual_direction: Calculated actual direction
            outcome: 'GOOD', 'BAD', or 'PERFECT'
        """
        # Update timeframe-specific log
        self._update_log_file(self.log_files[timeframe], timestamp,
                             actual_price, actual_direction, outcome)

        # Update combined log
        self._update_log_file(self.log_files['all'], timestamp,
                             actual_price, actual_direction, outcome)

    def _update_log_file(self, filepath, timestamp, actual_price, actual_direction, outcome):
        """Update a specific log file with validation data"""
        try:
            try:
                df = pd.read_csv(filepath)
            except pd.errors.EmptyDataError:
                # File is empty, nothing to update
                return

            # Find the prediction
            mask = df['timestamp'] == timestamp

            if mask.any():
                # Calculate price error
                predicted_price = df.loc[mask, 'predicted_price'].values[0]
                price_error_pct = ((actual_price - predicted_price) / predicted_price) * 100

                # Update fields
                df.loc[mask, 'actual_price'] = actual_price
                df.loc[mask, 'actual_direction'] = actual_direction
                df.loc[mask, 'outcome'] = outcome
                df.loc[mask, 'price_error_pct'] = price_error_pct

                # Save
                df.to_csv(filepath, index=False)
        except Exception as e:
            print(f"Error updating validation in {filepath}: {e}")

    def get_stats(self, timeframe='all'):
        """
        Get statistics for a specific timeframe.

        Args:
            timeframe: '15m', '1h', '4h', or 'all'

        Returns:
            dict: Statistics including accuracy, confidence, etc.
        """
        filepath = self.log_files.get(timeframe, self.log_files['all'])

        try:
            df = pd.read_csv(filepath)

            if len(df) == 0:
                return {'error': 'No predictions logged'}

            # Calculate statistics
            validated = df[df['outcome'].isin(['GOOD', 'BAD', 'PERFECT'])]

            stats = {
                'total_predictions': len(df),
                'validated_predictions': len(validated),
                'pending_predictions': len(df[df['outcome'] == 'PENDING']),

                'avg_confidence': df['confidence'].mean(),
                'avg_uncertainty': df['uncertainty'].mean(),

                'direction_distribution': {
                    'UP': len(df[df['predicted_direction'] == 'UP']),
                    'DOWN': len(df[df['predicted_direction'] == 'DOWN']),
                    'FLAT': len(df[df['predicted_direction'] == 'FLAT'])
                },

                'regime_distribution': df['regime'].value_counts().to_dict(),

                'gate2_rejection_rate': 1 - df['gate2_passed'].mean(),
                'caution_flag_rate': df['caution_flag'].mean(),

                'trades_executed': df['trade_executed'].sum()
            }

            # Accuracy if any validated
            if len(validated) > 0:
                correct = len(validated[validated['outcome'].isin(['GOOD', 'PERFECT'])])
                stats['accuracy'] = correct / len(validated)
                stats['correct_predictions'] = correct

            return stats

        except Exception as e:
            return {'error': str(e)}


class TradeLogger:
    """
    Dedicated trade logging separate from predictions.
    """

    def __init__(self, log_file="logs/trades.csv"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.init_log()

    def init_log(self):
        """Initialize trade log"""
        if not os.path.exists(self.log_file):
            columns = [
                'timestamp',
                'date',
                'action',  # BUY, SELL
                'price',
                'size',
                'value',
                'confidence',
                'timeframe',
                'predicted_direction',
                'balance_before',
                'balance_after',
                'position_before',
                'position_after',
                'equity_before',
                'equity_after',
                'pnl',
                'pnl_pct'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.log_file, index=False)

    def log_trade(self, action, price, size, confidence, timeframe,
                  predicted_direction, portfolio_before, portfolio_after):
        """
        Log a trade execution with full details.
        """
        timestamp = int(datetime.now().timestamp())

        record = {
            'timestamp': timestamp,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
            'price': price,
            'size': size,
            'value': price * size,
            'confidence': confidence,
            'timeframe': timeframe,
            'predicted_direction': predicted_direction,
            'balance_before': portfolio_before['balance'],
            'balance_after': portfolio_after['balance'],
            'position_before': portfolio_before['position'],
            'position_after': portfolio_after['position'],
            'equity_before': portfolio_before['equity'],
            'equity_after': portfolio_after['equity'],
            'pnl': portfolio_after['equity'] - portfolio_before['equity'],
            'pnl_pct': ((portfolio_after['equity'] - portfolio_before['equity']) /
                       portfolio_before['equity'] * 100)
        }

        try:
            df = pd.read_csv(self.log_file)
        except:
            self.init_log()
            df = pd.read_csv(self.log_file)

        # Use pd.concat with explicit column matching to avoid FutureWarning
        new_df = pd.DataFrame([record])
        # Ensure columns match
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[df.columns]

        df = pd.concat([df, new_df], ignore_index=True, copy=False)
        df.to_csv(self.log_file, index=False)


class SystemLogger:
    """
    System events and operational logging.
    """

    def __init__(self, log_file="logs/system_events.log"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_event(self, event_type, message, details=None):
        """
        Log a system event.

        Args:
            event_type: 'INFO', 'WARNING', 'ERROR', 'RETRAIN', etc.
            message: Human-readable message
            details: Optional dict with additional details
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        log_entry = f"[{timestamp}] [{event_type}] {message}"

        if details:
            log_entry += f" | Details: {json.dumps(details)}"

        log_entry += "\n"

        with open(self.log_file, 'a') as f:
            f.write(log_entry)
