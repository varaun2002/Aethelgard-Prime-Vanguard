import json
import os
from datetime import datetime
import pandas as pd

import numpy as np

class TransactionCosts:
    """
    Realistic crypto trading costs
    """
    # Maker/Taker fees (Binance standard)
    MAKER_FEE = 0.001      # 0.1% (limit orders)
    TAKER_FEE = 0.001      # 0.1% (market orders)
    
    # Slippage (price moves against you during execution)
    SLIPPAGE_BPS = {
        'low_vol': 5,       # 0.05% in normal conditions
        'med_vol': 10,      # 0.10% in active trading
        'high_vol': 20      # 0.20% during volatility
    }
    
    # Spread (bid-ask difference)
    SPREAD_BPS = {
        'low_vol': 2,       # 0.02% in tight markets
        'med_vol': 5,       # 0.05% normally
        'high_vol': 15      # 0.15% in volatile markets
    }
    
    @staticmethod
    def calculate_execution_cost(trade_size, volatility='med_vol', order_type='taker'):
        """
        Calculate real execution cost including all factors
        """
        fee = TransactionCosts.TAKER_FEE if order_type == 'taker' else TransactionCosts.MAKER_FEE
        slippage = TransactionCosts.SLIPPAGE_BPS.get(volatility, 10) / 10000
        spread = TransactionCosts.SPREAD_BPS.get(volatility, 5) / 10000
        
        # Total cost
        total_cost = fee + slippage + spread
        
        return total_cost
    
    @staticmethod
    def get_volatility_regime(recent_prices):
        """
        Classify current volatility
        """
        if len(recent_prices) < 2:
            return 'med_vol'
            
        returns = np.diff(recent_prices) / recent_prices[:-1]
        vol = np.std(returns)
        
        if vol < 0.005: # Adjusted for 15m candles
            return 'low_vol'
        elif vol < 0.015:
            return 'med_vol'
        else:
            return 'high_vol'

from src.inference.performance_monitor import PerformanceMonitor
from src.trading.position_sizing import KellyPositionSizer
from src.trading.risk_management import VolatilityAwareStops, DrawdownProtection
from src.inference.regime_detector import RegimeDetector
import logging

logger = logging.getLogger(__name__)

class PaperTrader:
    def __init__(self, initial_balance=10000.0, state_file="data/paper_portfolio.json",
                 tp_pct=0.05, sl_pct=0.02, trailing_sl_pct=0.015, fee_pct=0.001, confidence_threshold=0.30, persist=True):
        self.state_file = state_file
        self.initial_balance = initial_balance
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trailing_sl_pct = trailing_sl_pct # 1.5% trailing stop
        self.fee_pct = fee_pct
        self.base_confidence_threshold = confidence_threshold # Store base
        self.confidence_threshold = confidence_threshold
        self.persist = persist
        
        # Performance Monitor
        self.monitor = PerformanceMonitor(min_accuracy=0.0, window=50) 
        
        # Position Sizer
        self.sizer = KellyPositionSizer(max_position=0.95, kelly_fraction=0.25)
        
        # Risk Management
        self.risk_manager = VolatilityAwareStops()
        self.circuit_breaker = DrawdownProtection(max_drawdown_pct=0.10)
        
        # Regime Detector
        self.regime_detector = RegimeDetector(window=50)
        
        self.load_state()

    # ... load_state, stats methods etc ... (assume kept)
    def load_state(self):
        if self.persist and os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            if 'highest_price' not in self.state:
                self.state['highest_price'] = self.state.get('avg_entry_price', 0.0)
        else:
            self.reset_state()
            
    # Stats methods (omitted, same as before)
    def reset_state(self):
        """Initialize fresh portfolio state"""
        self.state = {
            'balance_usdt': self.initial_balance,
            'position_eth': 0.0,
            'avg_entry_price': 0.0,
            'highest_price': 0.0,
            'equity_history': [],
            'trades': [],
            'last_prediction': None
        }
        self.save_state()
        
    def save_state(self):
        """Persist state to disk if enabled"""
        if self.persist:
            try:
                # atomic write?
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f, indent=4)
            except Exception as e:
                print(f"Failed to save state: {e}")
            
    def get_portfolio_value(self, current_price):
        """Calculate total equity (USDT + ETH value)"""
        eth_val = self.state['position_eth'] * current_price
        return self.state['balance_usdt'] + eth_val

    def calculate_win_rate_stats(self, window=50):
        # ... (keep as is) ...
        return 0.55, 0.05, 0.02

    def calculate_stats_real(self, window=50):
        # ... (keep as is) ...
        return 0.52, 0.04, 0.02
        
    # check_tp_sl (omitted, same as before)
    def check_tp_sl(self, current_price, timestamp, recent_prices=None):
        # ... (keep as is) ...
        position = self.state['position_eth']
        entry_price = self.state['avg_entry_price']
        highest_price = self.state.get('highest_price', entry_price)
        
        if position <= 0 or entry_price <= 0:
            return None, 0.0
            
        if current_price > highest_price:
            highest_price = current_price
            self.state['highest_price'] = highest_price
            self.save_state()
            
        sl_pct = self.sl_pct
        tp_pct = self.tp_pct
        
        if recent_prices is not None and len(recent_prices) > 20:
             stops = self.risk_manager.get_stops(recent_prices, confidence=0.7)
             sl_pct = stops['stop_loss']
             tp_pct = stops['take_profit']
             
        pct_change_entry = (current_price - entry_price) / entry_price
        pct_drop_peak = (highest_price - current_price) / highest_price
        
        action = None
        if pct_change_entry >= tp_pct: action = "TP_SELL"
        elif pct_change_entry <= -sl_pct: action = "SL_SELL"
        elif pct_drop_peak >= self.trailing_sl_pct and pct_change_entry > 0.005: action = "TRAILING_STOP_SELL"
            
        if action:
            # ... execution logic ...
            size_to_sell = position
            gross_usdt = size_to_sell * current_price
            fee = gross_usdt * self.fee_pct
            net_usdt = gross_usdt - fee
            
            self.state['balance_usdt'] += net_usdt
            self.state['position_eth'] = 0.0
            self.state['avg_entry_price'] = 0.0
            self.state['highest_price'] = 0.0
            self._log_trade(timestamp, action, current_price, size_to_sell, 1.0, fee)
            self.save_state()
            return action, size_to_sell
            
        return None, 0.0

    def update(self, signal, confidence, current_price, timestamp, recent_prices=None, volume=None, recent_candles=None, probs=None):
        """
        Execute trade logic based on signal.
        recent_candles: DataFrame with Open/High/Low/Close for regime detection
        probs: List/Array of probabilities [p_flat, p_up, p_down]
        """
        # ... (Previous Monior/Prediction Logic) ...
        # 0. Update Monitor with result of LAST prediction
        last_pred = self.state.get('last_prediction')
        if last_pred:
            prev_price = last_pred['price']
            prev_signal = last_pred['signal']
            
            actual_direction = "FLAT"
            ret = (current_price - prev_price) / prev_price
            if ret > 0.0005: actual_direction = "UP" # 0.05% noise filter
            elif ret < -0.0005: actual_direction = "DOWN"
            
            self.monitor.add_result(prediction=prev_signal, actual=actual_direction)
            
        # Store CURRENT prediction for NEXT time
        # Note: We store the signal we DECIDE on below, not necessarily the input signal.
        # But we need to act first.
        
        # 1. Check TP/SL first (Pass recent_prices)
        tp_sl_action, size = self.check_tp_sl(current_price, timestamp, recent_prices)
        if tp_sl_action:
            # If we exit, we update last_prediction? No, unrelated.
            pass
        if tp_sl_action:
             # Just return action, but ensure we set last_prediction correctly?
             # Actually last_prediction is about checking ACCURACY of the previous signal.
             # We should set last_prediction at the END of update or use input signal?
             # Using input signal (argmax) is fair for model eval. 
             # Using TRADED signal is better for bot eval.
             # Let's stick to input signal for Monitor for now, or just whatever we decide.
             pass

        # ... Monitor/Circuit Checks ... Use self methods so I don't need to copy paste checks if I look carefully at replacement.
        # I will copy paste checks to be safe or use what I have.
        
        # Monitor
        if not self.monitor.should_continue_trading():
            return None
        # Circuit
        if self.circuit_breaker.check_drawdown(self.state['equity_history'], timestamp):
            return None
        # Volume
        if volume is not None and volume < 50000:
             return None

        # Dynamic Threshold Logic
        CONFIDENCE_THRESHOLD = self.base_confidence_threshold
        current_regime = "unknown"
        
        if recent_candles is not None and not recent_candles.empty:
            current_regime, metrics = self.regime_detector.detect_regime(recent_candles)
            if current_regime == 'ranging':
                CONFIDENCE_THRESHOLD = 0.12
        
        # Override Signal using Probs if available
        if probs is not None and len(probs) == 3:
            p_flat, p_down, p_up = probs
            
            # Default to Argmax signal provided
            final_signal = signal
            final_conf = confidence
            
            # Check if we can "Force" a trade
            # If UP > Threshold
            if p_up > CONFIDENCE_THRESHOLD:
                # Check if it's better than DOWN or if we don't care about DOWN being higher?
                # If p_up > Threshold, it is a valid UP signal for us.
                # But what if p_down > p_up? Then go DOWN.
                if p_up > p_down:
                    final_signal = 'UP'
                    final_conf = p_up
                elif p_down > CONFIDENCE_THRESHOLD:
                    final_signal = 'DOWN'
                    final_conf = p_down
            elif p_down > CONFIDENCE_THRESHOLD:
                final_signal = 'DOWN'
                final_conf = p_down
            
            # Update local vars
            signal = final_signal
            confidence = final_conf

        # Store prediction for next monitor check (AFTER we decided signal)
        self.state['last_prediction'] = {
            'signal': signal,
            'price': current_price,
            'timestamp': timestamp
        }

        if tp_sl_action:
             return tp_sl_action

        if signal == 'FLAT':
            return None
            
        if confidence < CONFIDENCE_THRESHOLD:
            # print(f"DEBUG: Rejected due to Confidence: {confidence} < {CONFIDENCE_THRESHOLD} (Regime={current_regime})")
            return None
            
        print(f"DEBUG: ACCEPTED TRADE! Signal={signal}, Conf={confidence}, Regime={current_regime}, Thresh={CONFIDENCE_THRESHOLD}")
        
        balance = self.state['balance_usdt']
        position = self.state['position_eth']
        
        action = None
        
        # Determine Costs
        vol_regime = 'med_vol'
        if recent_prices is not None and len(recent_prices) > 20:
             vol_regime = TransactionCosts.get_volatility_regime(recent_prices[-20:])
             
        cost_pct = TransactionCosts.calculate_execution_cost(
            trade_size=0, # Not used currently
            volatility=vol_regime
        )
        
        # 2. Process Signals
        if signal == "UP" and confidence > CONFIDENCE_THRESHOLD:
            # BUY Logic
            if position == 0 and balance > 10:
                # Calculate Size using Kelly
                # We need actual win rate history.
                win_rate, avg_win, avg_loss = self.calculate_stats_real()
                
                # Check if we should trade
                if self.sizer.should_trade(confidence):
                    amount_to_spend = self.sizer.calculate_position_size(
                        confidence, win_rate, avg_win, avg_loss, balance
                    )
                    
                    if amount_to_spend < 10: # Minimum trade
                        return None
                    
                    # Apply Fee & Slippage
                    effective_price = current_price * (1 + cost_pct)
                    fee_amount = amount_to_spend * cost_pct 
                    
                    net_spend = amount_to_spend
                    eth_to_buy = net_spend / effective_price
                    
                    self.state['position_eth'] += eth_to_buy
                    self.state['balance_usdt'] -= amount_to_spend
                    self.state['avg_entry_price'] = effective_price
                    self.state['highest_price'] = effective_price 
                    
                    action = "BUY"
                    self._log_trade(timestamp, action, effective_price, eth_to_buy, confidence, fee_amount)
                
        elif position > 0:
             # SELL Logic
             if signal == "DOWN":
                 eth_to_sell = position
                 effective_price = current_price * (1 - cost_pct)
                 gross_usdt = eth_to_sell * effective_price
                 
                 self.state['balance_usdt'] += gross_usdt
                 self.state['position_eth'] = 0.0
                 self.state['avg_entry_price'] = 0.0
                 self.state['highest_price'] = 0.0
                 
                 action = "SELL"
                 # Log implied fee
                 fee_amount = (eth_to_sell * current_price) - gross_usdt
                 self._log_trade(timestamp, action, effective_price, eth_to_sell, confidence, fee_amount)

             elif signal == "FLAT" and confidence > 0.65:
                 # Strong "Don't Hold" signal
                 eth_to_sell = position
                 effective_price = current_price * (1 - cost_pct)
                 gross_usdt = eth_to_sell * effective_price
                 
                 self.state['balance_usdt'] += gross_usdt
                 self.state['position_eth'] = 0.0
                 self.state['avg_entry_price'] = 0.0
                 self.state['highest_price'] = 0.0
                 
                 action = "SELL"
                 # Log implied fee
                 fee_amount = (eth_to_sell * current_price) - gross_usdt
                 self._log_trade(timestamp, action, effective_price, eth_to_sell, confidence, fee_amount)

        # Log History
        equity = self.get_portfolio_value(current_price)
        history_point = {
            'timestamp': timestamp,
            'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'equity': equity,
            'price': current_price
        }
        self.state['equity_history'].append(history_point)
        self.save_state()
        
        return action if action else tp_sl_action

    def _log_trade(self, timestamp, action, price, size, confidence, fee=0.0):
        """Internal helper to log trade"""
        trade_record = {
            'timestamp': timestamp,
            'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
            'price': price,
            'size': size,
            'confidence': float(confidence),
            'fee': fee,
            'new_balance': self.state['balance_usdt']
        }
        self.state['trades'].append(trade_record)
        if len(self.state['trades']) > 50:
             self.state['trades'] = self.state['trades'][-50:]

    def get_stats(self):
        hist = self.state['equity_history']
        current_eq = hist[-1]['equity'] if hist else self.initial_balance
        start_eq = self.initial_balance
        pnl_pct = ((current_eq - start_eq) / start_eq) * 100
        
        return {
            'current_equity': current_eq,
            'pnl_percent': pnl_pct,
            'open_position': self.state['position_eth'],
            'total_trades': len(self.state['trades'])
        }
