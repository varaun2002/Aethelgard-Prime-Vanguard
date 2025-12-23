import json
import os
from datetime import datetime
import pandas as pd

class PaperTrader:
    def __init__(self, initial_balance=10000.0, state_file="data/paper_portfolio.json",
                 tp_pct=0.05, sl_pct=0.02, trailing_sl_pct=0.015, fee_pct=0.001, confidence_threshold=0.45, persist=True):
        self.state_file = state_file
        self.initial_balance = initial_balance
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trailing_sl_pct = trailing_sl_pct # 1.5% trailing stop
        self.fee_pct = fee_pct
        self.confidence_threshold = confidence_threshold
        self.persist = persist
        self.load_state()

    def load_state(self):
        if self.persist and os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            # Migration for existing state files
            if 'highest_price' not in self.state:
                self.state['highest_price'] = self.state.get('avg_entry_price', 0.0)
        else:
            self.reset_state()
    
    def reset_state(self):
        self.state = {
            'balance_usdt': self.initial_balance,
            'position_eth': 0.0,
            'avg_entry_price': 0.0,
            'highest_price': 0.0, # Track peak for trailing stop
            'equity_history': [],
            'trades': []
        }
        if self.persist:
            self.save_state()

    def save_state(self):
        if self.persist:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=4)

    def get_portfolio_value(self, current_price):
        eth_value = self.state['position_eth'] * current_price
        return self.state['balance_usdt'] + eth_value

    def check_tp_sl(self, current_price, timestamp):
        """
        Check if we hit TP, SL, or Trailing Stop.
        """
        position = self.state['position_eth']
        entry_price = self.state['avg_entry_price']
        highest_price = self.state.get('highest_price', entry_price)
        
        if position <= 0 or entry_price <= 0:
            return None, 0.0
            
        # Update Highest Price
        if current_price > highest_price:
            highest_price = current_price
            self.state['highest_price'] = highest_price
            self.save_state()
            
        # Calculate pct change from entry (for fixed rules)
        pct_change_entry = (current_price - entry_price) / entry_price
        
        # Calculate pct dropout from peak (for trailing)
        pct_drop_peak = (highest_price - current_price) / highest_price
        
        action = None
        
        # 1. Take Profit (Fixed Moonbag Target)
        if pct_change_entry >= self.tp_pct:
            action = "TP_SELL"
            
        # 2. Stop Loss (Fixed Hard Floor)
        elif pct_change_entry <= -self.sl_pct:
            action = "SL_SELL"
            
        # 3. Trailing Stop
        elif pct_drop_peak >= self.trailing_sl_pct and pct_change_entry > 0.005: 
            # Only trigger trailing stop if we are at least slightly in profit (>0.5%)
            # This prevents getting chopped out immediately if price wiggles at entry
            action = "TRAILING_STOP_SELL"
            
        if action:
            size_to_sell = position
            gross_usdt = size_to_sell * current_price
            fee = gross_usdt * self.fee_pct
            net_usdt = gross_usdt - fee
            
            self.state['balance_usdt'] += net_usdt
            self.state['position_eth'] = 0.0
            self.state['avg_entry_price'] = 0.0
            self.state['highest_price'] = 0.0
            
            # Log it
            self._log_trade(timestamp, action, current_price, size_to_sell, 1.0, fee)
            self.save_state()
            return action, size_to_sell
            
        return None, 0.0

    def update(self, signal, confidence, current_price, timestamp):
        """
        Execute trade logic based on signal.
        """
        # 1. Check TP/SL first
        tp_sl_action, size = self.check_tp_sl(current_price, timestamp)
        if tp_sl_action:
            return tp_sl_action
            
        # Thresholds
        CONFIDENCE_THRESHOLD = self.confidence_threshold
        MAX_ALLOCATION = 0.95 # Keep 5% for fees/buffer
        
        balance = self.state['balance_usdt']
        position = self.state['position_eth']
        
        action = None
        
        # 2. Process Signals
        if signal == "UP" and confidence > CONFIDENCE_THRESHOLD:
            # BUY Logic
            # Only buy if we have no position (Simple mode) or add to position?
            # Let's stick to simple: Only buy if empty.
            if position == 0 and balance > 10:
                amount_to_spend = balance * MAX_ALLOCATION
                
                # Apply Fee
                fee = amount_to_spend * self.fee_pct
                net_spend = amount_to_spend - fee
                
                eth_to_buy = net_spend / current_price
                
                self.state['position_eth'] += eth_to_buy
                self.state['balance_usdt'] -= amount_to_spend
                self.state['avg_entry_price'] = current_price
                self.state['highest_price'] = current_price # Init peak
                
                action = "BUY"
                self._log_trade(timestamp, action, current_price, eth_to_buy, confidence, fee)
                
        elif (signal == "DOWN" or signal == "FLAT") and position > 0:
             # SELL Logic
             if signal == "DOWN" or (signal == "UP" and confidence < 0.4):
                 eth_to_sell = position
                 gross_usdt = eth_to_sell * current_price
                 
                 # Apply Fee
                 fee = gross_usdt * self.fee_pct
                 net_usdt = gross_usdt - fee
                 
                 self.state['balance_usdt'] += net_usdt
                 self.state['position_eth'] = 0.0
                 self.state['avg_entry_price'] = 0.0
                 self.state['highest_price'] = 0.0
                 
                 action = "SELL"
                 self._log_trade(timestamp, action, current_price, eth_to_sell, confidence, fee)

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
