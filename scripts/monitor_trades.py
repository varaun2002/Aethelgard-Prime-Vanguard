import time
import pandas as pd
import os
import sys
from datetime import datetime

# Add project root to path to allow imports from scripts module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate_live_predictions import LogValidator

LOG_FILES = [
    ("Main Bot", "logs/trades.csv"),
    ("Aggressive Bot", "logs/trades_aggressive.csv")
]

def get_last_line(filepath):
    if not os.path.exists(filepath): return 0
    with open(filepath, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_pos = f.tell()
    return last_pos

print("👀 Monitoring for new trades... (Ctrl+C to stop)")
print(f"Time: {datetime.now().strftime('%H:%M:%S')}")

# Get initial file sizes
file_pos = {name: os.path.getsize(path) if os.path.exists(path) else 0 for name, path in LOG_FILES}

# Initialize Validator
validator = LogValidator()
last_validate_time = 0
VALIDATE_INTERVAL = 60 # Validate every 60 seconds

trades_count = 0
MAX_TRADES = 10 # Monitor for a bit longer to be safe

try:
    while trades_count < MAX_TRADES:
        time.sleep(2)
        
        for name, path in LOG_FILES:
            if not os.path.exists(path): continue
            
            current_size = os.path.getsize(path)
            if current_size > file_pos[name]:
                # New data!
                with open(path, 'r') as f:
                    f.seek(file_pos[name])
                    new_lines = f.readlines()
                    
                for line in new_lines:
                    if line.strip():
                        print(f"\n🔔 [{name}] NEW TRADE DETECTED:")
                        print(line.strip())
                        trades_count += 1
                
                file_pos[name] = current_size
        
        # Run Validation periodically
        if time.time() - last_validate_time > VALIDATE_INTERVAL:
            try:
                validator.validate_all()
                last_validate_time = time.time()
            except Exception as e:
                print(f"Validation error: {e}")

except KeyboardInterrupt:
    print("\nStopped.")
