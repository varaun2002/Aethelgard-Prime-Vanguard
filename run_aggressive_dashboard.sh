#!/bin/bash
echo "Starting Vanguard Aggressive Dashboard (Watcher Disabled)..."
streamlit run dashboard/dashboard_15m_aggressive.py --server.port 8502 --server.fileWatcherType=none --server.headless=true
