import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import time

st.set_page_config(layout="wide", page_title="Profitability Validation")

def main():
    st.title("📊 Profitability Validation Dashboard")
    
    st.markdown("Watching `data/walk_forward_results.csv` for updates from Walk-Forward Validation...")
    
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            if os.path.exists("data/walk_forward_results.csv"):
                try:
                    df = pd.read_csv("data/walk_forward_results.csv")
                    if not df.empty:
                        # Summary Metrics
                        avg_acc = df['accuracy'].mean()
                        avg_profit = df['profit_pct'].mean()
                        total_return = (1 + df['profit_pct']).prod() - 1
                        avg_sharpe = df['sharpe'].mean()
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Avg Accuracy", f"{avg_acc:.1%}")
                        c2.metric("Avg Monthly Profit", f"{avg_profit:.2%}")
                        c3.metric("Total Backtest Return", f"{total_return:.2%}")
                        c4.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
                        
                        # Charts
                        col1, col2 = st.columns(2)
                        
                        # Profit per Fold
                        with col1:
                            st.subheader("Monthly Profitability")
                            fig_profit = go.Figure()
                            colors = ['green' if x > 0 else 'red' for x in df['profit_pct']]
                            fig_profit.add_trace(go.Bar(
                                x=df['fold'], 
                                y=df['profit_pct'],
                                marker_color=colors,
                                name='Profit'
                            ))
                            st.plotly_chart(fig_profit, use_container_width=True, key=f"profit_{time.time()}")
                            
                        # Equity Curve (Compounded)
                        with col2:
                            st.subheader("Simulated Equity Curve")
                            df['cum_return'] = (1 + df['profit_pct']).cumprod()
                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(
                                x=df['fold'], 
                                y=df['cum_return'],
                                mode='lines+markers',
                                fill='tozeroy',
                                name='Equity'
                            ))
                            st.plotly_chart(fig_eq, use_container_width=True, key=f"equity_{time.time()}")
                            
                        # Data Table
                        st.dataframe(df.style.format({
                            'accuracy': '{:.1%}',
                            'profit_pct': '{:.2%}',
                            'max_dd': '{:.2%}',
                            'sharpe': '{:.2f}'
                        }))
                    else:
                        st.info("Results file is empty.")
                except Exception as e:
                    st.error(f"Error reading results: {e}")
            else:
                st.warning("Waiting for results file to be created...")
                
        time.sleep(10)

if __name__ == "__main__":
    main()
