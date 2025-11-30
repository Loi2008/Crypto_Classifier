import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import ta

# Add src to path so we can import our modules
sys.path.append(os.path.abspath('src'))
import data_fetcher
import predict

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Crypto AI Trader")

st.title("🚀 Crypto Buy/Sell Classifier")
st.markdown("End-to-End Machine Learning Capstone Project")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol", value="BTCUSDT")
days_to_fetch = st.sidebar.slider("Days of History", min_value=100, max_value=2000, value=365)

# --- 1. FETCH DATA ---
@st.cache_data(ttl=600) # Cache data for 10 minutes to prevent spamming Binance
def get_data(sym, limit):
    # We use the existing fetcher but return the dataframe directly
    df = data_fetcher.fetch_binance_data(symbol=sym, limit=limit)
    
    # Basic Cleaning (same as processed_data.py)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# --- 2. APPLY FEATURES ON THE FLY ---
def add_features(df):
    df = df.copy()
    # Replicating logic from feature_generator.py
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["volatility"] = df["close"].pct_change().rolling(window=20).std()
    df["pct_change_1d"] = df["close"].pct_change()
    df["pct_change_7d"] = df["close"].pct_change(periods=7)
    df.dropna(inplace=True)
    return df

# Button to trigger analysis
if st.sidebar.button("Analyze Market"):
    with st.spinner(f"Fetching data for {symbol}..."):
        try:
            # 1. Get Data
            raw_df = get_data(symbol, days_to_fetch)
            
            # 2. Engineer Features
            processed_df = add_features(raw_df)
            
            # 3. Predict (Using the last 20 days for the chart, but only today for the big metric)
            # We use our modular predict.py script
            results_df = predict.predict_from_dataframe(processed_df)
            
            # Get latest prediction
            latest = results_df.iloc[-1]
            
            # --- DISPLAY METRICS ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${latest['close']:.2f}")
            
            # Color code the prediction
            pred_color = "off"
            if latest['prediction_text'] == "BUY": pred_color = "normal" # Green
            elif latest['prediction_text'] == "SELL": pred_color = "inverse" # Red
            
            col2.metric("AI Prediction", latest['prediction_text'], delta=None, delta_color=pred_color)
            col3.metric("Confidence", f"{latest['confidence']*100:.1f}%")
            col4.metric("RSI (14)", f"{latest['rsi']:.1f}")

            # --- DISPLAY CHART ---
            st.subheader("Technical Analysis & AI Signals")
            
            # Create interactive Plotly chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=('Price & SMA', 'RSI'),
                                row_width=[0.2, 0.7])

            # Candlestick
            fig.add_trace(go.Candlestick(x=results_df['open_time'],
                            open=results_df['open'], high=results_df['high'],
                            low=results_df['low'], close=results_df['close'], name='Price'), row=1, col=1)

            # SMAs
            fig.add_trace(go.Scatter(x=results_df['open_time'], y=results_df['sma_20'], 
                                     line=dict(color='blue', width=1), name='SMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=results_df['open_time'], y=results_df['sma_50'], 
                                     line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)

            # Buy/Sell Markers on Chart
            buy_signals = results_df[results_df['prediction_text'] == 'BUY']
            sell_signals = results_df[results_df['prediction_text'] == 'SELL']

            fig.add_trace(go.Scatter(x=buy_signals['open_time'], y=buy_signals['low']*0.98,
                                     mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                                     name='AI BUY'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=sell_signals['open_time'], y=sell_signals['high']*1.02,
                                     mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                                     name='AI SELL'), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=results_df['open_time'], y=results_df['rsi'], 
                                     line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
            
            # Add RSI Lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig.update_layout(height=800, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # --- DATA TABLE ---
            st.subheader("Recent Data & Predictions")
            display_cols = ['open_time', 'close', 'rsi', 'prediction_text', 'confidence']
            st.dataframe(results_df[display_cols].tail(10).sort_values('open_time', ascending=False))

        except Exception as e:
            st.error(f"Error analyzing market: {e}")
else:
    st.info("Click 'Analyze Market' in the sidebar to start.")