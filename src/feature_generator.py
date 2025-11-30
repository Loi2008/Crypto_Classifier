import pandas as pd
import ta
import os

def feature_generator():
    """
    Loads processed data, adds technical indicators, and saves 
    to 'data/feature_engineered/'.
    
    DOES NOT generate labels (that is now handled by labeler.py).
    """
    # --- 1. ROBUST PATH SETUP ---
    # Get the folder where THIS script is (src/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    
    # INPUT: We try to load the CLEANED data first
    input_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
    
    # OUTPUT: Feature Engineered folder
    output_dir = os.path.join(project_root, 'data', 'feature_engineered')
    output_path = os.path.join(output_dir, 'feature_engineered_data.csv')

    print(f"🚀 Starting feature engineering...")
    print(f"   Reading from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found at {input_path}")
        print("   Please run processed_data.py first.")
        return None
        
    df = pd.read_csv(input_path)
    
    # --- 2. PREP DATA (Fix Types) ---
    # Even though we cleaned it, CSVs lose datetime format, so we fix it again.
    if 'open_time' in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
    
    # Ensure numeric
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only OHLCV for calculation
    df = df[["open_time", "open", "high", "low", "close", "volume"]]

    # --- 3. CALCULATE INDICATORS ---
    print("   Calculating technical indicators...")
    
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    
    # Moving Averages
    df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    df["sma_200"] = ta.trend.SMAIndicator(df["close"], window=200).sma_indicator()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    
    # Volatility (Rolling Std Dev)
    df["volatility"] = df["close"].pct_change().rolling(window=20).std()
    
    # Returns (Features for the model, NOT targets)
    df["pct_change_1d"] = df["close"].pct_change()
    df["pct_change_7d"] = df["close"].pct_change(periods=7)

    # --- 4. CLEANUP & SAVE ---
    # We drop NaNs created by the indicators (like the first 200 rows for SMA200)
    # We do NOT generate labels here anymore.
    
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_len - len(df)
    
    # Create output directory
    #os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✅ Features generated.")
    print(f"   Dropped {dropped_rows} rows (warmup for indicators).")
    print(f"   Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    feature_generator()