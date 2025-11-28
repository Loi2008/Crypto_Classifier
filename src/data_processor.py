import pandas as pd
import os
import numpy as np
import ta  
 
def find_processed_data():
    base_dir = "Crypto_Classifier"
    processed_path = os.path.join(base_dir, "data", "processed", "processed_data.csv")
   
    if os.path.exists(processed_path):
        print(f"Found processed data at: {processed_path}")
        return processed_path
   
    # Fallback search
    search_paths = [
        "./Crypto_Classifier/data/processed/processed_data.csv",
        "Crypto_Classifier/data/processed/processed_data.csv",
        "./data/processed/processed_data.csv"
    ]
    for path in search_paths:
        if os.path.exists(path):
            print(f"Found processed data at: {path}")
            return path
   
    raise FileNotFoundError("processed_data.csv not found. Run data_processer.py first!")
def feature_generator(processed_path, symbol="BTCUSDT"):
    
    # Load processed data (same structure as data_processer.py)
    print(f"Loading from: {processed_path}")
    df = pd.read_csv(processed_path)
   
    # Clean column names (matches data_processer.py)
    df.columns = df.columns.str.strip().str.lower()
   
    # Ensure datetime columns (in case CSV stored as strings)
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'], errors='coerce')
   
    # Sort by time and set index
    df = df.sort_values('open_time').reset_index(drop=True)
    df.set_index('open_time', inplace=True)
   
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
   
    # ========== 1. RETURNS ==========
    print("Computing returns...")
    df['return_1d'] = df['close'].pct_change(1)           # 1-day return
    df['return_7d'] = df['close'].pct_change(7)           # 7-day return
    df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(365)  # Annualized volatility
   
    # ========== 2. TECHNICAL INDICATORS (ta library) ==========
    print("Computing technical indicators...")
   
    # RSI (14-period)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
   
    # MACD (12,26,9)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
   
    # Moving Averages
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
   
    # Bollinger Bands (20,2)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb.bollinger_mavg()
   
    # Stochastic Oscillator (14,3)
    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'],
        window=14, smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
   
    # ========== SAVE TO EXISTING feature engineered DIRECTORY ==========
    # Navigate to existing Crypto_Classifier/data/feature_engineered
    base_dir = "Crypto_Classifier"
    feature_engineered_dir = os.path.join(base_dir, "data", "feature_engineered")
   
    # Verify directory exists (as you confirmed)
    if not os.path.exists(feature_engineered_dir):
        raise FileNotFoundError(f"eng_data directory not found: {feature_engineered_dir}")
   
    output_path = os.path.join(feature_engineered_dir, f"feature_engineered_data.csv")
   
    # Reset index for saving (keep open_time as column)
    df_reset = df.reset_index()
    df_reset.to_csv(output_path, index=False)
   
    print(f"✅ Features saved to: {output_path}")
    print(f"Total columns: {df_reset.shape[1]} (added 17 feature columns)")
   
    # Feature summary
    feature_cols = ['return_1d', 'return_7d', 'volatility_20d', 'rsi', 'macd', 'sma_20', 'bb_width', 'stoch_k']
    print("\nFeature statistics (recent values):")
    print(df_reset[feature_cols].tail().round(4))
   
    return df_reset
 
# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    try:
        processed_file = find_processed_data()
        df_features = feature_generator(processed_file, "BTCUSDT")
        print("\n🎉 Feature engineering pipeline COMPLETE!")
        print("Output: Crypto_Classifier/data/eng_data/BTCUSDT_features.csv")
        print("Next: Train your crypto classifier model!")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Run these first:")
        print("1. python data_fetcher.py")
        print("2. python data_processer.py")
        print("3. pip install ta")
    except Exception as e:
        print(f"Error: {e}")
 