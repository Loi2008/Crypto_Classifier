import pandas as pd
import ta
import os

def feature_generator(input_path="Crypto_classifier/data/raw/raw_data.csv"):
    """
    Loads raw data, cleans it, adds technical indicators, and generates labels.
    """
    print("Starting feature engineering...")
    
    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"File {input_path} not found. Run data_fetcher.py first.")
        return None
        
    df = pd.read_csv(input_path)
    
    # 2. Cleaning & Typing
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    
    # Drop columns not needed for training
    df = df[["open_time", "open", "high", "low", "close", "volume"]]

    # 3. Feature Engineering (Technical Indicators)
    
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    
    # Simple Moving Averages
    df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    
    # Returns (Features)
    df["pct_change_1d"] = df["close"].pct_change()
    df["pct_change_7d"] = df["close"].pct_change(periods=7)
    df["volatility"] = df["close"].pct_change().rolling(window=20).std()

    # 4. Label Generation (Target)
    # Target: Return of the *next* day. 
    # Logic: > +2% (Buy/2), < -2% (Sell/0), Else (Hold/1)
    
    df["future_return"] = df["close"].pct_change().shift(-1)
    
    def get_label(ret):
        if ret > 0.02:
            return 2  # BUY
        elif ret < -0.02:
            return 0  # SELL
        else:
            return 1  # HOLD

    df["label"] = df["future_return"].apply(get_label)

    # 5. Cleanup
    # Drop rows with NaN (due to rolling windows or shifted targets)
    df.dropna(inplace=True)
    
    # Save Processed Data
    #os.makedirs("data/processed", exist_ok=True)
    output_path = "Crypto_Classifier/data/feature_engineered/feature_engineered_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Feature engineered data saved to {output_path}")
    print(f"Label Distribution:\n{df['label'].value_counts()}")
    
    return df

if __name__ == "__main__":
    feature_generator()