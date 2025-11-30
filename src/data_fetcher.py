import requests
import pandas as pd
import os

def fetch_binance_data(symbol="BTCUSDT", interval="1d", limit=1000):
    """
    Fetches historical kline data from Binance and saves it to data/raw.
    """
    print(f"Fetching {limit} rows of {interval} data for {symbol}...")
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_base_volume", "taker_quote_volume", "ignore"
        ]
        
        df = pd.DataFrame(data, columns=cols)
        
        # --- ROBUST PATH FIX ---
        # 1. Get the folder where THIS script (data_fetcher.py) lives (src/)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up one level to the project root (Crypto_Classifier/)
        project_root = os.path.dirname(current_script_dir)
        
        # 3. Construct the path to data/raw
        output_dir = os.path.join(project_root, 'data', 'raw')
        
        # 4. Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # -----------------------

        output_path = os.path.join(output_dir, "raw_data.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Data saved to {output_path}")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

if __name__ == "__main__":
    fetch_binance_data()


