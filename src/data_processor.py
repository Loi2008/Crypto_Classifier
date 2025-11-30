import pandas as pd
import os

def clean_raw_data():
    """
    Reads 'raw_data.csv', converts data types (timestamps & floats),
    and saves the clean version to 'data/processed/processed_data.csv'.
    """
    # --- 1. ROBUST PATH SETUP ---
    # Get the directory where THIS script is located (src/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root (Crypto_Classifier/)
    project_root = os.path.dirname(current_script_dir)
    
    # Define Input and Output paths based on the project root
    raw_path = os.path.join(project_root, 'data', 'raw', 'raw_data.csv')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    output_path = os.path.join(processed_dir, 'processed_data.csv')

    print(f"Reading raw data from: {raw_path}")

    # Check if raw file exists
    if not os.path.exists(raw_path):
        print(f"❌ Error: File not found at {raw_path}")
        print("Please run data_fetcher.py first.")
        return None

    # --- 2. LOAD DATA ---
    df = pd.read_csv(raw_path)

    # --- 3. CLEANING (Type Conversion) ---
    print("Cleaning data types...")

    # Convert timestamps (ms to datetime)
    if 'open_time' in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    if 'close_time' in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')

    # Convert numeric columns to floats (handling errors)
    potential_numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "num_trades", 
        "taker_base_volume", "taker_quote_volume", "ignore"
    ]

    for col in potential_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove any completely empty rows
    df.dropna(how='all', inplace=True)

    # --- 4. SAVE ---
    # Create the folder if it doesn't exist (Crucial fix!)
    os.makedirs(processed_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"✅ Successfully cleaned data.")
    print(f"   Rows: {len(df)}")
    print(f"   Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    clean_raw_data()