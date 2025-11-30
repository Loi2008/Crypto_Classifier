
import pandas as pd
import numpy as np
import os

def create_labels(method='dynamic', threshold=0.02, sensitivity=0.5):
    '''
    Generates labels for Buy/Sell/Hold.

    Args:
        method (str): 'fixed' (use hard % number) or 'dynamic' (use volatility).
        threshold (float): Used for 'fixed' method (e.g., 0.02 for 2%).
        sensitivity (float): Used for 'dynamic' method. Multiplier of volatility.
                             0.5 means if price moves > 0.5 * StdDev, it's a signal.
                             Lower = More signals (Fixes Imbalance).
                             Higher = Fewer signals (More precise).
    '''
    # --- PATH SETUP ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    input_path = os.path.join(project_root, 'data', 'feature_engineered', 'feature_engineered_data.csv')
    output_dir = os.path.join(project_root, 'data', 'labeled')
    output_path = os.path.join(output_dir, 'labeled_data.csv')

    print(f" Starting Smart Labeling ({method} mode)...")

    if not os.path.exists(input_path):
        print(f" Error: File not found at {input_path}")
        return None

    df = pd.read_csv(input_path)

    # 1. Calculate Future Return (The Target)
    df['future_return'] = df['close'].pct_change().shift(-1)

    # 2. Define Thresholds
    if method == 'dynamic':
        # Check if volatility exists, if not re-calculate
        if 'volatility' not in df.columns:
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()

        # Dynamic: Threshold changes every day based on market noise
        # We fill NaNs with the mean so we don't lose data
        volatility = df['volatility'].fillna(df['volatility'].mean())

        df['threshold_buy'] = volatility * sensitivity
        df['threshold_sell'] = -volatility * sensitivity
    else:
        # Fixed: Hardcoded number
        df['threshold_buy'] = threshold
        df['threshold_sell'] = -threshold

    # 3. Apply Labels
    # 2 = BUY, 0 = SELL, 1 = HOLD
    conditions = [
        (df['future_return'] > df['threshold_buy']),
        (df['future_return'] < df['threshold_sell'])
    ]

    df['label'] = np.select(conditions, [2, 0], default=1)

    # 4. Cleanup & Save
    df = df.dropna(subset=['future_return'])

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f" Labels Generated. Saved to: {output_path}")
    print("--- Class Distribution ---")
    print(df['label'].value_counts().sort_index())

    return df
