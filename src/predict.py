import joblib
import os
import pandas as pd
import numpy as np

def load_model(model_name):
    """
    Helper function to load a model.
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    
    # We add .pkl if the user forgot it
    if not model_name.endswith('.pkl'):
        model_name = f"{model_name}.pkl"
        
    model_path = os.path.join(project_root, 'models', model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
        
    return joblib.load(model_path)

def predict_from_dataframe(input_df, model_name="best_crypto_model"):
    """
    Predicts using the specified model.
    If no model_name is given, defaults to 'best_crypto_model'.
    """
    # 1. Load Model
    print(f"   🤖 Loading model: {model_name}...")
    model = load_model(model_name)
    
    # 2. Prepare Data
    df_clean = input_df.copy()
    
    # Columns to drop (Targets/Metadata)
    ignore_cols = [
        'open_time', 'close_time', 'ignore', 
        'future_return', 'label', 
        'threshold_buy', 'threshold_sell'
    ]
    
    # Filter columns
    drop_cols = [c for c in ignore_cols if c in df_clean.columns]
    X = df_clean.drop(columns=drop_cols)
    
    # 3. Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    # 4. Extract Confidence
    confidence = [probs[i][pred] for i, pred in enumerate(preds)]
    
    # 5. Attach results
    df_clean['predicted_label'] = preds
    df_clean['confidence'] = confidence
    
    label_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    df_clean['prediction_text'] = df_clean['predicted_label'].map(label_map)
    
    return df_clean

if __name__ == "__main__":
    print("Run this from your notebook!")