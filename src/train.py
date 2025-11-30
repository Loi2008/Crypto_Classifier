import pandas as pd
import numpy as np
import os
import joblib
import warnings

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

def train_models():
    """
    Trains all defined models on the Training set (70% of data)
    and saves EVERY model to the 'models/' folder.
    Does NOT evaluate or compare them.
    """
    # --- 1. SETUP ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    
    input_path = os.path.join(project_root, 'data', 'labeled', 'labeled_data.csv')
    model_dir = os.path.join(project_root, 'models')
    
    # Create models folder if missing
    os.makedirs(model_dir, exist_ok=True)

    print(" Starting Factory Training (Saving ALL models)...")
    
    if not os.path.exists(input_path):
        print(f" Error: File not found at {input_path}")
        return

    # --- 2. PREPARE DATA ---
    df = pd.read_csv(input_path)
    df.dropna(inplace=True)
    
    drop_cols = ['open_time', 'close_time', 'ignore', 'future_return', 'label', 'threshold_buy', 'threshold_sell']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    # Time Series Split: Train on first 70%
    # We reserve the rest (Validation + Test) for the evaluation script
    train_end = int(len(df) * 0.70)
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    print(f"   Training Data: {len(X_train)} rows (First 70%)")
    print(f"   Features: {len(feature_cols)}")

    # --- 3. DEFINE MODELS ---
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, eval_metric='mlogloss', random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, class_weight='balanced'),
        "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.05, depth=6, verbose=0, random_state=42, auto_class_weights='Balanced')
    }

    # --- 4. TRAIN & SAVE LOOP ---
    print(f"\n   🏃 Training {len(models)} models...")
    
    for name, model in models.items():
        try:
            print(f"      - Training {name}...", end=" ")
            model.fit(X_train, y_train)
            
            # Save the individual model
            save_path = os.path.join(model_dir, f"{name}.pkl")
            joblib.dump(model, save_path)
            print(f" Saved to models/{name}.pkl")
            
        except Exception as e:
            print(f" Failed: {e}")

    print(f"\n All models trained and saved to {model_dir}")

if __name__ == "__main__":
    train_models()