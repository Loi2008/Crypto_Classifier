
import pandas as pd
import numpy as np
import os
import joblib
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_models():
    # --- 1. SETUP ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    data_path = os.path.join(project_root, 'data', 'labeled', 'labeled_data.csv')
    model_dir = os.path.join(project_root, 'models')

    print(" Starting Model Evaluation Arena...")

    # --- 2. PREPARE TEST DATA ---
    if not os.path.exists(data_path):
        print(f" Error: Data not found at {data_path}")
        # FIX: Return 3 Nones so the notebook doesn't crash
        return None, None, None

    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    drop_cols = ['open_time', 'close_time', 'ignore', 'future_return', 'label', 'threshold_buy', 'threshold_sell']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df['label']

    test_start = int(len(df) * 0.85)
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    print(f"   Testing on {len(X_test)} rows (Last 15%)\n")

    # --- 3. EVALUATION LOOP ---
    best_acc = -1
    best_model_name = None
    best_preds = None

    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f" Error: Models folder not found at {model_dir}")
        return None, None, None

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and f != 'best_crypto_model.pkl']

    if not model_files:
        print(" No models found! Run train.py first.")
        # FIX: Return 3 Nones
        return None, None, None

    print(f"{'MODEL':<20} | {'ACCURACY':<10}")
    print("-" * 35)

    for filename in model_files:
        model_name = filename.replace('.pkl', '')
        model_path = os.path.join(model_dir, filename)

        try:
            model = joblib.load(model_path)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            print(f"{model_name:<20} | {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model_name = model_name
                best_preds = preds

        except Exception as e:
            print(f" Error loading {model_name}: {e}")

    # --- 4. SHOW WINNER ---
    if best_model_name is None:
        print(" No valid models could be evaluated.")
        return None, None, None

    print("-" * 35)
    print(f" WINNER: {best_model_name} (Accuracy: {best_acc:.4f})")

    # --- 5. SAVE BEST MODEL ---
    src_file = os.path.join(model_dir, f"{best_model_name}.pkl")
    dst_file = os.path.join(model_dir, "best_crypto_model.pkl")
    try:
        shutil.copyfile(src_file, dst_file)
        print(f" Copied winner to: {dst_file}")
    except Exception as e:
        print(f" Could not copy best model: {e}")

    # --- 6. TEXT REPORT ---
    print("\n Classification Report (Winner):")
    print(classification_report(y_test, best_preds, target_names=['SELL', 'HOLD', 'BUY']))

    return best_model_name, y_test, best_preds

if __name__ == "__main__":
    winner_name, y_true, y_pred = evaluate_models()

    if winner_name is not None:
        print("\n Generating Confusion Matrix Plot...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=['Pred SELL', 'Pred HOLD', 'Pred BUY'],
                    yticklabels=['Act SELL', 'Act HOLD', 'Act BUY'])
        plt.title(f'Confusion Matrix: {winner_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
