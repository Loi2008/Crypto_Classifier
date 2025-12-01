
# 📘 Crypto Buy/Sell Classification Capstone
### End-to-End Machine Learning Pipeline Using Real Binance Data

**Status:** Completed
**UI:** Streamlit Dashboard Included

---

## 1. Project Overview

This project implements a fully automated Machine Learning pipeline that predicts cryptocurrency market movements (Buy, Sell, or Hold). It fetches historical data from the **Binance API**, processes technical indicators, trains multiple ML models (XGBoost, Random Forest, CatBoost, etc.), and evaluates them to select the best performer.

### Key Features
*   **Real-time Data:** Fetches live/historical OHLCV data directly from Binance.
*   **Feature Engineering:** Calculates RSI, MACD, Bollinger Bands, and Moving Averages.
*   **Dynamic Labeling:** Solves class imbalance using volatility-based thresholds.
*   **Model Arena:** Trains 5 different algorithms and automatically promotes the winner.
*   **Interactive UI:** A web-based dashboard to visualize charts and AI signals.

---

## 2. Project Structure

```text
crypto-classifier/
│── data/
│   ├── raw/                  # raw_data.csv
│   ├── processed/            # processed_data.csv
│   ├── feature_engineered/   # feature_engineered_data.csv
│   ├── labeled/              # labeled_data.csv
│
│── notebooks/
│   ├── 01_fetch_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_binance_prediction.ipynb
│
│── src/
│   ├── data_fetcher.py       # Fetches from Binance
│   ├── data_processor.py     # Cleans data types
│   ├── feature_generator.py  # Calculates Indicators (RSI, MACD)
│   ├── labeler.py            # Generates Targets (Buy/Sell)
│   ├── train.py              # Trains all models
│   ├── evaluate.py           # Evaluates and picks winner
│   ├── predict.py            # Prediction Logic
│   ├── app.py                # Streamlit UI
│
│── models/                   # Saved .pkl models
│── README.md
│── requirements.txt
```

3. How to Run the Pipeline
You can run the entire workflow from the terminal using these commands in order:

Phase 1: Data Engineering
code
Bash
# 1. Fetch raw data
```python
src/data_fetcher.py
```
# 2. Clean data types
```python
src/data_processor.py
```
# 3. Generate Technical Indicators
```python
src/feature_generator.py
```
# 4. Generate Targets (Dynamic Imbalance Fix)
```python src/labeler.py
```
Phase 2: Machine Learning
code
Bash
# 5. Train Models 
```python
src/train.py
```
# 6. Evaluate & Select Best Model
```python
src/evaluate.py
```

Phase 3: User Interface
To launch the interactive dashboard:
code
```Bash
streamlit run src/app.py
```
#Notebooks Guide
- 01_fetch_data.ipynb: ETL process visualization.
- 02_feature_engineering.ipynb: Visualization of RSI, MACD, and Price.
- 03_model_training.ipynb: Training logs and validation scores.
- 04_evaluation.ipynb: Confusion matrix and performance report of the winner.
- 05_binance_prediction.ipynb: Testing the model on specific scenarios.
