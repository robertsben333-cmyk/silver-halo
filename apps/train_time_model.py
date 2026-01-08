import pandas as pd
import requests
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time, timedelta
import pytz
from dotenv import load_dotenv
import numpy as np
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY not found. Please set it in .env")
    sys.exit(1)

INPUT_FILE = "analysis-5/1/analysis_history.csv"
ARTIFACTS_DIR = "apps/model_artifacts"
OUTPUT_DIR = "analysis-5/1"

def get_minute_bars(ticker, date_str):
    ny_tz = pytz.timezone("America/New_York")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    start_dt = ny_tz.localize(datetime.combine(dt.date(), time(9, 30)))
    end_dt = ny_tz.localize(datetime.combine(dt.date(), time(16, 0)))
    
    ts_start = int(start_dt.timestamp() * 1000)
    ts_end = int(end_dt.timestamp() * 1000)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{ts_start}/{ts_end}"
    params = {
        "apiKey": API_KEY,
        "limit": 50000, 
        "sort": "asc",
        "adjusted": "true"
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                df['datetime'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ny_tz)
                return df
    except Exception as e:
        print(f"Error fetching data for {ticker} on {date_str}: {e}")
        
    return None

def load_model_and_predict(df):
    model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Model artifacts not found.")
        return df

    try:
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
            
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        X = df[feature_names]
        df['ModelScore'] = model.predict(X)
        print("Model predictions added.")
        
    except Exception as e:
        print(f"Error predicting with model: {e}")
        
    return df

def get_optimal_hold_time(bars_1m):
    """
    Find the time (in hours from open) that maximizes profit.
    """
    if bars_1m is None or bars_1m.empty:
        return 0, 0.0
        
    entry_price = bars_1m.iloc[0]['o']
    start_time = bars_1m.iloc[0]['datetime']
    
    # Calculate return at every minute
    # (Short: (Entry - Close) / Entry)
    prices = bars_1m['c'].values
    returns = (entry_price - prices) / entry_price
    
    # Find max return index
    max_idx = np.argmax(returns)
    max_return = returns[max_idx]
    
    # Calculate time delta in hours
    peak_time = bars_1m.iloc[max_idx]['datetime']
    duration_seconds = (peak_time - start_time).total_seconds()
    duration_hours = duration_seconds / 3600.0
    
    return duration_hours, max_return

def main():
    print("Loading analysis history...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"File not found: {INPUT_FILE}")
        return

    # 1. Add ModelScore
    df = load_model_and_predict(df)
    
    if 'ModelScore' not in df.columns:
        print("Error: Could not calculate ModelScore. Exiting.")
        return

    print(f"Analyzing {len(df)} trades to find optimal holding times...")
    
    # 2. Collect Data
    training_data = [] # List of dicts {ModelScore, OptimalTime}
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        score = row['ModelScore']
        
        bars = get_minute_bars(ticker, date_str)
        if bars is not None and not bars.empty:
            opt_time, max_ret = get_optimal_hold_time(bars)
            training_data.append({
                "ticker": ticker,
                "ModelScore": score,
                "OptimalTime": opt_time,
                "MaxReturn": max_ret
            })
            
    if not training_data:
        print("No valid data found.")
        return
        
    train_df = pd.DataFrame(training_data)
    
    # 3. Train Linear Regression (Time Model)
    # Filter out outliers? Maybe trades with MaxReturn < 0 (pure loss)?
    # User wants optimal holding time. If it's a loser, optimal time might be 0.
    # Let's keep all data to be robust.
    
    X = train_df[['ModelScore']]
    y = train_df['OptimalTime']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    corr = train_df['ModelScore'].corr(train_df['OptimalTime'])
    
    print("\n--- Time Prediction Model Results ---")
    print(f"Correlation (Score vs Time): {corr:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE: {mae:.2f} hours")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Save Model
    model_save_path = os.path.join(ARTIFACTS_DIR, "time_predictor.joblib")
    joblib.dump(model, model_save_path)
    print(f"Time Model saved to {model_save_path}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=train_df, x='ModelScore', y='OptimalTime', alpha=0.6)
    sns.regplot(data=train_df, x='ModelScore', y='OptimalTime', scatter=False, color='red')
    plt.title(f"Model Score vs Optimal Holding Time\nCorr: {corr:.2f}, Coef: {model.coef_[0]:.2f}")
    plt.xlabel("Predicted Return (Model Score) %")
    plt.ylabel("Optimal Holding Time (Hours)")
    plt.grid(True)
    
    plot_path = os.path.join(OUTPUT_DIR, "time_prediction_model.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
