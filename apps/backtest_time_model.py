import pandas as pd
import requests
import os
import sys
from datetime import datetime, time, timedelta
import pytz
from dotenv import load_dotenv
import numpy as np
import joblib
import json

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
    params = {"apiKey": API_KEY, "limit": 50000, "sort": "asc", "adjusted": "true"}
    
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

def load_models(df):
    # Load Return Predictor
    ret_model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    
    # Load Time Predictor
    time_model_path = os.path.join(ARTIFACTS_DIR, "time_predictor.joblib")
    
    if not os.path.exists(ret_model_path) or not os.path.exists(time_model_path):
        print("Models not found.")
        return df, None

    try:
        # 1. Predict Score
        ret_model = joblib.load(ret_model_path)
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
            
        for col in feature_names:
            if col not in df.columns: df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        X = df[feature_names]
        df['ModelScore'] = ret_model.predict(X)
        
        # 2. Predict Optimal Time
        time_model = joblib.load(time_model_path)
        # Time model was trained on simple 1D array of scores?
        # Let's check training script... yes: X = train_df[['ModelScore']]
        
        X_time = df[['ModelScore']]
        df['PredictedHoldTime'] = time_model.predict(X_time)
        
        print("Model predictions added (Score + HoldTime).")
        return df, time_model
        
    except Exception as e:
        print(f"Error loading/predicting: {e}")
        return df, None

def simulate_trade(bars, hold_hours):
    if bars is None or bars.empty: return 0.0
    
    entry_price = bars.iloc[0]['o']
    start_time = bars.iloc[0]['datetime']
    hard_stop_price = entry_price * 1.04
    
    # Calculate Exit Time
    exit_time = start_time + timedelta(hours=hold_hours)
    
    # Filter bars up to Exit Time
    trade_bars = bars[bars['datetime'] <= exit_time]
    
    # Check Hard Stop
    if trade_bars['h'].max() >= hard_stop_price:
        return -0.04
        
    # Exit at last available bar (if earlier than target, typically means EOD or end of data)
    # If target is beyond EOD (e.g. 6.5h), we just exit at EOD.
    exit_price = trade_bars.iloc[-1]['c']
    return (entry_price - exit_price) / entry_price

def main():
    print("Loading analysis history...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"File not found: {INPUT_FILE}")
        return

    df, _ = load_models(df)
    
    if 'PredictedHoldTime' not in df.columns:
        print("Prediction failed.")
        return

    # Filter High Conviction
    target_trades = df[df['ModelScore'] > 4.0].copy()
    print(f"Analyzing {len(target_trades)} High Conviction Trades...")
    
    returns = []
    
    for idx, row in target_trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        hold_time = row['PredictedHoldTime']
        
        # Clamp hold time? 
        # Market open 6.5 hours. 
        if hold_time < 0.1: hold_time = 0.1
        if hold_time > 6.4: hold_time = 6.4
        
        bars = get_minute_bars(ticker, date_str)
        if bars is not None:
            ret = simulate_trade(bars, hold_time)
            returns.append(ret)
            
    if not returns:
        print("No returns calculated.")
        return
        
    arr = np.array(returns) * 100
    print("\n--- Linear Time Model Results ---")
    print(f"Avg Return: {np.mean(arr):.2f}%")
    print(f"Median Return: {np.median(arr):.2f}%")
    print(f"Win Rate: {np.mean(arr > 0) * 100:.1f}%")
    print(f"Min: {np.min(arr):.2f}%")
    print(f"Max: {np.max(arr):.2f}%")

if __name__ == "__main__":
    main()
