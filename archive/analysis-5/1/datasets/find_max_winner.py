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

def get_minute_bars(ticker, date_str):
    ny_tz = pytz.timezone("America/New_York")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_dt = ny_tz.localize(datetime.combine(dt.date(), time(9, 30)))
    end_dt = ny_tz.localize(datetime.combine(dt.date(), time(16, 0)))
    ts_start = int(start_dt.timestamp() * 1000)
    ts_end = int(end_dt.timestamp() * 1000)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{ts_start}/{ts_end}"
    params = {"apiKey": API_KEY, "limit": 50000, "sort": "asc", "adjusted": "true"}
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("resultsCount", 0) > 0:
            df = pd.DataFrame(data["results"])
            df['datetime'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ny_tz)
            return df
    return None

def resample_bars(bars, timeframe='5min'):
    df = bars.set_index('datetime').copy()
    resampled = df.resample(timeframe).agg({
        'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'
    }).dropna()
    return resampled.reset_index()

def simulate(bars_1m):
    entry_price = bars_1m.iloc[0]['o']
    hard_stop_price = entry_price * 1.04
    bars_5m = resample_bars(bars_1m, '5min')
    consecutive_green = 0
    param = 3
    
    for idx, row in bars_5m.iterrows():
        if row['h'] >= hard_stop_price:
            return -0.04
        
        is_green = row['c'] > row['o']
        if is_green: consecutive_green += 1
        else: consecutive_green = 0
            
        if consecutive_green >= param:
            exit_price = row['c']
            return (entry_price - exit_price) / entry_price
            
    exit_price = bars_5m.iloc[-1]['c']
    return (entry_price - exit_price) / entry_price

def load_model_and_predict(df):
    ARTIFACTS_DIR = "../../apps/model_artifacts"
    model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    if not os.path.exists(model_path): return df
    model = joblib.load(model_path)
    with open(features_path, 'r') as f: feature_names = json.load(f)
    for col in feature_names:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    X = df[feature_names]
    df['ModelScore'] = model.predict(X)
    return df

def main():
    df = pd.read_csv("analysis_history.csv")
    df = load_model_and_predict(df)
    trades = df[df['ModelScore'] > 4.0].copy()
    
    max_ret = -1.0
    max_ticker = ""
    
    for idx, row in trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        bars = get_minute_bars(ticker, date_str)
        if bars is not None:
            ret = simulate(bars)
            if ret > max_ret:
                max_ret = ret
                max_ticker = ticker
                print(f"New Max: {ticker} ({ret*100:.2f}%)")
                
    print(f"FINAL MAX: {max_ticker} -> {max_ret*100:.2f}%")

if __name__ == "__main__":
    main()
