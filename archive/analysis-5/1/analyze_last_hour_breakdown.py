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

load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY not found. Please set it in .env")
    sys.exit(1)

INPUT_FILE = "analysis_history.csv"
OUTPUT_DIR = "."
ARTIFACTS_DIR = "../../apps/model_artifacts"

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
        print(f"Error fetching data: {e}")
    return None

def resample_bars(bars, timeframe='5min'):
    df = bars.set_index('datetime').copy()
    resampled = df.resample(timeframe).agg({
        'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'
    }).dropna()
    return resampled.reset_index()

def load_model_and_predict(df):
    model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    if not os.path.exists(model_path): return df
    try:
        model = joblib.load(model_path)
        with open(features_path, 'r') as f: feature_names = json.load(f)
        for col in feature_names:
            if col not in df.columns: df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        X = df[feature_names]
        df['ModelScore'] = model.predict(X)
        print("Model predictions added.")
    except: pass
    return df

def check_heuristic_status(bars_1m):
    """
    Run heuristic until 15:00 ET. 
    Return: 'Triggered' or 'Survivor'
    Logic: 5 Greens + Wait 1 bar + Profit Filter
    """
    if bars_1m is None or bars_1m.empty: return 'Unknown'
    
    entry_price = bars_1m.iloc[0]['o']
    hard_stop_price = entry_price * 1.04
    
    cutoff_time = bars_1m.iloc[0]['datetime'].replace(hour=15, minute=0, second=0, microsecond=0)
    
    bars_5m = resample_bars(bars_1m, '5min')
    
    consecutive_green = 0
    
    for i in range(len(bars_5m)):
        row = bars_5m.iloc[i]
        
        # Stop checking at 15:00
        if row['datetime'] >= cutoff_time:
            break
            
        # Hard Stop Check (if hit, we technically exited, so 'Triggered' or 'Stopped')
        if row['h'] >= hard_stop_price:
            return 'StoppedOut'
            
        is_green = row['c'] > row['o']
        if is_green: consecutive_green += 1
        else: consecutive_green = 0
        
        # Check Heuristic: 5 Greens
        if consecutive_green >= 5:
            # Wait 1 bar (5 mins)
            candidate_idx = i + 1
            if candidate_idx < len(bars_5m):
                # Check Hard Stop during wait
                if bars_5m.iloc[candidate_idx]['h'] >= hard_stop_price:
                     return 'StoppedOut'
                
                # Check Time Limit (don't trigger after 15:00)
                if bars_5m.iloc[candidate_idx]['datetime'] >= cutoff_time:
                    break
                    
                # Check Profit
                candidate_price = bars_5m.iloc[candidate_idx]['c']
                if candidate_price < entry_price:
                    return 'Triggered'
                    
    return 'Survivor'

def get_last_hour_return(bars_1m):
    if bars_1m is None or bars_1m.empty: return None
    
    df = bars_1m.set_index('datetime')
    current_date = df.index[0].date()
    ny_tz = pytz.timezone("America/New_York")
    
    t_1500 = ny_tz.localize(datetime.combine(current_date, time(15, 0)))
    
    # Get Price at 15:00
    subset = df[df.index >= t_1500]
    if subset.empty: return None
    p_1500 = subset.iloc[0]['o']
    
    # Get Price at Close (16:00)
    p_1600 = df.iloc[-1]['c']
    
    # Short Return
    return (p_1500 - p_1600) / p_1500

def main():
    print("Loading analysis history...")
    df = pd.read_csv(INPUT_FILE)
    df = load_model_and_predict(df)
    
    if 'ModelScore' in df.columns:
        trades = df[df['ModelScore'] > 4.0].copy()
    else:
        trades = df[df['finalScore'] < 0].copy()
        
    print(f"Analyzing {len(trades)} High Conviction trades...")
    
    results = []
    
    for idx, row in trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        bars = get_minute_bars(ticker, date_str)
        
        if bars is not None:
            status = check_heuristic_status(bars)
            lh_ret = get_last_hour_return(bars)
            
            if lh_ret is not None:
                results.append({
                    "ticker": ticker,
                    "status": status,
                    "LastHourReturn": lh_ret
                })
                
    res_df = pd.DataFrame(results)
    
    print("\n--- Last Hour Breakdown (15:00-16:00 ET) ---")
    
    # Group by Status
    summary = res_df.groupby('status')['LastHourReturn'].agg(['mean', 'count', 'median'])
    summary['mean'] = summary['mean'] * 100
    summary['median'] = summary['median'] * 100
    
    print(summary)
    
    # Calculate Weighted Average for "Portfolio" (Survivors Only)
    survivors = res_df[res_df['status'] == 'Survivor']
    
    print("\nInterpretation:")
    print(f"Survivors (Held to Close): {survivors['LastHourReturn'].mean()*100:.2f}% Avg Last Hour Return")
    
    # Also calculate All Trades (Baseline)
    print(f"All Trades (Baseline): {res_df['LastHourReturn'].mean()*100:.2f}% Avg Last Hour Return")
    
    # Save
    res_df.to_csv(os.path.join(OUTPUT_DIR, "last_hour_breakdown.csv"), index=False)

if __name__ == "__main__":
    main()
