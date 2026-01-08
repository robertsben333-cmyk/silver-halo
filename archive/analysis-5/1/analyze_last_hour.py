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
    
    # Fetch full day to be safe, though we only need afternoon
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

def load_model_and_predict(df):
    model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
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

def analyze_last_hour_returns(bars):
    """
    Returns short returns for:
    1. Last Hour (15:00 - 16:00)
    2. Last 30 Mins (15:30 - 16:00)
    """
    if bars is None or bars.empty:
        return None, None
        
    df = bars.set_index('datetime')
    
    # Define target times
    current_date = df.index[0].date()
    ny_tz = pytz.timezone("America/New_York")
    
    t_1500 = ny_tz.localize(datetime.combine(current_date, time(15, 0)))
    t_1530 = ny_tz.localize(datetime.combine(current_date, time(15, 30)))
    t_1600 = ny_tz.localize(datetime.combine(current_date, time(16, 0)))
    
    # Get prices closest to these times (using 'asof' logic or nearest index)
    # Reindexing to minutes makes this easier if data is sparse, but let's just find nearest.
    
    def get_price_at(target_time):
        # Allow +/- 5 mins tolerance? 
        # Actually, let's just take the first bar AFTER or ON the target time.
        # But for 16:00 we want the LAST bar.
        
        subset = df[df.index >= target_time]
        if not subset.empty:
            return subset.iloc[0]['o'] # Open price of that minute
        else:
            # If target is 16:00 and we have data up to 15:59:59, take last close
            if target_time.time() == time(16, 0):
                return df.iloc[-1]['c']
            return None

    p_1500 = get_price_at(t_1500)
    p_1530 = get_price_at(t_1530)
    p_1600 = df.iloc[-1]['c'] # Always take last available close for EOD
    
    # Calculate Short Returns: (Entry - Exit) / Entry
    # Entry = p_1500, Exit = p_1600
    
    ret_60m = None
    if p_1500 and p_1600:
        ret_60m = (p_1500 - p_1600) / p_1500
        
    ret_30m = None
    if p_1530 and p_1600:
        ret_30m = (p_1530 - p_1600) / p_1530
        
    return ret_60m, ret_30m

def main():
    print("Loading analysis history...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"File not found: {INPUT_FILE}")
        return

    df = load_model_and_predict(df)
    
    # Filter for High Conviction
    if 'ModelScore' in df.columns:
        trades = df[df['ModelScore'] > 4.0].copy()
        subset_name = "High Conviction (Pred > 4.0%)"
    else:
        trades = df[df['finalScore'] < 0].copy()
        subset_name = "Baseline (Score < 0)"

    print(f"Analyzing {len(trades)} trades ({subset_name})...")
    
    results = []
    
    for idx, row in trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        bars = get_minute_bars(ticker, date_str)
        
        if bars is not None:
            r60, r30 = analyze_last_hour_returns(bars)
            results.append({
                "ticker": ticker,
                "date": date_str,
                "Return_Last60m": r60,
                "Return_Last30m": r30
            })
            
    res_df = pd.DataFrame(results)
    
    print("\n--- Last Hour Analysis (Short Strategy) ---")
    print(f"Positive Return = Price Dropped (Keep Shorting)")
    print(f"Negative Return = Price Rallied (Should have Exited)")
    
    avg_60 = res_df['Return_Last60m'].mean() * 100
    avg_30 = res_df['Return_Last30m'].mean() * 100
    win_60 = (res_df['Return_Last60m'] > 0).mean() * 100
    win_30 = (res_df['Return_Last30m'] > 0).mean() * 100
    
    print(f"\nLast Hour (15:00-16:00 ET):")
    print(f"Avg Return: {avg_60:.2f}%")
    print(f"Win Rate: {win_60:.1f}%")
    
    print(f"\nLast 30 Mins (15:30-16:00 ET):")
    print(f"Avg Return: {avg_30:.2f}%")
    print(f"Win Rate: {win_30:.1f}%")
    
    # Save results
    res_df.to_csv(os.path.join(OUTPUT_DIR, "last_hour_analysis.csv"), index=False)
    
    # Plot distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(res_df['Return_Last60m']*100, kde=True, color='blue', label='Last 60m')
    sns.histplot(res_df['Return_Last30m']*100, kde=True, color='orange', label='Last 30m')
    plt.title(f"Distribution of Last Hour Short Returns\n{subset_name}")
    plt.xlabel("Return (%)")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(OUTPUT_DIR, "last_hour_distribution.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
