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

# Load environment variables
load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY not found. Please set it in .env")
    sys.exit(1)

INPUT_FILE = "analysis-5/1/analysis_history.csv"
OUTPUT_DIR = "analysis-5/1"

def get_minute_bars(ticker, date_str):
    """Fetch all 1-minute bars for a ticker on a specific date (9:30 - 16:00 ET)."""
    ny_tz = pytz.timezone("America/New_York")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Define Market Hours in UTC ms
    start_dt = ny_tz.localize(datetime.combine(dt.date(), time(9, 30)))
    end_dt = ny_tz.localize(datetime.combine(dt.date(), time(16, 0)))
    
    ts_start = int(start_dt.timestamp() * 1000)
    ts_end = int(end_dt.timestamp() * 1000)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{ts_start}/{ts_end}"
    params = {
        "apiKey": API_KEY,
        "limit": 50000, # Max limit to get all minutes
        "sort": "asc",
        "adjusted": "true"
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                # Convert 't' to datetime
                df['datetime'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ny_tz)
                return df
    except Exception as e:
        print(f"Error fetching data for {ticker} on {date_str}: {e}")
        
    return None

def simulate_trade_path(bars, stoploss_pct=0.04):
    """
    Simulate a short trade.
    Entry: Open of the first bar.
    Stoploss: If High > Entry * (1 + stoploss_pct)
    
    Returns a dictionary mapping 'MinutesFromOpen' (30, 60...) to ReturnPct.
    """
    if bars is None or bars.empty:
        return None
    
    entry_price = bars.iloc[0]['o']
    stop_price = entry_price * (1 + stoploss_pct)
    start_time = bars.iloc[0]['datetime']
    
    # Result containers
    timeline_returns = {}
    intervals = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390] # Every 30 mins up to 6.5h
    
    stopped_out = False
    stop_return = -stoploss_pct # Fixed return if stopped out
    
    # We will iterate through bars to check stoploss
    # For efficiency, we can check if High breached stop first
    
    current_interval_idx = 0
    
    for idx, row in bars.iterrows():
        if stopped_out:
            break
            
        # Check Stoploss
        if row['h'] >= stop_price:
            stopped_out = True
            # The return is locked at -4% from this moment onwards
            break
            
        # Check time
        time_elapsed = (row['datetime'] - start_time).total_seconds() / 60.0
        
        # If we passed an interval, record the return based on CLOSE of this bar
        # We handle multiple intervals if bars are sparse (though polygon minute bars are usually contiguous if liquid)
        while current_interval_idx < len(intervals) and time_elapsed >= intervals[current_interval_idx]:
             minutes = intervals[current_interval_idx]
             # Short Return: (Entry - Exit) / Entry
             # Exit = Current Close
             ret = (entry_price - row['c']) / entry_price
             timeline_returns[minutes] = ret
             current_interval_idx += 1
             
    # Fill remaining intervals
    for minutes in intervals:
        if minutes not in timeline_returns:
            if stopped_out:
                timeline_returns[minutes] = stop_return
            else:
                # If data ended before market close (e.g. halted, or today is incomplete), use last know
                # For "optimal timeframe" analysis, missing data is tricky. 
                # Let's assume -4% if stopped, else Last Close if market closed early? 
                # Or just NaN. NaN is safer to avoid skewing.
                # But if we were stopped out, we definitely know the return is -4%.
                pass
                
    return timeline_returns

def load_model_and_predict(df):
    """Load trained model and add 'ModelScore' column to df."""
    import joblib
    import json
    
    ARTIFACTS_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "..", "apps", "model_artifacts")
    model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Model artifacts not found. Skipping model predictions.")
        return df

    try:
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
            
        print(f"Loaded model from {model_path}")
        
        # Prepare Features
        # Need to handle potential missing cols or non-numeric
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0 # Default fallback
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        X = df[feature_names]
        df['ModelScore'] = model.predict(X)
        print("Model predictions added.")
        
    except Exception as e:
        print(f"Error predicting with model: {e}")
        
    return df

def main():
    print("Loading analysis history...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"File not found: {INPUT_FILE}")
        return

    # Add Model Predictions
    df = load_model_and_predict(df)
    
    # Define Subsets for Analysis
    subsets = {
        "Baseline (Score < 0)": df[df['finalScore'] < 0].copy(),
        "Stricter Score (Score < -0.5)": df[df['finalScore'] < -0.5].copy(),
    }
    
    if 'ModelScore' in df.columns:
        # High Conviction Model (Pred > 4.0%)
        # Note: ModelScore is %. e.g. 4.05
        # The threshold we found was ~4.05%
        subsets["Model High Conviction (Pred > 4.0%)"] = df[df['ModelScore'] > 4.0].copy()

    for name, trades in subsets.items():
        print(f"\n--- Analyzing Subset: {name} ({len(trades)} trades) ---")
        if trades.empty:
            print("No trades match criteria.")
            continue
            
        results = []
        for idx, row in trades.iterrows():
            ticker = row['ticker']
            date_str = row['date']
            
            # print(f"Processing {ticker}...", end=" ", flush=True) # Reduce spam
            bars = get_minute_bars(ticker, date_str)
            if bars is None: 
                continue
                
            path_metrics = simulate_trade_path(bars)
            if path_metrics:
                path_metrics['ticker'] = ticker
                path_metrics['date'] = date_str
                results.append(path_metrics)

        if not results:
            print("No results generated.")
            continue
            
        # Aggregate
        res_df = pd.DataFrame(results)
        melted = res_df.melt(id_vars=['ticker', 'date'], var_name='Minutes', value_name='Return')
        melted['ReturnPct'] = melted['Return'] * 100
        melted['Hours'] = melted['Minutes'] / 60.0
        
        stats = melted.groupby('Hours')['ReturnPct'].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
        print(stats)
        
        # Save Stats
        clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("<", "lt").replace(">", "gt")
        stats.to_csv(os.path.join(OUTPUT_DIR, f"exit_stats_{clean_name}.csv"), index=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=melted, x='Hours', y='ReturnPct', estimator='mean', errorbar=('ci', 95), marker='o', label='Mean')
        # Also plot Median?
        # sns.lineplot(data=melted, x='Hours', y='ReturnPct', estimator=np.median, errorbar=None, linestyle='--', label='Median')
        
        plt.title(f"Exit Analysis: {name}\n(N={len(trades)})")
        plt.xlabel("Hours Since Market Open")
        plt.ylabel("Return (%)")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.grid(True)
        plt.legend()
        
        plot_path = os.path.join(OUTPUT_DIR, f"exit_plot_{clean_name}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
