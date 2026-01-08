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

INPUT_FILE = "analysis_history.csv"
OUTPUT_DIR = "."

def get_minute_bars(ticker, date_str):
    """Fetch all 1-minute bars for a ticker on a specific date (9:30 - 16:00 ET)."""
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

def resample_bars(bars, timeframe='5min'):
    """Resample 1-min bars to X-min bars."""
    df = bars.set_index('datetime').copy()
    resampled = df.resample(timeframe).agg({
        'o': 'first',
        'h': 'max',
        'l': 'min',
        'c': 'last',
        'v': 'sum'
    }).dropna()
    return resampled.reset_index()

def simulate_dynamic_strategy(bars_1m, strategy_type, param):
    """
    Simulate a trade with dynamic exit logic.
    Entry: Open of first bar.
    Fixed Hard Stop: 4%
    
    Strategies:
    - 'consecutive_loss': Exit after 'param' consecutive green bars (Close > Open) on 5min chart.
    - 'trailing_stop': Exit if Price > LowestLow * (1 + param).
    - 'fixed_time': Exit after 'param' minutes (Baseline).
    """
    if bars_1m is None or bars_1m.empty:
        return None
        
    entry_price = bars_1m.iloc[0]['o']
    hard_stop_price = entry_price * 1.04 # 4% stoploss for short
    
    exit_price = None
    exit_reason = "EOD"
    
    # Pre-calculate bars
    bars_5m = resample_bars(bars_1m, '5min') if strategy_type == 'consecutive_loss' else None
    
    # State for Trailing Stop
    lowest_low = entry_price
    
    # State for Consecutive Loss
    consecutive_green = 0
    
    # We iterate minute by minute for accuracy on Hard Stop and Trailing Stop
    
    if strategy_type == 'consecutive_loss':
        for idx, row in bars_5m.iterrows():
            # 1. Check Hard Stop first (Optimization: Did high breach?)
            if row['h'] >= hard_stop_price:
                exit_price = hard_stop_price
                exit_reason = "Hard Stop (-4%)"
                return (entry_price - exit_price) / entry_price
            
            # 2. Check Strategy Signal
            is_green = row['c'] > row['o']
            if is_green:
                consecutive_green += 1
            else:
                consecutive_green = 0
                
            if consecutive_green >= param:
                exit_price = row['c']
                exit_reason = f"Consecutive Losses ({param})"
                return (entry_price - exit_price) / entry_price
                
        # If loop finishes, EOD exit
        exit_price = bars_5m.iloc[-1]['c']
        return (entry_price - exit_price) / entry_price

    elif strategy_type == 'trailing_stop':
        # param is e.g. 0.01 (1%)
        # Trigger trailing stop if High > LowestLow * (1 + param)
        
        for idx, row in bars_1m.iterrows():
            # 1. Check Hard Stop
            if row['h'] >= hard_stop_price:
                return -0.04
            
            # 2. Update Low
            if row['l'] < lowest_low:
                lowest_low = row['l']
                
            # 3. Check Trailing Stop
            rebound_trigger = lowest_low * (1 + param)
            
            if row['h'] >= rebound_trigger:
                exit_price = rebound_trigger
                return (entry_price - exit_price) / entry_price
                
        # EOD
        exit_price = bars_1m.iloc[-1]['c']
        return (entry_price - exit_price) / entry_price
        
    elif strategy_type == 'fixed_time':
        # param = minutes
        # Find bar closest to time
        start_time = bars_1m.iloc[0]['datetime']
        target_time = start_time + timedelta(minutes=param)
        
        # Check stoploss until target time
        # Slice bars up to target time
        onset_bars = bars_1m[bars_1m['datetime'] <= target_time]
        
        # Check max high
        if onset_bars['h'].max() >= hard_stop_price:
            return -0.04
            
        # Else exit at last bar close
        if not onset_bars.empty:
            exit_price = onset_bars.iloc[-1]['c']
            return (entry_price - exit_price) / entry_price
        else:
            return 0.0 # Should not happen

    return 0.0

def load_model_and_predict(df):
    """Load trained model and add 'ModelScore' column to df."""
    import joblib
    import json
    
    # FIXED PATH relative to project root
    ARTIFACTS_DIR = "../../apps/model_artifacts"
    model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    
    print(f"DEBUG: Looking for model at {os.path.abspath(model_path)}")
    
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

    # Filter for High Conviction trades
    if 'ModelScore' in df.columns:
        trades = df[df['ModelScore'] > 4.0].copy()
        subset_name = "High Conviction (Pred > 4.0%)"
    else:
        trades = df[df['finalScore'] < 0].copy()
        subset_name = "Baseline (Score < 0)"
        
    print(f"Analyzing {len(trades)} trades ({subset_name})...")
    
    strategies = [
        ('fixed_time', 120), # Baseline 2h
        ('consecutive_loss', 2), # 2 x 5min green bars
        ('consecutive_loss', 3), # 3 x 5min green bars
        ('trailing_stop', 0.01), # 1% Trailing
        ('trailing_stop', 0.02), # 2% Trailing
        ('trailing_stop', 0.03), # 3% Trailing
    ]
    
    results = {str(s): [] for s in strategies}
    
    for idx, row in trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        
        bars = get_minute_bars(ticker, date_str)
        if bars is None: continue
        
        for strat in strategies:
            ret = simulate_dynamic_strategy(bars, strat[0], strat[1])
            results[str(strat)].append(ret)
            
    # Compile Stats
    summary_data = []
    
    for strat, returns in results.items():
        if not returns: continue
        
        arr = np.array(returns) * 100
        
        # Parse name
        strat_tuple = eval(strat)
        name = f"{strat_tuple[0]} ({strat_tuple[1]})"
        
        summary_data.append({
            "Strategy": name,
            "Avg Return": np.mean(arr),
            "Median Return": np.median(arr),
            "Win Rate": np.mean(arr > 0) * 100,
            "Min": np.min(arr),
            "Max": np.max(arr)
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Avg Return", ascending=False)
    
    print("\nDynamic Exit Strategy Performance:")
    print(summary_df)
    
    stats_path = os.path.join(OUTPUT_DIR, "dynamic_exit_stats.csv")
    summary_df.to_csv(stats_path, index=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, y='Strategy', x='Avg Return', palette='viridis')
    plt.title(f"Avg Return by Exit Strategy\n{subset_name}")
    plt.xlabel("Average Return (%)")
    plt.grid(True, axis='x')
    
    safe_name = subset_name.replace(" ", "_").replace("(", "").replace(")", "").replace(">", "gt").replace("<", "lt").replace("%", "")
    plot_path = os.path.join(OUTPUT_DIR, f"dynamic_exit_comparison_{safe_name}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
