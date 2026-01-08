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
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY not found. Please set it in .env")
    sys.exit(1)

INPUT_FILE = "analysis_history.csv"
OUTPUT_DIR = "."

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

def resample_bars(bars, timeframe='5min'):
    df = bars.set_index('datetime').copy()
    resampled = df.resample(timeframe).agg({
        'o': 'first',
        'h': 'max',
        'l': 'min',
        'c': 'last',
        'v': 'sum'
    }).dropna()
    return resampled.reset_index()

def load_model_and_predict(df):
    ARTIFACTS_DIR = "../../apps/model_artifacts"
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

def simulate_heuristic_strategy(bars_1m, green_threshold=3, profit_filter=True, wait_period=0):
    """
    Simulate heuristic exit on 5m bars.
    - green_threshold: Number of consecutive green candles to trigger signal.
    - profit_filter: If True, only exit if Price < Entry (Profit > 0).
    - wait_period: Number of 5m bars to wait after signal before checking exit.
    """
    if bars_1m is None or bars_1m.empty:
        return None
        
    entry_price = bars_1m.iloc[0]['o']
    hard_stop_price = entry_price * 1.04
    
    # Resample to 5min
    bars_5m = resample_bars(bars_1m, '5min')
    
    consecutive_green = 0
    
    num_bars = len(bars_5m)
    
    for i in range(num_bars):
        row = bars_5m.iloc[i]
        
        # 1. Hard Stop Check
        if row['h'] >= hard_stop_price:
            return -0.04
            
        # 2. Strategy Logic
        is_green = row['c'] > row['o']
        
        if is_green:
            consecutive_green += 1
        else:
            consecutive_green = 0
            
        # Check Signal
        if consecutive_green >= green_threshold:
            # Signal Triggered.
            
            # Application of Wait Period
            candidate_idx = i + wait_period
            
            if candidate_idx < num_bars:
                candidate_price = bars_5m.iloc[candidate_idx]['c']
                
                # Check Hard Stop during wait period
                if wait_period > 0:
                     if bars_5m.iloc[i+1 : candidate_idx+1]['h'].max() >= hard_stop_price:
                         if candidate_idx > i: 
                            return -0.04

                # Check Profit Filter
                if profit_filter:
                    if candidate_price < entry_price:
                        # PROFITABLE -> EXIT
                        return (entry_price - candidate_price) / entry_price
                    else:
                        # NOT PROFITABLE -> IGNORE SIGNAL, HOLD
                        pass
                else:
                    # NO FILTER -> EXIT REGARDLESS
                    return (entry_price - candidate_price) / entry_price
            else:
                # Wait period goes beyond EOD -> EOD Exit
                pass

    # EOD Exit
    exit_price = bars_5m.iloc[-1]['c']
    return (entry_price - exit_price) / entry_price

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
        print("ModelScore not found. Using Baseline.")
        trades = df[df['finalScore'] < 0].copy()
        subset_name = "Baseline (Score < 0)"

    print(f"Analyzing {len(trades)} trades ({subset_name})...")
    
    # Parameter Grid
    p_greens = [2, 3, 4, 5]
    p_profit = [True, False]
    p_wait = [0, 1, 2] # 0, 5, 10 minutes
    
    results = {}
    
    trade_bars = []
    
    # Pre-fetch bars
    for idx, row in trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        bars = get_minute_bars(ticker, date_str)
        if bars is not None and not bars.empty:
            trade_bars.append({"ticker": ticker, "date": date_str, "bars": bars})

    print(f"Running Grid Search ({len(p_greens)*len(p_profit)*len(p_wait)} combinations)...")
    
    for greens in p_greens:
        for profit in p_profit:
            for wait in p_wait:
                strat_name = f"Count={greens} | Profit={profit} | Wait={wait*5}m"
                results[strat_name] = []
                
                for trade in trade_bars:
                    ret = simulate_heuristic_strategy(
                        trade['bars'], 
                        green_threshold=greens, 
                        profit_filter=profit, 
                        wait_period=wait
                    )
                    results[strat_name].append(ret)
                    
    # Compile Stats
    summary_data = []
    for strat, returns in results.items():
        if not returns: continue
        arr = np.array(returns) * 100
        summary_data.append({
            "Strategy": strat,
            "Greens": int(strat.split("=")[1].split("|")[0]),
            "ProfitFilter": "True" in strat,
            "Wait": int(strat.split("Wait=")[1].replace("m","")),
            "Avg Return": np.mean(arr),
            "Median Return": np.median(arr),
            "Win Rate": np.mean(arr > 0) * 100,
            "Min": np.min(arr),
            "Max": np.max(arr)
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Avg Return", ascending=False)
    
    print("\nTop 20 Heuristic Strategies:")
    print(summary_df.head(20).to_string())
    
    # Save Results
    stats_path = os.path.join(OUTPUT_DIR, "heuristic_exit_stats.csv")
    summary_df.to_csv(stats_path, index=False)
    
    # Plot Heatmap of Avg Return for Best Wait/Profit Combo
    # We'll plot Greens vs Wait for Profit=True (as that's the user preference)
    
    pivot_df = summary_df[summary_df['ProfitFilter'] == True].pivot(index='Greens', columns='Wait', values='Avg Return')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title(f"Avg Return (Profit Filter=True)\nConsecutive Greens vs Wait Period\n{subset_name}")
    
    plot_path = os.path.join(OUTPUT_DIR, "heuristic_exit_heatmap_profit_true.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Plot Comparison Bar Chart for Top 10
    plt.figure(figsize=(12, 8))
    top_10 = summary_df.head(10)
    sns.barplot(data=top_10, y='Strategy', x='Avg Return', palette='viridis')
    plt.title(f"Top 10 Heuristic Strategies\n{subset_name}")
    plt.xlabel("Average Return (%)")
    plt.axvline(5.58, color='red', linestyle='--', label='Benchmark (3x5m NoFilter): 5.58%')
    plt.legend()
    plt.tight_layout()
    
    plot_path_bar = os.path.join(OUTPUT_DIR, "heuristic_exit_top10.png")
    plt.savefig(plot_path_bar)
    print(f"Plot saved to {plot_path_bar}")

if __name__ == "__main__":
    main()
