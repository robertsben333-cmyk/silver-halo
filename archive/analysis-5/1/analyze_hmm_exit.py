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
from hmmlearn.hmm import GaussianHMM
import warnings

# Suppress sklearn/hmmlearn warnings
warnings.filterwarnings("ignore")

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

def load_model_and_predict(df):
    """Load trained model and add 'ModelScore' column to df."""
    ARTIFACTS_DIR = "../../apps/model_artifacts"
    model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print(f"Model artifacts not found at {model_path}. Skipping prediction.")
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

def train_hmm(training_data):
    """
    Train a 2-state Gaussian HMM on 1D array of returns.
    Returns (model, reversal_state_index)
    """
    X = training_data.reshape(-1, 1)
    
    # Train HMM with 2 states (Crash vs Reversal/Choppiness)
    print(f"Training HMM on {len(X)} data points (5min returns)...")
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(X)
    
    # Identify states
    means = model.means_.flatten()
    print(f"State Means: {means}")
    
    # Reversal state typically has higher (more positive) mean return than Crash state (negative mean)
    reversal_state = np.argmax(means)
    print(f"Reversal State Index: {reversal_state} (Mean: {means[reversal_state]:.5f})")
    
    return model, reversal_state

def simulate_hmm_strategy(bars_1m, model, reversal_state, threshold=0.9, persistence=1):
    """
    Simulate trading using HMM posterior probability on 5min bars.
    Exit if P(Reversal State) > Threshold for 'persistence' consecutive steps.
    """
    if bars_1m is None or bars_1m.empty:
        return None
        
    entry_price = bars_1m.iloc[0]['o']
    hard_stop_price = entry_price * 1.04
    
    # Resample to 5min for HMM logic
    bars_5m = resample_bars(bars_1m, '5min')
    
    # We iterate 5min bars
    prices = bars_5m['c'].values
    highs = bars_5m['h'].values
    
    returns_history = []
    consecutive_signals = 0
    
    for i in range(1, len(bars_5m)):
        # 1. Check Hard Stop
        if highs[i] >= hard_stop_price:
            return -0.04
            
        # 2. Calculate return
        ret = np.log(prices[i] / prices[i-1])
        returns_history.append(ret)
        
        # 3. Predict Regime
        if len(returns_history) < 5: 
            continue
            
        X_seq = np.array(returns_history).reshape(-1, 1)
        
        try:
            posteriors = model.predict_proba(X_seq)
            prob_reversal = posteriors[-1, reversal_state]
            
            if prob_reversal > threshold:
                consecutive_signals += 1
            else:
                consecutive_signals = 0
                
            # EXIT CONDITION:
            # 1. HMM Signal persists for 'persistence' steps
            # 2. We are currently in profit (Price < Entry) -> User Request
            if consecutive_signals >= persistence and prices[i] < entry_price:
                exit_price = prices[i]
                return (entry_price - exit_price) / entry_price
                
        except Exception:
            pass
            
    # EOD Exit
    exit_price = prices[-1]
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
    
    # 1. Collect Data for Training HMM (5min bars)
    all_returns = []
    trade_bars = {} # Cache bars
    
    print("Fetching data and preparing HMM training set (5min bars)...")
    for idx, row in trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        
        bars = get_minute_bars(ticker, date_str)
        if bars is not None and not bars.empty:
            trade_bars[ticker] = bars
            
            # Resample for training
            b5 = resample_bars(bars, '5min')
            if len(b5) > 1:
                prices = b5['c'].values
                rets = np.log(prices[1:] / prices[:-1])
                all_returns.extend(rets)
            
    if not all_returns:
        print("No return data found.")
        return
        
    training_data = np.array(all_returns)
    training_data = training_data[np.isfinite(training_data)]
    
    # 2. Train HMM
    model, reversal_state = train_hmm(training_data)
    
    # 3. Simulate Strategy
    print("\nSimulating HMM Strategy (5min Returns)...")
    # Test combinations of Threshold and Persistence
    strategies = [
        (0.8, 1), 
        (0.9, 1), 
        (0.95, 1),
        (0.8, 2), # Require 2 consecutive intervals (10 mins)
        (0.9, 2)
    ]
    
    results = {}
    
    for ticker, bars in trade_bars.items():
        for t, p in strategies:
            strat_name = f"HMM (T={t}, P={p})"
            if strat_name not in results: results[strat_name] = []
            
            ret = simulate_hmm_strategy(bars, model, reversal_state, threshold=t, persistence=p)
            results[strat_name].append(ret)
            
    # Compile Stats
    summary_data = []
    for strat, returns in results.items():
        if not returns: continue
        arr = np.array(returns) * 100
        summary_data.append({
            "Strategy": strat,
            "Avg Return": np.mean(arr),
            "Median Return": np.median(arr),
            "Win Rate": np.mean(arr > 0) * 100,
            "Min": np.min(arr),
            "Max": np.max(arr)
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Avg Return", ascending=False)
    
    print("\nHMM Strategy Performance:")
    print(summary_df)
    
    # Save Results
    stats_path = os.path.join(OUTPUT_DIR, "hmm_exit_stats_5min.csv")
    summary_df.to_csv(stats_path, index=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, y='Strategy', x='Avg Return', palette='magma')
    plt.title(f"HMM Exit (5min Returns) Performance\n{subset_name}")
    plt.xlabel("Average Return (%)")
    plt.axvline(5.58, color='green', linestyle='--', label='Consecutive (3x5m): 5.58%')
    plt.legend()
    plt.grid(True, axis='x')
    
    plot_path = os.path.join(OUTPUT_DIR, "hmm_exit_comparison_profit_filter.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
