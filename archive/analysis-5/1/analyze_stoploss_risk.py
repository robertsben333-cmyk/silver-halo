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

def analyze_trade(bars, stoploss_pct=0.04, exit_minutes=120):
    """
    Analyze a single trade path.
    Strategy: Short at Open.
    Check if Stoploss hit within first `exit_minutes`.
    Check Return at `exit_minutes`.
    """
    if bars is None or bars.empty:
        return None
    
    entry_price = bars.iloc[0]['o']
    stop_price = entry_price * (1 + stoploss_pct)
    start_time = bars.iloc[0]['datetime']
    
    # Determine the "Potential" result (if we simply held until exit time)
    # Find bar closest to exit_minutes
    # Calculate time delta for all bars
    bars['minutes_elapsed'] = (bars['datetime'] - start_time).dt.total_seconds() / 60.0
    
    # Filter bars within the window
    window_bars = bars[bars['minutes_elapsed'] <= exit_minutes]
    
    if window_bars.empty:
        return None
        
    # Check if Stoploss triggered in window
    max_high = window_bars['h'].max()
    min_low = window_bars['l'].min() # MFE Capture
    stop_hit = max_high >= stop_price
    
    # Exit Price: Close of the last bar in the window OR closest to exit_minutes if window is partial?
    # Ideally, we want the price EXACTLY at exit_minutes. 
    # If the last bar in window is < exit_minutes (e.g. data ends), use that.
    exit_bar = window_bars.iloc[-1]
    potential_exit_price = exit_bar['c']
    
    # Raw Return (No Stoploss)
    # Short: (Entry - Exit) / Entry
    potential_return = (entry_price - potential_exit_price) / entry_price
    
    # Realized Return (With Stoploss)
    if stop_hit:
        realized_return = -stoploss_pct
    else:
        realized_return = potential_return
        
    return {
        "EntryPrice": entry_price,
        "StopPrice": stop_price,
        "MaxHigh": max_high,
        "MinLow": min_low,
        "PotentialExitPrice": potential_exit_price,
        "PotentialReturn": potential_return,
        "StoppedOut": stop_hit,
        "RealizedReturn": realized_return
    }

def load_model_and_predict(df):
    """Load trained model and add 'ModelScore' column to df."""
    import joblib
    import json
    
    ARTIFACTS_DIR = "../../apps/model_artifacts"
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
    }
    
    if 'ModelScore' in df.columns:
        subsets["Model High Conviction (Pred > 4.0%)"] = df[df['ModelScore'] > 4.0].copy()

    summary_stats = []

    # Collect all trade data once
    processed_trades_data = {} # subset_name -> list of trade dicts
    
    for name, trades in subsets.items():
        print(f"\n--- Processing Data for Subset: {name} ({len(trades)} trades) ---")
        if trades.empty:
            continue
            
        subset_data = []
        for idx, row in trades.iterrows():
            ticker = row['ticker']
            date_str = row['date']
            
            bars = get_minute_bars(ticker, date_str)
            if bars is None: 
                continue
                
            # Use default 4% just to get the base metrics (MaxHigh, MinLow, Prices)
            # The specific stoploss_pct passed here doesn't matter for the raw metrics
            res = analyze_trade(bars, stoploss_pct=0.04) 
            if res:
                res['ticker'] = ticker
                res['date'] = date_str
                # Add ModelScore if available
                if 'ModelScore' in row:
                    res['ModelScore'] = row['ModelScore']
                
                subset_data.append(res)
        
        processed_trades_data[name] = pd.DataFrame(subset_data)

    # Now Analyze Sensitivity for each subset
    for name, df_trades in processed_trades_data.items():
        if df_trades.empty:
            continue
            
        print(f"\n=== Sensitivity Analysis: {name} ===")
        
        # 1. Output MFE Stats & Plot (Only need to do this once per subset)
        df_trades['MaxProfitPct'] = (df_trades['EntryPrice'] - df_trades['MinLow']) / df_trades['EntryPrice'] * 100
        print("\nDistribution of Maximum Favorable Excursion (Max Profit %):")
        print(df_trades['MaxProfitPct'].describe())
        
        clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("<", "lt").replace(">", "gt")
        
        # MFE Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df_trades['MaxProfitPct'], bins=20, kde=True, element="step")
        plt.title(f"Distribution of Max Potential Profit (2h window)\nSubset: {name}")
        plt.xlabel("Max Profit (%)")
        plt.savefig(os.path.join(OUTPUT_DIR, f"mfe_dist_{clean_name}.png"))
        plt.close()
        
        # Model Score vs MFE
        if 'ModelScore' in df_trades.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_trades, x='ModelScore', y='MaxProfitPct')
            sns.regplot(data=df_trades, x='ModelScore', y='MaxProfitPct', scatter=False, color='red')
            corr = df_trades['ModelScore'].corr(df_trades['MaxProfitPct'])
            plt.title(f"Model Score vs Max Profit %\nSubset: {name} (Corr: {corr:.2f})")
            plt.savefig(os.path.join(OUTPUT_DIR, f"model_vs_mfe_{clean_name}.png"))
            plt.close()

        # 2. Sensitivity Loop
        thresholds = np.arange(0.01, 0.155, 0.005) # 1% to 15% step 0.5%
        sensitivity_results = []
        
        for stop_pct in thresholds:
            # Calculate realized return for this specific threshold
            # Stop Price = Entry * (1 + stop_pct)
            # Stop Hit if MaxHigh >= Stop Price
            
            stop_price_series = df_trades['EntryPrice'] * (1 + stop_pct)
            is_stopped = df_trades['MaxHigh'] >= stop_price_series
            
            # Calculate Return
            # If stopped: -stop_pct
            # If not stopped: PotentialReturn
            
            realized_rets = np.where(is_stopped, -stop_pct, df_trades['PotentialReturn'])
            
            # Convert to %
            realized_rets_pct = realized_rets * 100
            
            avg_ret = np.mean(realized_rets_pct)
            median_ret = np.median(realized_rets_pct)
            win_rate = np.mean(realized_rets_pct > 0) * 100
            
            # Stopped Winner Rate:
            # A "Winner" is trade where PotentialReturn > 0
            # A "Stopped Winner" is PotentialReturn > 0 AND is_stopped is True
            potential_winners = df_trades['PotentialReturn'] > 0
            total_winners = potential_winners.sum()
            stopped_winners = (potential_winners & is_stopped).sum()
            
            stopped_winner_rate = (stopped_winners / total_winners * 100) if total_winners > 0 else 0
            
            sensitivity_results.append({
                "ThresholdPct": stop_pct * 100,
                "AvgReturn": avg_ret,
                "MedianReturn": median_ret,
                "WinRate": win_rate,
                "StoppedWinnerRate": stopped_winner_rate
            })
            
        stats_df = pd.DataFrame(sensitivity_results)
        print(stats_df.round(2).head()) # Preview
        
        stats_path = os.path.join(OUTPUT_DIR, f"sensitivity_stats_{clean_name}.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Sensitivity stats saved to {stats_path}")
        
        # Plot Sensitivity Curves
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        color = 'tab:blue'
        ax1.set_xlabel('Stoploss Threshold (%)')
        ax1.set_ylabel('Average Return (%)', color=color)
        ax1.plot(stats_df['ThresholdPct'], stats_df['AvgReturn'], color=color, marker='o', label='Avg Return')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        
        # Optimal Point Annotation
        max_ret_idx = stats_df['AvgReturn'].idxmax()
        optimal_stop = stats_df.loc[max_ret_idx, 'ThresholdPct']
        max_ret = stats_df.loc[max_ret_idx, 'AvgReturn']
        ax1.annotate(f'Optimal: {optimal_stop:.1f}% (Ret: {max_ret:.1f}%)', 
                     xy=(optimal_stop, max_ret), 
                     xytext=(optimal_stop, max_ret + 1),
                     arrowprops=dict(facecolor='black', shrink=0.05))

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Win Rate (%)', color=color)  # we already handled the x-label with ax1
        ax2.plot(stats_df['ThresholdPct'], stats_df['WinRate'], color=color, linestyle='--', label='Win Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"Stoploss Sensitivity Analysis: {name}")
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        plot_path = os.path.join(OUTPUT_DIR, f"sensitivity_plot_{clean_name}.png")
        plt.savefig(plot_path)
        print(f"Sensitivity Plot saved to {plot_path}")
        plt.close()

    # Save summary
    if summary_stats:
        pd.DataFrame(summary_stats).to_csv(os.path.join(OUTPUT_DIR, "stoploss_risk_summary.csv"), index=False)

if __name__ == "__main__":
    main()
