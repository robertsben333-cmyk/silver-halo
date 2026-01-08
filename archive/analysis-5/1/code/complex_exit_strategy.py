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

# Load environment variables
load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    # Try looking in parent dirs
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
    load_dotenv(dotenv_path)
    API_KEY = os.environ.get("POLYGON_API_KEY")

if not API_KEY:
    print("Error: POLYGON_API_KEY not found. Please set it in .env")
    sys.exit(1)

INPUT_FILE = r"C:\Users\XavierFriesen\.gemini\antigravity\playground\silver-halo\analysis-5\1\analysis_history.csv"
OUTPUT_DIR = r"C:\Users\XavierFriesen\.gemini\antigravity\playground\silver-halo\analysis-5\1"
CACHE_DIR = os.path.join(OUTPUT_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_ARTIFACTS_DIR = r"C:\Users\XavierFriesen\.gemini\antigravity\playground\silver-halo\apps\model_artifacts"

def get_minute_bars(ticker, date_str):
    """Fetch 1-minute bars, using local cache if available."""
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{date_str}.json")
    
    ny_tz = pytz.timezone("America/New_York")
    
    if os.path.exists(cache_file):
        try:
            df = pd.read_json(cache_file, orient='split')
            # Ensure datetime is timezone aware/converted correctly if lost in JSON
            if 'datetime' in df.columns:
                 # JSON serializes dates as strings usually
                 df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(ny_tz)
            return df
        except Exception as e:
            print(f"Error reading cache for {ticker}: {e}")

    # Fetch from Polygon
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
                # t is unix ms
                df['datetime'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ny_tz)
                
                # Save to cache
                # Serialize datetime appropriately
                df.to_json(cache_file, orient='split', date_format='iso')
                return df
    except Exception as e:
        print(f"Error fetching data for {ticker} on {date_str}: {e}")
        
    return None

def load_model_and_add_score(df):
    model_path = os.path.join(MODEL_ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(MODEL_ARTIFACTS_DIR, "model_features.json")
    
    if not os.path.exists(model_path):
        print("Model not found, cannot filter by score.")
        return df
        
    print("Loading model...")
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
        
    # Prepare features
    # Ensure all features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['ModelScore'] = model.predict(df[feature_names])
    return df

def simulate_complex_strategy(bars, check_time_minutes, threshold_pct, end_time_minutes=360, stoploss_pct=0.04):
    """
    Simulate the strategy:
    1. Enter at Open.
    2. Check Stoploss continuously.
    3. At check_time_minutes:
       - Calculate current return.
       - If Return < threshold_pct: EXIT (Close position).
       - Else: CONTINUE (Hold).
    4. If Continued: Exit at end_time_minutes (or earlier if Stoploss hit).
    
    Returns: Final Return of the trade.
    """
    if bars is None or bars.empty:
        return 0.0 # Or None? 0.0 implies break even, prone to error. None is better.
        
    entry_price = bars.iloc[0]['o']
    stop_price = entry_price * (1 + stoploss_pct)
    start_time = bars.iloc[0]['datetime']
    
    # Identify Check Time and End Time timestamps
    # We can iterate or search
    
    trade_return = None
    position_closed = False
    
    # We'll iterate row by row to be accurate with stoploss
    for idx, row in bars.iterrows():
        # Check Stoploss FIRST
        if row['h'] >= stop_price:
            return -stoploss_pct # Stopped out max loss
            
        time_elapsed = (row['datetime'] - start_time).total_seconds() / 60.0
        
        # Check if we passed the Check Time
        # We start checking "At or after" check_time
        if not position_closed and time_elapsed >= check_time_minutes:
             # Calculate Return NOW
             # Short Return: (Entry - CurrentClose) / Entry
             current_return = (entry_price - row['c']) / entry_price
             
             if current_return < (threshold_pct / 100.0):
                 # Threshold NOT met (Return is too low). EXIT.
                 return current_return
             else:
                 # Threshold met (Return is good). Continue holding.
                 # We mark that we passed the check, so we don't check again
                 # But we effectively just "do nothing" and let the loop continue to End Time
                 # Optimization: We could set a flag "passed_check = True" but we need to ensure we don't re-trigger this block
                 # The condition "time_elapsed >= check_time" will be true for ALL subsequent bars.
                 # We only want to check ONCE.
                 
                 # To handle "Checked Once", we can use a flag.
                 # But wait, this is inside a loop.
                 pass 
    
    # Re-logic for efficiency and correctness:
    # 1. Slice bars into "Before Check", "At Check", "After Check".
    
    # Let's find index of check_time
    # We need to find the specific bar that crosses the time threshold
    # Or just iterate.
    
    # Let's clean up logical flow:
    
    passed_check = False
    
    for idx, row in bars.iterrows():
        # SL
        if row['h'] >= stop_price:
            return -stoploss_pct
        
        time_elapsed = (row['datetime'] - start_time).total_seconds() / 60.0
        
        # Check Event
        if not passed_check and time_elapsed >= check_time_minutes:
            passed_check = True
            # Eval threshold
            current_return = (entry_price - row['c']) / entry_price
            if current_return < (threshold_pct / 100.0):
                # Cut losses/weak wins
                return current_return
            # Else continue
            
        # Exit Event (End Time)
        if time_elapsed >= end_time_minutes:
            final_return = (entry_price - row['c']) / entry_price
            return final_return
            
    # If data ends before End Time (and wasn't stopped out), use last close
    last_return = (entry_price - bars.iloc[-1]['c']) / entry_price
    return last_return

def main():
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Input file not found.")
        return

    # Filter with Model
    df = load_model_and_add_score(df)
    
    # Filter for User Criteria: "Predicting returns over 3.9%"
    # 3.9% = 3.9 in model score (based on previous script logic)
    SCORE_THRESHOLD = 3.9
    df_filtered = df[df['ModelScore'] > SCORE_THRESHOLD].copy()
    
    print(f"Trades with Score > {SCORE_THRESHOLD}%: {len(df_filtered)}")
    print(f"Model Score Stats: Min={df['ModelScore'].min():.2f}, Max={df['ModelScore'].max():.2f}, Median={df['ModelScore'].median():.2f}")
    
    if df_filtered.empty:
        print("No trades found.")
        return

    # Pre-fetch bars (Parallelize/Batch if needed, but for now linear is safer for API limits)
    print("Fetching minute bars...")
    bars_dict = {}
    for idx, row in df_filtered.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        key = (ticker, date_str)
        bars = get_minute_bars(ticker, date_str)
        if bars is not None and not bars.empty:
            bars_dict[key] = bars
            
    print(f"Loaded bars for {len(bars_dict)} trades.")
    
    # Sweep Parameters
    # Time: 30, 60, 90, 120 (every 30 mins) up to 240? Strategy says exit at 6h (360).
    # Cutoff checks usually make sense earlier. e.g. 1h, 2h.
    check_times = [60, 90, 120, 150, 180, 240, 300]
    
    # Thresholds: If return < X%, exit.
    # User asked for "much higher %" -> testing strictly higher thresholds
    thresholds = [-2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    
    results = []
    
    # Also calculate Baseline (Hold to 6h)
    print("Simulating baseline (Hold to 6h)...")
    baseline_returns = []
    for (ticker, date_str), bars in bars_dict.items():
        ret = simulate_complex_strategy(bars, check_time_minutes=9999, threshold_pct=-999, end_time_minutes=360)
        baseline_returns.append(ret)
    
    baseline_avg = np.mean(baseline_returns) * 100
    baseline_med = np.median(baseline_returns) * 100
    print(f"Baseline (6h) Mean: {baseline_avg:.2f}%, Median: {baseline_med:.2f}%")
    
    print("Running parameter sweep...")
    for ct in check_times:
        for th in thresholds:
            current_returns = []
            for (ticker, date_str), bars in bars_dict.items():
                ret = simulate_complex_strategy(bars, check_time_minutes=ct, threshold_pct=th, end_time_minutes=360)
                current_returns.append(ret)
            
            if not current_returns:
                continue
                
            avg_ret = np.mean(current_returns) * 100
            med_ret = np.median(current_returns) * 100
            min_ret = np.min(current_returns) * 100
            max_ret = np.max(current_returns) * 100
            win_rate = np.mean(np.array(current_returns) > 0) * 100
            
            results.append({
                'CheckTime_Min': ct,
                'Threshold_Pct': th,
                'MeanReturn': avg_ret,
                'MedianReturn': med_ret,
                'WinRate': win_rate,
                'Min': min_ret,
                'Max': max_ret,
                'Count': len(current_returns)
            })
            
    res_df = pd.DataFrame(results)
    
    # Save results
    res_csv = os.path.join(OUTPUT_DIR, "complex_exit_sweep.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"Sweep results saved to {res_csv}")
    
    # Plotting Heatmaps
    # Mean Return Heatmap
    pivot_mean = res_df.pivot(index='CheckTime_Min', columns='Threshold_Pct', values='MeanReturn')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_mean, annot=True, fmt=".2f", cmap="RdYlGn", center=baseline_avg)
    plt.title(f"Mean Return Strategy Comparison (Baseline: {baseline_avg:.2f}%)")
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_mean_return.png"))
    
    # Median Return Heatmap
    pivot_med = res_df.pivot(index='CheckTime_Min', columns='Threshold_Pct', values='MedianReturn')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_med, annot=True, fmt=".2f", cmap="RdYlGn", center=baseline_med)
    plt.title(f"Median Return Strategy Comparison (Baseline: {baseline_med:.2f}%)")
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_median_return.png"))
    
    # Win Rate Heatmap
    pivot_win = res_df.pivot(index='CheckTime_Min', columns='Threshold_Pct', values='WinRate')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_win, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Win Rate (%)")
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_win_rate.png"))

    print("Plots saved.")

    # --- 2h vs 6.5h Correlation Analysis ---
    print("\nRunning 2h vs 6.5h Analysis...")
    comparison_data = []
    
    for (ticker, date_str), bars in bars_dict.items():
        # Get Return at 2h (120 min)
        # We need to reuse logic or just simple check
        # We can reuse simulate with check_time, but we want the ACTUAL return at that time, not an exited one.
        # Actually simplest is to just inspect the bars directly.
        
        entry_price = bars.iloc[0]['o']
        start_time = bars.iloc[0]['datetime']
        stop_price = entry_price * (1 + 0.04) # 4% stop
        
        # We need:
        # 1. Did it hit stoploss BEFORE 2h? -> Return is -4%, Final is -4%
        # 2. Did it hit stoploss BETWEEN 2h and 6.5h? -> Return at 2h is X, Final is -4%
        # 3. No stoploss? -> Return at 2h is X, Final is Y
        
        # Find 2h bar (approx)
        # 2h = 120 mins
        # 6.5h = 390 mins
        
        ret_2h = None
        ret_final = None
        stopped_before_2h = False
        stopped_after_2h = False
        
        # Iterate to find states
        for idx, row in bars.iterrows():
            if row['h'] >= stop_price:
                # Stop triggered
                elapsed = (row['datetime'] - start_time).total_seconds() / 60.0
                if elapsed < 120:
                    ret_2h = -0.04
                    ret_final = -0.04
                    stopped_before_2h = True
                else:
                    # Stopped AFTER 2h
                    ret_final = -0.04
                    stopped_after_2h = True
                    # We still need to find what ret_2h WAS (it wasn't stopped yet)
                break
        
        # If not stopped before 2h, we need to find 2h return
        if not stopped_before_2h:
            # slice bars up to 120m
            # find bar closest to 120m
            target_time_2h = start_time + timedelta(minutes=120)
            # Find closest bar at or before target
            # bars is sorted
            
            # Simple approach: vector search
            # timestamps
            
            # Filter bars <= 120m
            mask_2h = (bars['datetime'] <= target_time_2h)
            if mask_2h.any():
                bar_2h = bars.loc[mask_2h].iloc[-1]
                ret_2h = (entry_price - bar_2h['c']) / entry_price
            else:
                 # No data before 2h?
                 ret_2h = 0.0 
        
        # If not stopped at all, find final return (390m or close)
        if not stopped_before_2h and not stopped_after_2h:
            # Final bar
            # 6.5h = 390m. Market close.
            # Just take last bar usually? Or strictly 390?
            # User said "6.5 hours", likely means End of Day.
            # My bars go to 16:00.
            final_bar = bars.iloc[-1]
            ret_final = (entry_price - final_bar['c']) / entry_price
            
        comparison_data.append({
            'ticker': ticker,
            'date': date_str,
            'Return_2h': ret_2h * 100,
            'Return_Final': ret_final * 100
        })

    comp_df = pd.DataFrame(comparison_data)
    
    # Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=comp_df, x='Return_2h', y='Return_Final', s=100, alpha=0.7)
    
    # Add correlation line
    # Fit linear regression
    if len(comp_df) > 1:
        m, b = np.polyfit(comp_df['Return_2h'], comp_df['Return_Final'], 1)
        x_range = np.array([comp_df['Return_2h'].min(), comp_df['Return_2h'].max()])
        plt.plot(x_range, m*x_range + b, color='red', linestyle='--', label=f'Trend (Slope={m:.2f})')
        
    plt.title(f"Return at 2h vs Final Return (6.5h)\nHigh Conviction (Score > {SCORE_THRESHOLD})")
    plt.xlabel("Return at 2 Hours (%)")
    plt.ylabel("Final Return at 6.5 Hours (%)")
    plt.axhline(0, color='black', alpha=0.3)
    plt.axvline(0, color='black', alpha=0.3)
    plt.grid(True)
    plt.legend()
    
    scatter_path = os.path.join(OUTPUT_DIR, "return_2h_vs_6.5h_scatter.png")
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")
    
    # Binned Plot
    # Bin 2h returns and calc avg final return
    # Bins: -infinity, -2, 0, 2, 4, 6, 8, infinity?
    # Or auto-bins
    bins = [-5, -2, 0, 2, 4, 6, 8, 10, 15]
    comp_df['Bin_2h'] = pd.cut(comp_df['Return_2h'], bins=bins)
    
    binned_stats = comp_df.groupby('Bin_2h', observed=True)['Return_Final'].agg(['mean', 'count']).reset_index()
    # Calculate midpoints for plotting
    binned_stats['Bin_Mid'] = binned_stats['Bin_2h'].apply(lambda x: x.mid)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=binned_stats, x='Bin_Mid', y='mean', marker='o', markersize=10)
    
    # Annotate counts
    for idx, row in binned_stats.iterrows():
        plt.text(row['Bin_Mid'], row['mean'] + 0.5, f"n={int(row['count'])}", ha='center')
        
    plt.title("Average Final Return vs 2h Return Buckets")
    plt.xlabel("Return at 2 Hours (%) - Binned")
    plt.ylabel("Average Final Return (6.5h) (%)")
    plt.grid(True)
    plt.axhline(0, color='black', alpha=0.3)
    
    binned_path = os.path.join(OUTPUT_DIR, "return_2h_vs_6.5h_binned.png")
    plt.savefig(binned_path)
    print(f"Binned plot saved to {binned_path}")

if __name__ == "__main__":
    main()
