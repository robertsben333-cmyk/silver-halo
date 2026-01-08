import pandas as pd
import numpy as np
import requests
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY not found in environment variables.")
    sys.exit(1)

# Fix path for Windows using os.path.join
INPUT_FILE = os.path.join("analysis-5", "1", "datasets", "stoploss_analysis_Baseline_Score_lt_0.csv")
OUTPUT_CSV = os.path.join("analysis-5", "1", "volatility_volume_stats.csv")
PLOT_DIR = os.path.join("analysis-5", "1", "plots")

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def get_polygon_daily_bars(ticker, end_date, days_back=30):
    """
    Fetch daily bars for the `days_back` period ending on (but not including) `end_date`.
    """
    # Calculate start date significantly back to ensure we get enough trading days
    # (30 trading days might require ~45 calendar days)
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - timedelta(days=days_back * 2) 
    
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d") # Up to yesterday
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {
        "apiKey": API_KEY,
        "limit": 50000,
        "sort": "desc" # Get most recent first
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("resultsCount", 0) > 0:
                results = data["results"]
                # We want the last 30 trading days
                return results[:days_back] 
    except Exception as e:
        print(f"Error fetching daily bars for {ticker}: {e}")
    
    return []

def get_polygon_intraday_bars(ticker, end_date, days_back=3):
    """
    Fetch 5-minute bars for the `days_back` period ending on (but not including) `end_date`.
    """
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - timedelta(days=days_back + 2) # Extra buffer for weekends
    
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Polygon V2 Aggs
    # Note: timestamps are strictly less than end time if we range well, 
    # but for daily date strings it includes the day. 
    # Just grab a chunk and filter if needed.
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{start_str}/{end_str}"
    params = {
        "apiKey": API_KEY,
        "limit": 50000,
        "sort": "desc" 
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("resultsCount", 0) > 0:
                results = data["results"]
                # Filter to only ensure we cover the requested days (approx)
                # For now just returning all sourced in that window as "short term history"
                return results
    except Exception as e:
        print(f"Error fetching intraday bars for {ticker}: {e}")
            
    return []

def calculate_metrics(daily_bars, intraday_bars):
    metrics = {
        "Volatility_30D": np.nan,
        "AvgVolume_30D": np.nan,
        "IntradayVolatility_3D": np.nan
    }
    
    # 1. Daily Metrics
    if daily_bars and len(daily_bars) > 5:
        closes = [b['c'] for b in daily_bars]
        volumes = [b['v'] * b['c'] for b in daily_bars] # Dollar Volume roughly
        
        # Calculate daily returns (reverse list to be chronological for calculation if needed, but std dev is invariant)
        # pct_change needs series
        closes_series = pd.Series(closes[::-1]) # Oldest to newest
        daily_returns = closes_series.pct_change().dropna()
        
        # Annualized Volatility
        vol_30d = daily_returns.std() * np.sqrt(252)
        metrics["Volatility_30D"] = vol_30d
        
        # Avg Dollar Volume
        metrics["AvgVolume_30D"] = np.mean(volumes)
        
    # 2. Intraday Metrics
    if intraday_bars and len(intraday_bars) > 10:
        # Just calculate std dev of 5-min returns over the whole window captured
        closes_intra = [b['c'] for b in intraday_bars]
        ts_intra = [b['t'] for b in intraday_bars]
        
        # Sort by time just in case
        sorted_pairs = sorted(zip(ts_intra, closes_intra))
        sorted_closes = [p[1] for p in sorted_pairs]
        
        intra_series = pd.Series(sorted_closes)
        intra_returns = intra_series.pct_change().dropna()
        
        # Pure std dev of returns (not annualized, just raw measure of 'choppiness')
        metrics["IntradayVolatility_3D"] = intra_returns.std()
        
    return metrics

import argparse

def main():
    parser = argparse.ArgumentParser(description='Analyze volatility and volume correlations.')
    parser.add_argument('--input', type=str, help='Path to input CSV file', default=INPUT_FILE)
    args = parser.parse_args()

    input_path = args.input
    # Adjust output filename based on input filename to avoid overwriting
    base_name = os.path.basename(input_path).replace(".csv", "")
    output_csv = os.path.join(os.path.dirname(input_path), f"volatility_stats_{base_name}.csv")
    plot_dir = os.path.join(os.path.dirname(input_path), "plots") # Ensure plots go to neighbor dir
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Target Input File: {os.path.abspath(input_path)}")
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Loading {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if 'RealizedReturn' not in df.columns:
        print("Error: 'RealizedReturn' column not found in input file.")
        return

    print(f"Found {len(df)} trades.")
    
    # Add columns
    df["Volatility_30D"] = np.nan
    df["AvgVolume_30D"] = np.nan
    df["IntradayVolatility_3D"] = np.nan
    
    for index, row in df.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        
        print(f"Processing {ticker} on {date_str}...", end=" ")
        
        # Fetch Data
        daily = get_polygon_daily_bars(ticker, date_str, days_back=30)
        intraday = get_polygon_intraday_bars(ticker, date_str, days_back=3)
        
        # Calc Metrics
        met = calculate_metrics(daily, intraday)
        
        df.at[index, "Volatility_30D"] = met["Volatility_30D"]
        df.at[index, "AvgVolume_30D"] = met["AvgVolume_30D"]
        df.at[index, "IntradayVolatility_3D"] = met["IntradayVolatility_3D"]
        
        print(f"Vol30: {met['Volatility_30D']:.4f}, Vol3D: {met['IntradayVolatility_3D']:.4f}")
        
    # Drop rows where metrics failed
    df_clean = df.dropna(subset=["Volatility_30D", "RealizedReturn"])
    
    print(f"\nAnalysis complete. Clean trades: {len(df_clean)}/{len(df)}")
    df.to_csv(output_csv, index=False)
    print(f"Saved stats to {output_csv}")
    
    if len(df_clean) < 3:
        print("Not enough data to correlation/plot.")
        return
        
    # Correlations
    print("\n--- Correlations with RealizedReturn ---")
    corrs = df_clean[["RealizedReturn", "Volatility_30D", "AvgVolume_30D", "IntradayVolatility_3D"]].corr(method='pearson')
    print(corrs["RealizedReturn"])
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    # 1. Volatility 30D
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_clean, x="Volatility_30D", y="RealizedReturn")
    plt.title(f"Realized Return vs 30-Day Volatility\n({base_name}, n={len(df_clean)})")
    plt.xlabel("Annualized Volatility (30D)")
    plt.ylabel("Realized Return")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"volatility_30d_{base_name}.png"))
    plt.close()
    
    # 2. Volume
    plt.figure(figsize=(10, 6))
    # Use log scale for volume often
    df_clean["LogVolume"] = np.log10(df_clean["AvgVolume_30D"])
    sns.regplot(data=df_clean, x="LogVolume", y="RealizedReturn")
    plt.title(f"Realized Return vs Log Avg Dollar Volume (30D)\n({base_name}, n={len(df_clean)})")
    plt.xlabel("Log10(Avg Dollar Volume)")
    plt.ylabel("Realized Return")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"volume_30d_{base_name}.png"))
    plt.close()
    
    # 3. Intraday Volatility
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_clean, x="IntradayVolatility_3D", y="RealizedReturn")
    plt.title(f"Realized Return vs Intraday Volatility (3D, 5min)\n({base_name}, n={len(df_clean)})")
    plt.xlabel("Std Dev of 5-min Returns (Intraday)")
    plt.ylabel("Realized Return")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"intraday_volatility_{base_name}.png"))
    plt.close()
    
    print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    main()
