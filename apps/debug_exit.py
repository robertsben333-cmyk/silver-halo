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

def simulate_old(bars_1m, param=3):
    entry_price = bars_1m.iloc[0]['o']
    hard_stop_price = entry_price * 1.04
    bars_5m = resample_bars(bars_1m, '5min')
    consecutive_green = 0
    
    for idx, row in bars_5m.iterrows():
        if row['h'] >= hard_stop_price:
            return -0.04
        
        is_green = row['c'] > row['o']
        if is_green:
            consecutive_green += 1
        else:
            consecutive_green = 0
            
        if consecutive_green >= param:
            exit_price = row['c']
            return (entry_price - exit_price) / entry_price
            
    exit_price = bars_5m.iloc[-1]['c']
    return (entry_price - exit_price) / entry_price

def simulate_new(bars_1m, green_threshold=3, profit_filter=False, wait_period=0):
    entry_price = bars_1m.iloc[0]['o']
    hard_stop_price = entry_price * 1.04
    bars_5m = resample_bars(bars_1m, '5min')
    
    consecutive_green = 0
    num_bars = len(bars_5m)
    
    for i in range(num_bars):
        row = bars_5m.iloc[i]
        
        if row['h'] >= hard_stop_price:
            return -0.04
            
        is_green = row['c'] > row['o']
        if is_green: consecutive_green += 1
        else: consecutive_green = 0
        
        if consecutive_green >= green_threshold:
            candidate_idx = i + wait_period
            
            if candidate_idx < num_bars:
                candidate_price = bars_5m.iloc[candidate_idx]['c']
                
                # Check Hard Stop during wait
                if wait_period > 0:
                     if bars_5m.iloc[i+1 : candidate_idx+1]['h'].max() >= hard_stop_price:
                        return -0.04

                if profit_filter:
                    if candidate_price < entry_price:
                        return (entry_price - candidate_price) / entry_price
                    else:
                        pass
                else:
                    return (entry_price - candidate_price) / entry_price

    exit_price = bars_5m.iloc[-1]['c']
    return (entry_price - exit_price) / entry_price

def main():
    df = pd.read_csv("analysis-5/1/analysis_history.csv")
    smx_trades = df[df['ticker'] == 'SMX']
    
    print(f"Found {len(smx_trades)} SMX trades.")
    
    for idx, row in smx_trades.iterrows():
        date_str = row['date']
        print(f"\nAnalyzing SMX on {date_str}...")
        
        bars = get_minute_bars('SMX', date_str)
        if bars is None: 
            print("No bars.")
            continue
            
        old_ret = simulate_old(bars, 3)
        new_ret = simulate_new(bars, 3, False, 0)
        
        print(f"Old Return: {old_ret*100:.2f}%")
        print(f"New Return: {new_ret*100:.2f}%")
        
        if abs(old_ret - new_ret) > 0.0001:
            print("!!! DISCREPANCY !!!")

if __name__ == "__main__":
    main()
