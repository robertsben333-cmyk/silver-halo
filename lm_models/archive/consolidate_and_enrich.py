import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("POLYGON_API_KEY")

def fetch_candle(ticker, date_str, use_snapshot_if_today=False):
    """
    Fetch daily candle for a specific date.
    """
    if not API_KEY: return None
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date_str}/{date_str}"
    params = {"apiKey": API_KEY, "adjusted": "true"}
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                c = results[0]
                return {'o': c.get('o'), 'h': c.get('h'), 'l': c.get('l'), 'c': c.get('c'), 'date': date_str}
        
        # Fallback for Today if requested
        if use_snapshot_if_today and date_str == today_str:
            print(f"  Fetching Snapshot for {ticker} ({date_str})...")
            snap_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
            snap_resp = requests.get(snap_url, params={"apiKey": API_KEY})
            if snap_resp.status_code == 200:
                snap = snap_resp.json().get('ticker', {})
                day = snap.get('day', {})
                c = day.get('c') or snap.get('lastTrade', {}).get('p')
                o = day.get('o')
                if c and o:
                    return {'o': o, 'h': day.get('h', o), 'l': day.get('l', o), 'c': c, 'date': date_str}
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        time.sleep(0.5)
        
    return None

def fetch_volume_30d(ticker, ref_date_str):
    # Reuse previous logic or simplified: T-1 backwards 30 days
    # For now, just placeholder or fetch 1 aggregate?
    # User focused on Return. I will assume 'avg_volume_30d' is already in file OR carry over.
    # The file has 'avg_volume_30d'.
    return None 

def process_file(input_path, output_path, strategy):
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Normalize Volume column name
    if 'avg_volume_30d' in df.columns:
        df['AvgVolume_30D'] = df['avg_volume_30d']
    
    # New columns
    new_cols = ['EntryPrice', 'StopPrice', 'MaxHigh', 'MinLow', 'PotentialExitPrice', 
                'PotentialReturn', 'StoppedOut', 'RealizedReturn', 'MaxProfitPct', 
                'RealizedReturn_T1', 'Date_T1']
    
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    print(f"Processing {len(df)} rows...")
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        analysis_date = row['date']
        
        # Calculate T+1 Date
        dt_analysis = datetime.strptime(analysis_date, "%Y-%m-%d")
        dt_t1 = dt_analysis + timedelta(days=1)
        # Skip weekends for T+1? Simple approach: +1 day, if Sat/Sun, Polygon returns empty, we try +2?
        # My previous script used range +1 to +10 and took first.
        # Let's replicate that logic 'get_next_trading_day'
        
        candle = None
        # Try finding next trading day (valid T+1)
        # We search +1 to +5 days
        for i in range(1, 6):
            target_dt = dt_analysis + timedelta(days=i)
            target_str = target_dt.strftime("%Y-%m-%d")
            
            # Check if future
            if target_dt > datetime.now() and target_str != datetime.now().strftime("%Y-%m-%d"):
                break # Cannot look into future
                
            candle = fetch_candle(ticker, target_str, use_snapshot_if_today=True)
            if candle:
                break
        
        if candle:
            # Assume Long Strategy for generic columns? Or Short?
            # User wants "merged_data.csv" which serves both?
            # Usually merged_data is strategy agnostic until training? No, separate folders.
            # But the user said "merged_data.csv" (singular).
            # I will calculate assuming SHORT for now (as that's the primary loser strategy)
            # OR better: I will leave Strategy-Specific columns (Profit, StoppedOut) blank/neutral
            # AND populate 'RealizedReturn_T1' as the Raw Return (Close - Open) / Open?
            # User uses `RealizedReturn_T1` for training. 
            # `adjust_target_and_train.py` computed Strategy Return.
            
            # Let's populate the physical metrics (OHLC) and let the model compute profit?
            # No, 'RealizedReturn' in the file IS the profit.
            # I will calculate Profit for SHORT strategy here (as default).
            
            o = candle['o']
            h = candle['h']
            l = candle['l']
            c = candle['c']
            
            # Simulated Short Metrics (T+1)
            stop_pct = 0.04
            entry = o
            stop_price = o * (1 + stop_pct)
            
            df.at[idx, 'EntryPrice'] = entry
            df.at[idx, 'StopPrice'] = stop_price
            df.at[idx, 'MaxHigh'] = h
            df.at[idx, 'MinLow'] = l
            df.at[idx, 'Date_T1'] = candle['date']
            
            # Logic based on Strategy
            stop_pct = 0.04
            
            if strategy == 'short':
                entry = o
                stop_price = o * (1 + stop_pct)
                stopped = h >= stop_price
                if stopped:
                    ret = -stop_pct
                else:
                    ret = (entry - c) / entry # Short Profit
            else: # Long
                entry = o
                stop_price = o * (1 - stop_pct)
                stopped = l <= stop_price
                if stopped:
                    ret = -stop_pct
                else:
                    ret = (c - entry) / entry # Long Profit

            df.at[idx, 'EntryPrice'] = entry
            df.at[idx, 'StopPrice'] = stop_price
            df.at[idx, 'MaxHigh'] = h
            df.at[idx, 'MinLow'] = l
            df.at[idx, 'Date_T1'] = candle['date']
            df.at[idx, 'StoppedOut'] = stopped
            df.at[idx, 'RealizedReturn'] = ret
            df.at[idx, 'RealizedReturn_T1'] = ret
            
            # Max Profit Pct
            if strategy == 'short':
                max_profit = (entry - l) / entry
            else:
                max_profit = (h - entry) / entry
            df.at[idx, 'MaxProfitPct'] = max_profit * 100
            
            # print(f"[{idx+1}/{len(df)}] {ticker} {analysis_date} -> T+1 {candle['date']}: Ret {ret:.2%}")
        else:
            # print(f"[{idx+1}/{len(df)}] {ticker} {analysis_date} -> No T+1 data found.")
            pass
            
        time.sleep(0.12)

    # Save
    print(f"Saving {strategy} dataset to {output_path}...")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=['long', 'short'])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_file("lm_models/full_history.csv", args.output, args.strategy)
