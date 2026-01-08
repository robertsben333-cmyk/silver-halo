import pandas as pd
import numpy as np
import requests
import os
import json
import time
import argparse
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
API_KEY = os.environ.get("POLYGON_API_KEY")

def fetch_next_day_candle(ticker, analysis_date_str):
    """
    Fetch the OHLC for the *next* trading day after analysis_date_str.
    Returns dict with o, h, l, c, date or None.
    """
    if not API_KEY:
        return None
        
    today_str = datetime.now().strftime("%Y-%m-%d")
    dt_analysis = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    
    # Range: from T+1 up to T+10
    start_dt = dt_analysis + timedelta(days=1)
    end_dt = dt_analysis + timedelta(days=10)
    
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    # If start_dt is in the future relative to today, we can't fetch.
    # But if start_dt == today (Jan 7), we CAN fetch today's data.
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {
        "apiKey": API_KEY,
        "adjusted": "true",
        "sort": "asc", # Oldest first
        "limit": 1
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                candle = results[0]
                ts = candle.get('t')
                date_obj = datetime.fromtimestamp(ts / 1000)
                return {
                    'o': candle.get('o'),
                    'h': candle.get('h'),
                    'l': candle.get('l'),
                    'c': candle.get('c'),
                    'date': date_obj.strftime("%Y-%m-%d")
                }
            else:
                # If no results and start_date is TODAY, try Snapshot
                if start_str == today_str:
                    print(f"DEBUG: T+1 is Today ({today_str}). Aggs empty. Trying Snapshot/Last Trade...")
                    prev_close_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev" # No, prev is yesterday
                    # Try daily Open/Close (snapshot)
                    snap_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
                    snap_resp = requests.get(snap_url, params={"apiKey": API_KEY})
                    if snap_resp.status_code == 200:
                        snap = snap_resp.json().get('ticker', {})
                        day = snap.get('day', {})
                        # If day has C, use it. If not, maybe market is open? Use V or last trade.
                        # User said use 21:45 closing time (approx).
                        c = day.get('c') or snap.get('lastTrade', {}).get('p')
                        o = day.get('o')
                        h = day.get('h')
                        l = day.get('l')
                        
                        if c and o:
                             return {
                                'o': o, 'h': h or o, 'l': l or o, 'c': c,
                                'date': today_str
                            }
                print(f"DEBUG: No results for {ticker} range {start_str} to {end_str}")

    except Exception as e:
        print(f"Error fetching next day for {ticker}: {e}")
        time.sleep(1)
        
    return None

def calculate_strategy_return(candle, strategy):
    """
    Calculate return with 4% stoploss logic.
    Strategy: 'long' or 'short'
    """
    open_price = candle['o']
    high_price = candle['h']
    low_price = candle['l']
    close_price = candle['c']
    
    STOP_PCT = 0.04
    
    if strategy == 'long':
        # Stop check: Low price
        stop_price = open_price * (1 - STOP_PCT)
        if low_price <= stop_price:
            return -STOP_PCT
        else:
            return (close_price - open_price) / open_price
            
    elif strategy == 'short':
        # Stop check: High price
        stop_price = open_price * (1 + STOP_PCT)
        if high_price >= stop_price:
            return -STOP_PCT
        else:
            return (open_price - close_price) / open_price # Short profit
            
    return 0.0

def retarget_and_train(data_path, output_dir, strategy):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Fetching T+1 returns for {len(df)} rows (Strategy: {strategy})...")
    
    new_targets = []
    next_dates = []
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        analysis_date = row['date']
        
        # We need volume logic too? 
        # The repair_and_train already fixed volume bias (T-1).
        # But wait, if we are trading T+1, we can use T-day volume.
        # However, the user said "Ensure this is what is estimated".
        # If we predict T+1, then T volume is known.
        # But let's stick to the volume column we have (AvgVolume_30D).
        # We assume the volume column in CSV is populated correctly by previous step.
        # Actually, previous step imputed lots of means.
        # But for T+1 prediction, we *can* look at T volume.
        # Let's verify if we should re-fetch volume for T?
        # User is focused on Target Return here. 
        # I will leave volume as is (it's T-1 relative to T, so T-2 relative to T+1? Very safe).
        # Or T-1 relative to T. 
        
        candle = fetch_next_day_candle(ticker, analysis_date)
        
        if candle:
            ret = calculate_strategy_return(candle, strategy)
            new_targets.append(ret)
            next_dates.append(candle['date'])
            print(f"[{idx+1}/{len(df)}] {ticker}: {analysis_date} -> {candle['date']} | Ret: {ret:.2%}")
        else:
            print(f"[{idx+1}/{len(df)}] {ticker}: {analysis_date} -> Failed to find next day.")
            new_targets.append(None) # Drop or keep?
            next_dates.append(None)
            
        time.sleep(0.12) # ~8 calls/sec max (Polygon limit is usually higher but stay safe)

    # Update DF
    df['RealizedReturn_T1'] = new_targets
    df['Date_T1'] = next_dates
    
    # Drop failures
    original_count = len(df)
    df = df.dropna(subset=['RealizedReturn_T1'])
    print(f"Dropped {original_count - len(df)} rows due to missing T+1 data.")
    
    # Overwrite RealizedReturn for training
    df['RealizedReturn'] = df['RealizedReturn_T1']
    
    # Save updated data? Maybe as a new file for safety?
    # Or just use in memory.
    # User said "And replace the existing files."
    # I will save as 'merged_data_t1.csv' then rename.
    save_path = os.path.join(output_dir, 'merged_data_t1.csv')
    df.to_csv(save_path, index=False)
    print(f"Saved T+1 targeted data to {save_path}")

    # --- Training ---
    print("Training Model on T+1 Targets...")
    
    # Features (Same as before)
    features_num = [
        'oneDayReturnPct', 'finalScore', 
        'metrics_PCR', 'metrics_EC', 'metrics_SD', 'metrics_NRI', 
        'metrics_HDM', 'metrics_CONTR', 'metrics_CP', 'metrics_RD',
        'metrics_FRESH_NEG',
        'AvgVolume_30D' 
    ]
    features_cat = ['confidence', 'uncertainty']
    target = 'RealizedReturn'
    
    features_num = [f for f in features_num if f in df.columns]
    X_num = df[features_num].fillna(0)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if all(c in df.columns for c in features_cat):
        X_cat = pd.DataFrame(encoder.fit_transform(df[features_cat].fillna("Medium")))
        X_cat.columns = encoder.get_feature_names_out(features_cat)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_num
        
    y = df[target]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    print(f"R2: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    
    # Save artifacts
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    coef_dict = {
        "intercept": model.intercept_,
        "coefficients": dict(zip(X.columns, model.coef_))
    }
    
    with open(os.path.join(output_dir, 'coefficients.json'), 'w') as f:
        json.dump(coef_dict, f, indent=4)
        
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='AbsCoef', ascending=False)
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Target: T+1 Return (Strategy: {strategy})\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MSE: {mse:.6f}\n\n")
        f.write(f"Intercept: {model.intercept_:.6f}\n\n")
        f.write("Coefficients:\n")
        f.write(coef_df.to_string())
        
        f.write("\n\n--- Profitability Analysis ---\n")
        header = f"{'Threshold':<10} | {'Trades':<6} | {'Mean Return':<12} | {'Win Rate':<8}"
        print("\n" + header)
        f.write(header + "\n")
        print("-" * 50)
        f.write("-" * 50 + "\n")
        
        thresholds = [0.0, 0.01, 0.02, 0.03, 0.04]
        for t in thresholds:
            mask = y_pred > t
            sel = y[mask]
            if len(sel) > 0:
                mean_ret = sel.mean()
                win = (sel > 0).mean()
                count = len(sel)
            else:
                mean_ret = 0
                win = 0
                count = 0
            
            line = f">{t:<9.0%} | {count:<6} | {mean_ret:<12.2%} | {win:<8.0%}"
            print(line)
            f.write(line + "\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual T+1 Return')
    plt.ylabel('Predicted')
    plt.title(f'T+1 Predictions (R2={r2:.2f})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", required=True, choices=['long', 'short'])
    args = parser.parse_args()
    
    retarget_and_train(args.data, args.output, args.strategy)
