import pandas as pd
import numpy as np
import requests
import os
import time
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("POLYGON_API_KEY")

def fetch_t1_metrics(ticker, analysis_date_str):
    """
    Fetch T+1 Open and Close.
    Returns: { 'o': float, 'c': float, 'h': float, 'l': float, 'date': str } or None
    """
    if not API_KEY: return None

    dt_analysis = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Search T+1 to T+5 for first trading day
    for i in range(1, 6):
        target_dt = dt_analysis + timedelta(days=i)
        target_str = target_dt.strftime("%Y-%m-%d")
        
        # Stop if future
        if target_dt > datetime.now() and target_str != today_str:
            return None # T+1 is in the future

        # 1. Try Aggregates
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{target_str}/{target_str}"
        params = {"apiKey": API_KEY, "adjusted": "true"}
        
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                res = resp.json().get("results", [])
                if res:
                    c = res[0]
                    return {'o': c.get('o'), 'c': c.get('c'), 'h': c.get('h'), 'l': c.get('l'), 'date': target_str}
        except Exception:
            pass
            
        # 2. If Today, Try Snapshot (Partial Day)
        if target_str == today_str:
            print(f"  Fetching Snapshot for {ticker} (T+1=Today)...")
            snap_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
            try:
                snap_resp = requests.get(snap_url, params={"apiKey": API_KEY})
                if snap_resp.status_code == 200:
                    snap = snap_resp.json().get('ticker', {})
                    day = snap.get('day', {})
                    c = day.get('c') or snap.get('lastTrade', {}).get('p')
                    o = day.get('o')
                    if c and o:
                         return {'o': o, 'c': c, 'h': day.get('h', o), 'l': day.get('l', o), 'date': target_str}
            except Exception:
                pass
                
    return None

def calculate_returns(entry_price, close_price, high_price, low_price):
    """
    Calculate generic returns.
    But we need Strategy Specifics with Stoploss.
    """
    pass

def train_model(df, target_col, strategy_name, output_dir):
    print(f"\n--- Training {strategy_name} Model ---")
    
    # Filter valid targets
    train_df = df.dropna(subset=[target_col]).copy()
    print(f"Training Rows: {len(train_df)}")
    
    if len(train_df) < 5:
        print("Not enough data to train.")
        return

    # Features
    features_num = [
        'oneDayReturnPct', 'finalScore', 
        'metrics_PCR', 'metrics_EC', 'metrics_SD', 'metrics_NRI', 
        'metrics_HDM', 'metrics_CONTR', 'metrics_CP', 'metrics_RD',
        'metrics_FRESH_NEG', 'avg_volume_30d' 
    ]
    features_cat = ['confidence', 'uncertainty']
    
    # Check features exist
    features_num = [f for f in features_num if f in train_df.columns]
    
    # Prepare X
    X_num = train_df[features_num].fillna(0)
    
    # OneHot Cat
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if all(c in train_df.columns for c in features_cat):
        X_cat = pd.DataFrame(encoder.fit_transform(train_df[features_cat].fillna("Medium")))
        X_cat.columns = encoder.get_feature_names_out(features_cat)
        # Reset index to concat correctly (VERY IMPORTANT)
        X = pd.concat([X_num.reset_index(drop=True), X_cat], axis=1)
    else:
        X = X_num.reset_index(drop=True)
        
    y = train_df[target_col].reset_index(drop=True)
    
    # Train
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"R2: {r2:.4f}, MSE: {mse:.6f}")
    
    # Save artifacts
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Coefficients
    coef_dict = {
        "intercept": model.intercept_,
        "coefficients": dict(zip(X.columns, model.coef_))
    }
    with open(os.path.join(output_dir, 'coefficients.json'), 'w') as f:
        json.dump(coef_dict, f, indent=4)
        
    # Summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model: {strategy_name}\nTarget: {target_col}\n")
        f.write(f"R2: {r2:.4f}\nMean Squared Error: {mse:.6f}\n\n")
        f.write(f"Intercept: {model.intercept_:.6f}\n\nCoefficients:\n")
        
        coef_df = pd.DataFrame({'Feature': X.columns, 'Val': model.coef_})
        coef_df['Abs'] = coef_df['Val'].abs()
        f.write(coef_df.sort_values('Abs', ascending=False).to_string())

def build_and_train():
    # 1. Load History
    input_path = "lm_models/full_history.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("Loading history...")
    df = pd.read_csv(input_path)
    
    # 2. Enrich with T+1
    print("Fetching T+1 Data...")
    
    stop_pct = 0.04
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        
        metrics = fetch_t1_metrics(ticker, date_str)
        
        if metrics:
            o = metrics['o']
            c = metrics['c']
            h = metrics['h']
            l = metrics['l']
            
            # --- Short Strategy ---
            # Entry = Open. Stop = Open * 1.04.
            # If High > Stop, Loss = -4%. Else Profit = (Entry - Close)/Entry
            entry_short = o
            stop_short = o * (1 + stop_pct)
            if h >= stop_short:
                 ret_short = -stop_pct
            else:
                 ret_short = (entry_short - c) / entry_short
                 
            # --- Long Strategy ---
            # Entry = Open. Stop = Open * 0.96.
            # If Low < Stop, Loss = -4%. Else Profit = (Close - Entry)/Entry
            entry_long = o
            stop_long = o * (1 - stop_pct)
            if l <= stop_long:
                 ret_long = -stop_pct
            else:
                 ret_long = (c - entry_long) / entry_long
            
            df.at[idx, 'Target_Short'] = ret_short
            df.at[idx, 'Target_Long'] = ret_long
            df.at[idx, 'T1_Date'] = metrics['date']
            
            # print(f"[{idx+1}/{len(df)}] {ticker}: Short={ret_short:.2%}, Long={ret_long:.2%}")
        else:
            # print(f"[{idx+1}/{len(df)}] {ticker}: No T+1 data.")
            pass
            
        time.sleep(0.12)
        
    # Save Enriched Data
    df.to_csv("lm_models/merged_data_enriched.csv", index=False)
    print("Enriched data saved to lm_models/merged_data_enriched.csv")
    
    # 3. Train Models
    # Short
    train_model(df, 'Target_Short', 'Short Strategy', 'lm_models/short')
    
    # Long
    train_model(df, 'Target_Long', 'Long Strategy', 'lm_models/long')

if __name__ == "__main__":
    build_and_train()
