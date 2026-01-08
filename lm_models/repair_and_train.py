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

# --- Config ---
API_KEY = os.environ.get("POLYGON_API_KEY")

def fetch_historical_30d_volume(ticker, date_str):
    """
    Fetch 30-day average dollar volume ending on date_str (inclusive or previous day).
    """
    if not API_KEY:
        return None
        
    # We want 30 trading days ending at date_str
    # Fetch 60 calendar days back to be safe
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Shift to T-1 to ensure no look-ahead bias (PRE-event liquidity)
    end_dt = dt - timedelta(days=1)
    end_str = end_dt.strftime("%Y-%m-%d")
    
    start_dt = end_dt - timedelta(days=60)
    start_str = start_dt.strftime("%Y-%m-%d")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {
        "apiKey": API_KEY,
        "adjusted": "true",
        "sort": "desc",
        "limit": 30
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if not results:
                return 0.0
            
            total_vol = 0.0
            count = 0
            for r in results:
                v = r.get('v', 0)
                c = r.get('c', 0)
                total_vol += (v * c)
                count += 1
            
            if count > 0:
                return total_vol / count
    except Exception as e:
        print(f"Error fetching vol for {ticker}: {e}")
        time.sleep(1) # Backoff
        
    return 0.0

def train_and_repair(data_path, output_dir, target_col='RealizedReturn'):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    
    # Check for AvgVolume_30D
    if 'avg_volume_30d' in df.columns and 'AvgVolume_30D' not in df.columns:
        df['AvgVolume_30D'] = df['avg_volume_30d']

    if 'AvgVolume_30D' not in df.columns:
        df['AvgVolume_30D'] = np.nan
        
    # Count missing
    # We treat 0.0, NaN, or empty as missing
    df['AvgVolume_30D'] = pd.to_numeric(df['AvgVolume_30D'], errors='coerce').fillna(0.0)
    
    # Only update zeros
    missing_indices = df[df['AvgVolume_30D'] == 0.0].index
    if len(missing_indices) > 0:
        print(f"Updating {len(missing_indices)} rows with missing volume...")
    
        updates_made = False
        for idx in missing_indices:
            row = df.loc[idx]
            ticker = row['ticker']
            date_str = row['date']
            print(f"[{idx+1}/{len(df)}] Fetching vol for {ticker} on {date_str}...")
            vol = fetch_historical_30d_volume(ticker, date_str)
            
            if vol and vol > 0:
                df.at[idx, 'AvgVolume_30D'] = vol
                updates_made = True
                print(f"  -> {vol:,.0f}")
            else:
                print("  -> Failed. Will impute with mean later.")
            
            time.sleep(0.2) # Rate limit protection

        # Impute remaining zeros with mean of non-zeros
        non_zero_vol = df[df['AvgVolume_30D'] > 0]['AvgVolume_30D']
        if len(non_zero_vol) > 0:
            mean_vol = non_zero_vol.mean()
            remaining_zeros = df[df['AvgVolume_30D'] == 0.0].index
            if len(remaining_zeros) > 0:
                print(f"Imputing {len(remaining_zeros)} rows with mean volume: {mean_vol:,.0f}")
                df.loc[remaining_zeros, 'AvgVolume_30D'] = mean_vol
                updates_made = True
            
        if updates_made:
            print(f"Overwriting {data_path} with updated data...")
            df.to_csv(data_path, index=False)
    
    # --- Training ---
    print(f"Training Model on target: {target_col}...")
    
    # Features
    features_num = [
        'oneDayReturnPct', 'finalScore', 
        'metrics_PCR', 'metrics_EC', 'metrics_SD', 'metrics_NRI', 
        'metrics_HDM', 'metrics_CONTR', 'metrics_CP', 'metrics_RD',
        'metrics_FRESH_NEG',
        'AvgVolume_30D' 
    ]
    
    features_cat = ['confidence', 'uncertainty']
    target = target_col
    
    # Verify Target Exists
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found in dataset. available: {list(df.columns)}")
        return

    # Filter existing cols
    features_num = [f for f in features_num if f in df.columns]
    
    X_num = df[features_num].fillna(0)
    
    # Categorical
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if all(c in df.columns for c in features_cat):
        X_cat = pd.DataFrame(encoder.fit_transform(df[features_cat].fillna("Medium"))) # Fill NA cats
        X_cat.columns = encoder.get_feature_names_out(features_cat)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_num
        
    y = df[target].fillna(0) # Fill target NaNs for robustness
    
    # Train
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
        
    # JSON
    coef_dict = {
        "intercept": model.intercept_,
        "coefficients": dict(zip(X.columns, model.coef_))
    }
    
    with open(os.path.join(output_dir, 'coefficients.json'), 'w') as f:
        json.dump(coef_dict, f, indent=4)
        
    # Summary
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='AbsCoef', ascending=False)
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Target: {target}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MSE: {mse:.6f}\n\n")
        f.write(f"Intercept: {model.intercept_:.6f}\n\n")
        f.write("Coefficients:\n")
        f.write(coef_df.to_string())
        
        # Profitability
        f.write("\n\n--- Profitability Analysis ---\n")
        header = f"{'Threshold':<10} | {'Trades':<6} | {'Mean Return':<12} | {'Std Dev':<10} | {'Win Rate':<8}"
        print("\n" + header)
        f.write(header + "\n")
        print("-" * 65)
        f.write("-" * 65 + "\n")
        
        # Expanded thresholds
        thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
        for t in thresholds:
            mask = y_pred > t
            sel = y[mask]
            # outliers filter - assuming simple returns, < 0.5 (50%)
            sel = sel[sel < 0.5] 
            
            if len(sel) > 0:
                mean_ret = sel.mean()
                std_dev = sel.std()
                win = (sel > 0).mean()
                count = len(sel)
            else:
                mean_ret = 0.0
                std_dev = 0.0
                win = 0.0
                count = 0
            
            line = f">{t:<9.0%} | {count:<6} | {mean_ret:<12.2%} | {std_dev:<10.2%} | {win:<8.0%}"
            print(line)
            f.write(line + "\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel(f'Actual {target}')
    plt.ylabel('Predicted')
    plt.title(f'Predictions (R2={r2:.2f})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_col", default="RealizedReturn")
    args = parser.parse_args()
    
    train_and_repair(args.data, args.output, args.target_col)
