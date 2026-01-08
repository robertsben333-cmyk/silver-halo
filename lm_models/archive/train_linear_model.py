import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os
import argparse
import json

def train(data_path, output_dir):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Filter rows with missing Target (e.g. Jan 7 rows where T+1 is future)
    target = 'RealizedReturn'
    # Also ensure we check 'RealizedReturn_T1' if present, but the consolidate script put it in RealizedReturn too.
    
    original_len = len(df)
    df = df.dropna(subset=[target])
    print(f"Filtered {original_len} -> {len(df)} rows (dropped NaNs in {target}).")
    
    if len(df) == 0:
        print("No training data available.")
        return

    # Features
    features_num = [
        'oneDayReturnPct', 'finalScore', 
        'metrics_PCR', 'metrics_EC', 'metrics_SD', 'metrics_NRI', 
        'metrics_HDM', 'metrics_CONTR', 'metrics_CP', 'metrics_RD',
        'metrics_FRESH_NEG',
        'AvgVolume_30D', 'avg_volume_30d'
    ]
    # Deduplicate volume if both exist
    if 'AvgVolume_30D' in df.columns and 'avg_volume_30d' in df.columns:
        features_num.remove('avg_volume_30d')
    elif 'avg_volume_30d' in df.columns and 'AvgVolume_30D' not in df.columns:
        # Renaming done in consolidate, but just in case
        df['AvgVolume_30D'] = df['avg_volume_30d']
        features_num.remove('avg_volume_30d')
        
    features_cat = ['confidence', 'uncertainty']
    
    features_num = [f for f in features_num if f in df.columns]
    print(f"Features: {features_num}")
    
    X_num = df[features_num].fillna(0)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if all(c in df.columns for c in features_cat):
        X_cat = pd.DataFrame(encoder.fit_transform(df[features_cat].fillna("Medium")))
        X_cat.columns = encoder.get_feature_names_out(features_cat)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_num
        
    y = df[target]
    
    # Train
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    print(f"R2: {r2:.4f}")
    
    # Artifacts
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
        f.write(f"Target: T+1 Return\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MSE: {mse:.6f}\n\n")
        f.write(f"Intercept: {model.intercept_:.6f}\n\n")
        f.write("Coefficients:\n")
        f.write(coef_df.to_string())
        
        # Profitability Analysis
        f.write("\n\n--- Profitability Analysis ---\n")
        header = f"{'Threshold':<10} | {'Trades':<6} | {'Mean Return':<12} | {'Win Rate':<8}"
        print("\n" + header)
        f.write(header + "\n")
        print("-" * 50)
        f.write("-" * 50 + "\n")
        
        thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
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
    args = parser.parse_args()
    
    train(args.data, args.output)
