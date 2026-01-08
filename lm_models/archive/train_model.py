import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os
import argparse
import json

def train_model(history_path, stats_path, output_dir):
    print(f"Training Model...")
    print(f"History (Features): {history_path}")
    print(f"Stats (Target/Vol): {stats_path}")
    print(f"Output Dir: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data
    try:
        df_hist = pd.read_csv(history_path)
        df_stats = pd.read_csv(stats_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Merge
    # df_hist: ticker, date, metrics_*, finalScore
    # df_stats: ticker, date, RealizedReturn, AvgVolume_30D
    
    # Normalize column names for merge if needed
    # Assuming standard 'ticker' and 'date' in both
    
    merged = pd.merge(df_hist, df_stats, on=['ticker', 'date'], how='inner')
    
    if merged.empty:
        print("Merged dataset is empty. Check ticker/date formats.")
        return
    
    print(f"Merged Data Size: {len(merged)}")
    
    # Save merged data for reference
    merged.to_csv(os.path.join(output_dir, 'merged_data.csv'), index=False)

    # 3. Feature Selection
    features_num = [
        'oneDayReturnPct', 'finalScore', 
        'metrics_PCR', 'metrics_EC', 'metrics_SD', 'metrics_NRI', 
        'metrics_HDM', 'metrics_CONTR', 'metrics_CP', 'metrics_RD',
        'metrics_V', # V is Volume? No, probably Volatility or something else in original model. 
                     # Wait, let's use the explicit 'AvgVolume_30D' we just added.
        'AvgVolume_30D' 
    ]
    
    # Optional features checking
    possible_features = ['sent_pro', 'sent_com', 'metrics_FRESH_NEG']
    for feat in possible_features:
        if feat in merged.columns:
            features_num.append(feat)

    # Filter only numeric features that actually exist in merged
    features_num = [f for f in features_num if f in merged.columns]
    print(f"Numeric Features: {features_num}")

    features_cat = ['confidence', 'uncertainty']
    
    # Prepare X and y
    # Fill NA with 0 for numeric features (common in this dataset structure)
    X_num = merged[features_num].fillna(0)
    
    # Encode Categoricals
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Reshape if necessary or dataframe
    if all(c in merged.columns for c in features_cat):
        X_cat = pd.DataFrame(encoder.fit_transform(merged[features_cat]))
        X_cat.columns = encoder.get_feature_names_out(features_cat)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        print(f"Warning: categorical features {features_cat} not found. Using numeric only.")
        X = X_num

    y = merged['RealizedReturn']
    
    # 4. Train
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 5. Evaluate
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    
    # 6. Coefficients
    coef_dict = {
        "intercept": model.intercept_,
        "coefficients": dict(zip(X.columns, model.coef_))
    }
    
    # Save Coefficients JSON
    with open(os.path.join(output_dir, 'coefficients.json'), 'w') as f:
        json.dump(coef_dict, f, indent=4)
        
    # readable summary
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='AbsCoef', ascending=False)
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MSE: {mse:.6f}\n\n")
        f.write(f"Intercept: {model.intercept_:.6f}\n\n")
        f.write("Coefficients:\n")
        f.write(coef_df.to_string())
        
    # 7. Threshold / Profit Analysis
    print("\n--- Profitability Analysis ---")
    thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    
    with open(summary_path, 'a') as f:
        f.write("\n\n--- Profitability Analysis ---\n")
        header = f"{'Threshold':<10} | {'Trades':<6} | {'Mean Return':<12} | {'Total Return':<12} | {'Win Rate':<8}"
        print(header)
        f.write(header + "\n")
        print("-" * 65)
        f.write("-" * 65 + "\n")
        
        for thresh in thresholds:
            mask = y_pred > thresh
            selected = y[mask]
            
            # Filter huge outliers > 50% for realistic average (as per prior user pref)
            selected_filtered = selected[selected < 0.50]
            
            if len(selected_filtered) > 0:
                mean_ret = selected_filtered.mean()
                total_ret = selected_filtered.sum()
                win_rate = (selected_filtered > 0).mean()
                count = len(selected_filtered)
            else:
                mean_ret = 0
                total_ret = 0
                win_rate = 0
                count = 0
                
            line = f">{thresh:<9.0%} | {count:<6} | {mean_ret:<12.2%} | {total_ret:<12.2%} | {win_rate:<8.0%}"
            print(line)
            f.write(line + "\n")

    # 8. Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Realized Return')
    plt.ylabel('Predicted Return')
    plt.title(f'Prediction Model (R2={r2:.2f}, w/ Volume)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'))
    plt.close()
    print(f"Analysis saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', required=True, help='Path to analysis history (features)')
    parser.add_argument('--stats', required=True, help='Path to volume/stats file (target)')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()
    
    train_model(args.history, args.stats, args.output)

if __name__ == "__main__":
    main()
