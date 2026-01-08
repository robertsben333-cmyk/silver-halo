
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import os

def analyze_deviation():
    # Paths
    history_path = '../analysis_history.csv'
    returns_path = 'full_returns_data.csv'
    
    if not os.path.exists(history_path) or not os.path.exists(returns_path):
        print("Input files not found.")
        return

    # Load Data
    df_hist = pd.read_csv(history_path)
    df_ret = pd.read_csv(returns_path)
    
    # Merge
    merged = pd.merge(df_hist, df_ret, left_on=['ticker', 'date'], right_on=['Ticker', 'AnalysisDate'])
    print(f"Merged Data Size: {len(merged)}")
    
    # Features
    features_num = [
        'oneDayReturnPct', 'finalScore', 
        'metrics_PCR', 'metrics_EC', 'metrics_SD', 'metrics_NRI', 
        'metrics_HDM', 'metrics_CONTR', 'metrics_CP', 'metrics_RD'
    ]
    if 'metrics_FRESH_NEG' in merged.columns: features_num.append('metrics_FRESH_NEG')
    features_cat = ['confidence', 'uncertainty']
    
    # Preprocessing
    X_num = merged[features_num].fillna(0)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = pd.DataFrame(encoder.fit_transform(merged[features_cat]))
    X_cat.columns = encoder.get_feature_names_out(features_cat)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    
    # Targets
    y_short = merged['ShortReturn_T1']
    y_long = merged['LongReturn_T1']
    y_day2 = merged['Day2Return'] # T+2
    
    # 1. Train Short Model
    model_short = LinearRegression()
    # Handle NaNs in targets if any (failed downloads)
    valid_short = y_short.notna()
    model_short.fit(X[valid_short], y_short[valid_short])
    pred_short = model_short.predict(X)
    merged['Pred_Short'] = pred_short
    merged['Dev_Short'] = pred_short - merged['ShortReturn_T1'] # Positive Deviation = Predicted > Actual (Disappointment)
    
    # 2. Train Long Model
    model_long = LinearRegression()
    valid_long = y_long.notna()
    model_long.fit(X[valid_long], y_long[valid_long])
    pred_long = model_long.predict(X)
    merged['Pred_Long'] = pred_long
    merged['Dev_Long'] = pred_long - merged['LongReturn_T1']
    
    # 3. Analysis: Worse Performance Events (Pred > 4% and Actual < 0)
    print("\n--- Deviation Analysis ---")
    
    # Short Deviations
    short_fail_mask = (merged['Pred_Short'] > 0.04) & (merged['ShortReturn_T1'] < 0)
    short_failures = merged[short_fail_mask]
    
    print(f"\nShort Strategy Failures (Pred > 4%, Actual < 0): {len(short_failures)}")
    if not short_failures.empty:
        corr_s = short_failures['Dev_Short'].corr(short_failures['Day2Return'])
        print(f"Correlation (Deviation vs T+2 Return) in Failures: {corr_s:.4f}")
        print("Detailed Failures (Top 5 Deviations):")
        print(short_failures[['Ticker', 'AnalysisDate', 'Pred_Short', 'ShortReturn_T1', 'Dev_Short', 'Day2Return']].sort_values('Dev_Short', ascending=False).head(5))

    # Long Deviations
    long_fail_mask = (merged['Pred_Long'] > 0.04) & (merged['LongReturn_T1'] < 0)
    long_failures = merged[long_fail_mask]
    
    print(f"\nLong Strategy Failures (Pred > 4%, Actual < 0): {len(long_failures)}")
    if not long_failures.empty:
        corr_l = long_failures['Dev_Long'].corr(long_failures['Day2Return'])
        print(f"Correlation (Deviation vs T+2 Return) in Failures: {corr_l:.4f}")
        print("Detailed Failures (Top 5 Deviations):")
        print(long_failures[['Ticker', 'AnalysisDate', 'Pred_Long', 'LongReturn_T1', 'Dev_Long', 'Day2Return']].sort_values('Dev_Long', ascending=False).head(5))

    # 4. Global Correlations
    valid_day2 = merged['Day2Return'].notna()
    global_corr_short = merged.loc[valid_day2, 'Dev_Short'].corr(merged.loc[valid_day2, 'Day2Return'])
    global_corr_long = merged.loc[valid_day2, 'Dev_Long'].corr(merged.loc[valid_day2, 'Day2Return'])
    
    print(f"\nGlobal Correlation (Short Deviation vs T+2): {global_corr_short:.4f}")
    print(f"Global Correlation (Long Deviation vs T+2): {global_corr_long:.4f}")
    
    # 5. Plots
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=merged, x='Dev_Short', y='Day2Return', hue=short_fail_mask, palette={False: 'blue', True: 'red'})
    plt.title(f'Short Deviation vs T+2 Return (Corr={global_corr_short:.2f})')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=merged, x='Dev_Long', y='Day2Return', hue=long_fail_mask, palette={False: 'green', True: 'red'})
    plt.title(f'Long Deviation vs T+2 Return (Corr={global_corr_long:.2f})')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    plt.savefig('deviation_analysis_plot.png')
    print("\nSaved deviation_analysis_plot.png")

if __name__ == "__main__":
    analyze_deviation()
