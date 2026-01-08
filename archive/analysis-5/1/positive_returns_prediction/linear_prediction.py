import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os

def run_linear_prediction():
    history_path = '../analysis_history.csv'
    returns_path = 'calculated_returns.csv'
    output_dir = '.'
    
    if not os.path.exists(history_path) or not os.path.exists(returns_path):
        print("Input files not found.")
        return

    # Load Data
    df_hist = pd.read_csv(history_path)
    df_ret = pd.read_csv(returns_path)
    
    # Merge
    # analysis_history: date, ticker
    # calculated_returns: AnalysisDate, Ticker
    merged = pd.merge(df_hist, df_ret, left_on=['ticker', 'date'], right_on=['Ticker', 'AnalysisDate'])
    
    if merged.empty:
        print("Merged dataset is empty. Check dates and tickers.")
        return
        
    print(f"Dataset Size: {len(merged)}")
    
    # Feature Selection
    # Continuous Features
    features_num = [
        'oneDayReturnPct', 'finalScore', 
        'metrics_PCR', 'metrics_EC', 'metrics_SD', 'metrics_NRI', 
        'metrics_HDM', 'metrics_CONTR', 'metrics_CP', 'metrics_RD',
        # 'sent_pro', 'sent_com' # These might be missing in history if not populated, check file
    ]
    
    # Check if sent_pro/com exist
    if 'sent_pro' in merged.columns:
        features_num.append('sent_pro')
    if 'sent_com' in merged.columns:
        features_num.append('sent_com')
        
    # Categorical Features to Encode
    features_cat = ['confidence', 'uncertainty']
    # FRESH_NEG is int (0/1), treat as num or cat? It is 0/1, so Num is fine.
    if 'metrics_FRESH_NEG' in merged.columns:
        features_num.append('metrics_FRESH_NEG')

    # Prepare X and y
    X_num = merged[features_num].fillna(0) # Simple imputation
    
    # Encode Categoricals
    # Map Low/Medium/High to 0, 0.5, 1 or OneHot?
    # Simple constraints often use Ordinal. 
    # Let's use OneHot for standard linear ref.
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = pd.DataFrame(encoder.fit_transform(merged[features_cat]))
    X_cat.columns = encoder.get_feature_names_out(features_cat)
    
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    y = merged['Return']
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    # Evaluation
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    
    # Coefficients
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='AbsCoef', ascending=False)
    
    print("\nTop Predictors:")
    print(coef_df.head(10))
    
    # Save Summary
    with open('linear_model_summary.txt', 'w') as f:
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MSE: {mse:.6f}\n\n")
        f.write(f"Intercept: {model.intercept_:.6f}\n\n")
        f.write("Coefficients:\n")
        f.write(coef_df.to_string())
        
    print(f"Intercept: {model.intercept_:.6f}")
        
    # Threshold Analysis
    print("\n--- Threshold Analysis ---")
    thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    print(f"{'Threshold':<10} | {'Trades':<6} | {'Mean Return':<12} | {'Total Return':<12} | {'Win Rate':<8}")
    print("-" * 65)
    
    analysis_results = []
    
    for thresh in thresholds:
        # Indices where prediction > threshold
        mask = y_pred > thresh
        selected_actuals = y[mask]
        
        # Filter outliers for reporting (Exclude > 50% returns)
        # User requested removing the ~55% (actually 50.8%) outlier for averages
        filtered_actuals = selected_actuals[selected_actuals < 0.50]
        
        if len(filtered_actuals) > 0:
            mean_ret = filtered_actuals.mean()
            total_ret = filtered_actuals.sum()
            win_rate = (filtered_actuals > 0).mean()
            count = len(filtered_actuals)
            
            # Count including outlier for transparency if needed, but user asked to remove it from avgs
            total_count_incl_outlier = len(selected_actuals)
        else:
            mean_ret = 0
            total_ret = 0
            win_rate = 0
            count = 0
            total_count_incl_outlier = 0
            
        print(f">{thresh:<9.0%} | {count:<6} (of {total_count_incl_outlier}) | {mean_ret:<12.2%} | {total_ret:<12.2%} | {win_rate:<8.0%}")
        analysis_results.append({
            'Threshold': thresh,
            'Trades': count,
            'MeanReturn': mean_ret,
            'TotalReturn': total_ret,
            'WinRate': win_rate
        })

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title(f'Linear Prediction (R2={r2:.2f})')
    plt.grid(True)
    plt.savefig('predicted_vs_actual.png')
    print("Saved predicted_vs_actual.png")

if __name__ == "__main__":
    run_linear_prediction()
