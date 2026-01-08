import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

# Configuration
INPUT_FILE = r"C:\Users\XavierFriesen\.gemini\antigravity\playground\silver-halo\analysis-5\1\analysis_history.csv"
OUTPUT_DIR = r"C:\Users\XavierFriesen\.gemini\antigravity\playground\silver-halo\analysis-5\1"
MODEL_EVAL_FILE = os.path.join(OUTPUT_DIR, "model_evaluation.txt")
STOPLOSS_PCT = 0.04

def train_and_evaluate():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # Data Preparation
    df['PriceAtOpen'] = pd.to_numeric(df['PriceAtOpen'], errors='coerce')
    df['PriceAt1730CET'] = pd.to_numeric(df['PriceAt1730CET'], errors='coerce')
    
    # Calculate Realized Return
    df['raw_return'] = (df['PriceAtOpen'] - df['PriceAt1730CET']) / df['PriceAtOpen']
    df['realized_return'] = df['raw_return'].apply(lambda x: max(x, -STOPLOSS_PCT))
    
    # Determine Features
    # Top correlated features from previous analysis
    features = ['sentiment', 'metrics_CP', 'finalScore', 'returnLikelihood', 'metrics_SD', 'metrics_NRI']
    
    # Ensure numeric
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows with missing values in features or target
    df_model = df.dropna(subset=features + ['realized_return']).copy()
    
    X = df_model[features]
    y = df_model['realized_return'] * 100 # Predict Percentage Return

    print(f"Training Data Size: {len(df_model)}")

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest (Reg)": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_model_name = ""
    best_correlation = -1
    best_predictions = None

    with open(MODEL_EVAL_FILE, "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("========================\n\n")

        for name, model in models.items():
            # Cross Validation Predictions
            y_pred_cv = cross_val_predict(model, X, y, cv=kf)
            
            # Metrics
            correlation = np.corrcoef(y_pred_cv, y)[0, 1]
            r2 = r2_score(y, y_pred_cv)
            
            f.write(f"Model: {name}\n")
            f.write(f"Correlation (CV Predictions vs Actual): {correlation:.4f}\n")
            f.write(f"R-Squared (CV): {r2:.4f}\n")
            f.write("-" * 30 + "\n")
            
            results.append({"Model": name, "Correlation": correlation, "R2": r2})
            
            # track best model
            if correlation > best_correlation:
                best_correlation = correlation
                best_model_name = name
                best_predictions = y_pred_cv

        
        # Train Best Model on Full Set for Final "Model Score"
        print(f"\nBest Model: {best_model_name} (Corr: {best_correlation:.2f})")
        final_model = models[best_model_name]
        final_model.fit(X, y)
        df_model['ModelScore'] = final_model.predict(X)
        
        # Save Model Artifacts
        import joblib
        ARTIFACTS_DIR = os.path.join(os.path.dirname(INPUT_FILE), "..", "..", "apps", "model_artifacts")
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
        joblib.dump(final_model, model_path)
        print(f"Model saved to: {model_path}")
        
        features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
        import json
        with open(features_path, 'w') as f:
            json.dump(features, f)
        print(f"Features saved to: {features_path}")
        
        # Save Model Predictions to verify "Model Score" effectiveness
        # We want to see if 'ModelScore' > Threshold gives us better returns
        
        # Plot Model Prediction vs Actual
        plt.figure(figsize=(6, 6))
        plt.scatter(df_model['ModelScore'], df_model['realized_return'] * 100, alpha=0.7, color='green')
        
        # Plot ideal line
        min_val = min(df_model['ModelScore'].min(), (df_model['realized_return'] * 100).min())
        max_val = max(df_model['ModelScore'].max(), (df_model['realized_return'] * 100).max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal Prediction')
        
        plt.title(f"{best_model_name}: Predicted vs Actual Returns\nCorrelation: {best_correlation:.2f}")
        plt.xlabel("Predicted Return (%)")
        plt.ylabel("Actual Realized Return (%)")
        plt.grid(True)
        plt.legend()
        
        OUTPUT_MODEL_PLOT = os.path.join(OUTPUT_DIR, "model_correlation.png")
        plt.savefig(OUTPUT_MODEL_PLOT)
        print(f"Model correlation plot saved to: {OUTPUT_MODEL_PLOT}")

        # Strategy Simulation with Model Score
        # Strategy: Long if Model Score > T? (Since Model Score = Expected Return)
        # OR Short if Model Score > T?
        # WAIT. We calculated Return for SHORT strategy.
        # So 'realized_return' is ALREADY the Short Return.
        # High 'realized_return' = Good Short.
        # So we want High 'Model Score'.
        # Strategy: Take trade if Model Score > Threshold.
        
        thresholds = np.linspace(df_model['ModelScore'].min(), df_model['ModelScore'].max(), 50)
        strat_results = []
        
        unique_days = df_model['date'].nunique() if 'date' in df_model.columns else 1
        if 'date' in df_model.columns:
             df_model['date'] = pd.to_datetime(df_model['date'])
             unique_days = df_model['date'].nunique()

        for t in thresholds:
            # Take trades where we Predict Return > T
            subset = df_model[df_model['ModelScore'] > t]
            if not subset.empty:
                avg_ret = subset['realized_return'].mean() * 100 # realized_return is already pct (0.05) but scaled to 100 for y
                # Wait, y was *100. realized_return col is 0.05.
                # So taking mean of realized_return * 100
                
                count = len(subset)
                trades_per_day = count / unique_days
                strat_results.append({'Threshold': t, 'AvgReturn': avg_ret, 'TradesPerDay': trades_per_day})
        
        strat_df = pd.DataFrame(strat_results)
        
        if not strat_df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(strat_df['Threshold'], strat_df['AvgReturn'], label='Avg Return (%)', color='purple')
            plt.xlabel("Model Score Threshold (Trade if Pred Return > T)")
            plt.ylabel("Average Actual Return (%)")
            plt.title(f"Strategy Optimization using {best_model_name}")
            plt.grid(True)
            
            ax2 = plt.gca().twinx()
            ax2.plot(strat_df['Threshold'], strat_df['TradesPerDay'], label='Trades/Day', color='gray', linestyle='--')
            ax2.set_ylabel("Avg Trades per Day")
            ax2.axhline(y=2.0, color='gray', linestyle=':', label='Target (2.0)')
            
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            OUTPUT_STRAT_PLOT = os.path.join(OUTPUT_DIR, "model_threshold_analysis.png")
            plt.savefig(OUTPUT_STRAT_PLOT)
            print(f"Model strategy plot saved to: {OUTPUT_STRAT_PLOT}")
            
            # Find optimal point
            valid_strat = strat_df[strat_df['TradesPerDay'] >= 2.0]
            if valid_strat.empty:
                 # Fallback to max return
                 best_row = strat_df.loc[strat_df['AvgReturn'].idxmax()]
                 print(f"\nOptimal Model Strategy (Max Return):")
            else:
                best_row = valid_strat.loc[valid_strat['AvgReturn'].idxmax()]
                print(f"\nOptimal Model Strategy (>= 2 trades/day):")
            
            print(f"Threshold (Predicted Return >): {best_row['Threshold']:.2f}%")
            print(f"Expected Avg Return: {best_row['AvgReturn']:.2f}%")
            print(f"Trades/Day: {best_row['TradesPerDay']:.2f}")

        # --- Time vs Accuracy Analysis ---
        print("\nRunning Time vs Accuracy Analysis...")
        # timestamp is ISO format. Market Open is 09:30 ET (15:30 CET usually).
        # We'll approximate Market Open as 15:30 on the 'date' day.
        # Assuming 'timestamp' and 'date' columns exist.
        
        if 'timestamp' in df_model.columns and 'date' in df_model.columns:
            try:
                # Calculate Absolute Error
                df_model['ModelError'] = (df_model['ModelScore'] - (df_model['realized_return'] * 100)).abs()
                
                # timestamps
                df_model['ts_dt'] = pd.to_datetime(df_model['timestamp'])
                
                # approximate open time (using 15:30 local/CET as proxy, or just set 9:30 ET if we had tz)
                # Let's assume the 'date' column is the trading day.
                # And let's assume 'timestamp' is in same timezone or convert both to UTC?
                # Simpler: Extract hour/minute from timestamp. 
                # If timestamp is 08:00 and Open is 09:30. Delta is 1.5 hours.
                # We need to correctly handle the Date part too (analysis could be Day-1).
                
                # Construct Open Datetime: Date + 15:30:00 (assuming CET/Local user time for strings)
                # Or 9:30 ET? The user seems to be in CET based on "PriceAt1730CET".
                # 15:30 CET is 9:30 ET.
                
                df_model['OpenDateTime'] = pd.to_datetime(df_model['date'].astype(str) + " 15:30:00")
                
                # If timestamp is timezone aware, start by stripping it for simple subtraction (or adding to Open)
                if df_model['ts_dt'].dt.tz is not None:
                     df_model['ts_dt'] = df_model['ts_dt'].dt.tz_localize(None)

                df_model['HoursToOpen'] = (df_model['OpenDateTime'] - df_model['ts_dt']).dt.total_seconds() / 3600.0
                
                # Filter reasonable range (e.g. within 24 hours) to avoid Bad Data skewing plots
                time_df = df_model[abs(df_model['HoursToOpen']) < 48].dropna(subset=['HoursToOpen', 'ModelError'])
                
                if not time_df.empty:
                    # Correlation
                    corr_time = time_df['HoursToOpen'].corr(time_df['ModelError'])
                    
                    print(f"Correlation (HoursToOpen vs ModelError): {corr_time:.4f}")
                    # Positive correlation => More time before open = Higher Error (Worse).
                    # Negative correlation => More time before open = Lower Error (Better?).
                    
                    plt.figure(figsize=(8, 6))
                    plt.scatter(time_df['HoursToOpen'], time_df['ModelError'], alpha=0.6, color='orange')
                    plt.title(f"Model Error vs. Hours Before Open (Corr: {corr_time:.2f})")
                    plt.xlabel("Hours Before Market Open (15:30 CET)")
                    plt.ylabel("Absolute Prediction Error (%)")
                    plt.grid(True)
                    plt.axvline(x=0, color='red', linestyle='--', label='Market Open')
                    plt.legend()
                    
                    OUTPUT_TIME = os.path.join(OUTPUT_DIR, "error_vs_time.png")
                    plt.savefig(OUTPUT_TIME)
                    print(f"Time accuracy plot saved to: {OUTPUT_TIME}")
                else:
                    print("No valid time data found for analysis.")

            except Exception as e:
                print(f"Error in Time Analysis: {e}")
                
        # --- Prediction Range vs Accuracy Analysis ---
        print("\nRunning Prediction Range vs Accuracy Analysis...")
        # We want to see if the model is more accurate when it predicts high returns vs low returns.
        # We'll plot ModelError vs ModelScore.
        
        try:
            # Ensure ModelError is calculated (might have failed in previous block if cols missing, so recalc safely)
            if 'realized_return' in df_model.columns and 'ModelScore' in df_model.columns:
                 df_model['ModelError'] = (df_model['ModelScore'] - (df_model['realized_return'] * 100)).abs()
                 
                 # Bin the predictions
                 # Create 5 bins based on Model Score quantiles or fixed ranges
                 df_model['PredBin'] = pd.qcut(df_model['ModelScore'], q=5, duplicates='drop')
                 
                 bin_stats = df_model.groupby('PredBin', observed=True)['ModelError'].agg(['mean', 'count']).reset_index()
                 bin_stats['BinMid'] = bin_stats['PredBin'].apply(lambda x: x.mid).astype(float)
                 
                 print("\nError by Prediction Range:")
                 print(bin_stats)
                 
                 # Plot
                 plt.figure(figsize=(8, 6))
                 
                 # Scatter of all points
                 plt.scatter(df_model['ModelScore'], df_model['ModelError'], alpha=0.4, color='gray', label='Individual Trades')
                 
                 # Line of avg error per bin
                 plt.plot(bin_stats['BinMid'], bin_stats['mean'], color='red', marker='o', linewidth=2, label='Avg Error per Range')
                 
                 plt.title("Model Accuracy vs. Predicted Return Range")
                 plt.xlabel("Predicted Return (%)")
                 plt.ylabel("Absolute Prediction Error (%)")
                 plt.grid(True)
                 plt.legend()
                 
                 OUTPUT_RANGE = os.path.join(OUTPUT_DIR, "error_vs_prediction_range.png")
                 plt.savefig(OUTPUT_RANGE)
                 print(f"Range accuracy plot saved to: {OUTPUT_RANGE}")
                 
        except Exception as e:
            print(f"Error in Range Analysis: {e}")

if __name__ == "__main__":
    train_and_evaluate()
