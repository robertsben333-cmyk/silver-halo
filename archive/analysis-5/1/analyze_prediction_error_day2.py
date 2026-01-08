import pandas as pd
import requests
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time, timedelta
import pytz
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY not found. Please set it in .env")
    sys.exit(1)

INPUT_FILE = "analysis_history.csv"
OUTPUT_DIR = "."

def get_minute_bars(ticker, date_str):
    """Fetch all 1-minute bars for a ticker on a specific date."""
    ny_tz = pytz.timezone("America/New_York")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    start_dt = ny_tz.localize(datetime.combine(dt.date(), time(9, 30)))
    end_dt = ny_tz.localize(datetime.combine(dt.date(), time(16, 0)))
    
    ts_start = int(start_dt.timestamp() * 1000)
    ts_end = int(end_dt.timestamp() * 1000)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{ts_start}/{ts_end}"
    params = {
        "apiKey": API_KEY,
        "limit": 50000,
        "sort": "asc",
        "adjusted": "true"
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                df['datetime'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ny_tz)
                return df
    except Exception as e:
        print(f"Error fetching data for {ticker} on {date_str}: {e}")
        
    return None

def find_next_trading_days(ticker, start_date_str, days_needed=2):
    """
    Find the next N valid trading days starting after start_date_str.
    Robustly checks if data exists to confirm day is valid.
    """
    found_days = []
    current_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    # Limit lookahead to avoid infinite loop
    max_lookahead = 10 
    attempts = 0
    
    while len(found_days) < days_needed and attempts < max_lookahead:
        current_date += timedelta(days=1)
        attempts += 1
        
        # Skip weekends simply
        if current_date.weekday() >= 5:
            continue
            
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Check if data exists
        bars = get_minute_bars(ticker, date_str)
        if bars is not None and not bars.empty:
            found_days.append((date_str, bars))
            
    return found_days

def load_model_and_predict(df):
    """Load trained model and add 'ModelScore' column to df."""
    import joblib
    import json
    
    ARTIFACTS_DIR = "../../apps/model_artifacts"
    model_path = os.path.join(ARTIFACTS_DIR, "return_predictor.joblib")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Model artifacts not found. Skipping model predictions.")
        return df

    try:
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
            
        print(f"Loaded model from {model_path}")
        
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        X = df[feature_names]
        df['ModelScore'] = model.predict(X)
        print("Model predictions added.")
        
    except Exception as e:
        print(f"Error predicting with model: {e}")
        
    return df

def main():
    print("Loading analysis history...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"File not found: {INPUT_FILE}")
        return

    # Add Model Predictions
    df = load_model_and_predict(df)
    
    results = []

    print(f"\nScanning {len(df)} tickers for Day 2 data...")
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        scan_date = row['date']
        
        # We need T=1 and T=2
        trading_days = find_next_trading_days(ticker, scan_date, days_needed=2)
        
        if len(trading_days) < 2:
            print(f"Skipping {ticker} on {scan_date}: Could not find 2 subsequent trading days.")
            continue
            
        t1_date, t1_bars = trading_days[0]
        t2_date, t2_bars = trading_days[1]
        
        # --- Metrics Calculation ---
        
        # T=1 (Strategy Day)
        # Strategy is Short, so 'Short Return'
        # But 'ModelScore' predicts 'Short Return' (Positive score = Good short)
        # WAIT. Model predicts Downside %? 
        # Checking analyze_next_day_strategy: "ModelScore" is used to pick shorts.
        # Ideally, ModelScore predicts Return. Let's assume ModelScore is Predicted Short Return %.
        
        t1_open = t1_bars.iloc[0]['o']
        t1_close = t1_bars.iloc[-1]['c']
        
        # Actual Short Return (T=1)
        t1_short_return = (t1_open - t1_close) / t1_open * 100
        
        # Model Prediction (T=1)
        # Note: ModelScore is in %.
        predicted_short_return = row.get('ModelScore', 0)
        
        # Error: "Mean error of the prediction".
        # If Model predicted 4% Short Return, and stock went UP 5% (Short Return = -5%)
        # Error = Actual - Predicted?
        # User: "Is the mean error... predictive of day after?"
        # Let's define Prediction Error = Actual Short Return - Predicted Return
        # If Error is Negative -> Stock performed WORSE (for short strategy) than expected. (i.e. stock price went UP higher than thought)
        # If Error is Positive -> Stock performed BETTER (dropped more) than expected.
        prediction_error = t1_short_return - predicted_short_return
        
        # Alternative Interpretation:
        # Maybe user means "Price Error" vs Model.
        # If Model predicts Drop, and Price Rises.
        # Let's track:
        # Stock Price Change T=1: (Close - Open)/Open
        # User asks: "if especially big losers of the short strategy, become big winners one day or two days later"
        # "Big loser of short strategy" = Stock Price Ripped UP on T=1.
        
        # T=2 (Day After)
        t2_open = t2_bars.iloc[0]['o']
        t2_close = t2_bars.iloc[-1]['c']
        
        # T2 Long Return (Winner?)
        t2_long_return = (t2_close - t2_open) / t2_open * 100
        
        results.append({
            "ticker": ticker,
            "scan_date": scan_date,
            "t1_date": t1_date,
            "t2_date": t2_date,
            "ModelScore": predicted_short_return,
            "T1_Short_Return": t1_short_return,
            "T1_Long_Return": -t1_short_return, # Approx
            "Prediction_Error": prediction_error,
            "T2_Long_Return": t2_long_return
        })

    if not results:
        print("No results found.")
        return

    res_df = pd.DataFrame(results)
    
    # Save Raw Data
    res_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_error_day2_data.csv"), index=False)
    print(f"Saved raw data to prediction_error_day2_data.csv")

    # Analysis 1: Correlation between Prediction Error and T2 Return
    corr = res_df['Prediction_Error'].corr(res_df['T2_Long_Return'])
    print(f"\nCorrelation (Prediction Error vs T2 Long Return): {corr:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=res_df, x='Prediction_Error', y='T2_Long_Return')
    sns.regplot(data=res_df, x='Prediction_Error', y='T2_Long_Return', scatter=False, color='red')
    plt.title(f"T1 Prediction Error vs T2 Long Return\nCorr: {corr:.2f}")
    plt.xlabel("T1 Prediction Error (Actual Short Ret - Model)\nNegative = Stock went UP (Short Failed)")
    plt.ylabel("T2 Long Return %")
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "error_vs_t2_return.png"))
    plt.close()
    
    # Analysis 2: "Big Losers of Short Strategy"
    # Identify trades where Short Strategy lost big (e.g. T1 Short Return < -5%)
    # Meaning Stock went UP > 5% on T1.
    
    big_losers = res_df[res_df['T1_Short_Return'] < -5].copy()
    print(f"\n--- Big Short Losers (Stock Ripped > 5% on T1) ---")
    print(f"Count: {len(big_losers)}")
    if not big_losers.empty:
        avg_t2 = big_losers['T2_Long_Return'].mean()
        win_rate_t2 = (big_losers['T2_Long_Return'] > 0).mean() * 100
        print(f"Avg T2 Return: {avg_t2:.2f}%")
        print(f"T2 Win Rate (Long): {win_rate_t2:.1f}%")
        print("\nTop Reversals (Big Losers -> T2 Winners):")
        print(big_losers[['ticker', 't1_date', 'T1_Short_Return', 'T2_Long_Return']].sort_values('T2_Long_Return', ascending=False).head())
    else:
        print("No big losers found (< -5% short return).")

    # Analysis 3: Error Quartiles
    res_df['Error_Quartile'] = pd.qcut(res_df['Prediction_Error'], 4, labels=['Large Negative Error', 'Negative Error', 'Positive Error', 'Large Positive Error'])
    print("\n--- Day 2 Performance by T1 Error Quartile ---")
    print(res_df.groupby('Error_Quartile')['T2_Long_Return'].agg(['mean', 'count', 'std']))

if __name__ == "__main__":
    main()
