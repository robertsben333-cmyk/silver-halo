import pandas as pd
import requests
import os
import sys
from datetime import datetime, time, timedelta
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY not found in environment variables.")
    sys.exit(1)

INPUT_DIR = "outputs"
OUTPUT_DIR = "analysis-5/1"
ANALYSIS_FILE = "analysis_history.csv"
DOWNSIDE_FILE = "downside_history.csv"

def get_polygon_price(ticker, timestamp_ms):
    """Fetch 1-minute bar for a specific timestamp."""
    # Convert ms timestamp to YYYY-MM-DD
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
    date_str = dt.strftime("%Y-%m-%d")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{timestamp_ms}/{timestamp_ms + 600000}" # Scan 10 mins ahead just in case
    params = {
        "apiKey": API_KEY,
        "limit": 1,
        "sort": "asc"
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("resultsCount", 0) > 0:
                result = data["results"][0]
                # Return the close price of that minute bar? or Open? 
                # Request says "Price at Open" and "Price at 17.30".
                # For specific timepoints, Close of that minute is usually best proxy for "price at that time".
                # For "Market Open", usually Open of the 9:30 bar is used.
                # Let's return entire object to handle logic outside.
                return result
    except Exception as e:
        print(f"Error fetching Polygon data for {ticker} at {dt}: {e}")
    
    return None

def is_trading_day(date_str):
    """Check if a date was a trading day using Polygon Market Status."""
    url = f"https://api.polygon.io/v1/marketstatus/upcoming?apiKey={API_KEY}"
    # This endpoint shows upcoming holidays. To check past status, we might just rely on data availability.
    # A better way for past dates is to check if we can get Grouped Daily bars for SPY on that day.
    
    try:
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        params = {"apiKey": API_KEY, "limit": 1} # Just need 1 result to confirm market was open
        resp = requests.get(url, params=params)
        if resp.status_code == 200 and resp.json().get("resultsCount", 0) > 0:
            return True
    except:
        pass
    return False

def get_market_open_timestamp(date_str):
    """Get timestamp for 9:30 AM ET on the given date."""
    ny_tz = pytz.timezone("America/New_York")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt_open = ny_tz.localize(datetime.combine(dt.date(), time(9, 30)))
    return int(dt_open.timestamp() * 1000)

def get_1730_cet_timestamp(date_str):
    """Get timestamp for 17:30 CET on the given date."""
    # CET can be CET or CEST. 'Europe/Paris' handles this.
    paris_tz = pytz.timezone("Europe/Paris")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt_1730 = paris_tz.localize(datetime.combine(dt.date(), time(17, 30)))
    return int(dt_1730.timestamp() * 1000)


# ... (Imports and constants remain the same) ...

def load_price_cache():
    """Load existing prices from already processed analysis_history.csv."""
    cache = {}
    path = os.path.join(OUTPUT_DIR, ANALYSIS_FILE)
    if os.path.exists(path):
        print(f"Loading price cache from {path}...")
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                key = (row['ticker'], row['date'])
                cache[key] = (row.get('PriceAtOpen'), row.get('PriceAt1730CET'))
            print(f"Loaded {len(cache)} price entries.")
        except Exception as e:
            print(f"Error loading price cache: {e}")
    return cache

def process_file(filename, columns, price_cache=None):
    print(f"Processing {filename}...")
    input_path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} does not exist. Skipping.")
        return

    try:
        df = pd.read_csv(input_path, header=None, names=columns, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # Infer or parsing date
    if 'date' not in df.columns:
        if 'timestamp' in df.columns:
            # Try to parse ISO timestamp
            df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        if 'date' not in df.columns or df['date'].isna().all():
             if 'run_id' in df.columns:
                 # Try run_id if timestamp failed or wasn't present
                 df['date'] = pd.to_datetime(df['run_id'], format='%Y%m%d_%H%M%S', errors='coerce').dt.strftime('%Y-%m-%d')
             
    # Fill any remaining NaNs if possible or report
    missing_dates = df['date'].isna().sum()
    if missing_dates > 0:
        print(f"Warning: {missing_dates} rows have invalid dates and will be dropped.")
        
    df = df.dropna(subset=['date'])

    cleaned_rows = []
    
    grouped = df.groupby('date')
    unique_dates = list(grouped.groups.keys())
    print(f"Found {len(unique_dates)} unique dates.")
    
    for date_val, group in grouped:
        print(f"  Date: {date_val}", end=" ", flush=True)
        
        if not is_trading_day(str(date_val)):
            print("-> Skipped (Non-trading/Holiday)")
            continue
            
        print("-> Processing")
        
        latest_10 = group.tail(10).copy()
        
        for idx, row in latest_10.iterrows():
            ticker = row['ticker']
            price_open = None
            price_1730 = None
            
            # Check Cache first
            if price_cache:
                cached = price_cache.get((ticker, str(date_val)))
                if cached:
                    price_open, price_1730 = cached
            
            # Fetch if missing
            if price_open is None or pd.isna(price_open):
                ts_open = get_market_open_timestamp(str(date_val))
                bar_open = get_polygon_price(ticker, ts_open)
                price_open = bar_open['o'] if bar_open else None
                
            if price_1730 is None or pd.isna(price_1730):
                ts_1730 = get_1730_cet_timestamp(str(date_val))
                bar_1730 = get_polygon_price(ticker, ts_1730)
                price_1730 = bar_1730['c'] if bar_1730 else None
            
            latest_10.at[idx, 'PriceAtOpen'] = price_open
            latest_10.at[idx, 'PriceAt1730CET'] = price_1730
            
        cleaned_rows.append(latest_10)

    if cleaned_rows:
        final_df = pd.concat(cleaned_rows)
        output_path = os.path.join(OUTPUT_DIR, filename)
        final_df.to_csv(output_path, index=False)
        print(f"Saved cleaned file to {output_path}")
    else:
        print("No data remained after cleaning.")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    analysis_columns = [
        "date", "ticker", "timestamp", "oneDayReturnPct", "finalScore",
        "nonFundamental", "news", "sentiment", "uncertainty", "confidence",
        "returnLikelihood", "evidenceCheckedCited", "reason",
        "metrics_EC", "metrics_PCR", "metrics_SD", "metrics_NRI",
        "metrics_HDM", "metrics_CONTR", "metrics_FRESH_NEG",
        "metrics_CP", "metrics_RD"
    ]
    # Process Analysis History (Source of Truth for Prices)
    process_file(ANALYSIS_FILE, analysis_columns)
    
    # Load Cache after processing analysis file
    price_cache = load_price_cache()
    
    downside_columns = [
        "run_id", "timestamp", "ticker", "dcl_prob", "scs_score", 
        "rank", "driver_category", "reason"
    ]
    # Process Downside History using Cache
    process_file(DOWNSIDE_FILE, downside_columns, price_cache=price_cache)

if __name__ == "__main__":
    main()
