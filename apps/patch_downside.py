import pandas as pd
import json
import os
import sys
from datetime import datetime
from clean_and_enrich import load_price_cache, get_market_open_timestamp, get_polygon_price, get_1730_cet_timestamp, OUTPUT_DIR, ANALYSIS_FILE

DOWNSIDE_FILE = "downside_history.csv"
SOURCE_REPORT_PATH = "outputs/20260105_113654/downside_report.json"
RUN_ID = "20260105_113654"
TIMESTAMP = "2026-01-05T11:36:54.000000" # Approx
DATE_VAL = "2026-01-05"

def patch_downside():
    csv_path = os.path.join(OUTPUT_DIR, DOWNSIDE_FILE)
    
    # 1. Load and clean existing file
    print(f"Loading {csv_path}...")
    try:
        # Read as strings to check format easily
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        cleaned_lines = []
        header = lines[0]
        cleaned_lines.append(header)
        
        for line in lines[1:]:
            parts = line.split(',')
            # Check if first column is likely a run_id (e.g. 2025...)
            # Analysis history rows start with 2025- or 2026- (date)
            # Downside history rows start with num_num
            first_col = parts[0]
            if "_" in first_col and not first_col.startswith("2025-") and not first_col.startswith("2026-"):
                cleaned_lines.append(line)
            else:
                print(f"Removing bad line: {line[:50]}...")
                
        # Write back cleaned lines
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        print("Cleaned bad rows.")
        
    except Exception as e:
        print(f"Error cleaning file: {e}")
        return

    # 2. Load Price Cache
    price_cache = load_price_cache()
    
    # 3. Load Source Data
    print(f"Loading source report from {SOURCE_REPORT_PATH}...")
    with open(SOURCE_REPORT_PATH, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
        
    # 4. Create new rows
    new_rows = []
    headers = [
        "run_id", "timestamp", "ticker", "dcl_prob", "scs_score", 
        "rank", "driver_category", "reason", "date", "PriceAtOpen", "PriceAt1730CET"
    ]
    
    for item in report_data:
        ticker = item['ticker']
        
        # Get prices
        price_open = None
        price_1730 = None
        
        cached = price_cache.get((ticker, DATE_VAL))
        if cached:
            price_open, price_1730 = cached
            
        # Fallback fetch if strictly needed (should be in cache if analysis processed)
        if price_open is None:
             print(f"Fetching open price for {ticker}...")
             ts = get_market_open_timestamp(DATE_VAL)
             bar = get_polygon_price(ticker, ts)
             price_open = bar['o'] if bar else ""
             
        if price_1730 is None:
             print(f"Fetching 17:30 price for {ticker}...")
             ts = get_1730_cet_timestamp(DATE_VAL)
             bar = get_polygon_price(ticker, ts)
             price_1730 = bar['c'] if bar else ""
             
        # Format row
        row = {
            "run_id": RUN_ID,
            "timestamp": TIMESTAMP,
            "ticker": ticker,
            "dcl_prob": item.get('downsideContinuationLikelihoodNextDay', '').replace('%',''),
            "scs_score": item.get('shortCandidateScore', ''),
            "rank": item.get('rank', ''),
            "driver_category": item.get('driverCategory', ''),
            "reason": item.get('reason', '').replace('\n', ' ').replace('"', "'"), # Basic CSV escape
            "date": DATE_VAL,
            "PriceAtOpen": price_open,
            "PriceAt1730CET": price_1730
        }
        
        # Handle quotes for CSV manually or use pandas
        new_rows.append(row)
        
    # Append using pandas to handle CSV escaping properly
    df_new = pd.DataFrame(new_rows)
    # Ensure columns order
    df_new = df_new[headers]
    
    df_new.to_csv(csv_path, mode='a', header=False, index=False)
    print("Appended new rows successfully.")

if __name__ == "__main__":
    patch_downside()
