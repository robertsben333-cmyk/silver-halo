import pandas as pd
import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Import utils
from model_utils import load_coefficients, fetch_historic_volume, calculate_model_prediction

load_dotenv()

HISTORY_FILE = os.path.join("outputs", "analysis_history.csv")
COLUMNS_ORDER = [
    "date", "ticker", "timestamp", "oneDayReturnPct", "finalScore",
    "modelReturnPrediction", 
    "nonFundamental", "news", "sentiment", "uncertainty", "confidence",
    "returnLikelihood", "evidenceCheckedCited", "reason",
    "metrics_EC", "metrics_PCR", "metrics_SD", "metrics_NRI",
    "metrics_HDM", "metrics_CONTR", "metrics_FRESH_NEG",
    "metrics_CP", "metrics_RD",
    "PriceAtOpen", "PriceAt1730CET", # From clean_and_enrich
    "lm_fit_long", "lm_fit_short", "avg_volume_30d" # User wants these strictly last
]

def is_vol_likely(val):
    try:
        f = float(val)
        return f > 1000 or f == 0 
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="Backfill analysis history with volume and split LM fits.")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving changes.")
    parser.add_argument("--limit", type=int, help="Limit number of rows to process.")
    args = parser.parse_args()
    
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Error: POLYGON_API_KEY not found.")
        sys.exit(1)
        
    if not os.path.exists(HISTORY_FILE):
        print(f"File not found: {HISTORY_FILE}")
        sys.exit(1)
        
    print(f"Reading {HISTORY_FILE}...")
    # Read CSV. Handle potential header issues. 
    # If the file was created by save_results, it has a header.
    # If it was cleaned by clean_and_enrich, it might NOT have a header?
    # Let's peek. Usually pandas sniffs it. 
    # But clean_and_enrich explicitly writes without index. 
    # wait, clean_and_enrich `final_df.to_csv(output_path, index=False)`. Pandas WRITES header by default.
    # So it should have a header.
    
    # Load raw lines to handle mixed schemas
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Loaded {len(lines)} lines.")
    
    # Schemas
    # 22-col: original
    schema_22 = [
        "date", "ticker", "timestamp", "oneDayReturnPct", "finalScore",
        "nonFundamental", "news", "sentiment", "uncertainty", "confidence",
        "returnLikelihood", "evidenceCheckedCited", "reason",
        "metrics_EC", "metrics_PCR", "metrics_SD", "metrics_NRI",
        "metrics_HDM", "metrics_CONTR", "metrics_FRESH_NEG",
        "metrics_CP", "metrics_RD"
    ]
    
    # 23-col: legacy (with modelReturnPrediction)
    schema_23 = [
        "date", "ticker", "timestamp", "oneDayReturnPct", "finalScore",
        "modelReturnPrediction",
        "nonFundamental", "news", "sentiment", "uncertainty", "confidence",
        "returnLikelihood", "evidenceCheckedCited", "reason",
        "metrics_EC", "metrics_PCR", "metrics_SD", "metrics_NRI",
        "metrics_HDM", "metrics_CONTR", "metrics_FRESH_NEG",
        "metrics_CP", "metrics_RD"
    ]
    
    # 26-col: new schema (previously defined with new cols in middle, now at end)
    # The FILE might have them in middle if we ran save_results recently?
    # Or if we ran the previous backfill script.
    # The previous backfill script wrote them in whatever order `df[final_order].to_csv` did.
    # Previous backfill logic used COLUMNS_ORDER which had them in MIDDLE.
    # So existing file likely has them in MIDDLE.
    # We must support reading that.
    
    schema_26_middle = [
        "date", "ticker", "timestamp", "oneDayReturnPct", "finalScore",
        "modelReturnPrediction", 
        "lm_fit_long", "lm_fit_short", "avg_volume_30d",
        "nonFundamental", "news", "sentiment", "uncertainty", "confidence",
        "returnLikelihood", "evidenceCheckedCited", "reason",
        "metrics_EC", "metrics_PCR", "metrics_SD", "metrics_NRI",
        "metrics_HDM", "metrics_CONTR", "metrics_FRESH_NEG",
        "metrics_CP", "metrics_RD"
    ]
    
    # If we already re-ordered, they might be at end (26 cols)
    schema_26_end = [
         "date", "ticker", "timestamp", "oneDayReturnPct", "finalScore",
        "modelReturnPrediction", 
        "nonFundamental", "news", "sentiment", "uncertainty", "confidence",
        "returnLikelihood", "evidenceCheckedCited", "reason",
        "metrics_EC", "metrics_PCR", "metrics_SD", "metrics_NRI",
        "metrics_HDM", "metrics_CONTR", "metrics_FRESH_NEG",
        "metrics_CP", "metrics_RD",
        "lm_fit_long", "lm_fit_short", "avg_volume_30d"
    ]
    
    parsed_rows = []
    
    import csv
    reader = csv.reader(lines)
    
    for row in reader:
        if not row: continue
        
        # Skip header if present
        if row[0] == "date":
            continue
            
        data = {}
        if len(row) == 22:
            data = dict(zip(schema_22, row))
        elif len(row) == 23:
            data = dict(zip(schema_23, row))
        elif len(row) == 26: 
             # Heuristic to detect middle vs end?
             # 'avg_volume_30d' is large number. 'nonFundamental' is 'Yes/No'.
             # Middle schema: idx 8 is avg_volume_30d.
             # End schema: idx 8 is nonFundamental.
             # Check idx 8.
             val_8 = row[8]
             # check if val_8 looks like volume (numeric) or nonFundamental (string yes/no)
             is_vol = False
             try:
                 float(val_8)
                 is_vol = True
             except:
                 pass
             
             if is_vol_likely(val_8): # Need helper? Just crude check
                  data = dict(zip(schema_26_middle, row))
             else:
                  data = dict(zip(schema_26_end, row))
                  
        else:
            # Maybe 28 with prices?
            print(f"Skipping row with {len(row)} columns: {row[:3]}")
            continue
        
        parsed_rows.append(data)
            
    df = pd.DataFrame(parsed_rows)
    print(f"Parsed {len(df)} valid rows.")
    
    # Clean numeric columns
    numeric_cols = [
        'oneDayReturnPct', 'finalScore', 'modelReturnPrediction', 'avg_volume_30d',
        'lm_fit_long', 'lm_fit_short'
    ]
    # Add metrics
    for c in df.columns:
        if c.startswith('metrics_'):
            numeric_cols.append(c)
            
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Ensure new columns exist
    new_cols = ['avg_volume_30d', 'lm_fit_long', 'lm_fit_short']
    for col in new_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    # Load Coefs
    long_coefs = load_coefficients('long')
    short_coefs = load_coefficients('short')
    
    if not long_coefs or not short_coefs:
        print("Error loading coefficients. Exiting.")
        sys.exit(1)
        
    # Process
    cache = {}
    
    updated_count = 0
    
    rows_to_proc = df.index
    if args.limit:
        rows_to_proc = rows_to_proc[:args.limit]
    
    total = len(rows_to_proc)
    print(f"Processing {total} rows...")
    
    for idx in rows_to_proc:
        row = df.loc[idx]
        
        ticker = row['ticker']
        date_str = str(row['date'])
        
        # 1. Fetch Volume
        # Check if already has volume? 
        # Only fetch if 'avg_volume_30d' is 0 or NaN.
        vol = row.get('avg_volume_30d', 0)
        if pd.isna(vol) or vol == 0:
            cache_key = (ticker, date_str)
            if cache_key in cache:
                vol = cache[cache_key]
            else:
                print(f"[{idx+1}/{total}] Fetching vol for {ticker} on {date_str}...")
                vol = fetch_historic_volume(ticker, date_str, api_key)
                cache[cache_key] = vol
            
            df.at[idx, 'avg_volume_30d'] = vol
            updated_count += 1
            
        # 2. Calculate Fits
        # We need metrics dictionary
        # metrics_* are columns.
        metrics = {}
        for col in df.columns:
            if col.startswith('metrics_'):
                key = col.replace('metrics_', '')
                metrics[key] = row[col]
        
        # Add return pct
        metrics['oneDayReturnPct'] = row.get('oneDayReturnPct', 0.0)
        
        # Fits
        final_score = row.get('finalScore', 0)
        confidence = row.get('confidence', 'Medium')
        uncertainty = row.get('uncertainty', 'Medium')
        
        fit_long = calculate_model_prediction(metrics, final_score, confidence, uncertainty, vol, long_coefs)
        fit_short = calculate_model_prediction(metrics, final_score, confidence, uncertainty, vol, short_coefs)
        
        df.at[idx, 'lm_fit_long'] = fit_long
        df.at[idx, 'lm_fit_short'] = fit_short
        
    print(f"Updated {updated_count} volume entries.")
    
    if args.dry_run:
        print("Dry run complete. Not saving.")
        print(df[['date', 'ticker', 'avg_volume_30d', 'lm_fit_long', 'lm_fit_short']].head())
    else:
        # Save
        # Reorder columns to match desire if possible, but keep extras
        # We explicitly want COLUMNS_ORDER first.
        # Ensure all COLUMNS_ORDER exist in df for safe selection
        for c in COLUMNS_ORDER:
            if c not in df.columns:
                # If PriceAtOpen missing, fill with NaN or empty?
                # Usually we want to keep it if it was there.
                # If it wasn't there, do we add it? 
                # Yes, user expects these columns.
                df[c] = "" # or NaN

        existing_cols = [c for c in COLUMNS_ORDER]
        extra_cols = [c for c in df.columns if c not in COLUMNS_ORDER]
        final_order = existing_cols + extra_cols
        
        df = df[final_order]
        df.to_csv(HISTORY_FILE, index=False)
        print(f"Saved updated history to {HISTORY_FILE}")

if __name__ == "__main__":
    main()
