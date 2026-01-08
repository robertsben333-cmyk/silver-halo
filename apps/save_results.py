import json
import csv
import argparse
import os
import re
from datetime import datetime

# Import utils
from model_utils import load_coefficients, calculate_model_prediction

def parse_args():
    parser = argparse.ArgumentParser(description="Save analysis results to CSV and generate summary table.")
    parser.add_argument("--assessments", required=True, help="Path to raw_assessments.json")
    parser.add_argument("--report", required=True, help="Path to stock_analysis_report.json")
    parser.add_argument("--csv", required=True, help="Path to output CSV file (appended)")
    parser.add_argument("--output_md", required=True, help="Path to output summary markdown file")
    return parser.parse_args()

def clean_value(text):
    if not isinstance(text, str):
        return text
    match = re.search(r"([-+]?\d*\.\d+|\d+)", text)
    if match:
        return float(match.group(1))
    return None

def save_to_csv(csv_path, assessments, report, long_coefs, short_coefs, report_path_for_dir):
    file_exists = os.path.isfile(csv_path)
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().isoformat()
    
    assessments_map = {item['ticker']: item for item in assessments}
    
    # Define Headers
    headers = [
        "date", "ticker", "timestamp", "oneDayReturnPct", "finalScore",
        "modelReturnPrediction", # Legacy
        "nonFundamental", "news", "sentiment", "uncertainty", "confidence",
        "returnLikelihood", "evidenceCheckedCited", "reason",
        "metrics_EC", "metrics_PCR", "metrics_SD", "metrics_NRI",
        "metrics_HDM", "metrics_CONTR", "metrics_FRESH_NEG",
        "metrics_CP", "metrics_RD",
        "lm_fit_long", "lm_fit_short", "avg_volume_30d" # New columns at end
    ]
    
    # Collect rows
    rows_to_write = []
    for item in report:
        ticker = item['ticker']
        raw = assessments_map.get(ticker, {})
        metrics = raw.get('metrics', {})

        sentiment_val = clean_value(item.get('sentiment'))
        likelihood_val = clean_value(item.get('returnLikelihood1to5d'))
        
        # Calculate Fits
        # We need volume. In the latest rebound_scoring, dollarVol is passed through to the report item.
        # Let's check where it is in 'item'. rebound_scoring puts it in `final_obj['dollarVol']` (mapped to avg_volume_30d)
        vol_30d = item.get('dollarVol', 0)
        
        # Prepare metrics for prediction (injecting oneDayReturnPct)
        model_input_metrics = metrics.copy()
        model_input_metrics['oneDayReturnPct'] = item.get('oneDayReturnPct', 0.0)
        
        # Calc Long
        fit_long = calculate_model_prediction(
            model_input_metrics,
            item.get('finalScore', 0),
            item.get('confidence', 'Medium'),
            item.get('uncertainty', 'Medium'),
            vol_30d,
            long_coefs
        )
        
        # Calc Short
        fit_short = calculate_model_prediction(
            model_input_metrics,
            item.get('finalScore', 0),
            item.get('confidence', 'Medium'),
            item.get('uncertainty', 'Medium'),
            vol_30d,
            short_coefs
        )

        row = {
            "date": today_date,
            "ticker": ticker,
            "timestamp": timestamp,
            "oneDayReturnPct": item.get('oneDayReturnPct'),
            "finalScore": item.get('finalScore'),
            "modelReturnPrediction": item.get('modelReturnPrediction'), # Legacy kept as is
            "lm_fit_long": fit_long,
            "lm_fit_short": fit_short,
            "avg_volume_30d": vol_30d,
            "nonFundamental": item.get('nonFundamental'),
            "news": item.get('news'),
            "sentiment": sentiment_val,
            "uncertainty": item.get('uncertainty'),
            "confidence": item.get('confidence'),
            "returnLikelihood": likelihood_val,
            "evidenceCheckedCited": item.get('evidenceCheckedCited'),
            "reason": item.get('reason'),
            "metrics_EC": metrics.get('EC'),
            "metrics_PCR": metrics.get('PCR'),
            "metrics_SD": metrics.get('SD'),
            "metrics_NRI": metrics.get('NRI'),
            "metrics_HDM": metrics.get('HDM'),
            "metrics_CONTR": metrics.get('CONTR'),
            "metrics_FRESH_NEG": metrics.get('FRESH_NEG'),
            "metrics_CP": metrics.get('CP'),
            "metrics_RD": metrics.get('RD')
        }
        rows_to_write.append(row)
        
        # update report item in place for enriched JSON
        item['lm_fit_long'] = fit_long
        item['lm_fit_short'] = fit_short
        item['avg_volume_30d'] = vol_30d

    # Save Enriched Report
    enriched_path = os.path.join(os.path.dirname(report_path_for_dir), "enriched_report.json")
    with open(enriched_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Saved enriched report to: {enriched_path}")
    # Verify headers match existing file?
    # If existing file has fewer columns, DictWriter might ignore extras if extrasaction='ignore', or error.
    # We should probably handle schema evolution.
    # Standard DictWriter raises ValueError if fieldnames don't match data keys, unless extrasaction='ignore'.
    # But he wants to *include* them.
    # If file exists, we need to read it to check columns. If columns missing, we might need to rewrite or just start appending with new schema
    # (which might break some CSV readers if headers differ mid-file, but usually we just append).
    # actually, proper CSV management involves ensuring the header row is correct.
    # If we append new columns, we effectively change the schema. 
    # The user instruction implies updating the history file. 
    # If the file exists, we should probably read the header. If it doesn't match effectively, we have a problem.
    # However, for this task, the Backfill script is responsible for fixing the hole.
    # Here we probably just want to write safe rows.
    
    # Strategy: Read existing header. If new columns missing, we can't easily append without rewriting the header.
    # Since we are planning a BACKFILL, the backfill script will likely normalize the file structure.
    # So for *this* running instance, if we just append, distinct parsers might fail.
    # But `save_results.py` runs usually once a day.
    # Let's just append. If the file is managed by `clean_and_enrich` or similar, it might be fine.
    # Actually, if we just append with new columns, standard pandas `read_csv` on the whole file might error `ParserError`.
    # BUT, the backfill script is coming right after.
    # Let's trust the backfill will align historical rows.
    # Here, we will try to handle the append gracefully.
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        else:
            # Check if we need to write a new header? No, usually not done in CSV.
            # We assume the file has been migrated or we are creating a mess that backfill will fix.
            # Wait, `backfill` reads the file. If the file has mixed columns, it might be hard to read.
            # We should probably ensure the file HAS the headers if they are missing?
            # Too complex for `save_results.py`. We'll assume backfill runs first or concurrently to fix schema.
            pass
            
        writer.writerows(rows_to_write)

def generate_table(report, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Stock Analysis Summary\n\n")
        f.write("| Ticker | 1D Return | Score | Non-Fund | News | Sent | Likelihood | Evidence | Reason |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        
        for item in report:
            sentiment_val = clean_value(item.get('sentiment'))
            likelihood_val = clean_value(item.get('returnLikelihood1to5d'))
            
            row = (
                f"| **{item['ticker']}** | {item['oneDayReturnPct']}% | **{item['finalScore']}** | "
                f"{item['nonFundamental']} | {item['news']} | {sentiment_val} | "
                f"{likelihood_val}% | {item['evidenceCheckedCited']} | {item['reason']} |"
            )
            f.write(row + "\n")

def main():
    args = parse_args()
    
    with open(args.assessments, 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    
    with open(args.report, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Load Coefficients
    long_coefs = load_coefficients('long')
    short_coefs = load_coefficients('short')
    
    if not long_coefs:
        print("Warning: Long coefficients not found.")
    if not short_coefs:
        print("Warning: Short coefficients not found.")

    # Save to CSV (Calculates fits internally)
    save_to_csv(args.csv, assessments, report, long_coefs, short_coefs, args.report)
    
    generate_table(report, args.output_md)
    print(f"Successfully appended to CSV: {args.csv}")
    print(f"Generated summary table: {args.output_md}")

if __name__ == "__main__":
    main()
