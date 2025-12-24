import csv
import argparse
import json
import os
from datetime import datetime

def save_results(csv_path, report_path):
    file_exists = os.path.isfile(csv_path)
    
    with open(report_path, 'r') as f:
        data = json.load(f)
        
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()
    
    headers = ["run_id", "timestamp", "ticker", "dcl_prob", "scs_score", "qsi_score", "rank", "driver_category", "reason"]
    
    rows_to_write = []
    
    for item in data:
        dcl_str = item.get('downsideContinuationLikelihoodNextDay', '0%').replace('%', '')
        try:
            dcl = float(dcl_str)
        except:
            dcl = 0.0
            
        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "ticker": item.get('ticker'),
            "dcl_prob": dcl,
            "scs_score": item.get('shortCandidateScore', 0.0),
            "qsi_score": item.get('qsi', 0),
            "rank": item.get('rank', 0),
            "driver_category": item.get('driverCategory', 'Unknown'),
            "reason": item.get('reason', '')
        }
        rows_to_write.append(row)
        
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows_to_write)
        
    print(f"Results appended to CSV: {csv_path} (RunID: {run_id})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to downside_report.json")
    parser.add_argument("--csv", required=True, help="Path to downside_history.csv")
    args = parser.parse_args()
    
    save_results(args.csv, args.report)

if __name__ == "__main__":
    main()
