import json
import csv
import argparse
import os
import re
from datetime import datetime

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

def save_to_csv(csv_path, assessments, report):
    file_exists = os.path.isfile(csv_path)
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().isoformat()
    
    assessments_map = {item['ticker']: item for item in assessments}
    
    # Define Headers
    headers = [
        "date", "ticker", "timestamp", "oneDayReturnPct", "finalScore",
        "nonFundamental", "news", "sentiment", "uncertainty", "confidence",
        "returnLikelihood", "evidenceCheckedCited", "reason",
        "metrics_EC", "metrics_PCR", "metrics_SD", "metrics_NRI",
        "metrics_HDM", "metrics_CONTR", "metrics_FRESH_NEG",
        "metrics_CP", "metrics_RD"
    ]
    
    # Collect rows
    rows_to_write = []
    for item in report:
        ticker = item['ticker']
        raw = assessments_map.get(ticker, {})
        metrics = raw.get('metrics', {})

        sentiment_val = clean_value(item.get('sentiment'))
        likelihood_val = clean_value(item.get('returnLikelihood1to5d'))

        row = {
            "date": today_date,
            "ticker": ticker,
            "timestamp": timestamp,
            "oneDayReturnPct": item.get('oneDayReturnPct'),
            "finalScore": item.get('finalScore'),
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

    # Append to CSV
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
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

    save_to_csv(args.csv, assessments, report)
    generate_table(report, args.output_md)
    print(f"Successfully appended to CSV: {args.csv}")
    print(f"Generated summary table: {args.output_md}")

if __name__ == "__main__":
    main()
