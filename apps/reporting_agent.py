import argparse
import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Send stock analysis report via email.")
    parser.add_argument("--summary_md", required=True, help="Path to summary_table.md (Contrarian)")
    parser.add_argument("--timing_json", required=True, help="Path to timing_output.json (Contrarian)")
    parser.add_argument("--downside_report", required=False, help="Path to downside_report.json")
    parser.add_argument("--downside_timing", required=False, help="Path to downside_timing_output.json")
    parser.add_argument("--recipient", default="xavierjjc@outlook.com", help="Email recipient")
    return parser.parse_args()

def md_table_to_html(md_content):
    """Converts a simple Markdown table to an HTML table."""
    lines = md_content.strip().split('\n')
    html_lines = ["<table border='1' style='border-collapse: collapse; width: 100%; font-size: 14px;'>"]
    
    in_table = False
    header_processed = False
    
    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            continue
        
        # Skip separator lines like |---|---|
        if '---' in line:
            header_processed = True
            continue

        cells = [c.strip() for c in line.split('|') if c]
        
        html_lines.append("<tr>")
        tag = "th" if not header_processed else "td"
        style = "padding: 8px; text-align: left; background-color: #f2f2f2;" if tag == "th" else "padding: 8px; text-align: left;"
        
        for cell in cells:
            # Simple bold parsing
            cell_html = cell.replace("**", "<b>").replace("**", "</b>")
            html_lines.append(f"<{tag} style='{style}'>{cell_html}</{tag}>")
        html_lines.append("</tr>")
        
    html_lines.append("</table>")
    return "\n".join(html_lines)

def json_timing_to_html(timing_data, title, top_n=4):
    """Converts timing output JSON to an HTML table."""
    if not isinstance(timing_data, list) or not timing_data:
        return f"<p>No {title} timing data available.</p>"

    # Sort if needed? Expected input is already sorted.
    # Limit to Top N
    display_data = timing_data[:top_n]

    html_lines = [f"<h3>{title} (Top {len(display_data)})</h3>"]
    html_lines.append("<table border='1' style='border-collapse: collapse; width: 100%; font-size: 14px;'>")
    
    # Headers
    headers = ["Ticker", "Action", "Score/Urgency", "Reasoning"]
    html_lines.append("<tr>")
    for h in headers:
        html_lines.append(f"<th style='padding: 8px; text-align: left; background-color: #e0e0e0;'>{h}</th>")
    html_lines.append("</tr>")

    # Rows
    for item in display_data:
        html_lines.append("<tr>")
        # Handle difference between Rebound (confidence/target) and Downside (urgency) schemas
        # Unified display for simplicity
        ticker = item.get('ticker', 'N/A')
        
        # Try finding Action/Timing
        action = item.get('timing', item.get('action', 'N/A'))
        
        # Try finding Score/Confidence/Urgency
        score_val = item.get('urgency_score')
        if score_val is None:
            score_val = f"{item.get('confidence', 'N/A')} (Tgt: {item.get('target_price', '-')})"
        
        reason = item.get('reasoning', 'N/A')

        row_data = [ticker, action, str(score_val), reason]
        
        for cell in row_data:
            html_lines.append(f"<td style='padding: 8px; text-align: left;'>{cell}</td>")
        html_lines.append("</tr>")
        
    html_lines.append("</table>")
    return "\n".join(html_lines)

def json_report_to_html(report_data, title, top_n=5):
    """Converts main report JSON to HTML table (for Downside since it has no MD summary)."""
    if not report_data:
        return ""
        
    display_data = report_data[:top_n]
    
    html_lines = [f"<h3>{title} (Top {len(display_data)})</h3>"]
    html_lines.append("<table border='1' style='border-collapse: collapse; width: 100%; font-size: 14px;'>")
    
    # Headers
    headers = ["Rank", "Ticker", "Score", "Probability", "Category"]
    html_lines.append("<tr>")
    for h in headers:
        html_lines.append(f"<th style='padding: 8px; text-align: left; background-color: #d1f2eb;'>{h}</th>")
    html_lines.append("</tr>")

    for item in display_data:
        html_lines.append("<tr>")
        # Adapting to Downside keys
        row_data = [
            str(item.get('rank', '-')),
            item.get('ticker', 'N/A'),
            str(item.get('shortCandidateScore', 'N/A')),
            item.get('downsideContinuationLikelihoodNextDay', 'N/A'),
            item.get('driverCategory', 'N/A')
        ]
        for cell in row_data:
            html_lines.append(f"<td style='padding: 8px; text-align: left;'>{cell}</td>")
        html_lines.append("</tr>")
    
    html_lines.append("</table>")
    return "\n".join(html_lines)

def send_email(subject, html_body, recipient):
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.office365.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")

    if not smtp_user or not smtp_pass:
        print("[WARNING] SMTP_USER or SMTP_PASS not set. Skipping email send.")
        return

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipient, msg.as_string())
        server.quit()
        print(f"Email sent successfully to {recipient}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

def main():
    args = parse_args()

    # --- 1. UPSIDE (Contrarian) Content ---
    upside_summary_html = "<p>No summary table found.</p>"
    if os.path.exists(args.summary_md):
        with open(args.summary_md, 'r', encoding='utf-8') as f:
            upside_summary_html = md_table_to_html(f.read())
            
    upside_timing_html = ""
    if os.path.exists(args.timing_json):
        try:
            with open(args.timing_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                upside_timing_html = json_timing_to_html(data, "Rebound Timing Advice", top_n=4)
        except Exception as e:
            upside_timing_html = f"<p>Error: {e}</p>"

    # --- 2. DOWNSIDE (Short) Content ---
    downside_report_html = ""
    if args.downside_report and os.path.exists(args.downside_report):
        try:
            with open(args.downside_report, 'r', encoding='utf-8') as f:
                data = json.load(f)
                downside_report_html = json_report_to_html(data, "Downside Candidates", top_n=5)
        except Exception as e:
            downside_report_html = f"<p>Error: {e}</p>"

    downside_timing_html = ""
    if args.downside_timing and os.path.exists(args.downside_timing):
        try:
            with open(args.downside_timing, 'r', encoding='utf-8') as f:
                data = json.load(f)
                downside_timing_html = json_timing_to_html(data, "Short Timing Advice", top_n=4)
        except Exception as e:
            downside_timing_html = f"<p>Error: {e}</p>"

    # --- Compose Body ---
    today_str = datetime.now().strftime("%Y-%m-%d")
    full_html = f"""
    <html>
    <body style='font-family: Arial, sans-serif;'>
        <h1 style='color: #2c3e50; text-align: center;'>Market Analysis Report ({today_str})</h1>
        <hr>
        
        <!-- SECTION 1: REBOUND OPPORTUNITIES -->
        <h2 style='color: #27ae60;'>🟢 Rebound Opportunities (Long)</h2>
        <p><i>Top contrarian plays based on Rebound Likelihood Score (RLS).</i></p>
        
        {upside_summary_html}
        {upside_timing_html}
        
        <br><hr><br>
        
        <!-- SECTION 2: SHORT OPPORTUNITIES -->
        <h2 style='color: #c0392b;'>🔴 Downside Opportunities (Short)</h2>
        <p><i>Candidates for persistent decline avoiding dead-cat bounces.</i></p>
        
        {downside_report_html}
        {downside_timing_html}

        <br><hr>
        <p style='color: #7f8c8d; font-size: 0.8em; text-align: center;'>Generated by Silver Halo Agent</p>
    </body>
    </html>
    """

    send_email(f"Stock Analysis Report - {today_str}", full_html, args.recipient)
    
    # Save preview
    debug_path = os.path.join(os.path.dirname(args.summary_md), "email_preview_dual.html")
    with open(debug_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"Email preview saved to: {debug_path}")

if __name__ == "__main__":
    main()
