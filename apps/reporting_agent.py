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
    parser.add_argument("--contrarian_report", required=False, help="Path to stock_analysis_report.json (for Model Picks)")
    parser.add_argument("--timing_json", required=True, help="Path to timing_output.json (Contrarian)")
    parser.add_argument("--recipient", default="xavierjjc@outlook.com", help="Email recipient")
    return parser.parse_args()

def clean_value(text):
    """Helper to extract float from string."""
    import re
    if not isinstance(text, str):
        return text
    match = re.search(r"([-+]?\d*\.\d+|\d+)", text)
    if match:
        return float(match.group(1))
    return None

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

def generate_model_picks_table(report_data, mode='long'):
    """Generates HTML table for Top Model Picks based on criteria."""
    if not report_data:
        return ""
    
    # Determine key
    pred_key = 'lm_fit_long' if mode == 'long' else 'lm_fit_short'
    title = "🟢 Top Long Picks" if mode == 'long' else "🔴 Top Short Picks"
    
    # Filter candidates (Optional: filter > 0 return?)
    # For now, show top entries regardless of threshold? 
    # Or stick to > 4.5%? User didn't specify threshold, just "top model picks".
    # Let's show top 5 for each.
    
    # Sort
    filtered = [x for x in report_data if x.get(pred_key) is not None]
    filtered.sort(key=lambda x: x.get(pred_key, -999), reverse=True)
    
    display_list = filtered[:5]
    
    if not display_list:
        return f"<h3>{title}</h3><p>No model data available.</p>"
        
    html_lines = [f"<h3>{title}</h3>"]
    html_lines.append("<table border='1' style='border-collapse: collapse; width: 100%; font-size: 14px; border: 2px solid #2980b9;'>")
    
    headers = ["Ticker", "Pred. Return", "Score", "Sentiment", "Reason"]
    html_lines.append("<tr>")
    for h in headers:
        bg_color = "#d4efdf" if mode == 'long' else "#fadbd8" # Greenish vs Reddish
        html_lines.append(f"<th style='padding: 8px; text-align: left; background-color: {bg_color}; color: #333;'>{h}</th>")
    html_lines.append("</tr>")
    
    for item in display_list:
        pred_val = item.get(pred_key, 0)
        pred_fmt = f"{pred_val*100:.2f}%"
        
        # Color logic
        pred_style = "font-weight: bold;"
        if pred_val > 0: pred_style += " color: green;"
        elif pred_val < 0: pred_style += " color: red;"
        
        score_val = item.get('finalScore', '-')
        sent_val = item.get('sentiment', '-')
        reason = item.get('reason', '')
        
        html_lines.append("<tr>")
        html_lines.append(f"<td style='padding: 8px;'><b>{item['ticker']}</b></td>")
        html_lines.append(f"<td style='padding: 8px; {pred_style}'>{pred_fmt}</td>")
        html_lines.append(f"<td style='padding: 8px;'>{score_val}</td>")
        html_lines.append(f"<td style='padding: 8px;'>{sent_val}</td>")
        html_lines.append(f"<td style='padding: 8px; font-size: 0.9em; color: #555;'>{reason}</td>")
        html_lines.append("</tr>")
        
    html_lines.append("</table><br>")
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
    
    # Look for enriched report
    # Typically in same dir as contrarian_report
    base_dir = os.path.dirname(args.summary_md)
    enriched_path = os.path.join(base_dir, "enriched_report.json")
    
    report_data = []
    
    if os.path.exists(enriched_path):
        try:
             with open(enriched_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
        except Exception as e:
            print(f"Error loading enriched report: {e}")
    elif args.contrarian_report and os.path.exists(args.contrarian_report):
         # Fallback to old report (might lack lm_fit keys)
         try:
             with open(args.contrarian_report, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
         except Exception as e:
            print(f"Error loading contrarian report: {e}")
            
    # Generate Tables
    long_table = generate_model_picks_table(report_data, mode='long')
    short_table = generate_model_picks_table(report_data, mode='short')

    # --- Compose Body ---
    today_str = datetime.now().strftime("%Y-%m-%d")
    full_html = f"""
    <html>
    <body style='font-family: Arial, sans-serif;'>
        <h1 style='color: #2c3e50; text-align: center;'>Market Analysis Report ({today_str})</h1>
        <hr>
        
        {long_table}
        <hr>
        {short_table}
        
        <br>
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
