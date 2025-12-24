import requests
import json
import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY not found in environment variables.")
    sys.exit(1)

def is_cet_between_17_and_20():
    """Return True when the current time in CET is between 17:00 and 20:00."""
    try:
        cet = ZoneInfo("Europe/Paris")
        now_cet = datetime.now().astimezone(cet)
        return 17 <= now_cet.hour < 20
    except Exception:
        return False

def fetch_losers(limit=10, min_price=15.0, date_str=None):
    print("--- Fetching common stock tickers from NYSE and NASDAQ ---")
    
    common_tickers = set()
    ticker_metadata = {}

    # 1. Load Common Stocks
    for exchange in ["XNYS", "XNAS"]:
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "apiKey": API_KEY, "type": "CS", "market": "stocks",
            "exchange": exchange, "active": "true", "limit": 1000,
        }
        
        print(f"Fetching tickers for {exchange}...")
        
        try:
            while url:
                resp = requests.get(url, params=params)
                if resp.status_code != 200:
                    print(f"  [ERROR] Failed to fetch data for {exchange}. Status Code: {resp.status_code}")
                    break
                data = resp.json()
                for result in data.get("results", []):
                    ticker = result.get("ticker")
                    if ticker and " " not in ticker and "." not in ticker:
                        ticker = ticker.upper()
                        common_tickers.add(ticker)
                        ticker_metadata[ticker] = {
                            "name": result.get("name", "N/A"),
                            "exchange": result.get("primary_exchange", exchange),
                        }
                url = data.get("next_url")
                params = {"apiKey": API_KEY}
        
        except requests.exceptions.RequestException as e:
            print(f"  [FATAL ERROR] A network error occurred while fetching {exchange}: {e}")
            url = None

    print(f"--- Loaded {len(common_tickers)} unique common stock tickers. ---")

    # 2. Fetch Data (Snapshot or Historical)
    raw_data = []
    if date_str:
        print(f"--- Fetching historical data for {date_str}... ---")
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        params = {"apiKey": API_KEY, "adjusted": "true"}
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                raw_data = resp.json().get("results", [])
            else:
                print(f"Error fetching history: {resp.status_code}")
        except Exception as e:
             print(f"Error: {e}")
    else:
        print("--- Fetching market snapshot... ---")
        snapshot_url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {"apiKey": API_KEY, "include_otc": "false"}
        try:
            resp = requests.get(snapshot_url, params=params)
            if resp.status_code == 200:
                raw_data = resp.json().get("tickers", [])
        except Exception as e:
            print(f"Error: {e}")

    calculated_losers = []

    # 3. Process Losers
    print("--- Processing data... ---")
    for item in raw_data:
        # Normalize keys (Snapshot vs Grouped)
        # Grouped: T, c, o, h, l, v, vw
        # Snapshot: ticker, day.c, day.o ...
        
        ticker = item.get("ticker") or item.get("T")
        if not ticker or ticker not in common_tickers:
            continue
            
        if date_str:
            # Historical Grouped
            close_price = item.get("c")
            open_price = item.get("o")
            if not close_price or not open_price or close_price < min_price:
                continue
            # Calculate change using Open vs Close (or if Grouped has prevClose? No it doesn't usually)
            # Actually Grouped Daily is OHLC. Change from Yesterday's Close is not provided directly.
            # We will use (Close - Open) / Open as a proxy for intraday, OR we can't accurately get Gap.
            # Wait, for "Big Losers", Close < Open is implied.
            # Let's use (Close - Open) / Open.
            change_pct = ((close_price - open_price) / open_price) * 100
            current_price = close_price
        else:
            # Snapshot
            current_price = item.get("lastTrade", {}).get("p") or item.get("day", {}).get("c")
            if not current_price or current_price < min_price:
                continue
            
            # Simple snapshot logic
            prev = item.get("prevDay", {}).get("c")
            if not prev: continue
            change_pct = ((current_price - prev) / prev) * 100

        if change_pct >= -4: # Strict -4% cutoff
            continue
            
        meta = ticker_metadata.get(ticker, {})
        calculated_losers.append({
            "ticker": ticker,
            "name": meta.get("name", "N/A"),
            "exchange": meta.get("exchange", "N/A"),
            "currentPrice": round(current_price, 2),
            "changePct": round(change_pct, 2),
            "yahooLink": f"https://finance.yahoo.com/quote/{ticker}",
        })

    sorted_losers = sorted(calculated_losers, key=lambda x: x["changePct"])
    return sorted_losers[:limit]

def main():
    parser = argparse.ArgumentParser(description="Fetch top stock losers.")
    parser.add_argument("--output", help="Path to save JSON output")
    parser.add_argument("--limit", type=int, default=10, help="Number of losers to fetch")
    parser.add_argument("--date", help="YYYY-MM-DD for historical fetch")
    args = parser.parse_args()

    losers = fetch_losers(limit=args.limit, date_str=args.date)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(losers, f, indent=2)
        print(f"Output saved to {args.output}")
    else:
        print(json.dumps(losers, indent=2))

if __name__ == "__main__":
    main()
