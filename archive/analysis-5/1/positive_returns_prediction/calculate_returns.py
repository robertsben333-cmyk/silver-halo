import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import time

def get_next_trading_day(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    next_day = date_obj + timedelta(days=1)
    while next_day.weekday() >= 5: # Skip Sat/Sun
        next_day += timedelta(days=1)
    # Simple holiday skip (New Years, Christmas) - can be expanded
    # For now, relying on yfinance returning no data or we can check simple list
    # 2025-12-25 is Th, 2026-01-01 is Th.
    holidays = ['2025-12-25', '2026-01-01', '2026-01-19'] # Jan 19 MLK
    while next_day.strftime('%Y-%m-%d') in holidays or next_day.weekday() >= 5:
         next_day += timedelta(days=1)
         if next_day.weekday() >= 5:
             continue
    return next_day.strftime('%Y-%m-%d')

def calculate_returns():
    input_path = '../analysis_history.csv'
    output_path = 'calculated_returns.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    results = []

    print(f"Processing {len(df)} entries from {input_path}...")

    # Unique ticker/date combinations to avoid re-fetching if duplicates exist
    # But analysis_history might have multiple entries per ticker if run multiple times?
    # The user said 'deduplicating entries to keep only the latest 10' in history.
    # We should process each row or just unique ticker/date pairs.
    # Assuming one trade per ticker per date.
    
    unique_trades = df[['ticker', 'date']].drop_duplicates()
    
    for index, row in unique_trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        
        next_trading_date = get_next_trading_day(date_str)
        print(f"[{index+1}/{len(unique_trades)}] {ticker} on {date_str} -> Next Day: {next_trading_date}")
        
        try:
            # Fetch data for next trading day
            # yfinance expects start, end (exclusive). So start=next_day, end=next_day+1
            start_date = next_trading_date
            end_date_obj = datetime.strptime(next_trading_date, '%Y-%m-%d') + timedelta(days=1)
            end_date = end_date_obj.strftime('%Y-%m-%d')
            
            data = yf.download(ticker, start=start_date, end=end_date, interval='1m', progress=False)
            
            if data.empty:
                print(f"  No data for {ticker} on {next_trading_date}")
                continue
                
            # Flatten columns if MultiIndex (common in new yfinance)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Reset index to access datetime
            data.reset_index(inplace=True)

            # Ensure Datetime column exists and is proper type
            if 'Date' in data.columns and 'Datetime' not in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
            if 'index' in data.columns and 'Datetime' not in data.columns:
                 data.rename(columns={'index': 'Datetime'}, inplace=True)

            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            # Identify Open Time (9:30 ET)
            # Check timezone of the first element to decide
            first_dt = data['Datetime'].iloc[0]
            if first_dt.tzinfo is None:
                 # Assume UTC if naive, then convert
                 data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            else:
                 # Convert to America/New_York
                 data['Datetime'] = data['Datetime'].dt.tz_convert('America/New_York')

            # Filter for market hours 09:30 - 16:00
            # Next day data might include pre-market if not careful, but yf usually allows filter.
            # Default auto_adjust=False, prepost=False.
            
            market_start = data['Datetime'].iloc[0].replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Filter data starting from 09:30
            session_data = data[data['Datetime'] >= market_start].copy()
            if session_data.empty:
                print("  No session data found >= 9:30")
                continue
                
            # Entry at Open of first bar (approx 9:30)
            entry_price = session_data.iloc[0]['Open']
            stop_loss_price = entry_price * 0.96 # 4% stop loss for LONG
            
            # Exit time target: 6.5 hours after open = 16:00 (End of Day)
            target_exit_time = market_start + timedelta(hours=6, minutes=30)
            
            exit_price = None
            outcome = ""
            
            # Iterate
            for i, bar in session_data.iterrows():
                # Check Stop Loss (Low <= SL)
                # Since we are LONG, if Price goes DOWN (Low <= SL), we sell.
                if bar['Low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    outcome = "Stop Loss"
                    break
                
                # Check Time Exit
                if bar['Datetime'] >= target_exit_time:
                    exit_price = bar['Open'] # Exit at open of the target bar
                    outcome = "Time Exit"
                    break
            
            # If session ended before target?
            if exit_price is None:
                exit_price = session_data.iloc[-1]['Close']
                outcome = "EOD Exit (Data ended)"
            
            # Calculate Long Return
            # Return = (Exit - Entry) / Entry
            ret_val = (exit_price - entry_price) / entry_price
            
            results.append({
                'Ticker': ticker,
                'AnalysisDate': date_str,
                'NextTradingDate': next_trading_date,
                'EntryPrice': entry_price,
                'ExitPrice': exit_price,
                'Outcome': outcome,
                'Return': ret_val
            })
            
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            continue

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} results to {output_path}")

if __name__ == "__main__":
    calculate_returns()
