
import pandas as pd
import yfinance as yf
import datetime
from datetime import timedelta
import os
import time

def get_next_trading_day(date_str):
    try:
        current_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None
        
    next_date = current_date + datetime.timedelta(days=1)
    
    # Simple check for weekends
    while next_date.weekday() >= 5: # 5=Sat, 6=Sun
        next_date += datetime.timedelta(days=1)
        
    # Check for holidays (Basic US Market holidays - truncated list for simplicity)
    # Ideally should use a proper market calendar library
    holidays = [
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', 
        '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01',
        '2025-11-27', '2025-12-25',
        '2026-01-01', '2026-01-19'
    ]
    
    while next_date.strftime('%Y-%m-%d') in holidays:
        next_date += datetime.timedelta(days=1)
        # Re-check weekend after skip
        while next_date.weekday() >= 5:
            next_date += datetime.timedelta(days=1)
            
    return next_date.strftime('%Y-%m-%d')

def calculate_full_data():
    input_path = '../analysis_history.csv'
    output_path = 'full_returns_data.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    results = []

    print(f"Processing {len(df)} entries...")
    unique_trades = df[['ticker', 'date']].drop_duplicates()
    
    for index, row in unique_trades.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        
        # Determine T+1
        t1_date = get_next_trading_day(date_str)
        if not t1_date:
            continue
            
        # Determine T+2
        t2_date = get_next_trading_day(t1_date)
        
        print(f"[{index+1}/{len(unique_trades)}] {ticker}: Analyzed {date_str}, T+1 {t1_date}, T+2 {t2_date}")
        
        try:
            # Fetch T+1 and T+2 Data (Need enough buffer)
            start_date = t1_date
            end_date_obj = datetime.datetime.strptime(t2_date, '%Y-%m-%d') + timedelta(days=5) # Fetch a few days ahead to capture T+2 properly
            end_date = end_date_obj.strftime('%Y-%m-%d')
            
            data = yf.download(ticker, start=start_date, end=end_date, interval='1m', progress=False)
            
            if data.empty:
                print(f"  No data for {ticker}")
                continue
                
            # Flatten MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.reset_index(inplace=True)
            
            # Standardize Datetime
            if 'Date' in data.columns and 'Datetime' not in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
            if 'index' in data.columns and 'Datetime' not in data.columns:
                 data.rename(columns={'index': 'Datetime'}, inplace=True)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            # Timezone
            first_dt = data['Datetime'].iloc[0]
            if first_dt.tzinfo is None:
                 data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            else:
                 data['Datetime'] = data['Datetime'].dt.tz_convert('America/New_York')

            # --- T+1 Calculation ---
            t1_market_start = datetime.datetime.strptime(t1_date, '%Y-%m-%d').replace(hour=9, minute=30)
            t1_market_start = pd.Timestamp(t1_market_start).tz_localize('America/New_York')
            
            t1_data = data[data['Datetime'].dt.date == t1_market_start.date()].copy()
            
            short_ret_t1 = None
            long_ret_t1 = None
            
            if not t1_data.empty:
                t1_open = t1_data.iloc[0]['Open']
                
                # Short Strategy T+1 (Sell Open, 4% SL, 2h Exit)
                sl_short_price = t1_open * 1.04
                exit_short_time = t1_data['Datetime'].iloc[0] + timedelta(hours=2)
                
                exit_short = None
                for _, bar in t1_data.iterrows():
                    if bar['High'] >= sl_short_price:
                        exit_short = sl_short_price
                        break
                    if bar['Datetime'] >= exit_short_time:
                        exit_short = bar['Open']
                        break
                if exit_short is None: exit_short = t1_data.iloc[-1]['Close']
                short_ret_t1 = (t1_open - exit_short) / t1_open

                # Long Strategy T+1 (Buy Open, 4% SL, 6.5h Exit)
                sl_long_price = t1_open * 0.96
                exit_long_time = t1_data['Datetime'].iloc[0] + timedelta(hours=6, minutes=30)
                
                exit_long = None
                for _, bar in t1_data.iterrows():
                    if bar['Low'] <= sl_long_price: # Fixed logic: LOW triggers SL on Long
                        exit_long = sl_long_price
                        break
                    if bar['Datetime'] >= exit_long_time:
                        exit_long = bar['Open'] # Or Close? Using Open of target bar for consistency
                        break
                if exit_long is None: exit_long = t1_data.iloc[-1]['Close'] # EOD
                long_ret_t1 = (exit_long - t1_open) / t1_open
            
            # --- T+2 Calculation ---
            t2_market_date = datetime.datetime.strptime(t2_date, '%Y-%m-%d').date()
            t2_market_start = pd.Timestamp(t2_market_date).tz_localize('America/New_York') + timedelta(hours=9, minutes=30)
            
            t2_data = data[data['Datetime'].dt.date == t2_market_date].copy()
            
            day2_ret_simple = None
            short_ret_t2_strat = None
            long_ret_t2_strat = None
            
            if not t2_data.empty:
                t2_open = t2_data.iloc[0]['Open']
                t2_close = t2_data.iloc[-1]['Close']
                day2_ret_simple = (t2_close - t2_open) / t2_open
                
                # --- T+2 Short Strategy (Sell Open, 4% SL, 2h Exit) ---
                sl_short_price_t2 = t2_open * 1.04
                # Use actual market start time for exit target
                exit_short_time_t2 = t2_data['Datetime'].iloc[0] + timedelta(hours=2)
                
                exit_short_t2 = None
                for _, bar in t2_data.iterrows():
                    if bar['High'] >= sl_short_price_t2:
                        exit_short_t2 = sl_short_price_t2
                        break
                    if bar['Datetime'] >= exit_short_time_t2:
                        exit_short_t2 = bar['Open']
                        break
                if exit_short_t2 is None: exit_short_t2 = t2_close
                short_ret_t2_strat = (t2_open - exit_short_t2) / t2_open

                # --- T+2 Long Strategy (Buy Open, 4% SL, 6.5h Exit) ---
                sl_long_price_t2 = t2_open * 0.96
                exit_long_time_t2 = t2_data['Datetime'].iloc[0] + timedelta(hours=6, minutes=30)
                
                exit_long_t2 = None
                for _, bar in t2_data.iterrows():
                    if bar['Low'] <= sl_long_price_t2:
                        exit_long_t2 = sl_long_price_t2
                        break
                    if bar['Datetime'] >= exit_long_time_t2:
                        exit_long_t2 = bar['Open']
                        break
                if exit_long_t2 is None: exit_long_t2 = t2_close
                long_ret_t2_strat = (exit_long_t2 - t2_open) / t2_open
            
            results.append({
                'Ticker': ticker,
                'AnalysisDate': date_str,
                'T1_Date': t1_date,
                'ShortReturn_T1': short_ret_t1,
                'LongReturn_T1': long_ret_t1,
                'T2_Date': t2_date,
                'Day2Return': day2_ret_simple,
                'ShortReturn_T2_Strat': short_ret_t2_strat,
                'LongReturn_T2_Strat': long_ret_t2_strat
            })
            
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} full results to {output_path}")

if __name__ == "__main__":
    calculate_full_data()
