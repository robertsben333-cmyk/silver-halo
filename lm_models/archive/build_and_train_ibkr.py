import sys
import os
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Add provided ibapi to path
sys.path.append(os.path.join(os.getcwd(), "ibapi"))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {} # reqId -> List[BarData]
        self.events = {} # reqId -> Event
        self.next_req_id = 1
        self.errors = {}

    def error(self, reqId, *args):
        # Handle variable signature (some versions include errorTime)
        # We just want to print the error
        print(f"Error {reqId}: {args}")
        if reqId != -1:
            # Try to grab errorString from args (usually 2nd or 3rd arg)
            errStr = str(args)
            self.errors[reqId] = errStr
            if reqId in self.events:
                self.events[reqId].set()

    def historicalData(self, reqId, bar: BarData):
        if reqId not in self.data:
            self.data[reqId] = []
        self.data[reqId].append(bar)

    def historicalDataEnd(self, reqId, start: str, end: str):
        if reqId in self.events:
            self.events[reqId].set()

def train_model(df, target_col, strategy_name, output_dir):
    print(f"\n--- Training {strategy_name} Model ---")
    train_df = df.dropna(subset=[target_col]).copy()
    print(f"Training Rows: {len(train_df)}")
    
    if len(train_df) < 5:
        print("Not enough data.")
        return

    features_num = [
        'oneDayReturnPct', 'finalScore', 
        'metrics_PCR', 'metrics_EC', 'metrics_SD', 'metrics_NRI', 
        'metrics_HDM', 'metrics_CONTR', 'metrics_CP', 'metrics_RD',
        'metrics_FRESH_NEG', 'avg_volume_30d' 
    ]
    features_cat = ['confidence', 'uncertainty']
    
    features_num = [f for f in features_num if f in train_df.columns]
    X_num = train_df[features_num].fillna(0)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if all(c in train_df.columns for c in features_cat):
        X_cat = pd.DataFrame(encoder.fit_transform(train_df[features_cat].fillna("Medium")))
        X_cat.columns = encoder.get_feature_names_out(features_cat)
        X = pd.concat([X_num.reset_index(drop=True), X_cat], axis=1)
    else:
        X = X_num.reset_index(drop=True)
        
    y = train_df[target_col].reset_index(drop=True)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"R2: {r2:.4f}, MSE: {mse:.6f}")
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    coef_dict = { "intercept": model.intercept_, "coefficients": dict(zip(X.columns, model.coef_)) }
    with open(os.path.join(output_dir, 'coefficients_ibkr.json'), 'w') as f:
        json.dump(coef_dict, f, indent=4)
        
    with open(os.path.join(output_dir, 'summary_ibkr.txt'), 'w') as f:
        f.write(f"Model: {strategy_name}\nR2: {r2:.4f}\nMSE: {mse:.6f}\n")
        f.write(f"Intercept: {model.intercept_}\nCoefficients:\n")
        coef_df = pd.DataFrame({'Feature': X.columns, 'Val': model.coef_})
        coef_df['Abs'] = coef_df['Val'].abs()
        f.write(coef_df.sort_values('Abs', ascending=False).to_string())

def main():
    # 1. Connect
    app = IBApp()
    app.connect("127.0.0.1", 7496, 999) # Client 999
    
    # Start thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    time.sleep(1) # wait for connect
    
    if not app.isConnected():
        print("Failed to connect to TWS/Gateway on port 7496.")
        return

    # 2. Load History
    input_path = "lm_models/full_history.csv"
    if not os.path.exists(input_path):
        print("Full history not found.")
        return
    df = pd.read_csv(input_path)
    
    print(f"Processing {len(df)} tickers...")
    
    stop_pct = 0.04
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        analysis_date = row['date'] # YYYY-MM-DD
        
        # Calculate T+1 target date
        # Strategy: We assume T+1 is NEXT Valid Trading Day.
        # IBKR query approach: Ask for 1 D bar ENDING on T+1 Close.
        # But we don't know T+1 date exactly if weekends/holidays (simple algo).
        # Better: T+2 date minus 1D duration?
        # Reliable: Set endDate = analysis_date + 5 days. Request Duration '5 D'. 
        # Then manually pick the bar with date > analysis_date.
        
        dt_analysis = datetime.strptime(analysis_date, "%Y-%m-%d")
        dt_end = dt_analysis + timedelta(days=5) # Look ahead window
        end_str = dt_end.strftime("%Y%m%d 23:59:59")
        
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = app.next_req_id
        app.next_req_id += 1
        app.events[req_id] = threading.Event()
        
        # Request 1 Week of daily bars ending 5 days after analysis
        # This ensures we cover T+1 even over weekends
        app.reqHistoricalData(req_id, contract, end_str, "1 W", "1 day", "TRADES", 1, 1, False, [])
        
        if not app.events[req_id].wait(timeout=5):
            print(f"[{ticker}] Timeout getting historical data.")
            continue
            
        if req_id in app.data:
            bars = app.data[req_id]
            # Find T+1 bar (First bar with date > analysis_date)
            t1_bar = None
            for b in bars:
                # b.date format usually YYYYMMDD
                b_date_str = b.date
                if len(b_date_str) == 8:
                    b_dt = datetime.strptime(b_date_str, "%Y%m%d")
                    if b_dt > dt_analysis:
                        t1_bar = b
                        break
            
            if t1_bar:
                o = t1_bar.open
                c = t1_bar.close
                h = t1_bar.high
                l = t1_bar.low
                
                # --- Short Strategy ---
                entry_short = o
                stop_short = o * (1 + stop_pct)
                if h >= stop_short:
                    ret_short = -stop_pct
                else:
                    ret_short = (entry_short - c) / entry_short
                    
                # --- Long Strategy ---
                entry_long = o
                stop_long = o * (1 - stop_pct)
                if l <= stop_long:
                    ret_long = -stop_pct
                else:
                    ret_long = (c - entry_long) / entry_long
                
                df.at[idx, 'Target_Short'] = ret_short
                df.at[idx, 'Target_Long'] = ret_long
                df.at[idx, 'T1_Date_IBKR'] = t1_bar.date
                
                # print(f"[{ticker}] T+1 {t1_bar.date}: S={ret_short:.2%} L={ret_long:.2%}")
            else:
                # print(f"[{ticker}] No T+1 bar found after {analysis_date}")
                pass
        else:
             # Check errors
             if req_id in app.errors:
                 print(f"[{ticker}] IBKR Error: {app.errors[req_id]}")

        # Pacing
        time.sleep(0.1) 
        
    app.disconnect()
    
    # Save
    df.to_csv("lm_models/merged_data_ibkr.csv", index=False)
    print("Saved lm_models/merged_data_ibkr.csv")
    
    # Train
    train_model(df, 'Target_Short', 'Short Strategy', 'lm_models/short')
    train_model(df, 'Target_Long', 'Long Strategy', 'lm_models/long')

if __name__ == "__main__":
    main()
