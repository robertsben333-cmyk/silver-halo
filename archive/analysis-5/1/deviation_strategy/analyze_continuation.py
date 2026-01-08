import pandas as pd
import os

def analyze_continuation():
    returns_path = 'full_returns_data.csv'
    
    if not os.path.exists(returns_path):
        print("full_returns_data.csv not found.")
        return

    df = pd.read_csv(returns_path)
    
    # Filter out SMX
    df = df[df['Ticker'] != 'SMX']
    
    print(f"Total Trades (excluding SMX): {len(df)}")
    
    print("\n--- Short Continuation Strategy ---")
    # Condition: Short Strategy on T+1 hit Stop Loss (Price Rallied > 4%)
    # This implies ShortReturn_T1 approx -0.04
    short_sl_cohort = df[df['ShortReturn_T1'] <= -0.039].copy()
    
    print(f"Short SL Cohort: {len(short_sl_cohort)} trades")
    
    if not short_sl_cohort.empty:
        # Strategy: Short again on T+2
        short_sl_cohort['Continuation_Return'] = short_sl_cohort['ShortReturn_T2_Strat']
        
        mean_ret = short_sl_cohort['Continuation_Return'].mean()
        win_rate = (short_sl_cohort['Continuation_Return'] > 0).mean()
        total_ret = short_sl_cohort['Continuation_Return'].sum()
        
        print(f"Mean Return: {mean_ret:.2%}")
        print(f"Total Return: {total_ret:.2%}")
        print(f"Win Rate: {win_rate:.0%}")
        print("Trades:")
        print(short_sl_cohort[['Ticker', 'AnalysisDate', 'ShortReturn_T1', 'Day2Return', 'Continuation_Return']])
    
    print("\n--- Long Continuation Strategy ---")
    # Condition: Long Strategy on T+1 hit Stop Loss (Price Dropped > 4%)
    # This implies LongReturn_T1 approx -0.04
    long_sl_cohort = df[df['LongReturn_T1'] <= -0.039].copy()
    
    print(f"Long SL Cohort: {len(long_sl_cohort)} trades")
    
    if not long_sl_cohort.empty:
        # Strategy: Long again on T+2
        long_sl_cohort['Continuation_Return'] = long_sl_cohort['LongReturn_T2_Strat']
        
        mean_ret = long_sl_cohort['Continuation_Return'].mean()
        win_rate = (long_sl_cohort['Continuation_Return'] > 0).mean()
        total_ret = long_sl_cohort['Continuation_Return'].sum()
        
        print(f"Mean Return: {mean_ret:.2%}")
        print(f"Total Return: {total_ret:.2%}")
        print(f"Win Rate: {win_rate:.0%}")
        print("Trades:")
        print(long_sl_cohort[['Ticker', 'AnalysisDate', 'LongReturn_T1', 'Day2Return', 'Continuation_Return']])

if __name__ == "__main__":
    analyze_continuation()
