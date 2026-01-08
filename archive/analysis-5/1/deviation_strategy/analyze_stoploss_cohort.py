
import pandas as pd
import os

def analyze_stoploss_cohort():
    returns_path = 'full_returns_data.csv'
    
    if not os.path.exists(returns_path):
        print("full_returns_data.csv not found.")
        return

    df = pd.read_csv(returns_path)
    
    print(f"Total Trades: {len(df)}")
    
    # Identify Short Strategy Stop-Losses on T1
    # Short SL is hit if price rises 4%. Return should be <= -0.04.
    # We'll use a threshold slightly loose to catch them (e.g. <= -0.035) just in case, 
    # but likely they are capped at -0.04 in calculation.
    sl_cohort = df[df['ShortReturn_T1'] <= -0.039]
    
    print(f"Short Stop-Loss Cohort (T1 Return <= -3.9%): {len(sl_cohort)} trades")
    
    if not sl_cohort.empty:
        mean_t2 = sl_cohort['Day2Return'].mean()
        win_rate_t2 = (sl_cohort['Day2Return'] > 0).mean()
        
        print(f"Mean T+2 Return (Long): {mean_t2:.4%}")
        print(f"Win Rate (T+2 > 0): {win_rate_t2:.1%}")
        
        print("\nIndividual Trades in Cohort:")
        print(sl_cohort[['Ticker', 'AnalysisDate', 'ShortReturn_T1', 'Day2Return']])
    else:
        print("No Stop-Loss trades found in this dataset.")

if __name__ == "__main__":
    analyze_stoploss_cohort()
