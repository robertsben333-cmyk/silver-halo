import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configuration
INPUT_FILE = r"C:\Users\XavierFriesen\.gemini\antigravity\playground\silver-halo\analysis-5\1\analysis_history.csv"
OUTPUT_DIR = r"C:\Users\XavierFriesen\.gemini\antigravity\playground\silver-halo\analysis-5\1"
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "backtest_results.png")
STOPLOSS_PCT = 0.04

def run_backtest():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # Ensure required columns exist
    required_cols = ['date', 'ticker', 'finalScore', 'PriceAtOpen', 'PriceAt1730CET']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return

    # Filter for valid trades (Score < 0)
    # Also ensure prices are numeric and not NaN
    df['PriceAtOpen'] = pd.to_numeric(df['PriceAtOpen'], errors='coerce')
    df['PriceAt1730CET'] = pd.to_numeric(df['PriceAt1730CET'], errors='coerce')
    
    # Filter: Score < 0 and valid prices
    trades = df[
        (df['finalScore'] < 0) & 
        (df['PriceAtOpen'].notna()) & 
        (df['PriceAt1730CET'].notna())
    ].copy()

    if trades.empty:
        print("No valid trades found (finalScore < 0 and valid prices).")
        return

    print(f"Found {len(trades)} potential trades.")

    # Sort by date to simulate timeline
    trades['date'] = pd.to_datetime(trades['date'])
    trades = trades.sort_values('date')

    # Calculate Returns
    # Short Strategy: Profit if Price goes DOWN.
    # Return = (Open - Close) / Open
    trades['raw_return'] = (trades['PriceAtOpen'] - trades['PriceAt1730CET']) / trades['PriceAtOpen']

    # Apply Stoploss
    # Max loss is STOPLOSS_PCT (e.g. 0.04). Loss is negative return.
    # So realized return cannot be less than -0.04.
    # max(return, -0.04)
    trades['realized_return'] = trades['raw_return'].apply(lambda x: max(x, -STOPLOSS_PCT))

    # Calculate Stoploss Hits
    trades['stoploss_triggered'] = trades['raw_return'] < -STOPLOSS_PCT

    # Equity Curve Calculation (Assuming equal position sizing, compounded)
    # Start with 1.0 (100%)
    # For "Equal amount of funds in each trade", strictly speaking, if we have multiple trades on the same day,
    # we split capital? Or do we assume 1 unit per trade?
    # User asked: "invest an equal amount of funds in each trade". 
    # Let's show simple cumulative return (Sum of % returns) AND Compounded Equity Curve.
    
    # Simple Sum Return
    total_return_simple = trades['realized_return'].sum()
    
    # Compounded Equity (Portfolio Simulation)
    # This is tricky without knowing concurrent trades. 
    # Let's assume sequential for the equity curve or just 1 trade at a time logic for simplicity, 
    # OR simpler: Cumulative Return Index = Product(1 + r)
    trades['equity_curve'] = (1 + trades['realized_return']).cumprod()

    # Statistics
    total_trades = len(trades)
    winners = len(trades[trades['realized_return'] > 0])
    losers = len(trades[trades['realized_return'] < 0])
    win_rate = (winners / total_trades) * 100 if total_trades > 0 else 0
    
    final_equity = trades['equity_curve'].iloc[-1]
    total_return_compounded = (final_equity - 1) * 100
    avg_return = trades['realized_return'].mean() * 100

    print("\n--- Backtest Results ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}% ({winners} W / {losers} L)")
    print(f"Stoploss Triggered: {trades['stoploss_triggered'].sum()} times")
    print(f"Average Return per Trade: {avg_return:.2f}%")
    print(f"Total Return (Simple Sum of %): {total_return_simple * 100:.2f}%")
    print(f"Total Return (Compounded): {total_return_compounded:.2f}%")
    print("------------------------")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(trades['date'], trades['equity_curve'], marker='o', linestyle='-')
    plt.title(f"Cumulative Return (Compounded)\nStrategy: Short if Score < 0, Exit 2hrs later, {STOPLOSS_PCT*100}% Stop")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Start = 1.0)")
    plt.grid(True)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PLOT)
    print(f"\nPlot saved to: {OUTPUT_PLOT}")

    # Plot Return vs Score
    plt.figure(figsize=(10, 6))
    plt.scatter(trades['finalScore'], trades['realized_return'] * 100, alpha=0.7)
    plt.title("Trade Return vs. Final Score")
    plt.xlabel("Final Score")
    plt.ylabel("Realized Return (%)")
    plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    OUTPUT_SCATTER = os.path.join(OUTPUT_DIR, "score_vs_return.png")
    plt.savefig(OUTPUT_SCATTER)
    print(f"Scatter plot saved to: {OUTPUT_SCATTER}")

    # Save detailed trade log
    output_csv = os.path.join(OUTPUT_DIR, "backtest_trades.csv")
    trades.to_csv(output_csv, index=False)
    print(f"Detailed trade log saved to: {output_csv}")

    # --- Threshold Sensitivity Analysis ---
    print("\nRunning Threshold Sensitivity Analysis...")
    
    # Reload all data to include positive scores
    df_all = pd.read_csv(INPUT_FILE)
    df_all['PriceAtOpen'] = pd.to_numeric(df_all['PriceAtOpen'], errors='coerce')
    df_all['PriceAt1730CET'] = pd.to_numeric(df_all['PriceAt1730CET'], errors='coerce')
    df_all = df_all.dropna(subset=['PriceAtOpen', 'PriceAt1730CET', 'finalScore']).copy()

    # Calculate Short Returns + Stoploss for ALL data points
    df_all['raw_return'] = (df_all['PriceAtOpen'] - df_all['PriceAt1730CET']) / df_all['PriceAtOpen']
    df_all['realized_return'] = df_all['raw_return'].apply(lambda x: max(x, -STOPLOSS_PCT))

    # Sweep thresholds
    min_score = df_all['finalScore'].min()
    max_score = df_all['finalScore'].max()
    thresholds = np.arange(np.floor(min_score), np.ceil(max_score) + 0.1, 0.1)
    
    unique_days = df_all['date'].nunique() if 'date' in df_all.columns else 1 # Ensure we have date count
    # Ensure 'date' is datetime in df_all
    if 'date' in df_all.columns:
         df_all['date'] = pd.to_datetime(df_all['date'])
         unique_days = df_all['date'].nunique()
    else:
        unique_days = 1 # Fallback
        
    print(f"Total Unique Trading Days: {unique_days}")

    results = []

    for t in thresholds:
        subset = df_all[df_all['finalScore'] < t]
        if not subset.empty:
            avg_ret = subset['realized_return'].mean() * 100
            count = len(subset)
            trades_per_day = count / unique_days
            results.append({'Threshold': t, 'AvgReturn': avg_ret, 'TradeCount': count, 'TradesPerDay': trades_per_day})
    
    res_df = pd.DataFrame(results)

    if not res_df.empty:
        # Find Optimal Threshold (TradesPerDay >= 2)
        # "Minimum 2 a 3 trades a day" -> Let's look for >= 2.0
        target_df = res_df[res_df['TradesPerDay'] >= 2.0]
        
        if not target_df.empty:
            optimal_row = target_df.loc[target_df['AvgReturn'].idxmax()]
            print("\n--- Optimization Result ---")
            print(f"Optimal Threshold (for >= 2 trades/day): {optimal_row['Threshold']:.2f}")
            print(f"Expected Avg Return per Trade: {optimal_row['AvgReturn']:.2f}%")
            print(f"Avg Trades per Day: {optimal_row['TradesPerDay']:.2f}")
            print(f"Total Trades: {int(optimal_row['TradeCount'])}")
        else:
            print("\n--- Optimization Result ---")
            print("No threshold found with >= 2.0 trades/day.")

        # Plot Threshold vs Avg Return and TradesPerDay
        plt.figure(figsize=(10, 6))
        
        # Plot Avg Return on primary y-axis
        plt.plot(res_df['Threshold'], res_df['AvgReturn'], label='Avg Return (%)', color='blue')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel("Score Threshold (Short if Score < T)")
        plt.ylabel("Average Return per Trade (%)")
        plt.title(f"Strategy Optimization (Unique Days: {unique_days})")
        plt.grid(True)
        
        # Plot Trades Per Day on secondary y-axis
        ax2 = plt.gca().twinx()
        ax2.plot(res_df['Threshold'], res_df['TradesPerDay'], label='Trades/Day', color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel("Avg Trades per Day")
        ax2.axhline(y=2.0, color='green', linestyle=':', label='Target (2.0)')
        
        # Combine legends
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        OUTPUT_THRESHOLD = os.path.join(OUTPUT_DIR, "threshold_analysis.png")
        plt.savefig(OUTPUT_THRESHOLD)
        print(f"Threshold analysis plot saved to: {OUTPUT_THRESHOLD}")
        
        # Save threshold data
        OUTPUT_THRESHOLD_CSV = os.path.join(OUTPUT_DIR, "threshold_analysis.csv")
        res_df.to_csv(OUTPUT_THRESHOLD_CSV, index=False)
        print(f"Threshold analysis data saved to: {OUTPUT_THRESHOLD_CSV}")

    # --- SCS Score Correlation Analysis ---
    print("\nRunning SCS Score Correlation Analysis...")
    DOWNSIDE_FILE = os.path.join(os.path.dirname(INPUT_FILE), "downside_history.csv")
    
    try:
        df_downside = pd.read_csv(DOWNSIDE_FILE)
        # Process timestamps to get date for merging
        # User warned date might not be universal, so use timestamp
        if 'timestamp' in df_downside.columns:
            df_downside['date_merge'] = pd.to_datetime(df_downside['timestamp']).dt.date
        elif 'date' in df_downside.columns:
            df_downside['date_merge'] = pd.to_datetime(df_downside['date']).dt.date
        else:
            print("Warning: No timestamp or date in downside_history.csv. Skipping SCS plot.")
            return

        # Prepare trades data for merge
        # calculating return for ALL potential trades (not just < 0 score, unless user wants that restriction? 
        # User said "plot return per trade", usually implies the trades we took. 
        # But for correlation, seeing all data is often useful. 
        # However, "return per trade" usually refers to the Strategy's trades. 
        # Let's stick to the 'trades' dataframe which is the filtered one (Score < 0). 
        # OR better: use df_all to show broad correlation if possible? 
        # "return per trade ... corresponding ticker ... downside_history ... prices in analysis_history"
        # Let's use the 'trades' df which contains the executed trades.
        
        trades_for_merge = trades.copy()
        trades_for_merge['date_merge'] = pd.to_datetime(trades_for_merge['date']).dt.date
        
        # Select relevant columns from downside
        # Handling potential duplicates in downside: duplicate (ticker, date) tuples?
        # Taking the latest entry per day if duplicates exist.
        df_downside_dedup = df_downside.sort_values('timestamp').groupby(['ticker', 'date_merge']).tail(1)
        
        merged_df = pd.merge(trades_for_merge, 
                             df_downside_dedup[['ticker', 'date_merge', 'scs_score']], 
                             on=['ticker', 'date_merge'], 
                             how='inner')
        
        if not merged_df.empty:
            plt.figure(figsize=(10, 6))
            plt.scatter(merged_df['scs_score'], merged_df['realized_return'] * 100, alpha=0.7, color='purple')
            plt.title("Trade Return vs. SCS Score")
            plt.xlabel("SCS Score")
            plt.ylabel("Realized Return (%)")
            plt.grid(True)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            OUTPUT_SCS = os.path.join(OUTPUT_DIR, "scs_vs_return.png")
            plt.savefig(OUTPUT_SCS)
            print(f"SCS Score correlation plot saved to: {OUTPUT_SCS}")
            print(f"Matched {len(merged_df)} trades with SCS scores.")
        else:
            print("Warning: No matches found between trades and downside_history.")
            
    except FileNotFoundError:
        print(f"Warning: {DOWNSIDE_FILE} not found. Skipping SCS plot.")

    # --- Comprehensive Correlation Analysis ---
    print("\nRunning Comprehensive Correlation Analysis...")
    
    # Define features to analyze
    # Standard columns
    feature_candidates = ['finalScore', 'news', 'confidence', 'returnLikelihood', 'sentiment', 'uncertainty']
    # Add all metrics_* columns from the dataframe
    metric_cols = [c for c in df_all.columns if c.startswith('metrics_')]
    feature_candidates.extend(metric_cols)
    
    # We need to analyze these against realized_return.
    # Note: df_all contains all data points (positive and negative scores). 
    # The correlation should be on the Full Dataset to see general predictive power.
    
    # Ensure columns are numeric
    for col in feature_candidates:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    correlation_results = []
    
    # Create correlations directory
    CORR_DIR = os.path.join(OUTPUT_DIR, "correlations")
    os.makedirs(CORR_DIR, exist_ok=True)

    for feature in feature_candidates:
        if feature not in df_all.columns:
            continue
            
        # Drop NaNs for this pair
        valid_data = df_all[[feature, 'realized_return']].dropna()
        if valid_data.empty:
            continue
            
        correlation = valid_data[feature].corr(valid_data['realized_return'])
        correlation_results.append({'Feature': feature, 'Correlation': correlation, 'Count': len(valid_data)})
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.scatter(valid_data[feature], valid_data['realized_return'] * 100, alpha=0.5)
        plt.title(f"Return vs {feature} (Corr: {correlation:.2f})")
        plt.xlabel(feature)
        plt.ylabel("Realized Return (%)")
        plt.grid(True)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        safe_name = feature.replace("/", "_")
        plt.savefig(os.path.join(CORR_DIR, f"corr_{safe_name}.png"))
        plt.close()

    # Also add SCS Score to the comparison list
    # Re-using the merged_df from SCS section if available
    try:
        # We need to re-run the merge logic for df_all (not just trades) to get full SCS correlation
        # or just use the subset we already have. 
        # Let's use the subset logic from the previous block but applied to df_all? 
        # Actually, let's just use what we have in 'merged_df' if it exists in scope?
        # Variable 'merged_df' is local to the previous block.
        # Let's re-do the merge quickly for the summary table.
        
        if 'df_downside_dedup' in locals():
             # process based on df_all instead of trades
             df_all_merge = df_all.copy()
             if 'date' in df_all_merge.columns:
                 df_all_merge['date_merge'] = pd.to_datetime(df_all_merge['date']).dt.date
             
             scs_merged = pd.merge(df_all_merge, 
                                 df_downside_dedup[['ticker', 'date_merge', 'scs_score']], 
                                 on=['ticker', 'date_merge'], 
                                 how='inner')
             
             if not scs_merged.empty:
                 scs_corr = scs_merged['scs_score'].corr(scs_merged['realized_return'])
                 correlation_results.append({'Feature': 'scs_score', 'Correlation': scs_corr, 'Count': len(scs_merged)})
    except Exception as e:
        print(f"Error calculating SCS correlation for summary: {e}")

    # Output Summary CSV
    corr_df = pd.DataFrame(correlation_results).sort_values('Correlation', ascending=False)
    SUMMARY_CSV = os.path.join(OUTPUT_DIR, "correlation_summary.csv")
    corr_df.to_csv(SUMMARY_CSV, index=False)
    
    print("\n--- Correlation Summary ---")
    print(corr_df)
    print(f"\nCorrelation plots saved to: {CORR_DIR}")
    print(f"Summary saved to: {SUMMARY_CSV}")

if __name__ == "__main__":
    run_backtest()
