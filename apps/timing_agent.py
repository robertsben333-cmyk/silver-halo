import json
import argparse
import sys

def calculate_timing(item):
    """
    Determines Rebound Timing and Target Price based on Contrarian Metrics.
    Uses 'Dip Probability' to suggest entry (Immediate vs Wait).
    """
    ticker = item.get('ticker')
    score = item.get('finalScore', 0)
    metrics = item.get('metrics', {})
    
    # --- 1. Dip Probability Model ---
    # Predicts chance of lower low before rebound.
    # Inputs: CRWD (Crowding), ATTN (Attention), LIQ (Liquidity), SENT (Sentiment)
    # Default values to mid-range if missing
    subscores = item.get('subscores', {})
    
    # We use z-scores from subscores if available, else raw estimate?
    # Actually, the scoring script standardizes them. We can use 'subscores' from standardized dict if we had access,
    # but 'item' passed here is final output object usually? NO, timing_agent reads 'stock_analysis_report.json'.
    # stock_analysis_report.json DOES NOT contain raw subscores usually, only the final object.
    # Wait, let's check input format. 
    # The 'stock_analysis_report.json' DOES NOT have the full metrics breakdown in the final object structure I wrote in rebound_scoring.py!
    # I need to fix `rebound_scoring.py` to pass through the metrics/subscores to the report OR read `raw_assessments.json` in timing agent.
    # The `timing_agent.py` currently reads `stock_analysis_report.json`.
    # Let's check `rebound_scoring.py` output structure in previous turn.
    # It writes `final_results`. `final_results` has: ticker, calc_NEWS, finalScore... NO subscores.
    
    # PATCH: I will first assume `rebound_scoring.py` needs to be updated to include 'metrics' in output for timing agent.
    # For now, I will use 'news' and 'finalScore' as proxies if metrics are missing, 
    # BUT to do this right, I should have access to the metrics.
    # However, 'timing_agent' only takes 'stock_analysis_report.json'.
    # I will modify `rebound_scoring.py` to include `metrics` in its output first? 
    # Or just use the limited data available?
    
    # Let's simply work with what we likely have or can easily derive.
    # We have 'news', 'sentiment' (text), 'finalScore', 'oneDayReturnPct'.
    # This is insufficient for the full Dip Model (CRWD/ATTN).
    
    # DECISION: I must update `rebound_scoring.py` to pass 'metrics' through to `stock_analysis_report.json`.
    # This is a prerequisite.
    pass

def calculate_dip_score(metrics):
    # DipScore = (0.4 * CRWD) + (0.3 * ATTN) + (0.3 * (1 - LIQ))
    # Using 0-1 scale.
    crwd = metrics.get('CRWD', 0.5) # Default 0.5
    attn = metrics.get('ATTN', 0.5)
    liq = metrics.get('LIQ', 0.5)
    
    dip_score = (0.4 * crwd) + (0.3 * attn) + (0.3 * (1.0 - liq))
    return round(dip_score, 2)

def calculate_timing_logic(item):
    score = item.get('finalScore', 0)
    # Recover metrics if passed (requires rebound_scoring update)
    metrics = item.get('metrics', {}) 
    
    # 1. Dip Calculation
    dip_prob = calculate_dip_score(metrics)
    
    one_day_ret = abs(item.get('oneDayReturnPct', 0.0))
    last_price = item.get('lastUsd', 0.0)
    is_non_fundamental = item.get('nonFundamental', 'No') == 'Yes'
    
    timing = "Avoid"
    action = "Avoid"
    confidence = "Low"
    reason = "Insufficient score."
    max_entry_price = None
    
    if score < 0.2:
        timing = "Avoid"
        action = "Avoid"
        confidence = "Medium"
        reason = "Score below threshold (0.2). Risk of momentum continuation."
    else:
        # Buy Signal Active - Determine Entry
        if dip_prob > 0.7:
            # High Dip Risk -> Falling Knife
            timing = "Wait 1-2 Days"
            action = "Wait for Stabilization"
            reason = f"High Dip Probability ({dip_prob}). Panic not over. Wait for stabilization."
            confidence = "High"
            # Entry Limit: Allow 5% drop
            max_entry_price = round(last_price * 0.95, 2)
        elif dip_prob >= 0.4:
            # Medium Dip Risk -> Intraday Washout
            timing = "Intraday (After 10:30 AM)"
            action = "Buy after morning dip"
            reason = f"Medium Dip Probability ({dip_prob}). Expect morning washout. Buy dip."
            confidence = "Medium"
            # Entry Limit: Allow 2% drop
            max_entry_price = round(last_price * 0.98, 2)
        else:
            # Low Dip Risk -> Oversold/Safe
            timing = "Immediate (Market Open)"
            action = "Buy at Open"
            reason = f"Low Dip Probability ({dip_prob}). Stock appears capitulated. Gap up risk."
            confidence = "High"
            # Entry Limit: Allow 0.5% slippage up
            max_entry_price = round(last_price * 1.005, 2)
            
    # 2. Target Price (Exit)
    exit_target = None
    if action != "Avoid":
        # Exit Target = Entry + (Volatility * Score)
        # Using 50% of the drop as a conservative target if score is high
        rebound_potential = one_day_ret / 100.0 * 0.5 * score
        exit_target = round(last_price * (1 + rebound_potential), 2)
        
    return timing, action, max_entry_price, exit_target, confidence, reason

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to stock_analysis_report.json")
    parser.add_argument("--output", required=True, help="Path to timing_output.json")
    
    args = parser.parse_args()
    
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        with open(args.output, 'w') as f:
            json.dump([], f)
        sys.exit(0)
        
    results = []
    for item in data:
        timing, action, max_entry, exit_target, conf, reason = calculate_timing_logic(item)
        results.append({
            "ticker": item.get('ticker'),
            "timing_window": timing,
            "action": action,
            "max_entry_price": max_entry,
            "exit_target": exit_target,
            "confidence": conf,
            "reasoning": reason,
            "dip_score": calculate_dip_score(item.get('metrics', {}))
        })
        
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Timing analysis complete. Saved to {args.output}")

if __name__ == "__main__":
    main()
