import json
import argparse
import sys

def calculate_squeeze_score(metrics):
    """
    Calculates probability of a short squeeze or significant bounce.
    Inputs: SFRI (Low=Squeeze), MPI (High=Panic Buy/Cover), QSI (Low=Junk often squeezes hard)
    """
    sfri = metrics.get('SFRI', 5)
    mpi = metrics.get('MPI', 5)
    qsi = metrics.get('QSI', 5)
    
    # Formula: (0.5 * (10 - SFRI)) + (0.3 * MPI) + (0.2 * (10 - QSI))
    # Higher Score = Higher Squeeze Risk
    
    score = (0.5 * (10 - sfri)) + (0.3 * mpi) + (0.2 * (10 - qsi))
    return round(score, 2)

def determine_timing(item):
    """
    Determines Short Timing based on Downside Metrics and Squeeze Probability.
    """
    ticker = item.get('ticker')
    metrics = item.get('metrics', {})
    last_price = item.get('lastUsd', 0.0)
    
    squeeze_score = calculate_squeeze_score(metrics)
    
    # Defaults
    action = "Avoid"
    urgency = 0
    min_short_price = None
    exit_target = None
    reason = "Assess carefully."
    
    # Baseline logic based on Squeeze Score
    if squeeze_score > 7.0:
        # High Squeeze Risk (e.g. Low Float + High Panic)
        action = "Wait / Avoid"
        urgency = 2
        reason = f"High Squeeze Risk ({squeeze_score}). Stock is volatile. Do not short low."
        # Limit: Only short if it rips 10%
        min_short_price = round(last_price * 1.10, 2)
        
    elif squeeze_score >= 4.0:
        # Medium Squeeze Risk (Dead Cat Bounce likely)
        action = "Short after bounce"
        urgency = 6
        reason = f"Medium Squeeze Risk ({squeeze_score}). Expect a bounce. Short into strength."
        # Limit: Wait for 3% bounce
        min_short_price = round(last_price * 1.03, 2)
        # Exit: Target 5% drop from current
        exit_target = round(last_price * 0.95, 2)
        
    else:
        # Low Squeeze Risk (Heavy/Broken)
        action = "Short immediately"
        urgency = 9
        reason = f"Low Squeeze Risk ({squeeze_score}). Stock is heavy. Continuation likely."
        # Limit: Short below current is okay (0.5% buffer)
        min_short_price = round(last_price * 0.995, 2)
        # Exit: Target 10% drop
        exit_target = round(last_price * 0.90, 2)
        
    return action, urgency, reason, squeeze_score, min_short_price, exit_target

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to downside_report.json")
    parser.add_argument("--output", required=True, help="Path to downside_timing_output.json")
    args = parser.parse_args()
    
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input: {e}")
        sys.exit(1)
        
    timing_results = []
    
    # Process each candidate
    # Process each candidate
    for item in data:
        action, urgency, reason, sqz_score, min_short, exit_tgt = determine_timing(item)
        
        timing_results.append({
            "ticker": item.get('ticker'),
            "action": action,
            "urgency_score": urgency,
            "squeeze_score": sqz_score,
            "min_short_price": min_short,
            "exit_target": exit_tgt,
            "driver_category": item.get('driverCategory'),
            "reasoning": reason
        })
        
    # Sort by Urgency (Highest first)
    timing_results.sort(key=lambda x: x['urgency_score'], reverse=True)
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(timing_results, f, indent=2)
        
    print(f"Downside timing analysis complete. Saved to {args.output}")

if __name__ == "__main__":
    main()
