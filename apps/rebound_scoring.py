import json
import argparse
import math
import statistics
import sys
import os
import requests
from datetime import datetime, timedelta

# Import utils
from model_utils import load_coefficients, fetch_historic_volume, calculate_model_prediction

# --- Constants & Config ---
RLS_PARAMS = {'a': 2.10, 'b': 1.35, 'c': 0.90, 'd': -0.20, 'e': 0.05}
BASE_WEIGHTS = {
    'RES': 0.22, 'LIQ': 0.20, 'FUND': 0.16, 
    'NEWS': 0.22, 'EVID': 0.10, 'ATTN': 0.05, 
    'CRWD': 0.03, 'CTX': 0.02
}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def clip(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def winsorize(data, lower_percentile=0.025, upper_percentile=0.975):
    if len(data) < 2: return data
    sorted_data = sorted(data)
    n = len(data)
    lower_idx = int(n * lower_percentile)
    upper_idx = int(n * upper_percentile)
    lower_val = sorted_data[lower_idx]
    upper_val = sorted_data[upper_idx]
    return [clip(x, lower_val, upper_val) for x in data]

def standardize(values):
    if len(values) < 2 or statistics.stdev(values) == 0:
        return [0.0] * len(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return [(x - mean) / stdev for x in values]

def compute_evid(metrics):
    # EVID_raw = 0.35·PCR + 0.20·EC + 0.15·SD + 0.15·NRI + 0.10·HDM − 0.15·CONTR
    raw = (0.35 * metrics.get('PCR', 0) + 
           0.20 * metrics.get('EC', 0) + 
           0.15 * metrics.get('SD', 0) + 
           0.15 * metrics.get('NRI', 0) + 
           0.10 * metrics.get('HDM', 0) - 
           0.15 * metrics.get('CONTR', 0))
    return 2 * raw - 1

def compute_news(metrics, fresh_neg):
    # NEWS_raw = 0.45·(1 − FRESH_NEG) + 0.30·CP + 0.15·NRI + 0.10·RD
    # CP is mapped to 0..1 in metrics input? Instructions say "CP: mean tone (-1..+1) mapped to [0,1]"
    # Assuming input 'CP' is already 0..1. If input is -1..1, we map it.
    # Let's assume input metrics are raw as defined in "Evidence Metrics" section of instructions.
    # "CP Consensus Polarity: mean tone (−1..+1) mapped to [0,1]" -> The AGENT should produce the 0..1 value.
    
    raw = (0.45 * (1 - fresh_neg) + 
           0.30 * metrics.get('CP', 0.5) + # Default neutral 0.5
           0.15 * metrics.get('NRI', 0) + 
           0.10 * metrics.get('RD', 0))
    
    if fresh_neg == 1 and metrics.get('PCR', 0) > 0.5: # Primary confirmed assumption if PCR is high? Or pass flag?
        # Instructions: "If FRESH_NEG = 1 and primary-confirmed"
        # We will assume fresh_neg=1 IMPLIES confirmation in the input data context or check PCR.
        # Let's check PCR > 0.0 to be safe, or relying on input flag.
        raw = min(raw, 0.20)
        
    return 2 * raw - 1

def compute_sent(metrics, sent_pro, sent_com):
    # SENT* = 0.65·PRO + 0.35·COM
    # PRO/COM inputs should be -1..1 or 0..1? 
    # Instructions: "PRO: professional/news sentiment = CP weighted by ..." -> The agent inputs PRO/COM.
    # Instructions: "SENT = clip(2·SENT* − 1, −1, +1)" implies SENT* is 0..1?
    # No, "PRO: ... = CP weighted ...". CP is 0..1. 
    # Let's assume PRO and COM are passed as 0..1 values.
    
    sent_star = 0.65 * sent_pro + 0.35 * sent_com
    return clip(2 * sent_star - 1, -1, 1)

def main():
    parser = argparse.ArgumentParser(description="Calculate Rebound Scores")

    parser.add_argument("--input", required=True, help="Path to raw_assessments.json")
    parser.add_argument("--output", required=True, help="Path to output stock_analysis_report.json")
    parser.add_argument("--vix", type=float, default=15.0, help="Current VIX value")
    parser.add_argument("--strategy", choices=['long', 'short'], default='long', help="Strategy model to use")
    args = parser.parse_args()
    
    # Load Coefficients
    coef_data = load_coefficients(args.strategy)
    if not coef_data:
        print("Failed to load coefficients. Exiting.")
        sys.exit(1)

    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input: {e}")
        sys.exit(1)
        
    api_key = os.environ.get("POLYGON_API_KEY")

    processed_tickers = []
    
    # Extract subscores for standardization
    subscores_map = {k: [] for k in BASE_WEIGHTS if k not in ['NEWS', 'EVID']}
    
    # 1. First Pass: Compute Per-Ticker Metrics (EVID, NEWS, SENT) and collect Subscores
    for item in data:
        metrics = item.get('metrics', {})
        subscores = item.get('subscores', {})
        
        # Calculate derived metrics
        evid = compute_evid(metrics)
        news = compute_news(metrics, metrics.get('FRESH_NEG', 0))
        sent = compute_sent(metrics, item.get('sent_pro', 0.5), item.get('sent_com', 0.5))
        
        item['calc_EVID'] = evid
        item['calc_NEWS'] = news
        item['calc_SENT'] = sent
        
        for k in subscores_map:
            subscores_map[k].append(subscores.get(k, 0))

    # 2. Standardize Subscores
    standardized_subscores = {}
    for k, v in subscores_map.items():
        # Winsorize first? Instructions: "Standardize subscores; winsorize at 2.5%..."
        # Usually winsorize THEN standardize or vice versa. 
        # "Standardize subscores; winsorize at 2.5% / 97.5%" -> Ambiguous order. 
        # Standard approach: Winsorize outliers, then Z-score.
        winsorized = winsorize(v)
        standardized = standardize(winsorized)
        standardized_subscores[k] = standardized

    # 3. Calculate Final Scores
    final_results = []
    
    # Dynamic Tilts
    weights = BASE_WEIGHTS.copy()
    if args.vix >= 20: # Or realized vol top quartile check (omitted for strictly VIX here)
        weights['LIQ'] += 0.06
        weights['ATTN'] -= 0.03
        weights['CRWD'] -= 0.03
        # Renormalize? Sum is still 1.0 (0.06 - 0.03 - 0.03 = 0).
    
    print(f"Processing {len(data)} items with {args.strategy} strategy...")

    for idx, item in enumerate(data):
        # Assemble weighted sum
        score = 0
        
        # Subscores
        for k in subscores_map:
            val = standardized_subscores[k][idx]
            # Tilt: Halve RES if earnings drop?
            w = weights[k]
            if k == 'RES' and item.get('is_earnings_drop', False):
                score += (val * w * 0.5) # "Halve RES contribution"
                # Where does the other half go? Renormalize? 
                # Instructions say "halve RES contribution". Usually implies total score reduces or other weights scale up.
                # Simplest interpreation: just reduce the impact component.
            else:
                score += val * w
                
        # Add NEWS and EVID
        score += item['calc_NEWS'] * weights['NEWS']
        score += item['calc_EVID'] * weights['EVID']
        
        # Gating
        # If FRESH_NEG=1 and confirmed: cap (RES+LIQ) <= 0 ...
        # Implementation: Check components. 
        # We need raw RES/LIQ standardized values to check this cap.
        res_z = standardized_subscores['RES'][idx]
        liq_z = standardized_subscores['LIQ'][idx]
        
        fresh_neg = item.get('metrics', {}).get('FRESH_NEG', 0)
        pcr = item.get('metrics', {}).get('PCR', 0)
        total_relevant = item.get('metrics', {}).get('total_relevant', 0)
        contr = item.get('metrics', {}).get('CONTR', 0)
        
        if fresh_neg == 1 and pcr > 0.5: # proxy for confirmed
            if (res_z * weights['RES'] + liq_z * weights['LIQ']) > 0:
                # How to "cap (RES+LIQ) <= 0"? 
                # Force their contribution to be max 0. 
                # We'll assume this means if their sum is positive, set it to 0.
                # Current contribution in 'score' might be mixed.
                # Let's re-calc contribution:
                current_res_liq = res_z * weights['RES'] + liq_z * weights['LIQ']
                if current_res_liq > 0:
                    score -= current_res_liq # Remove positive part
            
            # force NEWS <= -0.20
            if item['calc_NEWS'] > -0.20:
                score -= item['calc_NEWS'] * weights['NEWS']
                score += -0.20 * weights['NEWS']
                item['calc_NEWS'] = -0.20 # Update for RLS calc if used separately? 
                # RLS formula uses NEWS component.
                
        # Overrides on FRS (Final Score) directly
        if pcr < 0.40 and total_relevant >= 10:
            score -= 0.05
        
        if contr >= 0.25:
            score -= 0.08
            item['uncertainty'] = "High" # "Set Uncertainty >= Medium" -> Force High or Medium?
            
        frs = score
        
        # RLS Calculation
        # RLS = sigmoid(a·FRS + b·NEWS + c·SENT + d·(EC − 1.0) + e)
        # Using params from config
        p = RLS_PARAMS
        rls_logit = (p['a'] * frs + 
                     p['b'] * item['calc_NEWS'] + 
                     p['c'] * item['calc_SENT'] + 
                     p['d'] * (item.get('metrics', {}).get('EC', 0) - 1.0) + 
                     p['e'])
        rls_prob = sigmoid(rls_logit)
        
        # Buckets
        rls_pct = rls_prob * 100
        if rls_pct < 35.0: bucket = "Low"
        elif rls_pct < 65.0: bucket = "Medium"
        else: bucket = "High"
        
        rls_str = f"{rls_pct:.1f}% ({bucket})"
        
        # Uncertainty check
        uncertainty = item.get('uncertainty', 'Medium')
        if item.get('metrics', {}).get('SD', 0) < 0.3: uncertainty = "High" # Heuristic? Or explicit agent input?
        # Use agent input 'uncertainty' unless overridden.
        
        confidence = item.get('confidence', 'Medium')

        # Fetch Volume
        ticker = item['ticker']
        vol_30d = 0
        if api_key:
            # Rebound scoring is usually run 'now', so use today's date
            today_str = datetime.now().strftime("%Y-%m-%d")
            vol_30d = fetch_historic_volume(ticker, today_str, api_key)
        
        # Output Object
        final_obj = {
            "rank": 0, # Placeholder
            "ticker": item['ticker'],
            "company": item.get('company', '—'),
            "exchange": item.get('exchange', '—'),
            "sector": item.get('sector', '—'),
            "lastUsd": item.get('lastUsd', 0.0),
            "oneDayReturnPct": item.get('oneDayReturnPct', 0.0),
            "dollarVol": vol_30d, # Passed through?
            "nonFundamental": item.get('nonFundamental', 'No'),
            "news": round(item['calc_NEWS'], 2),
            "sentiment": f"{item['calc_SENT']:.2f}", # Label added by UI? Protocol says: "-0.32 (Bearish)"
            "finalScore": round(frs, 2),
            "sigma": 0.00, # Calculated from ensemble? Script doesn't have ensemble history. 
                           # We will use 0.0 or pass-through if agent provides Stdev of its conceptual ensemble.
            "uncertainty": uncertainty,
            "returnLikelihood1to5d": rls_str,
            "confidence": confidence,
            "reason": item.get('reason', ''),
            "evidenceCheckedCited": item.get('evidenceCheckedCited', ''),
            "metrics": item.get('metrics', {}) # Pass through for Timing Agent
        }
        
        # Add sentiment label
        s_val = item['calc_SENT']
        if s_val > 0.3: s_lbl = "Bullish"
        elif s_val < -0.3: s_lbl = "Bearish"
        else: s_lbl = "Neutral"
        final_obj['sentiment'] = f"{final_obj['sentiment']} ({s_lbl})"
        
        # Calculate Model Prediction
        # Need oneDayReturnPct in input if used
        # Note: 'oneDayReturnPct' is in the coefficients file, so we must add it to the model inputs.
        # But 'oneDayReturnPct' is a property of the item, not in metrics dict.
        # We'll pass it via kwargs or update metrics? 
        # Safer to manually inject it into the coef matching logic:
        # In calculate_model_prediction, we see: pred += metrics.get('oneDayReturnPct', 0) ...
        # So let's add it to metrics temporarily or handle it explicitly.
        
        model_input_metrics = item.get('metrics', {}).copy()
        model_input_metrics['oneDayReturnPct'] = item.get('oneDayReturnPct', 0.0)
        
        pred_return = calculate_model_prediction(
            model_input_metrics, 
            frs, 
            confidence,
            uncertainty,
            vol_30d,
            coef_data
        )
        final_obj['modelReturnPrediction'] = float(pred_return)
        
        final_results.append(final_obj)



# Update main loop to include prediction
# This replaces the loop end to insert the function call

    # Ranking
    final_results.sort(key=lambda x: x['finalScore'], reverse=True)
    for i, res in enumerate(final_results):
        res['rank'] = i + 1
        
    # Write Output
    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()
