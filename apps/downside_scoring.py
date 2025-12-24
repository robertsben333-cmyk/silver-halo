import json
import argparse
import sys

# Weights for DCL (Downside Continuation Likelihood)
# Content-first: Persistence of driver is most important.
DCL_WEIGHTS = {
    'SPI': 0.35, # Shock Persistence
    'MPI': 0.25, # Microstructure Pressure
    'OHI': 0.15, # Overhang
    'QSI': 0.25  # Quality & Sentiment (New: Fraud, Hate, Moat Decay)
}

def calculate_dcl(metrics):
    """
    Calculates Downside Continuation Likelihood (0-100%).
    Input metrics are expected to be 0-10 scale.
    """
    spi = metrics.get('SPI', 5)
    mpi = metrics.get('MPI', 5)
    ohi = metrics.get('OHI', 5)
    qsi = metrics.get('QSI', 5) # Default to mid if missing
    
    # Weighted sum (resulting in 0-10 scale)
    raw_score = (
        spi * DCL_WEIGHTS['SPI'] +
        mpi * DCL_WEIGHTS['MPI'] +
        ohi * DCL_WEIGHTS['OHI'] +
        qsi * DCL_WEIGHTS['QSI']
    )
    
    # Convert to probability (0-100%)
    # Map 0->0%, 5->50%, 10->100%
    return round(raw_score * 10, 1)

def calculate_scs(dcl, metrics, volatility):
    """
    Calculates Short Candidate Score (0-10 scale).
    SCS = DCL adjusted for Borrow, SqzRisk, liquidity.
    """
    # Base is DCL (0-1)
    base_score = dcl / 100.0 * 10.0 # 0-10
    
    # Adjustments
    sfri = metrics.get('SFRI', 5) # Short Feasibility & Squeeze Risk (10=Safe/Good, 0=Risky/Hard)
    
    # Penalty if SFRI is low (High squeeze risk or hard to borrow)
    # If SFRI < 5, penalize but clamp to min 0.1 to avoid 0 score masking signal
    if sfri < 5:
        penalty = max(0.1, sfri / 5.0)
        base_score *= penalty
    
    # Bonus for high liquidity/rational volatility?
    # No, strictly follow instructions: SCS decreases with Hard-to-borrow, Gap risk.
    
    return round(base_score, 2)

def main():
    parser = argparse.ArgumentParser(description="Downside Continuation Scoring")
    parser.add_argument("--input", required=True, help="Path to downside_assessments.json")
    parser.add_argument("--output", required=True, help="Path to downside_report.json")
    args = parser.parse_args()

    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input: {e}")
        sys.exit(1)

    results = []

    for item in data:
        metrics = item.get('metrics', {})
        
        # Calculate DCL
        dcl = calculate_dcl(metrics)
        
        # Calculate SCS
        scs = calculate_scs(dcl, metrics, 0)
        
        # Determine Uncertainty & Confidence based on scores or input
        # If SFRI is very low, Uncertainty is High
        if metrics.get('SFRI', 5) < 3:
            item['uncertainty'] = 'High'
        
        results.append({
            "rank": 0, # Placeholder
            "ticker": item.get('ticker'),
            "company": item.get('company'),
            "exchange": item.get('exchange'),
            "sector": item.get('sector'),
            "lastUsd": item.get('lastUsd'),
            "oneDayReturnPct": item.get('oneDayReturnPct'),
            "dollarVol": item.get('dollarVol', '—'),
            "driverCategory": item.get('driverCategory', 'Unknown'),
            "nonFundamental": item.get('nonFundamental', 'No'),
            "downsideContinuationLikelihoodNextDay": f"{dcl}%",
            "shortCandidateScore": scs,
            "uncertainty": item.get('uncertainty', 'Medium'),
            "confidence": item.get('confidence', 'Medium'),
            "qsi": metrics.get('QSI', 5), # Persist QSI for DB
            "reason": item.get('reason', ''),
            "evidenceCheckedCited": item.get('evidenceCheckedCited', '')
        })

    # Rank by SCS descending
    results.sort(key=lambda x: x['shortCandidateScore'], reverse=True)
    
    # Assign Ranks
    for i, res in enumerate(results):
        res['rank'] = i + 1

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Scoring complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
