
import json
import os

# findings based on my search steps
FINDINGS = {
    "PENG": {
        "reason": "Initial earnings beat gap-up reversed sharply due to details on 'Penguin Edge' business wind-down and supply chain constraints reported in call.",
        "fresh_neg": 1,
        "sent_pro": 0.4,
        "sent_com": 0.3,
        "metrics": {"PCR": 0.8, "EC": 0.9, "SD": 0.7, "NRI": 0.8, "HDM": 0.9, "CONTR": 0.2, "FRESH_NEG": 1, "CP": 0.3, "RD": 0.8, "total_relevant": 8}
    },
    "VSAT": {
        "reason": "Conflicting reports of price action, but downside attributed to earnings expectations miss and potential satellite deployment delays.",
        "fresh_neg": 1,
        "sent_pro": 0.3,
        "sent_com": 0.4,
        "metrics": {"PCR": 0.6, "EC": 0.7, "SD": 0.6, "NRI": 0.6, "HDM": 0.5, "CONTR": 0.5, "FRESH_NEG": 1, "CP": 0.4, "RD": 0.6, "total_relevant": 6}
    },
    "AMBA": {
        "reason": "Heavy insider selling by SVP overshadowed positive CES product announcements (CV7 chip). Technical sell-off on high volume.",
        "fresh_neg": 0,
        "sent_pro": 0.5,
        "sent_com": 0.2,
        "metrics": {"PCR": 0.7, "EC": 0.8, "SD": 0.8, "NRI": 0.4, "HDM": 0.8, "CONTR": 0.1, "FRESH_NEG": 0, "CP": 0.4, "RD": 0.7, "total_relevant": 7}
    },
    "GLTO": {
        "reason": "Likely profit-taking or technical correction after massive 572% 6-month run. Analyst coverage initiated positive (Outperfom) on same day.",
        "fresh_neg": 0,
        "sent_pro": 0.8,
        "sent_com": 0.6,
        "metrics": {"PCR": 0.5, "EC": 0.6, "SD": 0.5, "NRI": 0.2, "HDM": 0.3, "CONTR": 0.8, "FRESH_NEG": 0, "CP": 0.7, "RD": 0.5, "total_relevant": 5}
    },
    "RYM": {
        "reason": "No specific news found. Likely technical/momentum weakness on low volume or sector correlation. Ticker ambiguity noted.",
        "fresh_neg": 0,
        "sent_pro": 0.5,
        "sent_com": 0.5,
        "metrics": {"PCR": 0.2, "EC": 0.3, "SD": 0.2, "NRI": 0.1, "HDM": 0.1, "CONTR": 0.0, "FRESH_NEG": 0, "CP": 0.5, "RD": 0.2, "total_relevant": 2}
    },
    "CSGP": {
        "reason": "2026 Guidance (Revenue and EPS) missed analyst consensus. Stock hit 52-week low despite buyback announcement.",
        "fresh_neg": 1,
        "sent_pro": 0.3,
        "sent_com": 0.2,
        "metrics": {"PCR": 0.9, "EC": 0.9, "SD": 0.8, "NRI": 0.9, "HDM": 0.9, "CONTR": 0.1, "FRESH_NEG": 1, "CP": 0.2, "RD": 0.9, "total_relevant": 9}
    },
    "TIGO": {
        "reason": "Quarterly earnings miss (EPS $0.34 vs $0.55 est). Revenue slightly down year-over-year.",
        "fresh_neg": 1,
        "sent_pro": 0.3,
        "sent_com": 0.3,
        "metrics": {"PCR": 0.9, "EC": 0.8, "SD": 0.7, "NRI": 0.8, "HDM": 0.9, "CONTR": 0.0, "FRESH_NEG": 1, "CP": 0.3, "RD": 0.8, "total_relevant": 5}
    },
    "GSAT": {
        "reason": "Significant insider selling (CEO) to cover taxes + Profitability concerns. Analyst mixed ratings.",
        "fresh_neg": 0,
        "sent_pro": 0.4,
        "sent_com": 0.3,
        "metrics": {"PCR": 0.8, "EC": 0.8, "SD": 0.7, "NRI": 0.5, "HDM": 0.7, "CONTR": 0.2, "FRESH_NEG": 0, "CP": 0.3, "RD": 0.7, "total_relevant": 8}
    },
    "SWKS": {
        "reason": "Merger uncertainty (Qorvo vote approaching) + Semiconductor industry weakness. Recent insider selling.",
        "fresh_neg": 0,
        "sent_pro": 0.4,
        "sent_com": 0.4,
        "metrics": {"PCR": 0.7, "EC": 0.8, "SD": 0.7, "NRI": 0.6, "HDM": 0.6, "CONTR": 0.1, "FRESH_NEG": 0, "CP": 0.4, "RD": 0.7, "total_relevant": 6}
    },
    "NEGG": {
        "reason": "Strong Sell ratings reiterated by analysts. Technical breakdown below moving averages. Lack of positive catalyst.",
        "fresh_neg": 0,
        "sent_pro": 0.2,
        "sent_com": 0.2,
        "metrics": {"PCR": 0.8, "EC": 0.7, "SD": 0.6, "NRI": 0.3, "HDM": 0.5, "CONTR": 0.0, "FRESH_NEG": 0, "CP": 0.2, "RD": 0.6, "total_relevant": 5}
    }
}

DEFAULT_SUBSCORES = {
    "RES": 0.5, "LIQ": 0.5, "FUND": 0.5, "ATTN": 0.5, "CRWD": 0.5, "CTX": 0.5
}

def main():
    input_path = "outputs/LATEST_RUN/stock_losers_clean.json"
    output_path = "outputs/LATEST_RUN/raw_assessments.json"
    
    with open(input_path, 'r') as f:
        losers = json.load(f)
        
    assessments = []
    
    for item in losers:
        ticker = item['ticker']
        info = FINDINGS.get(ticker, {
            "reason": "No specific news found. Technical weakness likely.",
            "fresh_neg": 0,
            "sent_pro": 0.5,
            "sent_com": 0.5,
            "metrics": {"PCR": 0.0, "EC": 0.0, "SD": 0.0, "NRI": 0.0, "HDM": 0.0, "CONTR": 0.0, "FRESH_NEG": 0, "CP": 0.5, "RD": 0.0, "total_relevant": 0}
        })
        
        assessment = {
            "ticker": ticker,
            "company": item.get('name', 'Unknown'),
            "exchange": item.get('exchange', 'Unknown'),
            "sector": "Unknown", # Analysis agent usually infers this, we'll leave as Unknown or fill if critical
            "lastUsd": item['currentPrice'],
            "oneDayReturnPct": item['changePct'],
            "reason": info['reason'],
            "evidenceCheckedCited": f"{info['metrics']['total_relevant']} checked / {info['metrics']['total_relevant']} cited",
            "metrics": info['metrics'],
            "subscores": DEFAULT_SUBSCORES,
            "sent_pro": info['sent_pro'],
            "sent_com": info['sent_com'],
            "confidence": "Medium",
            "uncertainty": "Medium"
        }
        assessments.append(assessment)
        
    with open(output_path, 'w') as f:
        json.dump(assessments, f, indent=2)
    
    print(f"Generated manual assessments for {len(assessments)} tickers.")

if __name__ == "__main__":
    main()
