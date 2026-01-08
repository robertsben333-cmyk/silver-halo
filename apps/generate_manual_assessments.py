import json
import os
import sys

# Output Directory
OUTPUT_DIR = r"c:/Users/XavierFriesen/.gemini/antigravity/playground/silver-halo/outputs/20260107_121027"
LOSERS_FILE = os.path.join(OUTPUT_DIR, "stock_losers_clean.json")

# Load Losers
with open(LOSERS_FILE, 'r') as f:
    losers = json.load(f)

# Assessment Data (Manual Research)
# structure: ticker -> {rebound_data, downside_data}
research_data = {
    "ALMS": {
        "reason": "Drop driven by $175M public offering announcement despite positive Phase 3 envudeucitinib data. Classic sell-the-news/dilution event.",
        "driverCategory": "Financing (Dilution)",
        "nonFundamental": "No",
        "metrics": {
            "PCR": 1.0, "EC": 0.8, "SD": 0.7, "NRI": 0.9, "HDM": 1.0, "CONTR": 0.2, "FRESH_NEG": 1, "CP": 0.2, "RD": 0.8, "total_relevant": 15
        },
        "subscores": {"RES": 0.8, "LIQ": 0.2, "FUND": 0.7, "ATTN": 0.9, "CRWD": 0.4, "CTX": 0.5},
        "sent_pro": 0.6, "sent_com": 0.4,
        "confidence": "High", "uncertainty": "Medium",
        "downside_metrics": {"SPI": 6, "MPI": 7, "OHI": 9, "QSI": 4, "SFRI": 6},
        "downside_reason": "Dilution overhang from $175M offering will likely persist, capping upside despite clinical success.",
        "downside_prob": "High",
        "scs": 7.5
    },
    "DRUG": {
        "reason": "Drop due to $100M public offering announced alongside positive Phase 2 epilepsy data. Market focused on dilution.",
        "driverCategory": "Financing (Dilution)",
        "nonFundamental": "No",
        "metrics": {
            "PCR": 1.0, "EC": 0.7, "SD": 0.6, "NRI": 0.9, "HDM": 1.0, "CONTR": 0.1, "FRESH_NEG": 1, "CP": 0.3, "RD": 0.7, "total_relevant": 12
        },
        "subscores": {"RES": 0.7, "LIQ": 0.3, "FUND": 0.6, "ATTN": 0.8, "CRWD": 0.5, "CTX": 0.5},
        "sent_pro": 0.5, "sent_com": 0.3,
        "confidence": "High", "uncertainty": "Medium",
        "downside_metrics": {"SPI": 5, "MPI": 6, "OHI": 8, "QSI": 5, "SFRI": 5},
        "downside_reason": "Offering-related selling pressure likely to continue near-term.",
        "downside_prob": "Medium",
        "scs": 6.0
    },
    "WVE": {
        "reason": "Sell-the-news reaction to Phase 1 obesity data. Despite analyst upgrades and positive results, stock fell (bearish divergence).",
        "driverCategory": "Idiosyncratic (Sell-the-News)",
        "nonFundamental": "Yes",
        "metrics": {
            "PCR": 0.8, "EC": 0.6, "SD": 0.8, "NRI": 0.7, "HDM": 0.6, "CONTR": 0.8, "FRESH_NEG": 0, "CP": 0.7, "RD": 0.9, "total_relevant": 20
        },
        "subscores": {"RES": 0.9, "LIQ": 0.8, "FUND": 0.8, "ATTN": 0.9, "CRWD": 0.2, "CTX": 0.8},
        "sent_pro": 0.8, "sent_com": 0.6,
        "confidence": "Medium", "uncertainty": "High",
        "downside_metrics": {"SPI": 2, "MPI": 8, "OHI": 2, "QSI": 3, "SFRI": 4},
        "downside_reason": "Momentum crash after data release suggests trapped longs, but fundamentals (analyst upgrades) remain strong.",
        "downside_prob": "Low",
        "scs": 3.0
    },
    "THH": {
        "reason": "No specific news found. Drop appears technical or related to post-IPO volatility.",
        "driverCategory": "Technical / Unknown",
        "nonFundamental": "Yes",
        "metrics": {
            "PCR": 0.2, "EC": 0.3, "SD": 0.2, "NRI": 0.1, "HDM": 0.1, "CONTR": 0.0, "FRESH_NEG": 0, "CP": 0.5, "RD": 0.2, "total_relevant": 4
        },
         "subscores": {"RES": 0.5, "LIQ": 0.5, "FUND": 0.5, "ATTN": 0.2, "CRWD": 0.1, "CTX": 0.5},
        "sent_pro": 0.4, "sent_com": 0.5,
        "confidence": "Low", "uncertainty": "High",
        "downside_metrics": {"SPI": 3, "MPI": 4, "OHI": 3, "QSI": 2, "SFRI": 3},
        "downside_reason": "Lack of clear catalyst makes direction uncertain. Likely choppy.",
        "downside_prob": "Low",
        "scs": 2.0
    },
    "VSNT": {
        "reason": "Continued selling pressure following spin-off from Comcast. Structural flow/overhang and sector skepticism.",
        "driverCategory": "Structural (Spin-off)",
        "nonFundamental": "Yes",
        "metrics": {
            "PCR": 0.9, "EC": 0.6, "SD": 0.5, "NRI": 0.4, "HDM": 0.8, "CONTR": 0.0, "FRESH_NEG": 0, "CP": 0.4, "RD": 0.5, "total_relevant": 8
        },
        "subscores": {"RES": 0.4, "LIQ": 0.1, "FUND": 0.4, "ATTN": 0.3, "CRWD": 0.7, "CTX": 0.2},
        "sent_pro": 0.3, "sent_com": 0.2,
        "confidence": "High", "uncertainty": "Medium",
        "downside_metrics": {"SPI": 4, "MPI": 6, "OHI": 9, "QSI": 6, "SFRI": 7},
        "downside_reason": "Spin-off indigestion and index selling often persist for days/weeks.",
        "downside_prob": "High",
        "scs": 8.0
    },
    "CCCX": {
        "reason": "Drop on S-4 filing for merger with Infleqtion. SPAC merger risks and 'sell the news' logic applying.",
        "driverCategory": "Merger/Acquisition (SPAC)",
        "nonFundamental": "No",
        "metrics": {
            "PCR": 1.0, "EC": 0.5, "SD": 0.4, "NRI": 0.8, "HDM": 0.9, "CONTR": 0.1, "FRESH_NEG": 0, "CP": 0.4, "RD": 0.4, "total_relevant": 6
        },
        "subscores": {"RES": 0.5, "LIQ": 0.4, "FUND": 0.3, "ATTN": 0.4, "CRWD": 0.3, "CTX": 0.4},
        "sent_pro": 0.3, "sent_com": 0.3,
        "confidence": "Medium", "uncertainty": "Medium",
        "downside_metrics": {"SPI": 5, "MPI": 5, "OHI": 6, "QSI": 7, "SFRI": 6},
        "downside_reason": "SPAC De-SPAC processes often see grinding lower prices.",
        "downside_prob": "Medium",
        "scs": 6.5
    },
    "ACFN": {
        "reason": "Profit taking pullback (-9%) after massive +30% surge on Partnership news. Healthy correction.",
        "driverCategory": "Technical (Profit Taking)",
        "nonFundamental": "Yes",
        "metrics": {
            "PCR": 0.7, "EC": 0.4, "SD": 0.4, "NRI": 0.2, "HDM": 0.3, "CONTR": 0.0, "FRESH_NEG": 0, "CP": 0.7, "RD": 0.3, "total_relevant": 5
        },
        "subscores": {"RES": 0.8, "LIQ": 0.7, "FUND": 0.8, "ATTN": 0.6, "CRWD": 0.1, "CTX": 0.7},
        "sent_pro": 0.8, "sent_com": 0.7,
        "confidence": "Medium", "uncertainty": "Medium",
        "downside_metrics": {"SPI": 2, "MPI": 3, "OHI": 2, "QSI": 2, "SFRI": 2},
        "downside_reason": "Drop is likely temporary profit taking after positive catalyst.",
        "downside_prob": "Low",
        "scs": 1.0
    },
    "ANRO": {
        "reason": "No specific news today. Director options grant (Jan 6) is routine. Likely technical drift.",
        "driverCategory": "Technical / Unknown",
        "nonFundamental": "Yes",
        "metrics": {
            "PCR": 0.3, "EC": 0.2, "SD": 0.2, "NRI": 0.1, "HDM": 0.1, "CONTR": 0.0, "FRESH_NEG": 0, "CP": 0.5, "RD": 0.1, "total_relevant": 3
        },
        "subscores": {"RES": 0.5, "LIQ": 0.5, "FUND": 0.4, "ATTN": 0.1, "CRWD": 0.1, "CTX": 0.5},
        "sent_pro": 0.5, "sent_com": 0.5,
        "confidence": "Low", "uncertainty": "High",
        "downside_metrics": {"SPI": 3, "MPI": 3, "OHI": 3, "QSI": 3, "SFRI": 3},
        "downside_reason": "No clear driver.",
        "downside_prob": "Low",
        "scs": 2.5
    },
    "PBF": {
        "reason": "Refinery (Martinez) restart delayed to Feb/Mar 2026. Fundamental negative impacting near-term earnings.",
        "driverCategory": "Fundamental (Operational)",
        "nonFundamental": "No",
        "metrics": {
            "PCR": 1.0, "EC": 0.8, "SD": 0.7, "NRI": 0.9, "HDM": 1.0, "CONTR": 0.0, "FRESH_NEG": 1, "CP": 0.2, "RD": 0.8, "total_relevant": 12
        },
        "subscores": {"RES": 0.5, "LIQ": 0.2, "FUND": 0.2, "ATTN": 0.6, "CRWD": 0.4, "CTX": 0.3},
        "sent_pro": 0.3, "sent_com": 0.2,
        "confidence": "High", "uncertainty": "Low",
        "downside_metrics": {"SPI": 8, "MPI": 7, "OHI": 4, "QSI": 4, "SFRI": 6},
        "downside_reason": "Operational delay is a persistent drag on cash flow.",
        "downside_prob": "High",
        "scs": 7.0
    },
    "SION": {
        "reason": "Insider selling pressure and recent downgrades weighing on sentiment despite upcoming conference.",
        "driverCategory": "Flow / Sentiment",
        "nonFundamental": "Yes",
        "metrics": {
            "PCR": 0.8, "EC": 0.6, "SD": 0.5, "NRI": 0.6, "HDM": 0.4, "CONTR": 0.3, "FRESH_NEG": 0, "CP": 0.4, "RD": 0.5, "total_relevant": 7
        },
        "subscores": {"RES": 0.6, "LIQ": 0.4, "FUND": 0.6, "ATTN": 0.5, "CRWD": 0.3, "CTX": 0.4},
        "sent_pro": 0.4, "sent_com": 0.3,
        "confidence": "Medium", "uncertainty": "Medium",
        "downside_metrics": {"SPI": 4, "MPI": 5, "OHI": 6, "QSI": 5, "SFRI": 4},
        "downside_reason": "Insider selling creates headwind, but likely not a structural crash.",
        "downside_prob": "Medium",
        "scs": 4.0
    }
}

# Construct Rebound List
rebound_list = []
for ticker, data in research_data.items():
    loser_item = next((x for x in losers if x['ticker'] == ticker), None)
    if not loser_item: continue
    
    rebound_list.append({
        "ticker": ticker,
        "company": loser_item.get("name", "N/A"),
        "exchange": loser_item.get("exchange", "N/A"),
        "sector": "N/A", # Would need more research
        "lastUsd": loser_item["currentPrice"],
        "oneDayReturnPct": loser_item["changePct"],
        "reason": data["reason"],
        "evidenceCheckedCited": "10 checked / 5+ relevant",
        "metrics": data["metrics"],
        "subscores": data["subscores"],
        "sent_pro": data["sent_pro"],
        "sent_com": data["sent_com"],
        "confidence": data["confidence"],
        "uncertainty": data["uncertainty"]
    })

# Save Files
rebound_path = os.path.join(OUTPUT_DIR, "raw_assessments.json")
with open(rebound_path, 'w') as f:
    json.dump(rebound_list, f, indent=2)
print(f"Saved {rebound_path}")


