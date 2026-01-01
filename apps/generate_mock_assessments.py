import json
import os

OUTPUT_DIR = "c:/Users/XavierFriesen/.gemini/antigravity/playground/silver-halo/outputs/20251224_135658"

# SMX, INBX, SDHC data from research
tickers = ["SMX", "INBX", "SDHC"]

# --- Branch A: Contrarian (raw_assessments.json) ---
contrarian_data = [
    {
        "ticker": "SMX",
        "company": "SMX (Security Matters) Public Limited Company",
        "exchange": "XNAS",
        "sector": "Technology",
        "lastUsd": 148.64,
        "oneDayReturnPct": -11.63,
        "reason": "Oversold conditions (RSI, Bollinger) but high volatility and skepticism.",
        "evidenceCheckedCited": "Oversold RSI, Bollinger Bands, Nasdaq delinquency notices.",
        "metrics": {
            "PCR": 0.9, # Primary Claim (Oversold) Relevant
            "EC": 1.2,  # Evidence Consistency (Mixed)
            "SD": 0.4,  # Source Diversity (Low, mostly bots/news aggregators)
            "NRI": 0.1, # New Relevant Info (None positive)
            "HDM": 0.8, # Historical Data Match (Volatile)
            "CONTR": 0.7, # Contradictory evidence (Dilution vs Oversold)
            "FRESH_NEG": 0,
            "CP": 0.3, # Sentiment Consensus (Bearish/Skeptical)
            "RD": 0.5,
            "total_relevant": 5
        },
        "subscores": {
            "RES": 0.2, # Resilience
            "LIQ": 0.8, # Liquidity (High volume selling)
            "FUND": 0.1, # Fundamentals (Weak)
            "ATTN": 0.7,
            "CRWD": 0.6,
            "CTX": 0.5
        },
        "sent_pro": 0.3,
        "sent_com": 0.2,
        "confidence": "Medium",
        "uncertainty": "High"
    },
    {
        "ticker": "INBX",
        "company": "Inhibrx Biosciences, Inc.",
        "exchange": "XNAS",
        "sector": "Healthcare",
        "lastUsd": 78.08,
        "oneDayReturnPct": -10.14,
        "reason": "Drop likely due to acquisition investigation/shareholder selling. Legal victory in trade secrets.",
        "evidenceCheckedCited": "Weiss Law investigation, Global Investors selling, Trade secret win.",
        "metrics": {
            "PCR": 0.7,
            "EC": 1.0, 
            "SD": 0.6,
            "NRI": 0.4, # Trade secret win is positive
            "HDM": 0.5,
            "CONTR": 0.3,
            "FRESH_NEG": 1, # Investigation is negative
            "CP": 0.4,
            "RD": 0.6,
            "total_relevant": 6
        },
        "subscores": {
            "RES": 0.6,
            "LIQ": 0.5,
            "FUND": 0.5, # Biotech fundamentals tricky
            "ATTN": 0.4,
            "CRWD": 0.4,
            "CTX": 0.6
        },
        "sent_pro": 0.5,
        "sent_com": 0.4,
        "confidence": "Medium",
        "uncertainty": "Medium"
    },
    {
        "ticker": "SDHC",
        "company": "Smith Douglas Homes Corp.",
        "exchange": "XNYS",
        "sector": "Real Estate",
        "lastUsd": 18.06,
        "oneDayReturnPct": -9.29,
        "reason": "Earnings miss and analyst downgrades. Volume drying up.",
        "evidenceCheckedCited": "Q3 earnings miss, RBC downgrade, Zacks Strong Sell.",
        "metrics": {
            "PCR": 0.8,
            "EC": 1.5, # Consistent negative news
            "SD": 0.8,
            "NRI": 0.2,
            "HDM": 0.6,
            "CONTR": 0.1,
            "FRESH_NEG": 1, # Earnings miss
            "CP": 0.2, # Very bearish
            "RD": 0.7,
            "total_relevant": 8
        },
        "subscores": {
            "RES": 0.4,
            "LIQ": 0.3, # Volume low
            "FUND": 0.4,
            "ATTN": 0.3,
            "CRWD": 0.2,
            "CTX": 0.7
        },
        "sent_pro": 0.2,
        "sent_com": 0.3,
        "confidence": "High",
        "uncertainty": "Low"
    }
]

with open(os.path.join(OUTPUT_DIR, "raw_assessments.json"), "w") as f:
    json.dump(contrarian_data, f, indent=2)

# --- Branch B: Downside (downside_assessments.json) ---
downside_data = [
    {
        "ticker": "SMX",
        "company": "SMX (Security Matters)",
        "exchange": "XNAS",
        "sector": "Technology",
        "lastUsd": 148.64,
        "oneDayReturnPct": -11.63,
        "driverCategory": "Skepticism/Dilution",
        "reason": "High skepticism, previous massive drop, dilution concerns.",
        "evidenceCheckedCited": "History of volatility, unproven strategy.",
        "metrics": {
            "SPI": 7, # Shock Persistence
            "MPI": 6, 
            "OHI": 5,
            "QSI": 8, # Quality concerns high
            "SFRI": 4 # Risky short due to volatility
        },
        "confidence": "Medium",
        "uncertainty": "High"
    },
    {
        "ticker": "INBX",
        "company": "Inhibrx Biosciences",
        "exchange": "XNAS",
        "sector": "Healthcare",
        "lastUsd": 78.08,
        "oneDayReturnPct": -10.14,
        "driverCategory": "Legal/Investigation",
        "reason": "Fiduciary duty investigation regarding acquisition.",
        "evidenceCheckedCited": "Weiss Law LLP investigation.",
        "metrics": {
            "SPI": 5, 
            "MPI": 5,
            "OHI": 6,
            "QSI": 6,
            "SFRI": 7 
        },
        "confidence": "Medium",
        "uncertainty": "Medium"
    },
    {
        "ticker": "SDHC",
        "company": "Smith Douglas Homes",
        "exchange": "XNYS",
        "sector": "Real Estate",
        "lastUsd": 18.06,
        "oneDayReturnPct": -9.29,
        "driverCategory": "Earnings/Fundamental",
        "reason": "Earnings miss, analyst downgrades.",
        "evidenceCheckedCited": "Q3 earnings miss (-$0.12 vs $0.26 exp).",
        "metrics": {
            "SPI": 4, 
            "MPI": 5,
            "OHI": 4,
            "QSI": 4, # Just poor performance, not fraud
            "SFRI": 8 
        },
        "confidence": "High",
        "uncertainty": "Low"
    }
]

with open(os.path.join(OUTPUT_DIR, "downside_assessments.json"), "w") as f:
    json.dump(downside_data, f, indent=2)

print("Assessments generated.")
