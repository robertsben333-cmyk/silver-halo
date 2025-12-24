---
description: Run the U.S. Big Losers Cohort OSINT Rebound Analyzer workflow with strict quantitative scoring and enhanced evidence gathering.
---

# U.S. Big Losers Analysis Workflow

This workflow orchestrates the retrieval of stock loser data, deep-dive OSINT analysis (60+ sources), and strict quantitative scoring.

## 1. Setup & Data Service
Start the `stock_losers.py` service from the new `apps` directory.
Create a timestamped output folder for this run.

**Command:**
```powershell
// turbo-all
# Create Timestamped Folder
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
# Note: Adjust base path if needed, defaulting to playground root
$baseDir = "c:/Users/XavierFriesen/.gemini/antigravity/playground/silver-halo"
$outDir = "$baseDir/outputs/$ts"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
echo "OUTPUT_DIR=$outDir"

# Fetch Data
py "$baseDir/apps/stock_losers.py" --output "$outDir/stock_losers_clean.json" --limit 10
```

## 3. Parallel Research (Manual/Agentic)

**CRITICAL:** These are two separate mindset workflows. Do not conflate them.

### Branch A: Contrarian Reversal Scan (The Buyer)
**Role:** Senior Equity Research Analyst (Long Bias).
**Goal:** Find reasons the drop is an *overreaction*.
**Search Focus:** "Oversold", "reaction", "support levels", "insider buying", "slight miss".
**Action:** Perform search and create `raw_assessments.json`.

### Branch B: Downside Continuation Scan (The Short Seller)
**Role:** Forensic Accountant / Short Seller (Bear Bias).
**Goal:** Find reasons the drop is *just the beginning*.
**Search Focus:** "Fraud", "Investigation", "Accounting regularities", "Auditor resignation", "Dilution spiral", "Bankruptcy", "Lawsuit".
**Action:** Perform search and create `downside_assessments.json`.

## 4. Quantitative Scoring & Persistence

Execute the scoring scripts for both models and save results.

**Command:**
```powershell
// turbo-all
# Branch A: Contrarian Scoring
py "$baseDir/apps/rebound_scoring.py" --input "$outDir/raw_assessments.json" --output "$outDir/stock_analysis_report.json"
py "$baseDir/apps/save_results.py" --assessments "$outDir/raw_assessments.json" --report "$outDir/stock_analysis_report.json" --csv "$baseDir/outputs/analysis_history.csv" --output_md "$outDir/summary_table.md"

# Branch B: Downside Scoring
py "$baseDir/apps/downside_scoring.py" --input "$outDir/downside_assessments.json" --output "$outDir/downside_report.json"
# Downside Persistence
py "$baseDir/apps/downside_persistence.py" --report "$outDir/downside_report.json" --csv "$baseDir/outputs/downside_history.csv"

# -------------------------------------------------------------------------------------
# PHASE 4: Timing & Reporting (Combined)
# -------------------------------------------------------------------------------------

# 1. Timing Analysis (Branch A: Contrarian)
py "$baseDir/apps/timing_agent.py" --input "$outDir/stock_analysis_report.json" --output "$outDir/timing_output.json"

# 2. Timing Analysis (Branch B: Downside)
py "$baseDir/apps/downside_timing.py" --input "$outDir/downside_report.json" --output "$outDir/downside_timing_output.json"

# 3. Email Reporting (Dual Model)
py "$baseDir/apps/reporting_agent.py" --summary_md "$outDir/summary_table.md" --timing_json "$outDir/timing_output.json" --downside_report "$outDir/downside_report.json" --downside_timing "$outDir/downside_timing_output.json"
```
