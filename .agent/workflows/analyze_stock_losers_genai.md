---
description: Run the fully automated GenAI U.S. Big Losers Analysis Workflow using Gemini 3.0 API for research and timing.
---

# U.S. Big Losers GenAI Analysis Workflow

This workflow is the **API-driven** version of the stock analysis, using **Gemini 3.0** to perform the research, assessment generation, and timing logic.

## 1. Setup & Data Service
Start the `stock_losers.py` service.
Create a timestamped output folder for this run.

**Command:**
```powershell
// turbo-all
# Create Timestamped Folder
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$baseDir = "c:/Users/XavierFriesen/.gemini/antigravity/playground/silver-halo"
$outDir = "$baseDir/outputs/${ts}_genai"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
echo "OUTPUT_DIR=$outDir"

# Fetch Data
py "$baseDir/apps/stock_losers.py" --output "$outDir/stock_losers_clean.json" --limit 10
```

## 2. GenAI Research (Gemini 3.0 Flash/Pro)

Run the `analysis_agent.py` to perform dual-track research and generate assessments automatically using Gemini.

**Command:**
```powershell
// turbo-all
# Run Analysis Agent (Branch A & B)
py "$baseDir/apps/analysis_agent.py" --input "$outDir/stock_losers_clean.json" --output_dir "$outDir"
```

## 3. Quantitative Scoring

Execute the deterministic scoring scripts for both models.

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
```

## 4. GenAI Timing & Reporting

Use the **Gemini-powered** timing agents for enhanced reasoning.

**Command:**
```powershell
// turbo-all
# 1. Timing Analysis (Branch A: Contrarian - GenAI)
py "$baseDir/apps/timing_agent_genai.py" --input "$outDir/stock_analysis_report.json" --output "$outDir/timing_output.json"

# 2. Timing Analysis (Branch B: Downside - GenAI)
py "$baseDir/apps/downside_timing_genai.py" --input "$outDir/downside_report.json" --output "$outDir/downside_timing_output.json"

# 3. Email Reporting (Dual Model)
py "$baseDir/apps/reporting_agent.py" --summary_md "$outDir/summary_table.md" --timing_json "$outDir/timing_output.json" --downside_report "$outDir/downside_report.json" --downside_timing "$outDir/downside_timing_output.json"
```
