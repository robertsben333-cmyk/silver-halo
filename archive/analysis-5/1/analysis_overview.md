# Analysis Overview

This directory contains analysis scripts and potential trade data derived from the "Stock Losers" workflow. The scripts have been moved here for better organization.

## Directory Structure

- `plots/`: Contains all generated plots, organized by topic.
  - `exit_analysis/`: Optimal exit time analysis and comparisons.
  - `heatmaps/`: Heatmaps for heuristic exit parameters.
  - `mfe_analysis/`: Maximum Favorable Excursion (potential profit) distribution.
  - `sensitivity/`: Stoploss sensitivity analysis.
  - `last_hour/`: Analysis of returns in the last hour vs last 30 mins.
  - `returns/`: Return comparisons (e.g., 2h vs EOD).
  - `prediction_analysis/`: Plots related to prediction error and subsequent day returns.
- `analysis_history.csv`: Main input file containing trade history and model scores.

## Analysis Scripts

All scripts are configured to run from this directory (`analysis-5/1`) and usually require a `.env` file with `POLYGON_API_KEY` in the project root (or set in environment).

### Exit Strategy Analysis

| Script | Description | Suggested Usage |
| :--- | :--- | :--- |
| `analyze_exit_times.py` | Analyzes limits of "Fixed Time" exits (e.g., 30m, 1h, ... 6.5h). Produces "Optimal Exit Time" curves. | `python analyze_exit_times.py` |
| `analyze_dynamic_exit.py` | Compares dynamic exit logic: Trailing Stops, Consecutive Losses, vs Fixed Time. | `python analyze_dynamic_exit.py` |
| `analyze_heuristic_exits.py` | Tests "Heuristic" exits (Exit after N green bars) with optional Profit Filters. Generates heatmaps. | `python analyze_heuristic_exits.py` |
| `analyze_hmm_exit.py` | uses Hidden Markov Models (HMM) to detect "Reversal" regimes for exit. | `python analyze_hmm_exit.py` |

### Risk & Returns

| Script | Description | Suggested Usage |
| :--- | :--- | :--- |
| `analyze_stoploss_risk.py` | Analyzes the impact of different stoploss levels (1% to 15%) on Win Rate and Avg Return. Calculates MFE. | `python analyze_stoploss_risk.py` |
| `analyze_last_hour.py` | Calculates returns for Short positions held during 15:00-16:00 ET. Helps decide if one should hold to close. | `python analyze_last_hour.py` |
| `analyze_last_hour_breakdown.py` | Break downs last hour performance by "Survivor" status (heuristic didn't trigger) vs "Triggered". | `python analyze_last_hour_breakdown.py` |

### Prediction & Selection

| Script | Description | Suggested Usage |
| :--- | :--- | :--- |
| `analyze_prediction_error_day2.py` | correlates the Model's Day 1 prediction error with Day 2 returns. Checks if "failed shorts" become "long winners". | `python analyze_prediction_error_day2.py` |
| `find_max_winner.py` | Scans `analysis_history.csv` to find the single best potential trade (theoretical max return) based on simulation. | `python find_max_winner.py` |

## Data Requirements
- Most scripts expect `analysis_history.csv` in the current directory.
- Model artifacts are expected at `../../apps/model_artifacts`.
