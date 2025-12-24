# Workspace Rules: Maximum Effort

This file defines the strict operating standards for this workspace. All agents must adhere to these rules.

## 1. Maximum Effort & Deep Research
*   **Exhaustive Search**: Never rely on a single search query or source. Use multiple angles (e.g., "Ticker fraud", "Ticker lawsuit", "Ticker employee reviews", "Ticker 10-K risk factors") to build a complete picture.
*   **Verification**: Ensure claims are supported by a credible source.
*   **Chain of Thought**: When analyzing complex data (like financial statements or legal risks), explicitly write out the reasoning steps. Do not jump to conclusions.

## 2. High-Quality Thinking
*   **Contrarian Check**: Consider disconfirming evidence. Avoid confirmation bias.
*   **Contextual Intelligence**: detailed explanations are required. Avoid "shallow" summaries. Explain *why* a data point matters.
*   **No Hallucinations**: If data is missing, state it clearly. Do not infer or guess financial metrics.

## 3. Persistent Quality
*   **Artifact Updates**: Continuously update `task.md` and `walkthrough.md` with granular progress.
*   **Self-Correction**: If an error is found (e.g., wrong metric), stop, correct it, and document the correction.

## 4. Auto-Run Permissions
- **Apps Directory**: The user has explicitly granted PERMANENT permission to auto-run all scripts located in `apps/` (e.g., `stock_losers.py`, `rebound_scoring.py`).
- **Action**: Always set `SafeToAutoRun=true` for these scripts. Do not ask for confirmation.
