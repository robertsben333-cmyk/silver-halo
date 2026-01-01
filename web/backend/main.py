import os
import glob
import json
import subprocess
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Silver Halo Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIR = os.path.join(BASE_DIR, "web", "frontend")

# Create frontend dir if not exists (for safety)
# os.makedirs(FRONTEND_DIR, exist_ok=True) 

# Mount Static Files - This serves the frontend
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


class WorkflowResponse(BaseModel):
    status: str
    message: str
    output_dir: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Silver Halo Agent API is running"}

@app.post("/api/analyze", response_model=WorkflowResponse)
async def run_analysis():
    """
    Triggers the U.S. Big Losers Analysis Workflow.
    """
    try:
        # We'll use the PowerShell command from the workflow, but executed via subprocess
        # Ideally, we should import the python modules directly, but since the workflow uses CLI args, 
        # running them as subprocesses mimics the exact workflow interaction.
        
        # 1. Create timestamped folder
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUTPUTS_DIR, ts)
        os.makedirs(out_dir, exist_ok=True)
        
        # 2. Run stock_losers.py
        cmd_losers = [
            "python", os.path.join(BASE_DIR, "apps", "stock_losers.py"),
            "--output", os.path.join(out_dir, "stock_losers_clean.json"),
            "--limit", "10"
        ]
        subprocess.run(cmd_losers, check=True, cwd=BASE_DIR)

        # Note: The full workflow is quite long and involves manual steps or long-running processes.
        # For this initial button, we might just trigger the first step or a simplified version.
        # If the user wants the FULL workflow (including 'Parallel Research' which is manual/agentic),
        # we can't fully automate it in one blocking call without timeouts.
        # For now, we will run the automated data gathering part.
        
        return {
            "status": "success", 
            "message": "Analysis started (Data Gathering Phase). Check terminal for full progress.",
            "output_dir": out_dir
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script execution failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/api/runs")
async def get_runs():
    """
    Returns a list of available runs (timestamped folders in outputs/).
    """
    if not os.path.exists(OUTPUTS_DIR):
        return []
        
    runs = []
    for dirname in os.listdir(OUTPUTS_DIR):
        path = os.path.join(OUTPUTS_DIR, dirname)
        if os.path.isdir(path):
            runs.append(dirname)
    
    # Sort by timestamp descending (assuming folder name format YYYYMMDD_HHMMSS)
    runs.sort(reverse=True)
    return runs

@app.get("/api/results/latest")
async def get_latest_results():
    """
    Get results from the most recent run that has analysis data.
    """
    runs = await get_runs()
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found")
    
    # Iterate through runs to find the first one with meaningful data
    for run_id in runs:
        run_dir = os.path.join(OUTPUTS_DIR, run_id)
        # Check for key report files
        if os.path.exists(os.path.join(run_dir, "stock_analysis_report.json")) or \
           os.path.exists(os.path.join(run_dir, "downside_report.json")):
            return await get_results(run_id)
            
    # Fallback: if no full analysis found, just return the absolute latest (e.g. just losers)
    return await get_results(runs[0])

@app.get("/api/results/{run_id}")
async def get_results(run_id: str):
    """
    Aggregates data from a specific run folder.
    """
    run_dir = os.path.join(OUTPUTS_DIR, run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")
        
    data = {
        "run_id": run_id,
        "stock_losers": [],
        "analysis_report": [],
        "downside_report": [],
        "timing_output": [],
        "downside_timing_output": []
    }
    
    # Load standardized files if they exist
    files_map = {
        "stock_losers": "stock_losers_clean.json",
        "analysis_report": "stock_analysis_report.json",   # Longs/Rebounds
        "downside_report": "downside_report.json",         # Shorts
        "timing_output": "timing_output.json",             # Long Timing
        "downside_timing_output": "downside_timing_output.json" # Short Timing
    }
    
    for key, filename in files_map.items():
        filepath = os.path.join(run_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    content = json.load(f)
                    data[key] = content
            except Exception:
                data[key] = {"error": "Failed to parse JSON"}
                
    return data

@app.get("/api/stock/{symbol}/history")
async def get_stock_history(symbol: str):
    """
    Fetches historical price data for a graph.
    Uses yfinance.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        # Get 1 month of history for context
        hist = ticker.history(period="1mo")
        
        history_data = []
        for date, row in hist.iterrows():
            history_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": row["Close"],
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "volume": row["Volume"]
            })
            
        return history_data
    except ImportError:
         raise HTTPException(status_code=500, detail="yfinance not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
