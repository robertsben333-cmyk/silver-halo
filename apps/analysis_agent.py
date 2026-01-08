import argparse
import json
import os
import sys
import time
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from llm_utils import GenAIClient
from google.genai import types

# --- Pydantic Models for Rebound Analysis ---

class ReboundMetrics(BaseModel):
    PCR: float = Field(..., description="Primary Confirmation Ratio (0-1)")
    EC: float = Field(..., description="Evidence Coverage (0-1)")
    SD: float = Field(..., description="Source Diversity (0-1)")
    NRI: float = Field(..., description="Novelty-Recency Index (0-1)")
    HDM: float = Field(..., description="Headline-Driver Match (0-1)")
    CONTR: float = Field(..., description="Contradiction Penalty (0-1)")
    FRESH_NEG: int = Field(..., description="1 if fresh negative shock, else 0")
    CP: float = Field(..., description="Consensus Polarity (0-1 mapped)")
    RD: float = Field(..., description="Relevant Density (0-1)")
    total_relevant: int = Field(..., description="Total relevant sources found")

class ReboundSubscores(BaseModel):
    RES: float = Field(..., description="Residual Drop score")
    LIQ: float = Field(..., description="Liquidity score")
    FUND: float = Field(..., description="Fundamental Anchor score")
    ATTN: float = Field(..., description="Attention score")
    CRWD: float = Field(..., description="Crowding score")
    CTX: float = Field(..., description="Context score")

class ReboundAssessment(BaseModel):
    ticker: str
    company: str
    exchange: str
    sector: str
    lastUsd: float
    oneDayReturnPct: float
    reason: str = Field(..., description="Explanation of drop and rebound case")
    evidenceCheckedCited: str
    metrics: ReboundMetrics
    subscores: ReboundSubscores
    sent_pro: float = Field(..., description="Professional Sentiment (0-1)")
    sent_com: float = Field(..., description="Community Sentiment (0-1)")
    confidence: str = Field(..., description="Low, Medium, or High")
    uncertainty: str = Field(..., description="Low, Medium, or High")

class ReboundOutput(BaseModel):
    assessments: List[ReboundAssessment]



# --- Agent ---

def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="AI Analysis Agent")
    parser.add_argument("--input", required=True, help="Path to stock_losers.json")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    args = parser.parse_args()

    # Load Inputs
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            losers_data = json.load(f)
    except Exception as e:
        print(f"Error loading stock_losers.json: {e}")
        sys.exit(1)

    if not losers_data:
        print("No losers data found.")
        sys.exit(0)

    # Initialize Gemini
    try:
        genai_client = GenAIClient()
    except Exception as e:
        print(f"Failed to initialize Gemini Client: {e}")
        sys.exit(1)

    # Load Instructions
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rebound_instr_path = os.path.join(base_dir, "inputs", "instructions", "instructions.md")
    
    rebound_instructions = load_file(rebound_instr_path)

    print(f"--- Starting Analysis for {len(losers_data)} tickers ---", flush=True)

    # --- Rebound Analysis ---
    print("Running Rebound Analysis (Gemini 3.0 Flash)...", flush=True)
    rebound_prompt = f"""
    You are the Rebound Analyzer.
    
    Here is the Cohort JSON:
    {json.dumps(losers_data, indent=2)}
    
    Perform the analysis as described in the System Instructions.
    Provide the output matching the schema for 'raw_assessments.json' where each item has metrics, subscores, etc.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response_rebound = genai_client.generate_content(
                prompt=rebound_prompt,
                model="gemini-2.0-flash-exp",
                # thinking_level=types.ThinkingLevel.MEDIUM, # Not supported on 2.0-flash-exp
                system_instruction=rebound_instructions,
                response_schema=ReboundOutput
            )
            
            parsed_rebound = genai_client.parse_json_response(response_rebound)
            if parsed_rebound:
                # Handle Pydantic object
                if hasattr(parsed_rebound, 'model_dump'):
                    rebound_data = parsed_rebound.model_dump()
                else:
                    rebound_data = parsed_rebound
    
                rebound_list = rebound_data.get('assessments', rebound_data) if isinstance(rebound_data, dict) else rebound_data
                
                # Save
                out_path = os.path.join(args.output_dir, "raw_assessments.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(rebound_list, f, indent=2)
                print(f"Saved Rebound Analysis to {out_path}")
                break # Success
            else:
                print("Failed to parse Rebound Analysis response.")
                break 

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"Quota exceeded (Attempt {attempt+1}/{max_retries}). Retrying in 60s...")
                time.sleep(60)
            else:
                print(f"Error in Rebound Analysis: {e}")
                break

if __name__ == "__main__":
    main()
