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


# --- Pydantic Models for Downside Analysis ---

class DownsideMetrics(BaseModel):
    SPI: int = Field(..., description="Shock Persistence Index (0-10)")
    MPI: int = Field(..., description="Microstructure Pressure Index (0-10)")
    OHI: int = Field(..., description="Overhang Index (0-10)")
    QSI: int = Field(..., description="Quality & Sentiment Index (0-10)")
    SFRI: int = Field(..., description="Short Feasibility & Squeeze Risk (0-10)")

class DownsideAssessment(BaseModel):
    ticker: str
    company: str
    exchange: str
    sector: str
    lastUsd: float
    oneDayReturnPct: float
    driverCategory: str
    reason: str = Field(..., description="Reason for drop and downside continuation case")
    evidenceCheckedCited: str
    metrics: DownsideMetrics
    nonFundamental: str = Field(..., description="Yes or No")
    confidence: str = Field(..., description="Low, Medium, or High")
    uncertainty: str = Field(..., description="Low, Medium, or High")

class DownsideOutput(BaseModel):
    assessments: List[DownsideAssessment]


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
    downside_instr_path = os.path.join(base_dir, "inputs", "instructions", "downside_instructions.md")
    
    rebound_instructions = load_file(rebound_instr_path)
    downside_instructions = load_file(downside_instr_path)

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
    
    try:
        response_rebound = genai_client.generate_content(
            prompt=rebound_prompt,
            model="gemini-3-flash-preview",
            thinking_level=types.ThinkingLevel.MEDIUM,
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
        else:
            print("Failed to parse Rebound Analysis response.")

    except Exception as e:
        print(f"Error in Rebound Analysis: {e}")


    # --- Downside Analysis ---
    print("Running Downside Analysis (Gemini 3.0 Flash)...", flush=True)
    downside_prompt = f"""
    You are the Downside Analyzer.
    
    Here is the Cohort JSON:
    {json.dumps(losers_data, indent=2)}
    
    Perform the analysis as described in the System Instructions.
    Provide the output matching the schema for 'downside_assessments.json'.
    """

    try:
        response_downside = genai_client.generate_content(
            prompt=downside_prompt,
            model="gemini-3-flash-preview",
            thinking_level=types.ThinkingLevel.MEDIUM, # User requested highest thinking for instructions
            system_instruction=downside_instructions,
            response_schema=DownsideOutput
        )

        parsed_downside = genai_client.parse_json_response(response_downside)
        if parsed_downside:
             # Handle Pydantic object
            if hasattr(parsed_downside, 'model_dump'):
                downside_data = parsed_downside.model_dump()
            else:
                downside_data = parsed_downside

             # Unwrap if it's in the wrapper
            downside_list = downside_data.get('assessments', downside_data) if isinstance(downside_data, dict) else downside_data

            out_path = os.path.join(args.output_dir, "downside_assessments.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(downside_list, f, indent=2)
            print(f"Saved Downside Analysis to {out_path}")
        else:
             print("Failed to parse Downside Analysis response.")

    except Exception as e:
        print(f"Error in Downside Analysis: {e}")

if __name__ == "__main__":
    main()
