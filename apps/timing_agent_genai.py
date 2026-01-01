import argparse
import json
import os
import sys
from pydantic import BaseModel, Field
from typing import List, Optional
from llm_utils import GenAIClient
from google.genai import types

# --- Pydantic Models ---

class TimingResult(BaseModel):
    ticker: str
    timing_window: str = Field(..., description="e.g. 'Immediate (Market Open)', 'Wait 1-2 Days'")
    action: str = Field(..., description="e.g. 'Buy at Open', 'Buy after morning dip'")
    max_entry_price: float = Field(..., description="Specific price target for entry")
    exit_target: float = Field(..., description="Specific price target for exit")
    confidence: str = Field(..., description="Low, Medium, or High")
    reasoning: str
    dip_score: float = Field(..., description="Estimated Dip Risk Score (0-1)")

class TimingOutput(BaseModel):
    results: List[TimingResult]

# --- Agent ---

def load_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""

def main():
    parser = argparse.ArgumentParser(description="Timing Agent (GenAI)")
    parser.add_argument("--input", required=True, help="Path to stock_analysis_report.json")
    parser.add_argument("--output", required=True, help="Path to timing_output.json")
    args = parser.parse_args()

    # Load Inputs
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    except Exception as e:
        print(f"Error loading input: {e}")
        sys.exit(1)

    if not analysis_data:
        print("No analysis data found.")
        sys.exit(0)

    # Initialize Gemini
    try:
        genai_client = GenAIClient()
    except Exception as e:
        print(f"Failed to initialize Gemini Client: {e}")
        sys.exit(1)

    # Load Knowledge
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_path = os.path.join(base_dir, "inputs", "instructions", "timing_instructions.md")
    timing_knowledge = load_file(knowledge_path)

    print(f"--- Running Timing Analysis for {len(analysis_data)} items (Gemini 3.0 Pro) ---")

    prompt = f"""
    You are an expert market timing agent.
    
    Your goal is to determine the optimal Entry and Exit timing for the following stocks, which have been identified as Rebound Candidates.
    
    Context / Theory:
    {timing_knowledge}
    
    Stock Analysis Reports:
    {json.dumps(analysis_data, indent=2)}
    
    Instructions:
    Strictly follow the "Logic & Heuristics" in the Context above.
    Output a JSON object with a list of results adhering to the "Output Format" in the Context.
    """

    try:
        # User requested lower reasoning for timing
        response = genai_client.generate_content(
            prompt=prompt,
            model="gemini-3-flash-preview",
            thinking_level=types.ThinkingLevel.MINIMAL, 
            response_schema=TimingOutput
        )

        parsed_response = genai_client.parse_json_response(response)
        
        if parsed_response:
            if hasattr(parsed_response, 'model_dump'):
                data = parsed_response.model_dump()
            elif hasattr(parsed_response, 'dict'):
                 data = parsed_response.dict()
            else:
                data = parsed_response

             # Unwrap if it's in the wrapper
            results = data.get('results', data) if isinstance(data, dict) else data

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Saved Timing Analysis to {args.output}")
        else:
            print("Failed to parse Timing Analysis response.")

    except Exception as e:
        print(f"Error in Timing Analysis: {e}")
        # Failure Fallback: Empty list
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([], f)

if __name__ == "__main__":
    main()
