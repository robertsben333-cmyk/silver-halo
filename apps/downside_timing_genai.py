import argparse
import json
import os
import sys
from pydantic import BaseModel, Field
from typing import List, Optional
from llm_utils import GenAIClient
from google.genai import types

# --- Pydantic Models ---

class DownsideTimingResult(BaseModel):
    ticker: str
    action: str = Field(..., description="e.g. 'Short immediately', 'Short after bounce', 'Avoid'")
    urgency_score: int = Field(..., description="Urgency 0-10")
    squeeze_score: float = Field(..., description="Risk of squeeze (Score)")
    min_short_price: float = Field(..., description="Specific price target to initiate short")
    exit_target: float = Field(..., description="Specific price target to cover")
    driver_category: str
    reasoning: str

class DownsideTimingOutput(BaseModel):
    results: List[DownsideTimingResult]

# --- Agent ---

def load_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""

def main():
    parser = argparse.ArgumentParser(description="Downside Timing Agent (GenAI)")
    parser.add_argument("--input", required=True, help="Path to downside_report.json")
    parser.add_argument("--output", required=True, help="Path to downside_timing_output.json")
    args = parser.parse_args()

    # Load Inputs
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input: {e}")
        sys.exit(1)

    if not data:
        print("No downside data found.")
        sys.exit(0)

    # Initialize Gemini
    try:
        genai_client = GenAIClient()
    except Exception as e:
        print(f"Failed to initialize Gemini Client: {e}")
        sys.exit(1)

    # Load Knowledge (Instructions contain the framework for timing/scoring)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instr_path = os.path.join(base_dir, "inputs", "instructions", "downside_timing_instructions.md")
    downside_instructions = load_file(instr_path)

    print(f"--- Running Downside Timing Analysis for {len(data)} items (Gemini 3.0 Pro) ---")

    prompt = f"""
    You are an expert short-selling timing agent.
    
    Your goal is to determine the optimal Short Timing (Entry/Exit) for the following stocks.
    
    Context / Framework:
    {downside_instructions}
    
    Downside Reports:
    {json.dumps(data, indent=2)}
    
    Instructions:
    Strictly follow the "Logic & Heuristics" and "Scoring Logic" defined in the Context above to generate the response.
    Output a JSON object with a list of results adhering to the "Output Format" in the Context.
    """

    try:
        # User requested lower reasoning for timing
        response = genai_client.generate_content(
            prompt=prompt,
            model="gemini-3-flash-preview",
            thinking_level=types.ThinkingLevel.MINIMAL,
            response_schema=DownsideTimingOutput
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
            print(f"Saved Downside Timing Analysis to {args.output}")
        else:
            print("Failed to parse Downside Timing Analysis response.")

    except Exception as e:
        print(f"Error in Downside Timing Analysis: {e}")
        # Fallback
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([], f)

if __name__ == "__main__":
    main()
