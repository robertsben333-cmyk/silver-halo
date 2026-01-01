import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def test_model(model_name, thinking_level=None):
    print(f"Testing {model_name} (Thinking: {thinking_level})...")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    config_args = {}
    if thinking_level:
        config_args["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say hello.",
            config=types.GenerateContentConfig(**config_args)
        )
        print(f"SUCCESS: {model_name}")
        return True
    except Exception as e:
        print(f"FAILED: {model_name} - {e}")
        return False

if __name__ == "__main__":
    # Test 3.0 Pro
    test_model("gemini-3-pro-preview", types.ThinkingLevel.LOW)
    
    # Test 3.0 Flash
    test_model("gemini-3-flash-preview", types.ThinkingLevel.MEDIUM)
    
    # Test 2.5 Flash
    test_model("gemini-2.5-flash-001") # or similar name
