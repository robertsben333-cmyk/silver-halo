import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GenAIClient:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=self.api_key)

    def generate_content(self, prompt, model="gemini-3-pro-preview", thinking_level=None, system_instruction=None, response_schema=None):
        """
        Generates content using the Gemini API.

        Args:
            prompt (str): The input prompt.
            model (str): The model to use. Defaults to "gemini-3-pro-preview".
            thinking_level (types.ThinkingLevel): The thinking level. 
                                                  Defaults to None (model default).
                                                  Use types.ThinkingLevel.HIGH, LOW, etc.
            system_instruction (str): Optional system instruction.
            response_schema (type): Optional Pydantic model or type for structured output.

        Returns:
            response: The API response object.
        """
        
        config_args = {}
        
        if thinking_level:
            config_args["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
            
        if system_instruction:
            config_args["system_instruction"] = system_instruction

        if response_schema:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_args)

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            return response
        except Exception as e:
            print(f"Error generating content: {e}")
            raise

    def parse_json_response(self, response):
        """
        Helper to safely extract JSON from response.
        If structured output was used, it might be in response.parsed.
        Otherwise, try to parse response.text.
        """
        try:
            if hasattr(response, 'parsed') and response.parsed is not None:
                return response.parsed
            
            # Fallback to text parsing if standard json is returned in text
            import json
            text = response.text
            # Basic cleanup if markdown validation fails
            if text.startswith("```json"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text.rsplit("\n", 1)[0]
            return json.loads(text)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw text: {response.text}")
            return None
