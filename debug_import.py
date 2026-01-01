import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import google
    print(f"google path: {google.__path__}")
except ImportError as e:
    print(f"Error importing google: {e}")

try:
    from google import genai
    print("Successfully imported google.genai")
    print(genai.__file__)
except ImportError as e:
    print(f"Error importing google.genai: {e}")

try:
    import google_genai
    print("Imported google_genai (direct?)")
except ImportError:
    print("Could not import google_genai")
