import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    import google
    print(f"google path: {google.__path__}")
except ImportError as e:
    print(f"Error importing google: {e}")

try:
    from google import genai
    print("Successfully imported google.genai")
except ImportError as e:
    print(f"Error importing google.genai: {e}")
