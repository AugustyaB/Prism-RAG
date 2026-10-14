"""Quick test for LLM backends (LM Studio + Gemini)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from src.config import settings

print("=== Testing LM Studio ===")
try:
    url = settings.lmstudio_base_url.rstrip("/")
    r = requests.get(f"{url}/models", timeout=5)
    if r.status_code == 200:
        models = r.json()
        model_ids = [m["id"] for m in models.get("data", [])]
        print(f"Connected! Available models: {model_ids}")
        
        # Test generation
        from src.generation.generator import get_generator
        gen = get_generator("lmstudio")
        answer = gen.generate("Say hello in one sentence.", "No context needed.")
        print(f"Generation OK: {answer[:200]}")
    else:
        print(f"HTTP {r.status_code}")
except requests.exceptions.ConnectionError:
    print(f"Cannot connect to {settings.lmstudio_base_url}")
    print("Make sure LM Studio is running with the local server enabled.")
except Exception as e:
    print(f"Error: {e}")

print()
print("=== Testing Gemini ===")
try:
    if not settings.gemini_api_key:
        print("No API key set (RAG_GEMINI_API_KEY is empty)")
    else:
        from src.generation.generator import get_generator
        gen = get_generator("gemini")
        answer = gen.generate("Say hello in one sentence.", "No context needed.")
        print(f"Generation OK: {answer[:200]}")
except Exception as e:
    print(f"Error: {e}")
