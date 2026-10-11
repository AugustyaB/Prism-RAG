"""LLM backend integration for generating contextual answers."""

import json
from abc import ABC, abstractmethod
from loguru import logger
import requests

from src.config import settings

class BaseGenerator(ABC):
    """Abstract base class for LLM generation strategies."""
    
    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate an answer using the provided context.
        
        Args:
            query: The user query string.
            context: Formatted context blocks from the vector DB.
            
        Returns:
            The generated answer text.
        """
        pass
        
    def _build_prompt(self, query: str, context: str) -> str:
        """Construct a unified prompt isolating context from instructions."""
        return (
            "You are a helpful and precise RAG assistant. "
            "Use ONLY the provided context to answer the question. "
            "If the context does not contain the answer, say 'I cannot answer this based on the provided documents.' "
            "Do not hallucinate.\n\n"
            f"--- Context ---\n{context}\n\n"
            f"--- Question ---\n{query}\n\n"
            "Answer:"
        )


class OllamaGenerator(BaseGenerator):
    """Local LLM interaction using Ollama HTTP endpoints."""
    
    def generate(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        
        endpoint = f"{settings.ollama_base_url}/api/generate"
        logger.info(f"Querying Ollama API at {endpoint} with model {settings.ollama_model}")
        
        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ReadTimeout:
            logger.error("Ollama query timed out after 120 seconds.")
            return "Error: Local LLM timed out."
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error connecting to local Ollama service: {e}"


class LMStudioGenerator(BaseGenerator):
    """Local LLM via LM Studio's OpenAI-compatible API."""
    
    def generate(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        payload = {
            "model": settings.lmstudio_model,
            "messages": [
                {"role": "system", "content": "You are a helpful and precise RAG assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "stream": False
        }
        # Normalize base URL: ensure /v1 is in the path
        base = settings.lmstudio_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        endpoint = f"{base}/chat/completions"
        logger.info(f"Querying LM Studio at {endpoint} with model {settings.lmstudio_model}")
        
        try:
            response = requests.post(endpoint, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.ReadTimeout:
            logger.error("LM Studio query timed out after 180 seconds.")
            return "Error: Local LLM timed out."
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to LM Studio at {settings.lmstudio_base_url}")
            return "Error: LM Studio is not running. Start it and load a model."
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected LM Studio response format: {e}")
            return "Error: Unexpected response from LM Studio."
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            return f"Error connecting to LM Studio: {e}"


class GeminiGenerator(BaseGenerator):
    """Cloud LLM interaction utilizing Google GenAI API."""
    
    def __init__(self):
        super().__init__()
        from google import genai
        
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing for Gemini Provider.")
            
        self.client = genai.Client(api_key=settings.gemini_api_key)
        
    def generate(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        logger.info(f"Querying Gemini API with model {settings.gemini_model}")
        
        try:
            response = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error querying Gemini API: {e}"


def get_generator(provider: str | None = None) -> BaseGenerator:
    """Factory method to instantiate the correct generator.
    
    Args:
        provider: 'lmstudio', 'ollama', or 'gemini'. Defaults to settings.llm_provider.
    """
    provider = provider or settings.llm_provider
    provider = provider.lower()
    
    if provider == "lmstudio":
        return LMStudioGenerator()
    elif provider == "gemini":
        return GeminiGenerator()
    elif provider == "ollama":
        return OllamaGenerator()
    else:
        logger.warning(f"Unknown provider '{provider}', falling back to LM Studio.")
        return LMStudioGenerator()
