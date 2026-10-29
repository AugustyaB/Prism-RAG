"""LLM backend integration for generating contextual answers."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Generator
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

    def stream_generate(
        self, query: str, context: str = ""
    ) -> Generator[str, None, None]:
        """Yield tokens or text chunks as they are generated.

        Default fallback implementation splits the generated response into tokens.
        """
        full_answer = self.generate(query, context)
        tokens = re.findall(r"\S+\s*|\s+", full_answer)
        for token in tokens:
            yield token

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

    def generate_with_telemetry(
        self, query: str, context: str
    ) -> tuple[str, int, int]:
        """Generate answer while measuring and counting prompt and completion tokens."""
        from src.utils.telemetry import count_tokens

        prompt = self._build_prompt(query, context)
        prompt_tokens = count_tokens(prompt)
        answer = self.generate(query, context)
        completion_tokens = count_tokens(answer)
        return answer, prompt_tokens, completion_tokens


class OllamaGenerator(BaseGenerator):
    """Local LLM interaction using Ollama HTTP endpoints."""

    def generate(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context) if context else query
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }

        endpoint = f"{settings.ollama_base_url}/api/generate"
        logger.info(
            "Querying Ollama API at {} with model {}",
            endpoint,
            settings.ollama_model,
        )

        try:
            response = requests.post(endpoint, json=payload, timeout=(0.12, 120))
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ReadTimeout:
            logger.error("Ollama query timed out after 120 seconds.")
            return "Error: Local LLM timed out."
        except Exception as e:
            logger.error("Ollama generation failed: {}", e)
            return f"Error connecting to local Ollama service: {e}"

    def stream_generate(
        self, query: str, context: str = ""
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(query, context) if context else query
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": True,
        }
        endpoint = f"{settings.ollama_base_url}/api/generate"
        logger.info(
            "Streaming query from Ollama API at {} with model {}",
            endpoint,
            settings.ollama_model,
        )
        try:
            with requests.post(
                endpoint, json=payload, stream=True, timeout=(0.12, 120)
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.error("Ollama streaming failed: {}", e)
            yield from stream_extractive_answer(query, context)


class LMStudioGenerator(BaseGenerator):
    """Local LLM via LM Studio's OpenAI-compatible API."""

    def generate(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context) if context else query
        payload = {
            "model": settings.lmstudio_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful and precise RAG assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        }
        base = settings.lmstudio_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        endpoint = f"{base}/chat/completions"
        logger.info(
            "Querying LM Studio at {} with model {}",
            endpoint,
            settings.lmstudio_model,
        )

        try:
            response = requests.post(endpoint, json=payload, timeout=(0.12, 180))
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.ReadTimeout:
            logger.error("LM Studio query timed out after 180 seconds.")
            return "Error: Local LLM timed out."
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to LM Studio at {}", settings.lmstudio_base_url)
            return "Error: LM Studio is not running. Start it and load a model."
        except (KeyError, IndexError) as e:
            logger.error("Unexpected LM Studio response format: {}", e)
            return "Error: Unexpected response from LM Studio."
        except Exception as e:
            logger.error("LM Studio generation failed: {}", e)
            return f"Error connecting to LM Studio: {e}"

    def stream_generate(
        self, query: str, context: str = ""
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(query, context) if context else query
        payload = {
            "model": settings.lmstudio_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful and precise RAG assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "stream": True,
        }
        base = settings.lmstudio_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        endpoint = f"{base}/chat/completions"
        logger.info(
            "Streaming query from LM Studio at {} with model {}",
            endpoint,
            settings.lmstudio_model,
        )
        try:
            with requests.post(
                endpoint, json=payload, stream=True, timeout=(0.12, 180)
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        chunk_str = line_str[6:].strip()
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(chunk_str)
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except Exception:
                            continue
        except Exception as e:
            logger.error("LM Studio streaming failed: {}", e)
            yield from stream_extractive_answer(query, context)


class GeminiGenerator(BaseGenerator):
    """Cloud LLM interaction utilizing Google GenAI API."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        from google import genai

        effective_key = api_key or settings.gemini_api_key
        if not effective_key:
            raise ValueError("GEMINI_API_KEY is missing for Gemini Provider.")

        self.api_key = effective_key
        self.client = genai.Client(api_key=effective_key)
        self.model_name = settings.gemini_model

    def generate(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context) if context else query
        logger.info("Querying Gemini API with model {}", settings.gemini_model)

        try:
            response = self.client.models.generate_content(
                model=settings.gemini_model, contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error("Gemini generation failed: {}", e)
            return f"Error querying Gemini API: {e}"

    def stream_generate(
        self, query: str, context: str = ""
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(query, context) if context else query
        logger.info("Streaming query from Gemini API with model {}", settings.gemini_model)
        try:
            response = self.client.models.generate_content_stream(
                model=self.model_name, contents=prompt
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("Gemini streaming failed: {}", e)
            yield from stream_extractive_answer(query, context)


def synthesize_extractive_answer(query: str, context: str) -> str:
    """Synthesize a grounded extractive answer directly from retrieved parent context when local LLM is offline."""
    if not context or not context.strip():
        return "I cannot answer this based on the provided documents. No matching context available."

    blocks = [b.strip() for b in context.split("--- Document: ") if b.strip()]
    if not blocks:
        return "I cannot answer this based on the provided documents."

    summaries = []
    for b in blocks[:3]:
        lines = b.splitlines()
        header = lines[0].rstrip("-").strip() if lines else "Document Excerpt"
        if not header:
            header = "Document Excerpt"
        content_lines = [
            l.strip()
            for l in lines[1:]
            if l.strip() and not l.startswith("---")
        ]
        if content_lines:
            snippet = " ".join(content_lines[:4])
            if len(snippet) > 360:
                snippet = snippet[:357] + "..."
            summaries.append(f"**From {header}:**\n> {snippet}")
        elif lines:
            snippet = lines[0].rstrip("-").strip()
            if len(snippet) > 360:
                snippet = snippet[:357] + "..."
            summaries.append(f"> {snippet}")

    if not summaries and context.strip():
        snippet = context.strip()[:360]
        summaries.append(f"> {snippet}")

    return "\n\n".join(summaries)


def stream_extractive_answer(
    query: str, context: str = ""
) -> Generator[str, None, None]:
    """Stream an extractive grounded answer token-by-token."""
    answer = synthesize_extractive_answer(query, context)
    tokens = re.findall(r"\S+\s*|\s+", answer)
    for token in tokens:
        yield token


def get_generator(
    provider: str | None = None, api_key: str | None = None
) -> BaseGenerator:
    """Factory method to instantiate the correct generator.

    Args:
        provider: 'lmstudio', 'ollama', or 'gemini'. Defaults to settings.llm_provider.
        api_key: Optional runtime API key for cloud providers (e.g., Gemini).
    """
    provider = provider or settings.llm_provider
    provider = provider.lower()

    if provider == "lmstudio":
        return LMStudioGenerator()
    elif provider == "gemini":
        return GeminiGenerator(api_key=api_key)
    elif provider == "ollama":
        return OllamaGenerator()
    else:
        logger.warning(
            "Unknown provider '{}', falling back to LM Studio.", provider
        )
        return LMStudioGenerator()
