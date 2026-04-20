"""Centralized configuration for the RAG pipeline."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # --- Paths ---
    pdf_upload_dir: Path = Field(
        default=Path("./data/uploads"),
        description="Directory for uploaded PDF files",
    )
    extracted_dir: Path = Field(
        default=Path("./data/extracted"),
        description="Directory for extracted text output",
    )
    vector_db_dir: Path = Field(
        default=Path("./data/vectordb"),
        description="Directory for vector database storage",
    )

    # --- Page Classification ---
    text_threshold: int = Field(
        default=50,
        description="Minimum character count to classify a page as text-rich",
    )

    # --- Chunking ---
    max_chunk_size: int = Field(
        default=1000,
        description="Maximum chunk size in characters",
    )
    min_chunk_size: int = Field(
        default=100,
        description="Minimum chunk size in characters (smaller chunks are merged)",
    )
    overlap_size: int = Field(
        default=100,
        description="Character overlap between consecutive chunks",
    )

    # --- OCR ---
    ocr_dpi: int = Field(
        default=200,
        description="DPI for PDF page rendering before OCR",
    )
    ocr_languages: list[str] = Field(
        default=["en"],
        description="Languages for OCR recognition",
    )
    ocr_gpu: bool = Field(
        default=True,
        description="Use GPU for OCR (requires CUDA)",
    )
    ocr_timeout_seconds: int = Field(
        default=30,
        description="Maximum seconds per page for OCR before skipping",
    )
    ocr_max_pixmap_mb: int = Field(
        default=50,
        description="Maximum pixmap size in MB before reducing DPI",
    )

    # --- Embeddings ---
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence transformers embedding model",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for re-ranking search responses",
    )
    embedding_device: str = Field(
        default="cuda",
        description="Device to run embeddings on (cuda/cpu). Will fallback if cuda unavailable.",
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Number of chunks to embed simultaneously",
    )
    embedding_normalize: bool = Field(
        default=True,
        description="L2 normalize embeddings (required for cosine similarity)",
    )

    # --- Vector Database ---
    qdrant_collection_name: str = Field(
        default="rag_documents",
        description="Name of the Qdrant collection",
    )

    # --- LLMs / Generation ---
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama' or 'gemini'",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama local service",
    )
    ollama_model: str = Field(
        default="llama3",
        description="Local Ollama model name",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Gemini API key for cloud generation",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model formulation",
    )

    # --- Memory ---
    memory_warn_percent: float = Field(
        default=80.0,
        description="Log warning when process memory exceeds this % of total RAM",
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    model_config = {
        "env_prefix": "RAG_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.pdf_upload_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
