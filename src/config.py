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
    eval_benchmark_dir: Path = Field(
        default=Path("./data/benchmark_pdfs"),
        description="Directory for synthetic benchmark PDFs",
    )
    eval_dataset_path: Path = Field(
        default=Path("./data/eval/benchmark_dataset.json"),
        description="Path to 100-QA ground-truth dataset JSON",
    )
    eval_results_dir: Path = Field(
        default=Path("./data/eval/results"),
        description="Directory for evaluation run outputs",
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
    parent_chunk_size: int = Field(
        default=1200,
        description="Parent context block size in characters",
    )
    parent_overlap_size: int = Field(
        default=150,
        description="Overlap between consecutive parent blocks",
    )
    child_chunk_size: int = Field(
        default=300,
        description="Child chunk size in characters for dense/BM25 indexing",
    )
    child_overlap_size: int = Field(
        default=50,
        description="Overlap between consecutive child chunks",
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

    # --- Vector & Relational Databases ---
    qdrant_collection_name: str = Field(
        default="rag_documents_v2",
        description="Name of the active Qdrant collection",
    )
    qdrant_collection_v2: str = Field(
        default="rag_documents_v2",
        description="Name of the v2 hybrid Qdrant collection",
    )
    database_url: str = Field(
        default="sqlite:///./data/metadata.db",
        description="SQLAlchemy connection URL for relational metadata database",
    )

    # --- Async Ingestion Worker & Retrieval Safety ---
    max_ingestion_workers: int = Field(
        default=2,
        description="Bounded thread pool size for background PDF ingestion jobs",
    )
    max_job_retries: int = Field(
        default=2,
        description="Maximum automatic retry attempts on transient ingestion failure",
    )
    min_rerank_score: float = Field(
        default=0.15,
        description="Minimum Cross-Encoder reranker score before pre-LLM abstention",
    )

    # --- Authentication & Multi-Tenant RBAC Security ---
    jwt_secret_key: str = Field(
        default="enterprise-rag-super-secret-jwt-key-2026-change-in-production",
        description="Cryptographic secret key for signing JWT tokens",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="Cryptographic signing algorithm",
    )
    jwt_access_token_expire_minutes: int = Field(
        default=60,
        description="Lifetime of issued access tokens in minutes",
    )
    auth_enabled: bool = Field(
        default=True,
        description="Enforce JWT authentication and tenant filtering",
    )

    # --- LLMs / Generation ---
    llm_provider: str = Field(
        default="lmstudio",
        description="LLM provider: 'lmstudio', 'ollama', or 'gemini'",
    )
    # --- LM Studio (OpenAI-compatible local) ---
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL for LM Studio OpenAI-compatible API",
    )
    lmstudio_model: str = Field(
        default="gpt-oss-20b",
        description="Model loaded in LM Studio",
    )
    # --- Ollama ---
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama local service",
    )
    ollama_model: str = Field(
        default="llama3",
        description="Local Ollama model name",
    )
    # --- Gemini (Cloud) ---
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
        self.eval_benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        Path("./data/raw").mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
