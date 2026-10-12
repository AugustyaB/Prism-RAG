"""Local GPU-accelerated embeddings using sentence-transformers."""

from typing import List

from loguru import logger

from src.config import settings


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    """Generates dense vector embeddings using a local model.
    
    Lazily loads the model into VRAM/RAM on first use to save resources
    during pipeline initialization. Falls back to CPU if CUDA is unavailable.
    """

    def __init__(self):
        """Initialize the embedder configuration but defer model loading."""
        self._model = None
        self._reranker = None
        self.model_name = settings.embedding_model
        self.reranker_name = settings.reranker_model
        
        # Check if CUDA is requested and available
        self.device = settings.embedding_device.lower()
        try:
            import torch
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested for embeddings but Torch is not compiled "
                    "with CUDA. Falling back to CPU."
                )
                self.device = "cpu"
        except (ImportError, OSError) as e:
            logger.warning(
                "Could not import Torch ({}); deferring device configuration to CPU.",
                e,
            )
            self.device = "cpu"

    @property
    def reranker(self):
        """Lazily load the CrossEncoder model."""
        if self._reranker is None:
            logger.info("Loading reranker model '{}' on device '{}'", self.reranker_name, self.device)
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.reranker_name, device=self.device)
                logger.info("Reranker model loaded successfully")
            except Exception as e:
                logger.error("Failed to load reranker model: {}", e)
                raise RuntimeError(f"Reranker model initialization failed: {e}")
        return self._reranker

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """Score texts against a query using a Cross-Encoder.
        
        Args:
            query: The user query.
            texts: A list of string contexts to evaluate.
            
        Returns:
            A list of float scores representing relevance.
        """
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)
            
        pairs = [[query, text] for text in texts]
        scores = self.reranker.predict(pairs)
        return scores.tolist()

    @property
    def model(self):
        """Lazily load the SentenceTransformer model."""
        if self._model is None:
            logger.info(
                "Loading embedding model '{}' on device '{}'", 
                self.model_name, 
                self.device
            )
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name, 
                    device=self.device
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error("Failed to load embedding model: {}", e)
                raise RuntimeError(f"Embedding model initialization failed: {e}")
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string.
        
        Args:
            text: Raw string to embed.
            
        Returns:
            A dense vector as a list of python floats.
        """
        # Ensure we don't pass empty strings
        if not text or not text.strip():
            logger.warning("Attempted to embed empty string; returning empty list.")
            return []
            
        vector = self.model.encode(
            text,
            batch_size=1,
            normalize_embeddings=settings.embedding_normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return vector.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query, prepending the asymmetric BGE v1.5 retrieval instruction prefix."""
        if not query or not query.strip():
            return []

        cleaned = query.strip()
        if "bge" in self.model_name.lower() and not cleaned.startswith(BGE_QUERY_PREFIX):
            cleaned = f"{BGE_QUERY_PREFIX}{cleaned}"

        return self.embed_text(cleaned)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings concurrently.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of dense vectors (each is a list of floats).
        """
        if not texts:
            return []
            
        # Filter purely empty strings out to avoid model warnings,
        # but keep track so we return parallel arrays if needed. 
        # For our architecture, chunks are already noise-filtered.
        
        try:
            vectors = self.model.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=settings.embedding_normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return vectors.tolist()
        except Exception as e:
            logger.error("Failed to process embedding batch of size {}: {}", len(texts), e)
            raise
