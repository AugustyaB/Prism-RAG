"""Qdrant integration for persistent vector storage."""

import uuid
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import settings
from src.embeddings.embedder import LocalEmbedder
from src.ingestion.pipeline import IngestionResult


class VectorStore:
    """Manages persistent vector storage using Qdrant.
    
    Provides capabilities to ingest chunks along with their
    embeddings and metadata into a local persistent database.
    """

    def __init__(self, embedder: LocalEmbedder):
        """Initialize the Qdrant persistent client and collection.
        
        Args:
            embedder: Initialized LocalEmbedder instance for generating vectors.
        """
        self.embedder = embedder
        
        # Initialize persistent client at vector_db_dir
        logger.info(f"Connecting to Qdrant at {settings.vector_db_dir}")
        self.client = QdrantClient(path=str(settings.vector_db_dir))
        
        # Test an embedding to get exact vector dimension dynamically
        # or we could hardcode 384 for bge-small. Dynamic is safer.
        test_emb = self.embedder.embed_text("test")
        dim = len(test_emb)
        self.vector_size = dim
        
        col_name = settings.qdrant_collection_name
        
        if not self.client.collection_exists(col_name):
            logger.info(f"Creating collection '{col_name}' with size {dim}")
            self.client.create_collection(
                collection_name=col_name,
                vectors_config=models.VectorParams(
                    size=dim, 
                    distance=models.Distance.COSINE if settings.embedding_normalize else models.Distance.EUCLID
                ),
            )
        else:
            count = self.client.count(collection_name=col_name).count
            logger.info(f"Connected to collection '{col_name}' (chunks: {count})")

    def add_document(self, ingestion_result: IngestionResult) -> None:
        """Add an ingested document's chunks to the vector database.
        
        Args:
            ingestion_result: Result from IngestionPipeline.process_pdf.
        """
        if not ingestion_result.chunks:
            logger.warning(
                f"No chunks to add for {ingestion_result.source_pdf}."
            )
            return

        texts = []
        metadatas = []
        points = []

        # Generate embeddings
        logger.info(
            f"Embedding {len(ingestion_result.chunks)} chunks for {ingestion_result.source_pdf}..."
        )
        for chunk in ingestion_result.chunks:
            texts.append(chunk.text)
            
        embeddings = self.embedder.embed_batch(texts)

        for chunk, embedding in zip(ingestion_result.chunks, embeddings):
            chunk_idx = chunk.metadata.get("chunk_index", 0)
            # Create a UUID based on the file_hash and chunk_idx to enforce uniqueness
            # Qdrant accepts UUID string or int
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{ingestion_result.file_hash}_{chunk_idx}"))
            
            # Pack text into the payload
            payload = chunk.metadata.copy()
            payload["text"] = chunk.text
            payload["file_hash"] = ingestion_result.file_hash
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points
        )
        logger.info(
            f"Successfully stored {len(points)} chunks in Vector DB."
        )

    def delete_document(self, file_hash: str) -> None:
        """Delete all chunks belonging to a specific document by hash.
        
        Args:
            file_hash: The SHA-256 hash of the parent PDF.
        """
        col_name = settings.qdrant_collection_name
        self.client.delete(
            collection_name=col_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_hash",
                            match=models.MatchValue(value=file_hash)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted chunks for document {file_hash}.")

    def search(self, query: str, initial_k: int = 20, final_k: int = 5) -> list[dict]:
        """Perform a two-stage retrieval (Dense vector search + CrossEncoder rerank).
        
        Args:
            query: The text to search for.
            initial_k: Number of dense vectors to pull from Qdrant.
            final_k: Number of highest-relevance contexts to retain after reranking.
            
        Returns:
            A sorted list of dictionaries containing 'text', 'score', and metadata.
        """
        if not query.strip():
            return []
            
        col_name = settings.qdrant_collection_name
        
        # Stage 1: Dense Retrieval
        try:
            query_emb = self.embedder.embed_text(query)
            response = self.client.query_points(
                collection_name=col_name,
                query=query_emb,
                limit=initial_k,
                with_payload=True
            )
            qdrant_results = response.points
        except Exception as e:
            logger.error(f"Search failed during dense retrieval: {e}")
            return []
            
        if not qdrant_results:
            return []
            
        # Stage 2: Cross-Encoder Reranking
        texts = [hit.payload.get("text", "") for hit in qdrant_results]
        try:
            rerank_scores = self.embedder.rerank(query, texts)
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to default dense sorting.")
            rerank_scores = [hit.score for hit in qdrant_results]
            
        # Combine and sort by reranking score descending
        enriched_results = []
        for hit, text, r_score in zip(qdrant_results, texts, rerank_scores):
            meta = hit.payload.copy()
            meta.pop("text", None)
            enriched_results.append({
                "text": text,
                "score": float(r_score),
                "metadata": meta,
                "vector_score": float(hit.score)
            })
            
        enriched_results.sort(key=lambda x: x["score"], reverse=True)
        return enriched_results[:final_k]
