"""Qdrant Hybrid Vector Store — Named Dense + Sparse BM25 IDF Vectors, RRF (k=60) & Parent Deduplication (L1)."""

from __future__ import annotations

from typing import Any
import uuid

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import settings
from src.embeddings.embedder import LocalEmbedder
from src.embeddings.sparse_bm25 import SparseBM25Encoder
from src.ingestion.pipeline import IngestionResult

VALID_RETRIEVAL_MODES = frozenset({"vector_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"})
RRF_K = 60.0


class VectorStore:
    """Manages persistent hybrid vector storage using Qdrant (`rag_documents_v2`).

    Stores both Dense (`BAAI/bge-small-en-v1.5`, 384-d) and Sparse (`BM25` with Qdrant `Modifier.IDF`)
    named vectors, supporting 4 retrieval modes (`vector_only`, `bm25_only`, `hybrid_rrf`, `hybrid_rerank`),
    Reciprocal Rank Fusion (`k=60`), and Parent Block Deduplication.
    """

    def __init__(self, embedder: LocalEmbedder, collection_name: str | None = None):
        """Initialize the Qdrant persistent client and hybrid `rag_documents_v2` collection."""
        self.embedder = embedder
        self.sparse_encoder = SparseBM25Encoder()
        self.collection_name = collection_name or settings.qdrant_collection_v2

        logger.info("Connecting to Qdrant at {}", settings.vector_db_dir)
        self.client = QdrantClient(path=str(settings.vector_db_dir))

        test_emb = self.embedder.embed_text("test")
        dim = len(test_emb) if test_emb else 384
        self.vector_size = dim

        self._ensure_hybrid_collection(self.collection_name, dim)

    def _ensure_hybrid_collection(self, col_name: str, dim: int) -> None:
        """Create or verify the named-vector (`dense` + `sparse` IDF) Qdrant collection."""
        if self.client.collection_exists(col_name):
            info = self.client.get_collection(col_name)
            vectors_cfg = info.config.params.vectors
            if isinstance(vectors_cfg, dict) and "dense" in vectors_cfg:
                count = self.client.count(collection_name=col_name).count
                logger.info("Connected to hybrid collection '{}' (chunks: {})", col_name, count)
                return
            logger.warning(
                "Collection '{}' lacks named 'dense'/'sparse' schema; recreating as v2 hybrid collection",
                col_name,
            )
            self.client.delete_collection(col_name)

        logger.info("Creating hybrid collection '{}' (dense dim={}, sparse=IDF)", col_name, dim)
        self.client.create_collection(
            collection_name=col_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dim,
                    distance=(
                        models.Distance.COSINE
                        if settings.embedding_normalize
                        else models.Distance.EUCLID
                    ),
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def add_document(
        self,
        ingestion_result: IngestionResult,
        org_id: str = "default_org",
        visibility: str = "ORG",
        allowed_roles: list[str] | None = None,
    ) -> None:
        """Index an ingested document's child chunks with Dense + Sparse vectors and parent/RBAC metadata."""
        if not ingestion_result.chunks:
            logger.warning("No chunks to add for {}.", ingestion_result.source_pdf)
            return

        doc_org_id = getattr(ingestion_result, "org_id", None) or org_id or "default_org"
        doc_visibility = getattr(ingestion_result, "visibility", None) or visibility or "ORG"
        doc_roles = (
            getattr(ingestion_result, "allowed_roles", None)
            or allowed_roles
            or ["ADMIN", "MANAGER", "ANALYST", "VIEWER"]
        )

        texts = [chunk.text for chunk in ingestion_result.chunks]

        logger.info(
            "Embedding {} child chunks (Dense + Sparse BM25) for {}...",
            len(texts),
            ingestion_result.source_pdf,
        )
        dense_embeddings = self.embedder.embed_batch(texts)
        sparse_embeddings = self.sparse_encoder.encode_batch(texts)

        points: list[models.PointStruct] = []
        for idx, (chunk, dense_vec, (sp_indices, sp_values)) in enumerate(
            zip(ingestion_result.chunks, dense_embeddings, sparse_embeddings)
        ):
            chunk_idx = chunk.metadata.get("chunk_index", idx)
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_OID,
                    f"{doc_org_id}_{ingestion_result.file_hash}_{chunk_idx}",
                )
            )

            payload = chunk.metadata.copy()
            payload["text"] = chunk.text
            payload["parent_id"] = payload.get(
                "parent_id", f"{ingestion_result.source_pdf}::parent_{chunk_idx}"
            )
            payload["parent_text"] = payload.get("parent_text", chunk.text)
            payload["page_number"] = payload.get("page_number", 1)
            payload["page_end"] = payload.get("page_end", payload["page_number"])
            payload["source_pdf"] = payload.get("source_pdf", ingestion_result.source_pdf)
            payload["file_hash"] = ingestion_result.file_hash
            payload["org_id"] = payload.get("org_id", doc_org_id)
            payload["visibility"] = payload.get("visibility", doc_visibility)
            payload["allowed_roles"] = payload.get("allowed_roles", doc_roles)

            vector_payload: dict[str, Any] = {"dense": dense_vec}
            if sp_indices:
                vector_payload["sparse"] = models.SparseVector(
                    indices=sp_indices,
                    values=sp_values,
                )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector_payload,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info("Successfully stored {} hybrid chunks in '{}'.", len(points), self.collection_name)

    def delete_document(self, file_hash: str) -> None:
        """Delete all chunks belonging to a specific document by SHA-256 hash."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_hash",
                            match=models.MatchValue(value=file_hash),
                        )
                    ]
                )
            ),
        )
        logger.info("Deleted chunks for document {} from '{}'.", file_hash, self.collection_name)

    def search(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 4,
        mode: str = "hybrid_rerank",
        query_filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Execute multi-mode retrieval (`vector_only`, `bm25_only`, `hybrid_rrf`, `hybrid_rerank`)
        with Reciprocal Rank Fusion (`k=60`), Parent Block Deduplication, and Cross-Encoder Reranking.
        """
        if not query or not query.strip():
            return []

        if mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval mode '{mode}'. Choose from {sorted(VALID_RETRIEVAL_MODES)}")

        dense_hits: list[ Any ] = []
        sparse_hits: list[ Any ] = []

        # 1. Stage-1 Dense Retrieval (uses BGE query instruction prefix via embed_query)
        if mode in {"vector_only", "hybrid_rrf", "hybrid_rerank"}:
            try:
                query_emb = (
                    self.embedder.embed_query(query)
                    if hasattr(self.embedder, "embed_query")
                    else self.embedder.embed_text(query)
                )
                if query_emb:
                    dense_resp = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_emb,
                        using="dense",
                        query_filter=query_filter,
                        limit=initial_k,
                        with_payload=True,
                    )
                    dense_hits = list(dense_resp.points)
            except Exception as exc:
                logger.error("Dense search leg failed: {}", exc)

        # 2. Stage-1 Sparse BM25 IDF Retrieval
        if mode in {"bm25_only", "hybrid_rrf", "hybrid_rerank"}:
            try:
                sp_idx, sp_val = self.sparse_encoder.encode_query(query)
                if sp_idx:
                    sparse_resp = self.client.query_points(
                        collection_name=self.collection_name,
                        query=models.SparseVector(indices=sp_idx, values=sp_val),
                        using="sparse",
                        query_filter=query_filter,
                        limit=initial_k,
                        with_payload=True,
                    )
                    sparse_hits = list(sparse_resp.points)
            except Exception as exc:
                logger.error("Sparse BM25 search leg failed: {}", exc)

        # 3. Combine / Score Stage-1 Candidates
        candidates: list[dict[str, Any]] = []
        if mode == "vector_only":
            for hit in dense_hits:
                candidates.append({"hit": hit, "stage1_score": float(hit.score)})
        elif mode == "bm25_only":
            for hit in sparse_hits:
                candidates.append({"hit": hit, "stage1_score": float(hit.score)})
        else:
            # Reciprocal Rank Fusion (k=60) over 1-indexed Dense and Sparse rankings
            hit_map: dict[str, Any] = {}
            rrf_scores: dict[str, float] = {}
            raw_tiebreak: dict[str, float] = {}

            for rank_d, hit in enumerate(dense_hits, start=1):
                pid = str(hit.id)
                hit_map[pid] = hit
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (RRF_K + rank_d))
                score_d = float(hit.score)
                raw_tiebreak[pid] = max(raw_tiebreak[pid], score_d) if pid in raw_tiebreak else score_d

            for rank_s, hit in enumerate(sparse_hits, start=1):
                pid = str(hit.id)
                hit_map[pid] = hit
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (RRF_K + rank_s))
                score_s = float(hit.score)
                raw_tiebreak[pid] = max(raw_tiebreak[pid], score_s) if pid in raw_tiebreak else score_s

            sorted_pids = sorted(
                rrf_scores.keys(),
                key=lambda pid: (rrf_scores[pid], raw_tiebreak.get(pid, 0.0)),
                reverse=True,
            )
            for pid in sorted_pids:
                candidates.append(
                    {
                        "hit": hit_map[pid],
                        "stage1_score": float(rrf_scores[pid]),
                    }
                )

        if not candidates:
            return []

        # 4. Parent Block Deduplication (keep highest Stage-1 scoring child per parent_id)
        deduped_candidates: list[dict[str, Any]] = []
        seen_parents: set[tuple[str, str]] = set()

        for item in candidates:
            hit = item["hit"]
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")
            file_hash = payload.get("file_hash", "")
            dedup_key = (file_hash, str(parent_id)) if parent_id else ("__point__", str(hit.id))

            if dedup_key in seen_parents:
                continue
            seen_parents.add(dedup_key)
            deduped_candidates.append(item)
            if len(deduped_candidates) >= initial_k:
                break

        # 5. Stage-2 Cross-Encoder Reranking (only for mode == "hybrid_rerank")
        if mode == "hybrid_rerank":
            rerank_texts = [
                (c["hit"].payload or {}).get("parent_text")
                or (c["hit"].payload or {}).get("text", "")
                for c in deduped_candidates
            ]
            try:
                rerank_scores = list(self.embedder.rerank(query, rerank_texts))
                if len(rerank_scores) != len(deduped_candidates):
                    raise RuntimeError(
                        f"Reranker returned {len(rerank_scores)} scores for {len(deduped_candidates)} candidates"
                    )
            except Exception as exc:
                logger.error("Reranking failed: {}. Falling back to Stage-1 RRF scores.", exc)
                rerank_scores = [c["stage1_score"] for c in deduped_candidates]
        else:
            rerank_scores = [c["stage1_score"] for c in deduped_candidates]

        enriched_results: list[dict[str, Any]] = []
        for item, final_score in zip(deduped_candidates, rerank_scores):
            hit = item["hit"]
            payload = (hit.payload or {}).copy()
            child_text = payload.pop("text", "")
            parent_text = payload.get("parent_text") or child_text
            parent_id = payload.get("parent_id")

            enriched_results.append(
                {
                    "text": child_text,
                    "parent_text": parent_text,
                    "parent_id": parent_id,
                    "score": float(final_score),
                    "metadata": payload,
                    "vector_score": float(item["stage1_score"]),
                }
            )

        enriched_results.sort(key=lambda x: x["score"], reverse=True)
        return enriched_results[:final_k]
