"""Retrieves, filters, and formats Parent-Child query contexts with a Two-Tier Abstention Gate (L3)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from loguru import logger

from src.auth.dependencies import build_qdrant_rbac_filter
from src.config import settings
from src.database.vector_store import VectorStore, combine_filters


class Retriever:
    """Manages multi-mode context retrieval and enforces the Pre-LLM Abstention Gate."""

    def __init__(self, vector_store: VectorStore):
        """Initialize Retriever with a VectorStore instance."""
        self.vector_store = vector_store
        self.last_retrieval_meta: dict[str, Any] = {}

    def retrieve(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 4,
        mode: str = "hybrid_rerank",
        query_filter: Any = None,
        user: Any = None,
    ) -> dict[str, Any]:
        """Execute hybrid retrieval, enforce Tier-1 Pre-LLM Abstention, and format Parent context blocks."""
        logger.info("Retrieving context for query: '{}' (mode={})", query, mode)

        active_filter = query_filter
        if user is not None:
            rbac_filter = build_qdrant_rbac_filter(user)
            active_filter = combine_filters(query_filter, rbac_filter)

        results = self.vector_store.search(
            query,
            initial_k=initial_k,
            final_k=final_k,
            mode=mode,
            query_filter=active_filter,
        )

        # Tier 1a: Empty Retrieval Abstention
        if not results:
            reason = (
                "RBAC_EMPTY_RETRIEVAL"
                if (settings.auth_enabled and user is not None)
                else "EMPTY_RETRIEVAL"
            )
            logger.warning("Tier-1 Abstention ({}) for query: '{}'", reason, query)
            outcome = {
                "context": "",
                "results": [],
                "abstained": True,
                "abstention_reason": reason,
                "top_score": None,
                "mode": mode,
            }
            self.last_retrieval_meta = outcome
            return outcome

        top_score = float(results[0].get("score", 0.0))

        # Tier 1b: Pre-LLM Cross-Encoder Rerank Score Threshold Gate
        if mode == "hybrid_rerank" and top_score < settings.min_rerank_score:
            logger.warning(
                "Tier-1 Abstention (LOW_SCORE): top_score={:.3f} < min_rerank_score={:.3f}",
                top_score,
                settings.min_rerank_score,
            )
            outcome = {
                "context": "",
                "results": [],
                "abstained": True,
                "abstention_reason": "LOW_SCORE",
                "top_score": top_score,
                "mode": mode,
            }
            self.last_retrieval_meta = outcome
            return outcome

        context_blocks: list[str] = []
        for hit in results:
            meta = hit.get("metadata", {})
            source = meta.get("source_pdf", "Unknown Source")
            page_start = meta.get("page_number", "?")
            page_end = meta.get("page_end", page_start)
            page_label = (
                f"{page_start}-{page_end}"
                if page_end is not None and page_end != page_start
                else f"{page_start}"
            )
            block_text = (
                hit.get("parent_text")
                or meta.get("parent_text")
                or hit.get("text", "")
            ).strip()

            block = f"--- Document: {source} (Page: {page_label}) ---\n{block_text}"
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        logger.info("Assembled {} parent context blocks (top_score={:.3f}).", len(context_blocks), top_score)

        outcome = {
            "context": formatted_context,
            "results": results,
            "abstained": False,
            "abstention_reason": None,
            "top_score": top_score,
            "mode": mode,
        }
        self.last_retrieval_meta = outcome
        return outcome

    def retrieve_context(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 4,
        mode: str = "hybrid_rerank",
        query_filter: Any = None,
        user: Any = None,
    ) -> Tuple[str, List[Dict]]:
        """Backward-compatible wrapper returning `(formatted_context, results)`."""
        outcome = self.retrieve(
            query=query,
            initial_k=initial_k,
            final_k=final_k,
            mode=mode,
            query_filter=query_filter,
            user=user,
        )
        return outcome["context"], outcome["results"]
