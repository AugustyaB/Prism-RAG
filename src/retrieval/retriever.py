"""Retrieves, filters, and formats Parent-Child query contexts with a Two-Tier Abstention Gate (L3)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from loguru import logger

from src.auth.dependencies import build_qdrant_rbac_filter
from src.config import settings
from src.database.vector_store import VectorStore, combine_filters
from src.retrieval.crag import CRAGGrader
from src.retrieval.multi_query import MultiQueryDecomposer


def _clean_text_artifacts(text: str) -> str:
    """Sanitize Unicode replacement characters and extraction artifacts."""
    if not text:
        return ""
    text = text.replace("\ufffd", " ")
    text = re.sub(r"(?<=\n)\?\s+(?=[A-Z0-9])", "• ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def compute_information_novelty(
    candidate_tokens: set[str],
    seen_token_sets: list[set[str]],
) -> float:
    """Compute Information Novelty Index (INI) in [0.0, 1.0] using Jaccard set distance.

    INI(d_k, S) = 1.0 - max_{S_i in S} (|T(d_k) cap S_i| / |T(d_k) cup S_i|)
    Returns 1.0 if seen_token_sets is empty or candidate_tokens has zero overlap with seen sets.
    """
    if not candidate_tokens:
        return 0.0
    if not seen_token_sets:
        return 1.0
    max_sim = 0.0
    for seen in seen_token_sets:
        if not seen:
            continue
        intersection_len = len(candidate_tokens & seen)
        union_len = len(candidate_tokens | seen)
        if union_len > 0:
            sim = intersection_len / union_len
            if sim > max_sim:
                max_sim = sim
    return round(max(0.0, 1.0 - max_sim), 4)


class Retriever:
    """Manages multi-mode context retrieval, multi-query RRF decomposition, and CRAG self-reflection."""

    def __init__(
        self,
        vector_store: VectorStore,
        decomposer: MultiQueryDecomposer | None = None,
        crag_grader: CRAGGrader | None = None,
    ) -> None:
        """Initialize Retriever with a VectorStore instance and optional decomposer/grader."""
        self.vector_store = vector_store
        self.decomposer = decomposer or MultiQueryDecomposer(vector_store)
        self.crag_grader = crag_grader or CRAGGrader()
        self.last_retrieval_meta: dict[str, Any] = {}

    @property
    def embedder(self):
        """Convenience property exposing the vector store's dense embedder."""
        return self.vector_store.embedder

    def retrieve(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 4,
        mode: str = "hybrid_rerank",
        query_filter: Any = None,
        user: Any = None,
        collector: Any = None,
        enable_multi_query: bool = False,
        enable_crag: bool = True,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid retrieval, multi-query RRF expansion, and CRAG self-reflection."""
        logger.info("Retrieving context for query: '{}' (mode={})", query, mode)

        active_filter = query_filter
        if user is not None:
            rbac_filter = build_qdrant_rbac_filter(user)
            active_filter = combine_filters(query_filter, rbac_filter)

        multi_query_meta: dict[str, Any] = {"decomposed": False}

        # 1. Multi-Query Expansion & Parallel RRF Retrieval (if requested)
        if (enable_multi_query or mode == "multi_query") and self.decomposer.should_decompose(query):
            results, multi_query_meta = self.decomposer.search_and_merge(
                query=query,
                initial_k=initial_k,
                final_k=final_k,
                mode="hybrid_rerank" if mode == "multi_query" else mode,
                query_filter=active_filter,
                collector=collector,
            )
        else:
            effective_mode = "hybrid_rerank" if mode == "multi_query" else mode
            results = self.vector_store.search(
                query,
                initial_k=initial_k,
                final_k=final_k,
                mode=effective_mode,
                query_filter=active_filter,
                collector=collector,
                alpha=alpha,
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
                "multi_query": multi_query_meta,
                "crag": {"grade": "low", "action": "abstain", "reason": reason},
            }
            self.last_retrieval_meta = outcome
            return outcome

        top_score = float(results[0].get("score", 0.0))

        # Tier 1b: CRAG Self-Reflection & 1-shot Query Reformulation
        crag_decision: dict[str, Any] = {
            "grade": "high",
            "action": "proceed",
            "reason": "HIGH_CONFIDENCE",
            "confidence": top_score,
            "loop_count": 0,
        }
        if enable_crag and mode in {"hybrid_rerank", "multi_query"}:
            crag_decision = self.crag_grader.grade(top_score, results, loop_count=0)
            if crag_decision["action"] == "reformulate":
                logger.info(
                    "CRAG triggered 1-shot query reformulation (top_score={:.3f})", top_score
                )
                reformulated_q = self.crag_grader.reformulate_query(query, results)
                crag_decision["reformulated_query"] = reformulated_q

                # Retry retrieval once with reformulated query
                retry_results = self.vector_store.search(
                    reformulated_q,
                    initial_k=initial_k,
                    final_k=final_k,
                    mode="hybrid_rerank",
                    query_filter=active_filter,
                    collector=collector,
                )
                if retry_results:
                    retry_top_score = float(retry_results[0].get("score", 0.0))
                    retry_decision = self.crag_grader.grade(
                        retry_top_score, retry_results, loop_count=1
                    )
                    retry_decision["reformulated_query"] = reformulated_q
                    crag_decision = retry_decision
                    if retry_top_score > top_score:
                        results = retry_results
                        top_score = retry_top_score

        # Pre-LLM Cross-Encoder Rerank Score Threshold Gate
        if mode in {"hybrid_rerank", "multi_query"} and (
            top_score < settings.min_rerank_score or crag_decision.get("action") == "abstain"
        ):
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
                "multi_query": multi_query_meta,
                "crag": crag_decision,
            }
            self.last_retrieval_meta = outcome
            return outcome

        # In hybrid_rerank mode, retain top-ranked positive relevance chunks (score >= 0.0)
        # while pruning completely irrelevant negative-logit noise
        if mode in {"hybrid_rerank", "multi_query"} and top_score >= settings.min_rerank_score:
            filtered_results = [
                hit for idx, hit in enumerate(results) if idx < 2 or float(hit.get("score", 0.0)) >= 0.0
            ]
            if len(filtered_results) < len(results):
                logger.info(
                    "Filtered {} negative noise chunks (< 0.0); retained {} high-confidence chunks.",
                    len(results) - len(filtered_results),
                    len(filtered_results),
                )
                results = filtered_results

        # Dynamic Context Pruning via Information Novelty Index (INI)
        # Yield(d_k) = s_rerank * INI >= tau_yield (0.05).
        # Chunk 1 (top hit) is unconditionally preserved. Subsequent chunks with Yield < tau_yield are pruned.
        tau_yield = 0.05
        pruned_results: list[dict[str, Any]] = []
        seen_token_sets: list[set[str]] = []

        for idx, hit in enumerate(results):
            raw_block = (
                hit.get("parent_text")
                or (hit.get("metadata") or {}).get("parent_text")
                or hit.get("text", "")
            )
            cand_tokens = set(re.findall(r"\b\w+\b", raw_block.lower()))
            score = float(hit.get("score", 0.0))

            if idx == 0:
                ini = 1.0
                yield_val = score
                keep = True
            else:
                ini = compute_information_novelty(cand_tokens, seen_token_sets)
                yield_val = score * ini
                keep = yield_val >= tau_yield

            hit["ini"] = round(ini, 4)
            hit["yield"] = round(yield_val, 4)

            if keep:
                pruned_results.append(hit)
                if cand_tokens:
                    seen_token_sets.append(cand_tokens)
            else:
                logger.info(
                    "Context Pruned chunk idx={} (score={:.3f}, ini={:.3f}, yield={:.3f} < tau_yield={:.2f})",
                    idx,
                    score,
                    ini,
                    yield_val,
                    tau_yield,
                )

        if pruned_results:
            results = pruned_results

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
            raw_block = (
                hit.get("parent_text") or meta.get("parent_text") or hit.get("text", "")
            )
            block_text = _clean_text_artifacts(raw_block)
            if "text" in hit and isinstance(hit["text"], str):
                hit["text"] = _clean_text_artifacts(hit["text"])
            if "parent_text" in hit and isinstance(hit["parent_text"], str):
                hit["parent_text"] = _clean_text_artifacts(hit["parent_text"])

            block = f"--- Document: {source} (Page: {page_label}) ---\n{block_text}"
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        logger.info(
            "Assembled {} parent context blocks (top_score={:.3f}).", len(context_blocks), top_score
        )

        outcome = {
            "context": formatted_context,
            "results": results,
            "abstained": False,
            "abstention_reason": None,
            "top_score": top_score,
            "mode": mode,
            "multi_query": multi_query_meta,
            "crag": crag_decision,
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
