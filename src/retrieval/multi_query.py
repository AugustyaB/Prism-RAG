"""Multi-Query RRF Decomposition Engine (L3)."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from src.database.vector_store import RRF_K, VectorStore

MULTI_PART_INDICATORS = (
    "and",
    "compare",
    "summary",
    "versus",
    "vs",
    "difference",
    "between",
    "contrast",
    "overview",
    "both",
)


class MultiQueryDecomposer:
    """Decomposes complex user queries into multi-perspective sub-queries

    and executes parallel retrieval merged via native Reciprocal Rank Fusion (k=60).
    """

    def __init__(self, vector_store: VectorStore, generator: Any = None) -> None:
        self.vector_store = vector_store
        self.generator = generator

    def should_decompose(self, query: str) -> bool:
        """Determine if a query qualifies for multi-query decomposition."""
        if not query or not query.strip():
            return False
        clean_q = query.strip()
        if len(clean_q) > 25:
            return True
        words = set(re.findall(r"\b[a-zA-Z]+\b", clean_q.lower()))
        return any(ind in words for ind in MULTI_PART_INDICATORS)

    def decompose(self, query: str, timeout: float = 1.2) -> list[str]:
        """Generate 3 perspective sub-queries with a 1.2s timeout and deterministic fallback."""
        if not self.should_decompose(query):
            return [query]

        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._generate_perspectives, query)
                sub_queries = future.result(timeout=timeout)
                if sub_queries and len(sub_queries) == 3:
                    logger.info(
                        "MultiQueryDecomposer generated 3 sub-queries in {:.1f}ms: {}",
                        (time.perf_counter() - t0) * 1000.0,
                        sub_queries,
                    )
                    return sub_queries
        except Exception as exc:
            logger.debug(
                "Fast LLM decomposition timed out or failed ({}); using deterministic fallback.", exc
            )

        return self._heuristic_perspectives(query)

    def _generate_perspectives(self, query: str) -> list[str]:
        """Prompt generator for 3 perspective queries."""
        generator = self.generator
        if generator is None:
            from src.generation.generator import get_generator

            generator = get_generator()

        prompt = (
            "You are an expert search query decomposer. "
            "Given the user query, generate exactly 3 focused search queries from different perspectives:\n"
            "1. High-level problem statement & overview perspective\n"
            "2. Technical methodology & implementation perspective\n"
            "3. Experimental findings & quantitative results perspective\n\n"
            f"User Query: {query}\n\n"
            "Respond with ONLY the 3 queries, one per line. Do NOT include numbers, bullets, or commentary."
        )
        raw_output = generator.generate(
            query="Decompose query into 3 search perspectives", context=prompt
        )
        cleaned_lines = [
            re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            for line in raw_output.strip().splitlines()
            if line.strip()
            and not line.strip().startswith(("-", "*", "Note", "Here", "I cannot", "Error"))
        ]
        valid_queries = [
            line for line in cleaned_lines if len(line) >= 5 and "error" not in line.lower()[:25]
        ]
        if len(valid_queries) >= 3:
            return valid_queries[:3]
        raise ValueError(f"Generator produced insufficient perspectives ({len(valid_queries)})")

    def _heuristic_perspectives(self, query: str) -> list[str]:
        """Deterministic 3-perspective decomposition when LLM is offline or timed out."""
        clean = re.sub(r"[?!.,]", "", query).strip()
        return [
            f"{clean} problem statement overview architecture",
            f"{clean} technical methodology implementation",
            f"{clean} experimental findings results benchmarks",
        ]

    def search_and_merge(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 4,
        mode: str = "hybrid_rrf",
        query_filter: Any = None,
        collector: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Decompose query, execute parallel retrieval for sub-queries, and merge via RRF (k=60)."""
        sub_queries = self.decompose(query)
        if len(sub_queries) <= 1:
            search_mode = "hybrid_rerank" if mode == "multi_query" else mode
            results = self.vector_store.search(
                query=query,
                initial_k=initial_k,
                final_k=final_k,
                mode=search_mode,
                query_filter=query_filter,
                collector=collector,
            )
            return results, {"decomposed": False, "sub_queries": [query], "count": len(results)}

        all_sub_results: list[list[dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
            futures = [
                pool.submit(
                    self.vector_store.search,
                    q,
                    initial_k=initial_k,
                    final_k=final_k * 2,
                    mode="hybrid_rrf",
                    query_filter=query_filter,
                    collector=collector,
                )
                for q in sub_queries
            ]
            for f in futures:
                try:
                    all_sub_results.append(f.result())
                except Exception as exc:
                    logger.warning("Sub-query retrieval failed: {}", exc)
                    all_sub_results.append([])

        rrf_scores: dict[str, float] = {}
        candidate_map: dict[str, dict[str, Any]] = {}

        for hits in all_sub_results:
            for rank, hit in enumerate(hits):
                pid = str(hit.get("id"))
                if not pid:
                    continue
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))
                if pid not in candidate_map:
                    candidate_map[pid] = hit

        sorted_pids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused_candidates: list[dict[str, Any]] = []
        for pid in sorted_pids:
            cand = dict(candidate_map[pid])
            cand["fused_rrf_score"] = rrf_scores[pid]
            cand["score"] = rrf_scores[pid]
            fused_candidates.append(cand)

        deduped: list[dict[str, Any]] = []
        seen_parents: set[tuple[str, str]] = set()
        for hit in fused_candidates:
            meta = hit.get("metadata", {})
            parent_id = hit.get("parent_id") or meta.get("parent_id")
            file_hash = hit.get("file_hash") or meta.get("file_hash", "")
            key = (str(file_hash), str(parent_id)) if parent_id else ("__point__", str(hit.get("id")))
            if key in seen_parents:
                continue
            seen_parents.add(key)
            deduped.append(hit)
            if len(deduped) >= initial_k:
                break

        # Re-score candidates against the original query using Cross-Encoder if in rerank mode
        target_mode = "hybrid_rerank" if mode == "multi_query" else mode
        if target_mode == "hybrid_rerank" and deduped:
            try:
                rerank_texts = [d.get("parent_text") or d.get("text", "") for d in deduped]
                if hasattr(self.vector_store, "embedder") and hasattr(
                    self.vector_store.embedder, "rerank"
                ):
                    scores = list(self.vector_store.embedder.rerank(query, rerank_texts))
                    for item, s in zip(deduped, scores):
                        item["score"] = float(s)
                    deduped.sort(key=lambda x: x["score"], reverse=True)
            except Exception as exc:
                logger.error(
                    "Cross-encoder reranking of fused candidates failed: {}. Preserving RRF scores.",
                    exc,
                )

        final_results = deduped[:final_k]
        meta_info = {
            "decomposed": True,
            "sub_queries": sub_queries,
            "total_candidates": len(candidate_map),
            "final_count": len(final_results),
        }
        return final_results, meta_info
