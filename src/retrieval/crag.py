"""Corrective RAG (CRAG) Self-Reflection & Query Reformulation Engine (L3)."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from src.config import settings


class CRAGGrader:
    """Evaluates candidate retrieval relevance, flags marginal confidence,

    and executes 1-shot query reformulation with strict loop bounds (loop_count <= 1).
    """

    HIGH_THRESHOLD = 0.15  # Score >= 0.15: High confidence -> Proceed directly to synthesis
    LOW_THRESHOLD = 0.05  # Score < 0.05: Low confidence -> Factual abstention
    # 0.05 <= Score < 0.15: Marginal confidence -> 1-shot Query Reformulation

    def __init__(self, generator: Any = None) -> None:
        self.generator = generator

    def grade(
        self,
        top_score: float | None,
        results: list[dict[str, Any]],
        loop_count: int = 0,
    ) -> dict[str, Any]:
        """Evaluate retrieval candidates and return grading decision with top-margin gap.

        Args:
            top_score: Highest Cross-Encoder rerank score from retrieval.
            results: List of candidate chunk hits.
            loop_count: Number of previous reformulation attempts (max 1).

        Returns:
            Dictionary containing grade, action, confidence, margin_gap, and loop guard.
        """
        if not results or top_score is None:
            return {
                "grade": "low",
                "action": "abstain",
                "reason": "EMPTY_RETRIEVAL",
                "confidence": 0.0,
                "margin_gap": 0.0,
                "loop_count": loop_count,
            }

        score = float(top_score)
        second_score = float(results[1].get("score", 0.0)) if len(results) > 1 else 0.0
        margin_gap = max(0.0, score - second_score)

        if score >= self.HIGH_THRESHOLD:
            reason = "HIGH_CONFIDENCE_DECISIVE" if margin_gap >= 0.10 else "HIGH_CONFIDENCE_SYNTHESIS"
            return {
                "grade": "high",
                "action": "proceed",
                "reason": reason,
                "confidence": score,
                "margin_gap": round(margin_gap, 4),
                "loop_count": loop_count,
            }

        if score < self.LOW_THRESHOLD:
            return {
                "grade": "low",
                "action": "abstain",
                "reason": "LOW_SCORE",
                "confidence": score,
                "margin_gap": round(margin_gap, 4),
                "loop_count": loop_count,
            }

        # Marginal confidence: 0.05 <= score < 0.15
        if loop_count == 0:
            return {
                "grade": "marginal",
                "action": "reformulate",
                "reason": "MARGINAL_CONFIDENCE_RETRY",
                "confidence": score,
                "margin_gap": round(margin_gap, 4),
                "loop_count": loop_count,
            }

        # Loop count already >= 1: no further retries permitted (hard guard)
        if score >= settings.min_rerank_score:
            action = "proceed"
            reason = "POST_RETRY_PASS"
        else:
            action = "abstain"
            reason = "LOW_SCORE"

        return {
            "grade": "marginal",
            "action": action,
            "reason": reason,
            "confidence": score,
            "margin_gap": round(margin_gap, 4),
            "loop_count": loop_count,
        }

    def reformulate_query(
        self,
        query: str,
        results: list[dict[str, Any]] | None = None,
    ) -> str:
        """Reformulate query by extracting core technical entities and domain nouns.

        Attempts fast LLM reformulation; falls back cleanly to rule-based entity extraction.
        """
        try:
            generator = self.generator
            if generator is None:
                from src.generation.generator import get_generator

                generator = get_generator()

            prompt = (
                "You are an expert query reformulation assistant. "
                "The following search query yielded ambiguous or marginal document matches. "
                "Rewrite the query into a more specific, high-precision search query focusing on "
                "key technical nouns, standard acronyms, and core entities. "
                "Do NOT include conversational filler, quotes, or preambles.\n\n"
                f"Original Query: {query}\n\n"
                "Reformulated Query:"
            )
            reformulated = generator.generate(
                query="Reformulate query for precision retrieval", context=prompt
            )
            cleaned = reformulated.strip().strip('"').strip("'")
            if cleaned and len(cleaned) >= 5 and not cleaned.lower().startswith("error"):
                logger.info("CRAGGrader LLM reformulated query: '{}' -> '{}'", query, cleaned)
                return cleaned
        except Exception as exc:
            logger.debug(
                "CRAG LLM reformulation unavailable ({}); using extractive entity filter.", exc
            )

        return self._heuristic_reformulate(query, results)

    def _heuristic_reformulate(
        self,
        query: str,
        results: list[dict[str, Any]] | None = None,
    ) -> str:
        """Deterministic entity and noun phrase extraction for offline query reformulation."""
        filler_patterns = [
            r"\b(can you please|could you please|please|tell me about|what is|what are|how does|how do)\b",
            r"\b(explain|describe|give me|find me|show me)\b",
            r"\b(in the document|based on the context|in detail)\b",
        ]
        cleaned = query
        for pat in filler_patterns:
            cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "these",
            "those",
            "have",
            "has",
            "been",
            "will",
            "would",
            "should",
            "could",
            "about",
            "tell",
            "me",
            "you",
            "can",
            "please",
        }
        tokens = [
            t
            for t in re.findall(r"\b[A-Za-z0-9]+(?:[-._][A-Za-z0-9]+)*\b", cleaned)
            if t.lower() not in stopwords
        ]

        chunk_hints: list[str] = []
        if results:
            for hit in results[:2]:
                text = hit.get("text", "")
                caps = re.findall(r"\b[A-Z][A-Z0-9_.-]{2,}\b", text)
                for c in caps[:2]:
                    if c not in chunk_hints and c.lower() not in stopwords:
                        chunk_hints.append(c)

        combined_terms = tokens + chunk_hints
        if combined_terms:
            reformulated = " ".join(combined_terms)
            logger.info("CRAGGrader heuristic reformulated query: '{}' -> '{}'", query, reformulated)
            return reformulated
        return query
