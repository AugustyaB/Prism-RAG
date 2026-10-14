"""Unit tests for Phase 2.10 Wave 2: Multi-Query RRF Decomposition & CRAG Self-Reflection Engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.crag import CRAGGrader
from src.retrieval.multi_query import MultiQueryDecomposer
from src.retrieval.retriever import Retriever


# ── 1. MultiQueryDecomposer Tests ──────────────────────────────────────────


def test_should_decompose_heuristics():
    """Verify query qualification heuristics (length > 25 chars or multi-part indicators)."""
    decomposer = MultiQueryDecomposer(vector_store=MagicMock())

    # Short query without indicators -> False
    assert decomposer.should_decompose("What is RAG?") is False
    assert decomposer.should_decompose("") is False

    # Short query with indicator -> True
    assert decomposer.should_decompose("BERT and RoBERTa") is True
    assert decomposer.should_decompose("Compare Qdrant vs Faiss") is True

    # Long query > 25 chars -> True
    assert decomposer.should_decompose("Explain the mathematical foundations of transformer attention mechanisms") is True


def test_decompose_perspectives_fallback():
    """Verify 3-perspective decomposition generates problem, methodology, and results perspectives."""
    decomposer = MultiQueryDecomposer(vector_store=MagicMock())

    sub_queries = decomposer.decompose(
        "Compare vector search and BM25 lexical search efficiency"
    )
    assert len(sub_queries) == 3
    # Check that perspectives encompass overview, methodology, and experimental results
    assert any("overview" in q or "problem" in q for q in sub_queries)
    assert any("methodology" in q or "implementation" in q for q in sub_queries)
    assert any("findings" in q or "benchmarks" in q for q in sub_queries)


def test_search_and_merge_rrf_fusion():
    """Verify parallel retrieval across sub-queries and Reciprocal Rank Fusion (k=60) candidate merge."""
    mock_vs = MagicMock()
    # Simulate return from sub-queries
    mock_vs.search.side_effect = [
        [
            {"id": "doc1", "score": 0.85, "text": "Overview of attention", "parent_id": "p1"},
            {"id": "doc2", "score": 0.70, "text": "Basics of vectors", "parent_id": "p2"},
        ],
        [
            {"id": "doc1", "score": 0.90, "text": "Overview of attention", "parent_id": "p1"},
            {"id": "doc3", "score": 0.65, "text": "Methodology", "parent_id": "p3"},
        ],
        [
            {"id": "doc3", "score": 0.80, "text": "Methodology", "parent_id": "p3"},
            {"id": "doc4", "score": 0.50, "text": "Benchmarks", "parent_id": "p4"},
        ],
    ]

    decomposer = MultiQueryDecomposer(vector_store=mock_vs)
    results, meta = decomposer.search_and_merge(
        query="Compare attention mechanisms and evaluate computational complexity",
        initial_k=10,
        final_k=3,
        mode="hybrid_rrf",
    )

    assert meta["decomposed"] is True
    assert len(meta["sub_queries"]) == 3
    assert len(results) <= 3
    # doc1 appeared in rank 0 in query 1 and query 2, so it must have the highest fused RRF score
    assert results[0]["id"] == "doc1"
    assert "fused_rrf_score" in results[0]


# ── 2. CRAGGrader Tests ───────────────────────────────────────────────────


def test_crag_grading_tiers():
    """Verify CRAG 3-tier grading: High (>=0.15), Marginal (0.05-0.15), and Low (<0.05)."""
    grader = CRAGGrader()

    # 1. High confidence -> Proceed
    high_res = grader.grade(top_score=0.25, results=[{"id": "1", "score": 0.25}], loop_count=0)
    assert high_res["grade"] == "high"
    assert high_res["action"] == "proceed"
    assert high_res["confidence"] == 0.25

    # 2. Marginal confidence -> 1-shot Reformulation (loop_count == 0)
    marg_res = grader.grade(top_score=0.10, results=[{"id": "1", "score": 0.10}], loop_count=0)
    assert marg_res["grade"] == "marginal"
    assert marg_res["action"] == "reformulate"

    # 3. Low confidence -> Clean Abstention
    low_res = grader.grade(top_score=-0.12, results=[{"id": "1", "score": -0.12}], loop_count=0)
    assert low_res["grade"] == "low"
    assert low_res["action"] == "abstain"
    assert low_res["reason"] == "LOW_SCORE"

    # 4. Empty retrieval -> Abstain
    empty_res = grader.grade(top_score=None, results=[], loop_count=0)
    assert empty_res["grade"] == "low"
    assert empty_res["action"] == "abstain"
    assert empty_res["reason"] == "EMPTY_RETRIEVAL"


def test_crag_loop_guard():
    """Verify hard guard prevents infinite reformulation loops (loop_count <= 1)."""
    grader = CRAGGrader()

    # If score is marginal (0.10) but loop_count is already 1:
    # Must NOT reformulate again!
    retry_res = grader.grade(top_score=0.10, results=[{"id": "1", "score": 0.10}], loop_count=1)
    assert retry_res["action"] != "reformulate"
    assert retry_res["loop_count"] == 1


def test_crag_heuristic_reformulation():
    """Verify heuristic query reformulation strips conversational filler and isolates technical entities."""
    grader = CRAGGrader()

    query = "Can you please tell me about POL-SEC-104.2 key rotation policies in detail"
    reformulated = grader.reformulate_query(query)
    assert "Can you please" not in reformulated
    assert "in detail" not in reformulated
    assert "POL-SEC-104.2" in reformulated
    assert "rotation" in reformulated


# ── 3. Integrated Retriever Tests ──────────────────────────────────────────


def test_retriever_multi_query_and_crag_integration():
    """Verify Retriever integrates MultiQueryDecomposer and CRAGGrader."""
    mock_vs = MagicMock()
    mock_vs.search.return_value = [
        {
            "id": "c1",
            "score": 0.35,
            "text": "Cross-Encoder scores pair relevance.",
            "parent_text": "Cross-Encoder scores pair relevance across document blocks.",
            "metadata": {"source_pdf": "doc.pdf", "page_number": 1, "page_end": 1},
        }
    ]

    retriever = Retriever(vector_store=mock_vs)

    # 1. High confidence query
    outcome = retriever.retrieve(
        query="What is cross-encoder scoring?",
        mode="hybrid_rerank",
        enable_crag=True,
    )
    assert outcome["abstained"] is False
    assert outcome["crag"]["grade"] == "high"
    assert outcome["crag"]["action"] == "proceed"
    assert "Cross-Encoder" in outcome["context"]

    # 2. Low confidence query triggers abstention
    mock_vs.search.return_value = [
        {
            "id": "c2",
            "score": -2.5,
            "text": "Completely irrelevant content.",
            "metadata": {"source_pdf": "doc.pdf", "page_number": 1},
        }
    ]
    low_outcome = retriever.retrieve(
        query="What is the orbital speed of Triton?",
        mode="hybrid_rerank",
        enable_crag=True,
    )
    assert low_outcome["abstained"] is True
    assert low_outcome["abstention_reason"] == "LOW_SCORE"
    assert low_outcome["crag"]["grade"] == "low"
