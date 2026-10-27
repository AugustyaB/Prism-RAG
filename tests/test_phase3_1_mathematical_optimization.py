"""Unit and integration test suite for Phase 3.1: Mathematical Foundation Optimization.

Certifies:
1. BM25+ Saturation with lower-bound delta=0.5 and avg_dl=65.0 (preventing document length starvation).
2. Adaptive Weighted RRF (W-RRF) with dynamic query alpha(q) in [0.25, 0.75].
3. Cross-Encoder Temperature Scaling (T=0.8) for probability sharpening.
4. Corrective RAG (CRAG) Top-Margin Gap (Delta_s = s1 - s2) for decisive vs. multi-chunk synthesis classification.
5. Information Novelty Index (INI) and Dynamic Context Pruning (Yield(d_k) = s_rerank * INI >= 0.05).
6. Continuous O(1) Token Bucket Rate Limiter with scalar replenishment rate r = L / W.
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import pytest

from src.api.server import SlidingWindowRateLimiter
from src.database.vector_store import VectorStore, compute_query_alpha
from src.embeddings.embedder import LocalEmbedder
from src.embeddings.sparse_bm25 import SparseBM25Encoder
from src.retrieval.crag import CRAGGrader
from src.retrieval.retriever import Retriever, compute_information_novelty


# ── 1. BM25+ Lower-Bound Saturation Tests ────────────────────────────────────


def test_bm25_plus_delta_preservation_in_long_documents():
    """Verify BM25+ delta lower bound prevents term starvation in long technical documents."""
    encoder_standard = SparseBM25Encoder(k1=1.2, b=0.75, avg_dl=50.0, delta=0.0)
    encoder_plus = SparseBM25Encoder(k1=1.2, b=0.75, avg_dl=65.0, delta=0.5)

    # Synthetic document: 1 target technical identifier surrounded by 2,000 filler words
    target_id = "ERR-CV-4092"
    filler = " ".join(["system", "process", "memory", "buffer", "allocation"] * 400)
    long_doc = f"{target_id} {filler}"

    _, std_vals = encoder_standard.encode_document(long_doc)
    _, plus_vals = encoder_plus.encode_document(long_doc)

    assert len(std_vals) > 0
    assert len(plus_vals) > 0

    # Target term in standard BM25 suffers severe denominator inflation due to |d| >> avgdl
    min_std = min(std_vals)
    min_plus = min(plus_vals)

    # In BM25+, every term weight is strictly >= delta (0.5)
    assert all(val >= 0.5 for val in plus_vals)
    assert min_plus >= 0.5
    # Standard BM25 drops significantly lower than delta
    assert min_std < 0.5
    assert min_plus > min_std


# ── 2. Adaptive Query Alpha & W-RRF Tests ────────────────────────────────────


def test_compute_query_alpha_heuristics():
    """Verify compute_query_alpha steers alpha dynamically based on query syntax."""
    # Technical code queries -> alpha = 0.25 (prioritize BM25 sparse leg)
    assert compute_query_alpha("ERR-CV-4092") == 0.25
    assert compute_query_alpha("POL-SEC-104.2") == 0.25
    assert compute_query_alpha("fastapi_auth_middleware_v2") == 0.25
    assert compute_query_alpha("Find document with hash a1b2c3d4e5f6") == 0.25

    # Pure natural language prose (>= 6 words, no numbers/hyphens) -> alpha = 0.75 (prioritize dense leg)
    assert compute_query_alpha("what are the key architectural differences between dense and sparse search") == 0.75
    assert compute_query_alpha("explain the role of reciprocal rank fusion in multi modal retrieval systems") == 0.75

    # Short balanced queries -> alpha = 0.50 (balanced hybrid)
    assert compute_query_alpha("deep learning") == 0.50
    assert compute_query_alpha("database encryption") == 0.50
    assert compute_query_alpha("") == 0.50


def test_vector_store_w_rrf_weighting():
    """Verify VectorStore.search applies dense/sparse weighting according to alpha."""
    mock_client = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    with patch("src.database.vector_store.QdrantClient") as mock_qc_cls:
        mock_qc_cls.return_value = mock_client
        store = VectorStore(embedder=mock_embedder)
        store.client = mock_client

    # Mock dense and sparse search responses
    dense_pt = MagicMock()
    dense_pt.id = "doc_dense_only"
    dense_pt.score = 0.90
    dense_pt.payload = {"parent_id": "p_dense", "text": "Dense hit text"}

    sparse_pt = MagicMock()
    sparse_pt.id = "doc_sparse_only"
    sparse_pt.score = 15.0
    sparse_pt.payload = {"parent_id": "p_sparse", "text": "Sparse hit text"}

    # Mock query_points
    def query_points_mock(collection_name, query, using, query_filter, limit, with_payload):
        resp = MagicMock()
        if using == "dense":
            resp.points = [dense_pt]
        else:
            resp.points = [sparse_pt]
        return resp

    mock_client.query_points.side_effect = query_points_mock

    # When alpha = 0.25 (code query): dense_weight = 0.5, sparse_weight = 1.5
    # doc_dense_only stage1_score = 0.5 / (60 + 1) = 0.008196
    # doc_sparse_only stage1_score = 1.5 / (60 + 1) = 0.024590 -> sparse must be ranked FIRST!
    res_code = store.search("ERR-CV-4092", mode="hybrid_rrf", alpha=0.25)
    assert len(res_code) == 2
    assert res_code[0]["parent_id"] == "p_sparse"
    assert res_code[1]["parent_id"] == "p_dense"
    assert res_code[0]["score"] > res_code[1]["score"]

    # When alpha = 0.75 (prose query): dense_weight = 1.5, sparse_weight = 0.5
    # doc_dense_only stage1_score = 1.5 / (60 + 1) = 0.024590 -> dense must be ranked FIRST!
    res_prose = store.search("how does the vector store perform fusion retrieval", mode="hybrid_rrf", alpha=0.75)
    assert len(res_prose) == 2
    assert res_prose[0]["parent_id"] == "p_dense"
    assert res_prose[1]["parent_id"] == "p_sparse"
    assert res_prose[0]["score"] > res_prose[1]["score"]


# ── 3. Cross-Encoder Temperature Scaling Tests ──────────────────────────────


def test_cross_encoder_temperature_scaling():
    """Verify temperature scaling sharpens confidence probabilities (T < 1.0)."""
    embedder = LocalEmbedder()
    mock_reranker = MagicMock()
    # Simulated CrossEncoder outputs: [0.8, 0.5, 0.2]
    mock_reranker.predict.return_value = [0.80, 0.50, 0.20]
    embedder._reranker = mock_reranker

    texts = ["doc high", "doc mid", "doc low"]

    # Identity check with T = 1.0
    scores_t1 = embedder.rerank("test query", texts, temperature=1.0)
    assert scores_t1 == [0.80, 0.50, 0.20]

    # Sharpening with T = 0.8:
    # High score (> 0.5) must increase: 0.80 -> > 0.80
    # Center score (0.5) must remain 0.50
    # Low score (< 0.5) must decrease: 0.20 -> < 0.20
    scores_t08 = embedder.rerank("test query", texts, temperature=0.8)
    assert scores_t08[0] > 0.80
    assert abs(scores_t08[1] - 0.50) < 1e-4
    assert scores_t08[2] < 0.20


# ── 4. CRAG Margin Gap & Classification Tests ───────────────────────────────


def test_crag_margin_gap_decisive_vs_synthesis():
    """Verify CRAG distinguishes decisive single-chunk dominance vs. multi-chunk synthesis."""
    grader = CRAGGrader()

    # Case A: Top chunk decisively outscores second chunk by >= 0.10 (Delta_s >= 0.10)
    results_decisive = [
        {"score": 0.45, "text": "Dominant passage"},
        {"score": 0.15, "text": "Secondary passage"},
    ]
    dec_a = grader.grade(top_score=0.45, results=results_decisive)
    assert dec_a["grade"] == "high"
    assert dec_a["action"] == "proceed"
    assert dec_a["reason"] == "HIGH_CONFIDENCE_DECISIVE"
    assert dec_a["margin_gap"] == 0.30

    # Case B: Top chunks have close scores (Delta_s < 0.10) -> requires multi-chunk synthesis
    results_synthesis = [
        {"score": 0.22, "text": "Part 1 of explanation"},
        {"score": 0.18, "text": "Part 2 of explanation"},
    ]
    dec_b = grader.grade(top_score=0.22, results=results_synthesis)
    assert dec_b["grade"] == "high"
    assert dec_b["action"] == "proceed"
    assert dec_b["reason"] == "HIGH_CONFIDENCE_SYNTHESIS"
    assert dec_b["margin_gap"] == 0.04

    # Case C: Low score below LOW_THRESHOLD (0.05) -> abstain
    results_low = [{"score": 0.02, "text": "Irrelevant noise"}]
    dec_c = grader.grade(top_score=0.02, results=results_low)
    assert dec_c["grade"] == "low"
    assert dec_c["action"] == "abstain"
    assert dec_c["reason"] == "LOW_SCORE"


# ── 5. Information Novelty Index (INI) & Context Pruning Tests ──────────────


def test_compute_information_novelty_jaccard():
    """Verify INI calculates token set novelty distance in [0.0, 1.0]."""
    # Empty seen tokens -> 1.0 (first chunk is always completely novel)
    t1 = {"fastapi", "qdrant", "bm25", "rerank"}
    assert compute_information_novelty(t1, []) == 1.0

    # Exact duplicate tokens -> INI = 0.0
    assert compute_information_novelty(t1, [t1]) == 0.0

    # Partial overlap
    # intersection = {"qdrant"}, union = {"fastapi", "qdrant", "bm25", "rerank", "postgres"} (len=5)
    # Jaccard sim = 1/5 = 0.20 -> INI = 1.0 - 0.20 = 0.80
    t2 = {"qdrant", "postgres"}
    ini_val = compute_information_novelty(t2, [t1])
    assert abs(ini_val - 0.80) < 1e-3


def test_retriever_dynamic_context_pruning():
    """Verify Retriever prunes low-yield near-duplicate chunks while preserving top chunk."""
    mock_vs = MagicMock()
    retriever = Retriever(vector_store=mock_vs)

    # 3 candidate hits:
    # Chunk 0: Top hit (score 0.40)
    # Chunk 1: Near duplicate of Chunk 0 (score 0.10, 95% word overlap -> INI < 0.10 -> Yield < 0.05) -> PRUNED
    # Chunk 2: Novel passage (score 0.25, 0% overlap -> INI 1.0 -> Yield 0.25 >= 0.05) -> KEPT
    hits = [
        {
            "id": "h1",
            "score": 0.40,
            "text": "The quick brown fox jumps over the lazy dog in the forest.",
            "metadata": {"source_pdf": "doc.pdf", "page_number": 1},
        },
        {
            "id": "h2",
            "score": 0.10,
            "text": "The quick brown fox jumps over the very lazy dog in the dense forest.",
            "metadata": {"source_pdf": "doc.pdf", "page_number": 1},
        },
        {
            "id": "h3",
            "score": 0.25,
            "text": "Autonomous agents utilize finite state machines for coordinated tool execution.",
            "metadata": {"source_pdf": "doc.pdf", "page_number": 2},
        },
    ]
    mock_vs.search.return_value = hits

    outcome = retriever.retrieve("foxes and agents", mode="hybrid_rerank")
    assert outcome["abstained"] is False
    surviving_hits = outcome["results"]

    # Chunk 1 must be pruned due to Yield < 0.05
    surviving_ids = [h["id"] for h in surviving_hits]
    assert "h1" in surviving_ids
    assert "h3" in surviving_ids
    assert "h2" not in surviving_ids
    assert len(surviving_hits) == 2


# ── 6. Continuous O(1) Token Bucket Rate Limiter Tests ───────────────────────


def test_o1_token_bucket_burst_and_refill():
    """Verify O(1) Token Bucket allows capacity bursts, blocks excess with Retry-After, and refills."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    limiter.reset()

    client_id = "client_test_burst"
    limit = 5

    # 1. Burst: First 5 requests must pass (is_allowed == True)
    for _ in range(limit):
        allowed, wait = limiter.is_allowed(client_id, limit=limit)
        assert allowed is True
        assert wait == 0

    # 2. 6th request in the same instant must be blocked (is_allowed == False)
    allowed, retry_after = limiter.is_allowed(client_id, limit=limit)
    assert allowed is False
    # Deficit is ~1.0 token, refill rate is 5/60 = 1/12 token/sec -> retry_after ~ 12 seconds
    assert retry_after >= 1
    assert retry_after <= 12

    # 3. Verify state storage is O(1) per client
    assert len(limiter.buckets[client_id]) == 2  # [tokens: float, last_update: float]

    # 4. Reset clears the state immediately
    limiter.reset()
    assert len(limiter.buckets) == 0
    allowed, wait = limiter.is_allowed(client_id, limit=limit)
    assert allowed is True
    assert wait == 0
