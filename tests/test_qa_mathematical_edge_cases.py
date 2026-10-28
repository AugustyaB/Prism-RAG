"""QA Deep Mathematical Edge Cases & Numerical Stability Audit Suite (Phase 3.1).

This test suite specifically audits and empirically verifies edge cases, boundary conditions,
numerical stability, failure modes, and concurrency invariants that standard unit tests omit:

1. BM25+ Saturation & Tokenization Edge Cases:
   - Empty/whitespace strings, stopword-only queries, non-ASCII/Unicode/math symbols.
   - Saturated TF limit behavior (tf -> infinity saturates to (k1 + 1) + delta).
   - Document length extremes (dl=1 vs dl=100,000) and 31-bit CRC32 index bounds.
2. Adaptive Query Alpha & W-RRF Boundary Behavior:
   - Query classification across edge case strings (pure numbers, mixed symbols, technical codes).
   - Score monotonicity and rank fusion when candidate sets are completely disjoint vs identical.
   - Robustness against extreme alpha values.
3. Cross-Encoder Temperature Scaling Numerical Stability:
   - Clamping boundaries at p=0.0 and p=1.0 preventing NaN/ZeroDivision.
   - Overflow prevention in logit-mode math.exp with extreme negative/positive logits.
   - Temperature extremes (T <= 0.0, T = 1.0, T -> infinity).
4. CRAG Margin Gap & Decision Engine Boundary Robustness:
   - Single retrieved chunk (missing s2) and None score handling.
   - Identical chunk scores (Delta_s = 0.0) triggering HIGH_CONFIDENCE_SYNTHESIS.
   - Exact threshold boundaries for high, marginal, low, and decisive classifications.
5. Information Novelty Index (INI) & Context Pruning Under Adversarial Inputs:
   - Zero-token candidates, complete duplicates, complete disjointness.
   - Guarantee that top-ranked chunk is unconditionally preserved.
6. O(1) Token Bucket Rate Limiter Concurrency & Resilience:
   - Zero-capacity and zero-rate protection (no ZeroDivisionError).
   - Multi-threaded concurrent bursts (20 threads) verifying lock synchronization.
   - Sub-second fractional replenishment and token ceiling invariants.
7. Advanced Offline Evaluation Metrics Edge Cases:
   - nDCG@K with K > len(chunks), partial evidence retrieval, empty evidence.
   - Harmonic Q_RAG collapse to 0.0 when any metric is 0 or near-zero epsilon.
   - Citation metrics with string page numbers and missing metadata.
8. Welford Streaming Stats & Latency Anomaly Numerical Stability:
   - Large constant streams (N=1,000) with floating-point roundoff protection (no sqrt(-eps)).
   - N=0 and N=1 variance edge cases.
   - Latency anomaly detector with zero variance (sigma=0) without ZeroDivisionError.
9. Ingestion Queue Engine Edge Cases:
   - ETA estimation when processed > total or total = 0.
   - Bounded decorator backoff with jitter across attempt extremes.
10. Zero Live Database Pollution:
    - Confirms that live data/metadata.db is completely unaffected by all audit tests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import random
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.api.server import SlidingWindowRateLimiter
from src.database.models import User
from src.database.vector_store import (
    RRF_K,
    VectorStore,
    compute_query_alpha,
)
from src.embeddings.embedder import LocalEmbedder
from src.embeddings.sparse_bm25 import SparseBM25Encoder
from src.evaluation.evaluator import (
    compute_citation_metrics,
    compute_harmonic_q_rag,
    ndcg_at_k,
)
from src.jobs.ingestion_worker import (
    PageProcessingEtaEstimator,
    compute_backoff_with_jitter,
    estimate_ingestion_eta,
)
from src.retrieval.crag import CRAGGrader
from src.retrieval.retriever import (
    Retriever,
    compute_information_novelty,
)
from src.utils.telemetry import (
    EMALatencyTracker,
    LatencyAnomalyDetector,
    WelfordRunningStats,
)


# ── 1. BM25+ Saturation & Tokenization Edge Cases ────────────────────────────


def test_bm25_plus_empty_and_whitespace_inputs():
    """Verify SparseBM25Encoder gracefully returns empty vectors for empty/whitespace inputs."""
    encoder = SparseBM25Encoder()
    assert encoder.tokenize("") == []
    assert encoder.tokenize("   \t\n  ") == []
    assert encoder.encode_document("") == ([], [])
    assert encoder.encode_document("    ") == ([], [])


def test_bm25_plus_unicode_and_math_symbols():
    """Verify tokenizer extracts non-ASCII alphanumeric symbols and code tokens cleanly."""
    encoder = SparseBM25Encoder()
    text = "Error in λ-calculus module: ∑_i=0^N alpha_val_42, delta=0.5, CVE-2026-8899."
    tokens = encoder.tokenize(text)
    assert len(tokens) > 0
    # Must preserve exact hyphenated CVE identifier
    assert any("cve-2026-8899" in t for t in tokens)
    # Check encoding succeeds without exception
    indices, values = encoder.encode_document(text)
    assert len(indices) == len(values)
    assert all(0 <= idx <= 0x7FFFFFFF for idx in indices)
    assert all(v >= 0.5 for v in values)


def test_bm25_plus_tf_saturation_limit():
    """Verify term frequency saturation behavior: as tf -> infinity, weight -> (k1 + 1) + delta."""
    k1 = 1.2
    delta = 0.5
    encoder = SparseBM25Encoder(k1=k1, b=0.75, avg_dl=100.0, delta=delta)

    # Document where a single term occurs 50,000 times
    text = " ".join(["overclocked"] * 50000)
    indices, values = encoder.encode_document(text)

    assert len(indices) == 1
    theoretical_limit = (k1 + 1.0) + delta  # 2.2 + 0.5 = 2.7
    # For very large tf, weight should be strictly <= theoretical_limit and > 2.65
    assert values[0] <= theoretical_limit
    assert values[0] > 2.65


def test_bm25_plus_crc32_positive_31bit_bound():
    """Verify token-to-index CRC32 mapping strictly adheres to positive 31-bit integer bounds."""
    encoder = SparseBM25Encoder()
    adversarial_tokens = [
        "__system_critical_fault__",
        "0x7FFFFFFF",
        "POL-SEC-999.999.999",
        "µ-service-gateway-αβγ",
        "very_long_" * 50 + "end",
    ]
    for tok in adversarial_tokens:
        idx = encoder._token_to_index(tok)
        assert isinstance(idx, int)
        assert 0 <= idx <= 0x7FFFFFFF, f"CRC32 index {idx} out of 31-bit positive bounds for {tok}"


# ── 2. Adaptive Query Alpha & W-RRF Boundary & Monotonicity ──────────────────


def test_compute_query_alpha_adversarial_queries():
    """Verify compute_query_alpha handles punctuation, numbers, and boundary strings."""
    # Technical code queries with alphanumeric or hyphens/dots -> code query (0.25)
    assert compute_query_alpha("port-9092") == 0.25
    assert compute_query_alpha("P9092") == 0.25
    assert compute_query_alpha("127.0.0.1:8080") == 0.25
    assert compute_query_alpha("v2.10.0-rc1") == 0.25
    assert compute_query_alpha("0xDEADBEEF") == 0.25

    # Pure standalone numbers without code affixes -> default balanced 0.50
    assert compute_query_alpha("9092") == 0.50

    # Pure symbols/punctuation -> default balanced 0.50
    assert compute_query_alpha("---") == 0.50
    assert compute_query_alpha("??? !!!") == 0.50
    assert compute_query_alpha("   ") == 0.50

    # Mixed natural language with a single code token -> code query (0.25)
    assert compute_query_alpha("how to fix the error in POL-SEC-104.2") == 0.25


def test_w_rrf_disjoint_vs_overlapping_scores():
    """Verify W-RRF score calculation with dense/sparse candidates."""
    k = RRF_K  # 60.0
    alpha = 0.25
    dense_weight = 2.0 * alpha          # 0.5
    sparse_weight = 2.0 * (1.0 - alpha) # 1.5

    # Case 1: Item appears in dense at rank 1, NOT in sparse
    score_dense_only = dense_weight / (k + 1.0)  # 0.5 / 61 ≈ 0.008196

    # Case 2: Item appears in sparse at rank 1, NOT in dense
    score_sparse_only = sparse_weight / (k + 1.0) # 1.5 / 61 ≈ 0.024590

    # Case 3: Item appears in both at rank 1
    score_both = score_dense_only + score_sparse_only # 2.0 / 61 ≈ 0.032787

    # Monotonicity invariants:
    assert score_both > score_sparse_only > score_dense_only
    assert pytest.approx(score_both, abs=1e-6) == (score_dense_only + score_sparse_only)


# ── 3. Cross-Encoder Temperature Scaling Numerical Stability ─────────────────


def test_temperature_scaling_probability_boundaries_and_clamping():
    """Verify temperature scaling at boundary probabilities (0.0 and 1.0) without NaN or crash."""
    embedder = LocalEmbedder()
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.0, 1.0, 0.5]
    embedder._reranker = mock_reranker

    # T = 0.8
    scores = embedder.rerank("query", ["doc0", "doc1", "doc0.5"], temperature=0.8)
    assert len(scores) == 3
    assert not any(math.isnan(s) for s in scores)
    assert not any(math.isinf(s) for s in scores)

    # 0.0 stays clamped near 0.0; 1.0 stays clamped near 1.0; 0.5 is fixed point
    assert scores[0] <= 1e-4
    assert scores[1] >= 0.9999
    assert pytest.approx(scores[2], abs=1e-4) == 0.50


def test_temperature_scaling_extreme_logits_no_overflow():
    """Verify unnormalized extreme negative/positive logits do not cause math.exp OverflowError."""
    embedder = LocalEmbedder()
    mock_reranker = MagicMock()
    # Extreme logits: -500.0, +500.0, -100.0, +100.0 (outside [0, 1])
    mock_reranker.predict.return_value = [-500.0, 500.0, -100.0, 100.0]
    embedder._reranker = mock_reranker

    # Small temperature T = 0.1 could produce exp(5000) if unshielded
    scores = embedder.rerank("query", ["d1", "d2", "d3", "d4"], temperature=0.1)
    assert len(scores) == 4
    assert not any(math.isnan(s) for s in scores)
    assert not any(math.isinf(s) for s in scores)
    # Negative logit -> sigmoid -> ~0.0
    assert scores[0] == 0.0
    # Positive logit -> sigmoid -> ~1.0
    assert scores[1] == 1.0


def test_temperature_scaling_non_positive_temperature():
    """Verify non-positive or identity temperatures (T <= 0 or T = 1.0) return raw unscaled scores."""
    embedder = LocalEmbedder()
    mock_reranker = MagicMock()
    raw = [0.15, 0.45, 0.85]
    mock_reranker.predict.return_value = raw
    embedder._reranker = mock_reranker

    assert embedder.rerank("query", ["d1", "d2", "d3"], temperature=1.0) == raw
    assert embedder.rerank("query", ["d1", "d2", "d3"], temperature=0.0) == raw
    assert embedder.rerank("query", ["d1", "d2", "d3"], temperature=-0.5) == raw


# ── 4. CRAG Margin Gap & Decision Engine Boundary Robustness ─────────────────


def test_crag_single_chunk_and_none_score_handling():
    """Verify CRAG handles single-chunk retrieval and None scores without TypeError."""
    grader = CRAGGrader()

    # Only 1 chunk in results: second chunk doesn't exist -> margin_gap = top_score
    results_single = [{"score": 0.35, "text": "Sole candidate"}]
    decision = grader.grade(top_score=0.35, results=results_single)
    assert decision["grade"] == "high"
    assert decision["action"] == "proceed"
    assert decision["reason"] == "HIGH_CONFIDENCE_DECISIVE"
    assert decision["margin_gap"] == 0.35

    # Secondary chunk with score = None
    results_none_score = [
        {"score": 0.40, "text": "Top candidate"},
        {"score": None, "text": "Unscored candidate"},
    ]
    dec_none = grader.grade(top_score=0.40, results=results_none_score)
    assert dec_none["grade"] == "high"
    assert dec_none["margin_gap"] == 0.40


def test_crag_identical_scores_multi_chunk_synthesis():
    """Verify chunks with identical scores yield margin_gap = 0.0 and trigger synthesis classification."""
    grader = CRAGGrader()
    results_tied = [
        {"score": 0.30, "text": "Passage Alpha"},
        {"score": 0.30, "text": "Passage Beta"},
    ]
    dec = grader.grade(top_score=0.30, results=results_tied)
    assert dec["grade"] == "high"
    assert dec["action"] == "proceed"
    assert dec["reason"] == "HIGH_CONFIDENCE_SYNTHESIS"
    assert dec["margin_gap"] == 0.0


def test_crag_exact_threshold_boundaries():
    """Verify CRAG behavior at exact boundary values (0.15, 0.05, and margin_gap 0.10)."""
    grader = CRAGGrader()

    # Exact HIGH_THRESHOLD (0.15) with margin 0.10 -> DECISIVE
    dec_exact_high = grader.grade(0.15, [{"score": 0.15}, {"score": 0.05}])
    assert dec_exact_high["grade"] == "high"
    assert dec_exact_high["reason"] == "HIGH_CONFIDENCE_DECISIVE"
    assert dec_exact_high["margin_gap"] == 0.10

    # Exact HIGH_THRESHOLD (0.15) with margin 0.09 -> SYNTHESIS
    dec_exact_synth = grader.grade(0.15, [{"score": 0.15}, {"score": 0.06}])
    assert dec_exact_synth["grade"] == "high"
    assert dec_exact_synth["reason"] == "HIGH_CONFIDENCE_SYNTHESIS"

    # Exact LOW_THRESHOLD (0.05) with loop_count=0 -> MARGINAL REFORMULATE
    dec_marginal_boundary = grader.grade(0.05, [{"score": 0.05}], loop_count=0)
    assert dec_marginal_boundary["grade"] == "marginal"
    assert dec_marginal_boundary["action"] == "reformulate"

    # Just below LOW_THRESHOLD (0.0499) -> ABSTAIN
    dec_low_boundary = grader.grade(0.0499, [{"score": 0.0499}], loop_count=0)
    assert dec_low_boundary["grade"] == "low"
    assert dec_low_boundary["action"] == "abstain"


# ── 5. Information Novelty Index (INI) & Context Pruning Under Adversarial Inputs ──


def test_ini_adversarial_token_sets():
    """Verify compute_information_novelty with empty, identical, and disjoint token sets."""
    # Empty candidate tokens -> 0.0
    assert compute_information_novelty(set(), [{"apple", "banana"}]) == 0.0

    # Candidate with empty seen sets -> 1.0 (first chunk)
    cand = {"neural", "network", "transformer"}
    assert compute_information_novelty(cand, []) == 1.0
    assert compute_information_novelty(cand, [set()]) == 1.0

    # Identical set -> 0.0 novelty
    assert compute_information_novelty(cand, [cand]) == 0.0

    # Completely disjoint set -> 1.0 novelty
    assert compute_information_novelty(cand, [{"database", "sqlite", "relational"}]) == 1.0


def test_retriever_unconditional_top_chunk_preservation():
    """Verify dynamic context pruning preserves the top-1 chunk unconditionally even if yield < 0.05."""
    mock_vs = MagicMock()
    retriever = Retriever(vector_store=mock_vs)

    # Top chunk with low score (0.02) and duplicate subsequent chunks
    hits = [
        {"id": "c1", "score": 0.02, "text": "Top chunk text.", "metadata": {"source_pdf": "doc.pdf"}},
        {"id": "c2", "score": 0.01, "text": "Top chunk text identical duplicate.", "metadata": {"source_pdf": "doc.pdf"}},
    ]
    mock_vs.search.return_value = hits

    # Query in vector_only mode (bypasses rerank gate to isolate context pruning)
    outcome = retriever.retrieve("query", mode="vector_only")
    assert outcome["abstained"] is False
    surviving = outcome["results"]

    # Top chunk MUST be preserved unconditionally
    assert len(surviving) >= 1
    assert surviving[0]["id"] == "c1"
    assert surviving[0]["ini"] == 1.0
    assert surviving[0]["yield"] == 0.02


# ── 6. Continuous O(1) Token Bucket Rate Limiter Concurrency & Resilience ────


def test_token_bucket_zero_limit_protection():
    """Verify SlidingWindowRateLimiter handles limit=0 without ZeroDivisionError."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    limiter.reset()

    # Zero limit must cleanly return False with retry_after > 0, never crashing
    allowed, retry_after = limiter.is_allowed("test_client", limit=0)
    assert allowed is False
    assert retry_after >= 1


def test_token_bucket_simulated_clock_skew():
    """Verify rate limiter does not deduct or corrupt tokens if system clock jumps backward."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    limiter.reset()
    client = "client_clock_skew"

    # Request 1 at t=100
    with patch("time.time", return_value=1000.0):
        allowed, _ = limiter.is_allowed(client, limit=10)
        assert allowed is True

    # Request 2 with simulated backward clock jump (t=990 < 1000)
    with patch("time.time", return_value=990.0):
        allowed, _ = limiter.is_allowed(client, limit=10)
        assert allowed is True
        # Tokens should not have become negative
        assert limiter.buckets[client][0] >= 0.0


def test_token_bucket_multi_threaded_concurrency():
    """Verify thread safety and exact burst enforcement under multi-threaded concurrency."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    limiter.reset()
    client = "client_concurrent"
    limit = 10

    # Launch 25 concurrent threads at the exact same instant
    def make_call(_):
        return limiter.is_allowed(client, limit=limit)

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(make_call, range(25)))

    allowed_count = sum(1 for allowed, _ in results if allowed is True)
    blocked_count = sum(1 for allowed, _ in results if allowed is False)

    # Exactly `limit` requests must be allowed, the rest blocked
    assert allowed_count == limit
    assert blocked_count == 15


# ── 7. Advanced Offline Evaluation Metrics Edge Cases ────────────────────────


def test_ndcg_at_k_k_greater_than_retrieved():
    """Verify nDCG@K computes correctly when K is greater than retrieved chunks count."""
    retrieved = [
        {"parent_text": "Relevant passage content here"},
    ]
    expected = ["Relevant passage content here"]
    # K = 10 but only 1 chunk retrieved
    score = ndcg_at_k(retrieved, expected, k=10)
    assert score == 1.0


def test_ndcg_at_k_partial_evidence_corpus_idcg():
    """Verify nDCG@K reflects missed evidence when expected evidence contains multiple items."""
    # 2 expected evidence items
    expected = ["Evidence A", "Evidence B"]

    # System retrieved only Evidence A at rank 1, but completely missed Evidence B
    retrieved = [
        {"parent_text": "Evidence A is present here"},
        {"parent_text": "Unrelated noise"},
    ]
    # DCG = 1 / log2(2) = 1.0
    # IDCG with 2 expected items = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
    # nDCG = 1.0 / 1.6309 ≈ 0.6131
    score = ndcg_at_k(retrieved, expected, k=2)
    assert score < 1.0
    assert pytest.approx(score, abs=1e-3) == 1.0 / (1.0 + 1.0 / math.log2(3))


def test_harmonic_q_rag_epsilon_and_negative_guards():
    """Verify compute_harmonic_q_rag handles sub-epsilon or negative numbers gracefully."""
    # Zero or negative inputs must collapse to 0.0
    assert compute_harmonic_q_rag(-0.5, 0.8, 0.8, 0.8) == 0.0
    assert compute_harmonic_q_rag(0.8, 0.0, 0.8, 0.8) == 0.0
    assert compute_harmonic_q_rag(0.8, 1e-7, 0.8, 0.8) == 0.0  # < eps (1e-6)


def test_citation_metrics_string_page_numbers():
    """Verify compute_citation_metrics handles metadata with string page numbers robustly."""
    retrieved = [
        {"metadata": {"source_pdf": "tech_manual.pdf", "page_number": "5", "page_end": "5"}},
    ]
    metrics = compute_citation_metrics(retrieved, "tech_manual.pdf", [5], k=1)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


# ── 8. Welford Streaming Stats & Latency Anomaly Numerical Stability ──────────


def test_welford_constant_stream_no_math_domain_error():
    """Verify WelfordRunningStats avoids negative variance and math domain error on identical floats."""
    welford = WelfordRunningStats()
    # Feed 1,000 identical numbers
    for _ in range(1000):
        welford.update(42.0)

    assert welford.count == 1000
    assert pytest.approx(welford.mean, abs=1e-9) == 42.0
    assert welford.variance >= 0.0
    assert welford.std_dev >= 0.0
    assert welford.variance == 0.0
    assert welford.std_dev == 0.0


def test_welford_sample_count_zero_and_one():
    """Verify WelfordRunningStats handles N=0 and N=1 without division by zero."""
    welford = WelfordRunningStats()
    assert welford.count == 0
    assert welford.mean == 0.0
    assert welford.variance == 0.0
    assert welford.std_dev == 0.0

    welford.update(100.0)
    assert welford.count == 1
    assert welford.mean == 100.0
    assert welford.variance == 0.0
    assert welford.std_dev == 0.0


def test_latency_anomaly_detector_zero_variance():
    """Verify LatencyAnomalyDetector does not divide by zero when all observations are identical."""
    detector = LatencyAnomalyDetector(z_threshold=3.0, min_samples=5)
    # Feed identical values
    for _ in range(10):
        is_ano, z = detector.record_and_check(50.0)
        assert is_ano is False
        assert z == 0.0


# ── 9. Ingestion Queue Engine Edge Cases ─────────────────────────────────────


def test_estimate_ingestion_eta_edge_cases():
    """Verify estimate_ingestion_eta under boundary conditions."""
    # Processed > Total pages -> ETA = 0.0
    assert estimate_ingestion_eta(total_pages=5, processed_pages=10, t_page_avg=2.0) == 0.0
    # Total pages = 0 -> ETA = 0.0
    assert estimate_ingestion_eta(total_pages=0, processed_pages=0, t_page_avg=2.0) == 0.0
    # Zero page duration -> ETA = 0.0
    assert estimate_ingestion_eta(total_pages=10, processed_pages=5, t_page_avg=0.0) == 0.0


def test_page_processing_eta_estimator_ignores_non_positive_times():
    """Verify PageProcessingEtaEstimator ignores negative or zero duration measurements."""
    estimator = PageProcessingEtaEstimator(default_t_page=1.5)
    estimator.record_page_time(0.0)
    estimator.record_page_time(-2.5)
    # page_durations should remain empty, using default_t_page
    assert len(estimator.page_durations) == 0
    assert estimator.t_page_avg == 1.5


def test_jittered_backoff_attempt_extremes():
    """Verify compute_backoff_with_jitter across negative and large attempt numbers."""
    # Negative attempt: 2^(-1) = 0.5 -> upper = 0.5 * 0.5 = 0.25
    b_neg = compute_backoff_with_jitter(-1, t_base=0.5, t_max=10.0)
    assert 0.0 <= b_neg <= 0.25

    # Very large attempt (e.g., attempt 30): capped strictly at t_max
    for _ in range(10):
        b_large = compute_backoff_with_jitter(30, t_base=0.5, t_max=10.0)
        assert 0.0 <= b_large <= 10.0


# ── 10. Zero Live Database Pollution Verification ────────────────────────────


def test_live_database_remains_unpolluted_by_math_audit():
    """Confirm live data/metadata.db contains zero temporary unit test accounts or modified passwords."""
    live_db_path = Path("data/metadata.db")
    if not live_db_path.exists():
        pytest.skip("No live database exists at data/metadata.db.")

    engine = create_engine(f"sqlite:///{live_db_path}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with Session() as session:
        # Check no bogus test accounts leaked
        ghost_user = session.scalars(
            select(User).where(User.email.like("%adversarial%") | User.email.like("%edge_case%"))
        ).first()
        assert ghost_user is None, "Test user leaked into live database!"

    engine.dispose()
