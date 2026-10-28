"""Unit and integration tests for Phase 3.1 Wave 2: Advanced Offline Evaluation Metrics, Anomaly Telemetry & Queue Optimization."""

from __future__ import annotations

import math
import statistics
from unittest.mock import MagicMock

import pytest

from src.evaluation.evaluator import (
    EvaluationEngine,
    compute_citation_metrics,
    compute_harmonic_q_rag,
    ndcg_at_k,
)
from src.jobs.ingestion_worker import (
    PageProcessingEtaEstimator,
    compute_backoff_with_jitter,
    estimate_ingestion_eta,
)
from src.utils.telemetry import (
    EMALatencyTracker,
    LatencyAnomalyDetector,
    TelemetryCollector,
    WelfordRunningStats,
    reset_global_telemetry_trackers,
)


# ── 1. Normalized Discounted Cumulative Gain (nDCG@K) Tests ─────────────────


def test_ndcg_at_k_perfect_ranking():
    """Verify nDCG@K is 1.0 when relevant item is ranked at position 1."""
    retrieved = [
        {"parent_text": "Alpha error code 0x7A occurred in network buffer"},
        {"parent_text": "Unrelated paragraph about system cooling fans"},
        {"parent_text": "Another unrelated section about power supply"},
    ]
    expected_evidence = ["0x7A occurred in network buffer"]
    score = ndcg_at_k(retrieved, expected_evidence, k=3)
    assert score == 1.0


def test_ndcg_at_k_discounted_ranking():
    """Verify nDCG@K applies logarithmic rank discount when relevant item is lower."""
    # Relevant item at position 2 (0-indexed 1)
    retrieved = [
        {"parent_text": "Unrelated paragraph about system cooling fans"},
        {"parent_text": "Alpha error code 0x7A occurred in network buffer"},
        {"parent_text": "Another unrelated section about power supply"},
    ]
    expected_evidence = ["0x7A occurred in network buffer"]
    score = ndcg_at_k(retrieved, expected_evidence, k=3)
    # DCG = 1 / log2(3) ≈ 0.6309, IDCG = 1 / log2(2) = 1.0 -> nDCG ≈ 0.6309
    expected_dcg = 1.0 / math.log2(3)
    assert pytest.approx(score, abs=1e-3) == expected_dcg


def test_ndcg_at_k_zero_relevance_and_edge_cases():
    """Verify nDCG@K handles zero matches and empty inputs gracefully."""
    retrieved = [
        {"parent_text": "Unrelated paragraph about system cooling fans"},
        {"parent_text": "Another unrelated section about power supply"},
    ]
    expected = ["critical fault nuclear core breach"]
    assert ndcg_at_k(retrieved, expected, k=2) == 0.0

    # Empty retrieved returns 0.0
    assert ndcg_at_k([], expected, k=2) == 0.0

    # Empty expected evidence returns 1.0
    assert ndcg_at_k(retrieved, [], k=2) == 1.0


# ── 2. Harmonic Composite Quality Score (Q_RAG) Tests ───────────────────────


def test_compute_harmonic_q_rag_balanced():
    """Verify harmonic mean across balanced evaluation metrics."""
    assert compute_harmonic_q_rag(1.0, 1.0, 1.0, 1.0) == 1.0
    assert compute_harmonic_q_rag(0.8, 0.8, 0.8, 0.8) == 0.8


def test_compute_harmonic_q_rag_single_metric_collapse():
    """Verify harmonic mean strictly collapses to 0.0 when any component collapses."""
    # Faithfulness collapse
    assert compute_harmonic_q_rag(0.95, 0.90, 0.92, 0.0) == 0.0

    # Recall collapse
    assert compute_harmonic_q_rag(0.0, 0.90, 0.92, 0.88) == 0.0

    # Abstention F1 collapse
    assert compute_harmonic_q_rag(0.95, 0.0, 0.92, 0.88) == 0.0

    # Citation Accuracy collapse
    assert compute_harmonic_q_rag(0.95, 0.90, 0.0, 0.88) == 0.0


def test_compute_harmonic_q_rag_penalizes_asymmetry():
    """Verify harmonic mean severely penalizes asymmetric metric degradation."""
    # Arithmetic mean = (0.9 + 0.9 + 0.9 + 0.3) / 4 = 0.75
    # Harmonic mean = 4 / (1/0.9 + 1/0.9 + 1/0.9 + 1/0.3) = 4 / (3.333 + 3.333) = 4 / 6.666 ≈ 0.60
    q = compute_harmonic_q_rag(0.9, 0.9, 0.9, 0.3)
    assert q < 0.65
    assert q > 0.55


# ── 3. Citation Precision, Recall & F1 Tests ─────────────────────────────────


def test_compute_citation_metrics_exact_match():
    """Verify citation precision, recall, and F1 with exact document and page matches."""
    retrieved = [
        {"metadata": {"source_pdf": "tech_manual.pdf", "page_number": 5, "page_end": 5}},
        {"metadata": {"source_pdf": "tech_manual.pdf", "page_number": 6, "page_end": 6}},
    ]
    # Expected: tech_manual.pdf pages 5 and 6
    metrics = compute_citation_metrics(retrieved, "tech_manual.pdf", [5, 6], k=2)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_citation_metrics_partial_precision():
    """Verify precision drops when retrieved chunks include irrelevant documents."""
    retrieved = [
        {"metadata": {"source_pdf": "tech_manual.pdf", "page_number": 5, "page_end": 5}},
        {"metadata": {"source_pdf": "random_policy.pdf", "page_number": 1, "page_end": 1}},
    ]
    # 1 valid out of 2 retrieved -> Precision = 0.5; 1 valid out of 1 expected page -> Recall = 1.0
    metrics = compute_citation_metrics(retrieved, "tech_manual.pdf", [5], k=2)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    # F1 = 2 * 0.5 * 1.0 / 1.5 = 1.0 / 1.5 ≈ 0.6667
    assert pytest.approx(metrics["f1"], abs=1e-3) == 0.6667


def test_compute_citation_metrics_no_match_and_edge_cases():
    """Verify citation metrics return zeros on completely mismatched sources."""
    retrieved = [
        {"metadata": {"source_pdf": "other.pdf", "page_number": 1}},
    ]
    metrics = compute_citation_metrics(retrieved, "tech_manual.pdf", [5], k=1)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

    # No expected doc returns 1.0
    assert compute_citation_metrics(retrieved, None, [], k=1) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


# ── 4. Welford's Streaming Algorithm Tests ─────────────────────────────────


def test_welford_running_stats_mathematical_accuracy():
    """Verify Welford streaming stats match python statistics library exactly."""
    samples = [45.2, 52.1, 48.9, 61.4, 47.0, 50.8, 55.3, 49.6, 53.2]
    welford = WelfordRunningStats()
    for s in samples:
        welford.update(s)

    expected_mean = statistics.mean(samples)
    expected_variance = statistics.variance(samples)
    expected_std_dev = statistics.stdev(samples)

    assert pytest.approx(welford.mean, abs=1e-6) == expected_mean
    assert pytest.approx(welford.variance, abs=1e-6) == expected_variance
    assert pytest.approx(welford.std_dev, abs=1e-6) == expected_std_dev
    assert welford.count == len(samples)


# ── 5. Z-Score Latency Anomaly Detector Tests ───────────────────────────────


def test_latency_anomaly_detector_threshold_and_detection():
    """Verify Z-Score anomaly detector ignores early samples and catches 3-sigma spikes."""
    detector = LatencyAnomalyDetector(z_threshold=3.0, min_samples=5)

    # Feed normal latencies around 50ms with small variance (std_dev ≈ 1.5ms)
    baseline = [50.0, 51.0, 49.0, 50.5, 49.5, 50.2, 50.8, 49.8]
    for lat in baseline:
        is_ano, z = detector.record_and_check(lat)
        assert is_ano is False
        assert z < 3.0

    # Inject a 3-sigma+ latency anomaly (e.g., 80ms when mean ≈ 50, std_dev ≈ 0.6)
    is_ano, z = detector.record_and_check(85.0)
    assert is_ano is True
    assert z >= 3.0
    assert detector.anomaly_count == 1


# ── 6. Exponential Moving Average (EMA) Drift Tracker Tests ─────────────────


def test_ema_latency_tracker_smoothing():
    """Verify EMA smoothing formula: EMA_t = beta * EMA_{t-1} + (1 - beta) * latency_t."""
    tracker = EMALatencyTracker(beta=0.90)

    # First point initializes EMA
    e1 = tracker.update("embed", 100.0)
    assert e1 == 100.0

    # Second point: 0.90 * 100 + 0.10 * 200 = 90 + 20 = 110.0
    e2 = tracker.update("embed", 200.0)
    assert e2 == 110.0

    # Third point: 0.90 * 110 + 0.10 * 300 = 99 + 30 = 129.0
    e3 = tracker.update("embed", 300.0)
    assert e3 == 129.0

    # Independent stage
    s1 = tracker.update("search", 50.0)
    assert s1 == 50.0

    all_emas = tracker.to_dict()
    assert all_emas["embed"] == 129.0
    assert all_emas["search"] == 50.0


# ── 7. Exponential Backoff with Decorrelated Jitter Tests ───────────────────


def test_compute_backoff_with_jitter_bounds():
    """Verify jittered exponential backoff stays within [0, min(t_max, t_base * 2^attempt)]."""
    t_base = 0.5
    t_max = 8.0

    # Attempt 0: max upper is 0.5 * 1 = 0.5
    for _ in range(20):
        b0 = compute_backoff_with_jitter(0, t_base=t_base, t_max=t_max)
        assert 0.0 <= b0 <= 0.5

    # Attempt 2: max upper is 0.5 * 4 = 2.0
    for _ in range(20):
        b2 = compute_backoff_with_jitter(2, t_base=t_base, t_max=t_max)
        assert 0.0 <= b2 <= 2.0

    # Attempt 10: uncapped would be 0.5 * 1024 = 512, but must be capped at t_max (8.0)
    for _ in range(20):
        b10 = compute_backoff_with_jitter(10, t_base=t_base, t_max=t_max)
        assert 0.0 <= b10 <= t_max


# ── 8. Ingestion ETA Estimator Tests ─────────────────────────────────────────


def test_estimate_ingestion_eta_math():
    """Verify ETA in seconds: ETA_sec = (N_total - N_done) * t_page_avg."""
    # 10 total pages, 4 processed, 2.0 seconds per page -> 6 * 2.0 = 12.0s
    assert estimate_ingestion_eta(total_pages=10, processed_pages=4, t_page_avg=2.0) == 12.0

    # Done >= total -> 0.0s
    assert estimate_ingestion_eta(total_pages=10, processed_pages=10, t_page_avg=2.0) == 0.0
    assert estimate_ingestion_eta(total_pages=10, processed_pages=12, t_page_avg=2.0) == 0.0


def test_page_processing_eta_estimator_class():
    """Verify PageProcessingEtaEstimator maintains dynamic running average."""
    estimator = PageProcessingEtaEstimator(default_t_page=2.0)
    assert estimator.estimate(total_pages=5, processed_pages=1) == 8.0  # 4 * 2.0

    # Record actual page durations
    estimator.record_page_time(1.0)
    estimator.record_page_time(3.0)
    # Average = (1.0 + 3.0) / 2 = 2.0
    assert estimator.t_page_avg == 2.0
    assert estimator.estimate(total_pages=10, processed_pages=5) == 10.0  # 5 * 2.0


# ── 9. TelemetryCollector Integration with Anomalies & EMA ──────────────────


def test_telemetry_collector_anomaly_and_ema_integration():
    """Verify TelemetryCollector records stage EMAs and includes anomaly flags."""
    reset_global_telemetry_trackers()

    collector = TelemetryCollector()
    collector.record_stage("embed", 25.0)
    collector.record_stage("search", 40.0)
    collector.record_stage("rerank", 55.0)
    collector.record_stage("llm", 120.0)

    data = collector.to_dict()
    assert "stage_emas" in data
    assert "is_anomaly" in data
    assert "z_score" in data
    assert data["stage_emas"]["embed"] == 25.0
    assert data["stage_emas"]["search"] == 40.0
    assert data["stage_emas"]["rerank"] == 55.0
    assert data["stage_emas"]["llm"] == 120.0


# ── 10. EvaluationEngine Metric Aggregation Integration ─────────────────────


def test_evaluation_engine_aggregate_metrics_includes_new_scores():
    """Verify EvaluationEngine._aggregate_metrics computes nDCG@4, Citation F1, and Q_RAG."""
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    engine = EvaluationEngine.__new__(EvaluationEngine)
    engine.vector_store = mock_vs
    engine.retriever = mock_retriever

    synthetic_results = [
        {
            "id": "q1",
            "category": "semantic",
            "should_abstain": False,
            "abstained": False,
            "recall_at_4": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
            "ndcg_at_4": 1.0,
            "citation_accuracy": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "citation_f1": 1.0,
            "faithfulness": 0.95,
            "latency_ms": 45.0,
        },
        {
            "id": "q2",
            "category": "unanswerable",
            "should_abstain": True,
            "abstained": True,
            "recall_at_4": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
            "ndcg_at_4": 1.0,
            "citation_accuracy": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "citation_f1": 1.0,
            "faithfulness": 1.0,
            "latency_ms": 15.0,
        },
    ]

    summary = engine._aggregate_metrics("hybrid_rerank", synthetic_results, [45.0, 15.0])
    metrics = summary["metrics"]

    assert "ndcg_at_4" in metrics
    assert "citation_precision" in metrics
    assert "citation_recall" in metrics
    assert "citation_f1" in metrics
    assert "q_rag" in metrics
    assert metrics["ndcg_at_4"] == 1.0
    assert metrics["citation_f1"] == 1.0
    assert metrics["q_rag"] > 0.9
