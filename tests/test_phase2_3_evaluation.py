"""Phase 2.3 QA Test Suite: Automated RAG Evaluation Engine, Metrics, Benchmark Suite & CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluate import build_markdown_report, format_table
from src.evaluation.evaluator import (
    EvaluationEngine,
    citation_accuracy,
    compute_abstention_confusion,
    compute_answer_faithfulness,
    mean_reciprocal_rank,
    normalize_text,
    recall_at_k,
)


# ============================================================================
# 1. BENCHMARK DATASET INTEGRITY & CATEGORY BALANCE AUDIT
# ============================================================================


def test_benchmark_dataset_integrity_and_schema():
    """Verify data/eval/benchmark_dataset.json adheres to schema and exact category breakdown."""
    dataset_path = Path("data/eval/benchmark_dataset.json")
    assert dataset_path.exists(), f"Benchmark dataset missing at {dataset_path}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 100, f"Expected 100 QA items, got {len(data)}"

    categories: dict[str, int] = {}
    should_abstain_count = 0

    for idx, item in enumerate(data):
        assert "id" in item and item["id"].startswith("Q"), f"Item {idx} missing valid ID"
        assert "question" in item and len(item["question"].strip()) > 5
        assert "category" in item
        assert "should_abstain" in item

        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1

        if item["should_abstain"]:
            should_abstain_count += 1
            assert item["expected_doc"] is None, f"Unanswerable item {item['id']} has expected_doc"
            assert item["expected_pages"] == [], f"Unanswerable item {item['id']} has expected_pages"
            assert item["expected_evidence"] == []
        else:
            assert item["expected_doc"] is not None, f"Answerable item {item['id']} missing expected_doc"
            assert len(item["expected_evidence"]) > 0, f"Answerable item {item['id']} missing evidence"

    assert categories == {
        "semantic": 40,
        "lexical": 30,
        "ocr": 15,
        "unanswerable": 15,
    }, f"Category breakdown mismatch: {categories}"
    assert should_abstain_count == 15


def test_benchmark_pdfs_exist_and_are_valid():
    """Verify the 3 programmatic benchmark PDFs exist and have the expected page counts."""
    import fitz

    bench_dir = Path("data/benchmark_pdfs")
    expected_files = {
        "tech_manual.pdf": 4,
        "compliance_policy.pdf": 4,
        "scanned_operations_log.pdf": 3,
    }

    for fname, expected_pages in expected_files.items():
        pdf_path = bench_dir / fname
        assert pdf_path.exists(), f"Missing benchmark PDF: {pdf_path}"
        doc = fitz.open(str(pdf_path))
        try:
            assert doc.page_count == expected_pages, (
                f"{fname} has {doc.page_count} pages, expected {expected_pages}"
            )
        finally:
            doc.close()


# ============================================================================
# 2. EVALUATION METRIC UNIT TESTS & MATHEMATICAL VALIDATION
# ============================================================================


def test_recall_at_k_and_mrr_metrics():
    """Verify Recall@K and MRR on perfect, zero, and rank-k retrieval results."""
    chunks = [
        {"text": "Irrelevant introductory paragraph.", "parent_text": "Intro text."},
        {"text": "AetherDB Raft communicates on Port 9093.", "parent_text": "Consensus on Port 9093."},
        {"text": "Client connections bind on Port 9092.", "parent_text": "Clients connect to Port 9092."},
        {"text": "Metrics exposed on Port 8080.", "parent_text": "Prometheus on Port 8080."},
    ]

    # Perfect Hit at Rank 1
    r_perfect = recall_at_k(chunks[1:], ["port 9093"], k=4)
    mrr_perfect = mean_reciprocal_rank(chunks[1:], ["port 9093"], max_k=4)
    assert r_perfect == 1.0
    assert mrr_perfect == 1.0

    # Hit at Rank 2 (chunks[1] matches)
    r_rank2 = recall_at_k(chunks, ["port 9093"], k=4)
    mrr_rank2 = mean_reciprocal_rank(chunks, ["port 9093"], max_k=4)
    assert r_rank2 == 1.0
    assert mrr_rank2 == 0.5

    # Hit at Rank 3
    mrr_rank3 = mean_reciprocal_rank(chunks, ["port 9092"], max_k=4)
    assert mrr_rank3 == pytest.approx(1.0 / 3.0)

    # Zero Hit
    r_zero = recall_at_k(chunks, ["nonexistent term xyz"], k=4)
    mrr_zero = mean_reciprocal_rank(chunks, ["nonexistent term xyz"], max_k=4)
    assert r_zero == 0.0
    assert mrr_zero == 0.0

    # Edge cases: empty chunks or empty evidence
    assert recall_at_k([], ["any"]) == 0.0
    assert recall_at_k(chunks, []) == 1.0
    assert mean_reciprocal_rank([], ["any"]) == 0.0
    assert mean_reciprocal_rank(chunks, []) == 0.0


def test_citation_accuracy_and_non_numeric_page_guard():
    """Verify citation accuracy with page ranges and tolerance against non-numeric page values."""
    chunks = [
        {
            "metadata": {
                "source_pdf": "tech_manual.pdf",
                "page_number": 2,
                "page_end": 3,
            }
        },
        {
            "metadata": {
                "source_pdf": "compliance_policy.pdf",
                "page_number": "?",  # Corrupt / unknown page number
                "page_end": "?",
            }
        },
    ]

    # Matching document and overlapping page (Page 2 overlaps [2, 3])
    acc_match = citation_accuracy(chunks, "tech_manual.pdf", [2], k=4)
    assert acc_match == 1.0

    # Matching document but non-overlapping page (Page 4 not in [2, 3])
    acc_mismatch_page = citation_accuracy(chunks, "tech_manual.pdf", [4], k=4)
    assert acc_mismatch_page == 0.0

    # Non-matching document
    acc_mismatch_doc = citation_accuracy(chunks, "other.pdf", [2], k=4)
    assert acc_mismatch_doc == 0.0

    # Non-numeric page metadata must be handled gracefully without crashing
    acc_corrupt = citation_accuracy(chunks[1:], "compliance_policy.pdf", [1], k=4)
    assert acc_corrupt == 0.0


def test_abstention_confusion_matrix_and_f1():
    """Verify confusion matrix, accuracy, precision, recall, and F1 calculations."""
    items = [
        {"should_abstain": True, "abstained": True},    # TP
        {"should_abstain": True, "abstained": True},    # TP
        {"should_abstain": False, "abstained": True},   # FP (unwanted abstention)
        {"should_abstain": False, "abstained": False},  # TN
        {"should_abstain": False, "abstained": False},  # TN
        {"should_abstain": True, "abstained": False},   # FN (missed trap)
    ]

    conf = compute_abstention_confusion(items)
    assert conf["tp"] == 2
    assert conf["fp"] == 1
    assert conf["tn"] == 2
    assert conf["fn"] == 1

    # Precision = 2 / 3 = 0.6667
    assert conf["precision"] == pytest.approx(2.0 / 3.0, abs=1e-3)
    # Recall = 2 / 3 = 0.6667
    assert conf["recall"] == pytest.approx(2.0 / 3.0, abs=1e-3)
    # F1 = 0.6667
    assert conf["f1"] == pytest.approx(2.0 / 3.0, abs=1e-3)
    # Accuracy = 4 / 6 = 0.6667
    assert conf["accuracy"] == pytest.approx(4.0 / 6.0, abs=1e-3)

    # Edge cases: empty list, zero division guards
    empty_conf = compute_abstention_confusion([])
    assert empty_conf["f1"] == 0.0
    assert empty_conf["accuracy"] == 0.0


def test_answer_faithfulness():
    """Verify deterministic lexical token grounding calculation between answer and context."""
    context = (
        "AetherDB v3.4 Raft consensus communicates across Port 9093. "
        "Client applications connect to Port 9092 for read and write queries."
    )

    # Fully grounded answer
    grounded_ans = "Raft consensus communicates on Port 9093 while clients use Port 9092."
    f_high = compute_answer_faithfulness(grounded_ans, context)
    assert f_high >= 0.70

    # Model abstention statement is 100% faithful
    abstain_ans = "I cannot answer this based on the provided documents."
    assert compute_answer_faithfulness(abstain_ans, context) == 1.0

    # Hallucinated answer with zero context grounding
    hallucinated_ans = "Bananas grow on tropical trees in equatorial regions."
    f_low = compute_answer_faithfulness(hallucinated_ans, context)
    assert f_low == 0.0

    # Empty inputs
    assert compute_answer_faithfulness("", context) == 0.0
    assert compute_answer_faithfulness(grounded_ans, "") == 0.0


# ============================================================================
# 3. EVALUATION ENGINE INTEGRATION & REPORT GENERATION TEST
# ============================================================================


def test_evaluation_engine_mock_integration_across_all_modes(tmp_path: Path):
    """Test EvaluationEngine running on a synthetic mini-benchmark dataset across all 4 modes."""
    mini_dataset = [
        {
            "id": "Q001",
            "question": "What port does Raft gossip communicate over?",
            "category": "lexical",
            "expected_doc": "tech_manual.pdf",
            "expected_pages": [1],
            "expected_evidence": ["port 9093"],
            "should_abstain": False,
        },
        {
            "id": "Q002",
            "question": "What is the capital city of planet Neptune?",
            "category": "unanswerable",
            "expected_doc": None,
            "expected_pages": [],
            "expected_evidence": [],
            "should_abstain": True,
        },
    ]

    dataset_file = tmp_path / "mini_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(mini_dataset, f)

    # Mock VectorStore & Retriever
    mock_vs = MagicMock()
    mock_retriever = MagicMock()

    def mock_retrieve(query, initial_k=20, final_k=4, mode="hybrid_rerank"):
        if "neptune" in query.lower():
            return {
                "context": "",
                "results": [],
                "abstained": True,
                "abstention_reason": "LOW_SCORE",
                "top_score": -3.5,
                "mode": mode,
            }
        else:
            return {
                "context": "Raft consensus communicates on Port 9093.",
                "results": [
                    {
                        "text": "Raft consensus communicates on Port 9093.",
                        "parent_text": "AetherDB Cluster Guide: Raft consensus communicates on Port 9093.",
                        "score": 2.8 if "rerank" in mode else 0.85,
                        "metadata": {
                            "source_pdf": "tech_manual.pdf",
                            "page_number": 1,
                            "page_end": 1,
                        },
                    }
                ],
                "abstained": False,
                "abstention_reason": None,
                "top_score": 2.8,
                "mode": mode,
            }

    mock_retriever.retrieve = mock_retrieve

    engine = EvaluationEngine(
        vector_store=mock_vs,
        retriever=mock_retriever,
        dataset_path=dataset_file,
    )

    comparison = engine.compare_modes(
        modes=["vector_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"],
        limit=2,
        skip_generation=True,
    )

    assert len(comparison["modes"]) == 4
    rerank_metrics = comparison["modes"]["hybrid_rerank"]["metrics"]
    assert rerank_metrics["recall_at_4"] == 1.0
    assert rerank_metrics["mrr"] == 1.0
    assert rerank_metrics["abstention"]["f1"] == 1.0

    # Verify Markdown report and ASCII table formatting from evaluate.py CLI
    md_report = build_markdown_report(comparison)
    assert "# 📊 Project-RAG_LLM — Comprehensive Benchmark Evaluation Report" in md_report
    assert "`hybrid_rerank`" in md_report

    ascii_table = format_table(["Header A", "Header B"], [["val1", "val2"]])
    assert "Header A | Header B" in ascii_table
