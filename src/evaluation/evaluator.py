"""Automated RAG Evaluation Engine with Retrieval, Abstention, Citation & Faithfulness Metrics (L3)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from typing import Any

from loguru import logger

from src.config import settings
from src.database.vector_store import VectorStore
from src.retrieval.retriever import Retriever


def normalize_text(text: str) -> str:
    """Normalize whitespace and lower-case text for robust containment matching."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def recall_at_k(retrieved_chunks: list[dict[str, Any]], expected_evidence: list[str], k: int = 4) -> float:
    """Compute Recall@K: 1.0 if any of top-K chunks contain any expected evidence string, else 0.0."""
    if not expected_evidence:
        return 1.0
    if not retrieved_chunks:
        return 0.0

    top_k_chunks = retrieved_chunks[:k]
    for chunk in top_k_chunks:
        chunk_text = normalize_text(
            chunk.get("parent_text") or (chunk.get("metadata") or {}).get("parent_text") or chunk.get("text", "")
        )
        for ev in expected_evidence:
            norm_ev = normalize_text(ev)
            if norm_ev and norm_ev in chunk_text:
                return 1.0

    return 0.0


def mean_reciprocal_rank(
    retrieved_chunks: list[dict[str, Any]], expected_evidence: list[str], max_k: int = 10
) -> float:
    """Compute Reciprocal Rank: 1 / rank of the first chunk in top-max_k containing expected evidence."""
    if not expected_evidence or not retrieved_chunks:
        return 0.0

    for rank, chunk in enumerate(retrieved_chunks[:max_k], start=1):
        chunk_text = normalize_text(
            chunk.get("parent_text") or (chunk.get("metadata") or {}).get("parent_text") or chunk.get("text", "")
        )
        for ev in expected_evidence:
            norm_ev = normalize_text(ev)
            if norm_ev and norm_ev in chunk_text:
                return 1.0 / float(rank)

    return 0.0


def citation_accuracy(
    retrieved_results: list[dict[str, Any]],
    expected_doc: str | None,
    expected_pages: list[int],
    k: int = 4,
) -> float:
    """Compute Citation Accuracy: 1.0 if top-K retrieved chunk metadata matches document and page overlap."""
    if not expected_doc:
        return 1.0
    if not retrieved_results:
        return 0.0

    for hit in retrieved_results[:k]:
        meta = hit.get("metadata") or {}
        source = meta.get("source_pdf") or hit.get("source_pdf")
        if source and Path(source).name.lower() == Path(expected_doc).name.lower():
            if not expected_pages:
                return 1.0
            p_start = meta.get("page_number")
            p_end = meta.get("page_end", p_start)
            if p_start is not None and p_end is not None:
                try:
                    start_i = int(p_start)
                    end_i = int(p_end)
                    for exp_page in expected_pages:
                        if start_i <= int(exp_page) <= end_i:
                            return 1.0
                except (ValueError, TypeError):
                    continue

    return 0.0


def ndcg_at_k(
    retrieved_chunks: list[dict[str, Any]],
    expected_evidence: list[str],
    k: int = 4,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K).

    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    nDCG@K = DCG@K / IDCG@K
    """
    if not expected_evidence:
        return 1.0
    if not retrieved_chunks:
        return 0.0

    norm_evs = [normalize_text(ev) for ev in expected_evidence if ev.strip()]
    if not norm_evs:
        return 1.0

    rels: list[float] = []
    for chunk in retrieved_chunks[:k]:
        chunk_text = normalize_text(
            chunk.get("parent_text") or (chunk.get("metadata") or {}).get("parent_text") or chunk.get("text", "")
        )
        is_rel = any(ev in chunk_text for ev in norm_evs)
        rels.append(1.0 if is_rel else 0.0)

    # DCG@K: sum_{i=1}^K rel_i / log2(i + 1), where i is 1-indexed (i=1 -> idx=0 -> log2(2))
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels))

    # Ideal DCG: theoretical best ranking placing all expected relevant items at the top
    ideal_count = min(len(norm_evs), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_count))
    if idcg <= 0.0:
        return 0.0

    return round(min(1.0, dcg / idcg), 4)


def compute_harmonic_q_rag(
    recall: float,
    abstain_f1: float,
    citation_acc: float,
    faithfulness: float,
) -> float:
    """Harmonic Composite Quality Score (Q_RAG).

    Q_RAG = 4 / (1/Recall + 1/Abstain_F1 + 1/CitationAcc + 1/Faithfulness)
    Penalizes single-metric collapse to near-zero.
    """
    eps = 1e-6
    if (
        recall <= eps
        or abstain_f1 <= eps
        or citation_acc <= eps
        or faithfulness <= eps
    ):
        return 0.0

    denom = (1.0 / recall) + (1.0 / abstain_f1) + (1.0 / citation_acc) + (1.0 / faithfulness)
    return round(4.0 / denom, 4)


def compute_citation_metrics(
    retrieved_results: list[dict[str, Any]],
    expected_doc: str | None,
    expected_pages: list[int],
    k: int = 4,
) -> dict[str, float]:
    """Compute Citation Precision, Recall, and Citation F1 over top-K chunks.

    P_cite = |C_valid| / |C_total|
    R_cite = |C_valid| / |C_expected|
    F1_cite = 2 * P * R / (P + R)
    """
    if not expected_doc:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not retrieved_results:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    hits = retrieved_results[:k]
    c_total = len(hits)
    if c_total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    exp_doc_name = Path(expected_doc).name.lower()
    c_valid = 0
    for hit in hits:
        meta = hit.get("metadata") or {}
        source = meta.get("source_pdf") or hit.get("source_pdf")
        if source and Path(source).name.lower() == exp_doc_name:
            if not expected_pages:
                c_valid += 1
            else:
                p_start = meta.get("page_number")
                p_end = meta.get("page_end", p_start)
                if p_start is not None and p_end is not None:
                    try:
                        s_i = int(p_start)
                        e_i = int(p_end)
                        if any(s_i <= int(ep) <= e_i for ep in expected_pages):
                            c_valid += 1
                    except (ValueError, TypeError):
                        continue

    # P_cite = |C_valid| / |C_total|
    p_cite = c_valid / float(c_total)

    # R_cite = |C_valid| / |C_expected|
    c_expected = max(1, len(expected_pages)) if expected_pages else 1
    r_cite = min(1.0, c_valid / float(c_expected))

    # F1_cite = 2 * P * R / (P + R)
    if (p_cite + r_cite) > 0.0:
        f1_cite = (2.0 * p_cite * r_cite) / (p_cite + r_cite)
    else:
        f1_cite = 0.0

    return {
        "precision": round(p_cite, 4),
        "recall": round(r_cite, 4),
        "f1": round(f1_cite, 4),
    }


def compute_abstention_confusion(
    items: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Calculate confusion matrix (TP, FP, TN, FN) and Accuracy, Precision, Recall, F1 for abstention."""
    tp = fp = tn = fn = 0
    for it in items:
        should_abstain = bool(it.get("should_abstain", False))
        abstained = bool(it.get("abstained", False))
        if should_abstain and abstained:
            tp += 1
        elif not should_abstain and abstained:
            fp += 1
        elif not should_abstain and not abstained:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    if total == 0:
        return {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def compute_answer_faithfulness(answer: str, context: str) -> float:
    """Compute deterministic lexical token & sentence grounding between generated answer and context."""
    if not answer or not answer.strip():
        return 0.0

    norm_ans = normalize_text(answer)
    if "cannot answer this based on the provided documents" in norm_ans:
        return 1.0

    if not context or not context.strip():
        return 0.0

    norm_ctx = normalize_text(context)
    words = [w for w in re.findall(r"\b[a-z0-9_-]+\b", norm_ans) if len(w) > 2]
    if not words:
        return 1.0

    matches = sum(1 for w in words if w in norm_ctx)
    return round(matches / float(len(words)), 4)


class EvaluationEngine:
    """Evaluates RAG retrieval modes against ground-truth datasets with multi-metric scoring."""

    def __init__(
        self,
        vector_store: VectorStore,
        retriever: Retriever,
        generator: Any = None,
        dataset_path: Path | str | None = None,
    ):
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator
        self.dataset_path = Path(dataset_path or settings.eval_dataset_path)
        self.dataset = self.load_dataset(self.dataset_path)

    @staticmethod
    def load_dataset(path: Path | str) -> list[dict[str, Any]]:
        """Load and validate the 100-QA benchmark dataset JSON."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at: {target}")

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Benchmark dataset must be a JSON array of QA items")

        logger.info("Loaded {} benchmark QA items from {}", len(data), target)
        return data

    def evaluate_mode(
        self,
        mode: str = "hybrid_rerank",
        limit: int | None = None,
        skip_generation: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a single retrieval mode over the benchmark dataset."""
        logger.info("Starting benchmark evaluation for mode '{}' (limit={})...", mode, limit)
        eval_items = self.dataset[:limit] if limit is not None else self.dataset

        results: list[dict[str, Any]] = []
        latencies: list[float] = []

        for item in eval_items:
            qid = item["id"]
            question = item["question"]
            category = item["category"]
            expected_doc = item.get("expected_doc")
            expected_pages = item.get("expected_pages") or []
            expected_evidence = item.get("expected_evidence") or []
            should_abstain = bool(item.get("should_abstain", False))

            t0 = time.perf_counter()
            ret_outcome = self.retriever.retrieve(
                query=question,
                initial_k=20,
                final_k=4,
                mode=mode,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            latencies.append(elapsed_ms)

            retrieved_chunks = ret_outcome.get("results") or []
            abstained = bool(ret_outcome.get("abstained", False))
            abstention_reason = ret_outcome.get("abstention_reason")
            context = ret_outcome.get("context", "")

            # If question is answerable, compute retrieval metrics
            if not should_abstain:
                r_at_4 = recall_at_k(retrieved_chunks, expected_evidence, k=4)
                r_at_10 = recall_at_k(retrieved_chunks, expected_evidence, k=10)
                mrr = mean_reciprocal_rank(retrieved_chunks, expected_evidence, max_k=10)
                ndcg = ndcg_at_k(retrieved_chunks, expected_evidence, k=4)
                cite_acc = citation_accuracy(retrieved_chunks, expected_doc, expected_pages, k=4)
                cite_mets = compute_citation_metrics(retrieved_chunks, expected_doc, expected_pages, k=4)
            else:
                # For unanswerable trap questions, retrieval recall is N/A or matched when abstained
                r_at_4 = 1.0 if abstained else 0.0
                r_at_10 = 1.0 if abstained else 0.0
                mrr = 1.0 if abstained else 0.0
                ndcg = 1.0 if abstained else 0.0
                cite_acc = 1.0 if abstained else 0.0
                cite_mets = {
                    "precision": 1.0 if abstained else 0.0,
                    "recall": 1.0 if abstained else 0.0,
                    "f1": 1.0 if abstained else 0.0,
                }

            # Answer generation & faithfulness
            answer = ""
            faithfulness = 1.0 if abstained else 0.0
            if not skip_generation and self.generator:
                if abstained:
                    answer = "I cannot answer this based on the provided documents."
                    faithfulness = 1.0
                else:
                    try:
                        answer = self.generator.generate(question, context)
                        faithfulness = compute_answer_faithfulness(answer, context)
                    except Exception as gen_err:
                        logger.error("Generation failed for {}: {}", qid, gen_err)
                        answer = f"Error: {gen_err}"
                        faithfulness = 0.0

            item_result = {
                "id": qid,
                "question": question,
                "category": category,
                "should_abstain": should_abstain,
                "abstained": abstained,
                "abstention_reason": abstention_reason,
                "recall_at_4": r_at_4,
                "recall_at_10": r_at_10,
                "mrr": mrr,
                "ndcg_at_4": ndcg,
                "citation_accuracy": cite_acc,
                "citation_precision": cite_mets["precision"],
                "citation_recall": cite_mets["recall"],
                "citation_f1": cite_mets["f1"],
                "faithfulness": faithfulness,
                "latency_ms": elapsed_ms,
                "answer": answer,
                "top_score": ret_outcome.get("top_score"),
                "retrieved_count": len(retrieved_chunks),
            }
            results.append(item_result)

        # Compute aggregate metrics
        summary = self._aggregate_metrics(mode, results, latencies)
        return summary

    def _aggregate_metrics(
        self, mode: str, results: list[dict[str, Any]], latencies: list[float]
    ) -> dict[str, Any]:
        """Aggregate item results into overall and per-category metric summaries."""
        total_items = len(results)
        answerable_items = [r for r in results if not r["should_abstain"]]
        unanswerable_items = [r for r in results if r["should_abstain"]]

        # Overall retrieval metrics (over answerable items)
        if answerable_items:
            overall_r4 = sum(r["recall_at_4"] for r in answerable_items) / len(answerable_items)
            overall_r10 = sum(r["recall_at_10"] for r in answerable_items) / len(answerable_items)
            overall_mrr = sum(r["mrr"] for r in answerable_items) / len(answerable_items)
            overall_ndcg = sum(r["ndcg_at_4"] for r in answerable_items) / len(answerable_items)
            overall_cite = sum(r["citation_accuracy"] for r in answerable_items) / len(answerable_items)
            overall_cite_p = sum(r["citation_precision"] for r in answerable_items) / len(answerable_items)
            overall_cite_r = sum(r["citation_recall"] for r in answerable_items) / len(answerable_items)
            overall_cite_f1 = sum(r["citation_f1"] for r in answerable_items) / len(answerable_items)
        else:
            overall_r4 = overall_r10 = overall_mrr = overall_ndcg = overall_cite = overall_cite_p = overall_cite_r = overall_cite_f1 = 1.0

        overall_faith = (
            sum(r["faithfulness"] for r in results) / total_items if total_items > 0 else 1.0
        )

        # Abstention metrics across all items
        abstention_conf = compute_abstention_confusion(results)
        abstain_f1 = float(abstention_conf.get("f1", 0.0))

        # Harmonic Composite Quality Score (Q_RAG)
        q_rag = compute_harmonic_q_rag(overall_r4, abstain_f1, overall_cite, overall_faith)

        # Latency metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_lat = sorted(latencies)
        p50_latency = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0.0
        p95_latency = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)] if sorted_lat else 0.0

        # Category breakdowns
        categories: dict[str, Any] = {}
        for cat in ("semantic", "lexical", "ocr", "unanswerable"):
            cat_results = [r for r in results if r["category"] == cat]
            if not cat_results:
                continue

            if cat == "unanswerable":
                cat_abstain = sum(1 for r in cat_results if r["abstained"]) / len(cat_results)
                categories[cat] = {
                    "count": len(cat_results),
                    "abstention_rate": round(cat_abstain, 4),
                    "recall_at_4": round(cat_abstain, 4),
                    "mrr": round(cat_abstain, 4),
                    "ndcg_at_4": round(cat_abstain, 4),
                }
            else:
                cat_r4 = sum(r["recall_at_4"] for r in cat_results) / len(cat_results)
                cat_mrr = sum(r["mrr"] for r in cat_results) / len(cat_results)
                cat_ndcg = sum(r["ndcg_at_4"] for r in cat_results) / len(cat_results)
                cat_cite = sum(r["citation_accuracy"] for r in cat_results) / len(cat_results)
                cat_cite_f1 = sum(r["citation_f1"] for r in cat_results) / len(cat_results)
                categories[cat] = {
                    "count": len(cat_results),
                    "recall_at_4": round(cat_r4, 4),
                    "mrr": round(cat_mrr, 4),
                    "ndcg_at_4": round(cat_ndcg, 4),
                    "citation_accuracy": round(cat_cite, 4),
                    "citation_f1": round(cat_cite_f1, 4),
                }

        return {
            "mode": mode,
            "total_items": total_items,
            "answerable_count": len(answerable_items),
            "unanswerable_count": len(unanswerable_items),
            "metrics": {
                "recall_at_4": round(overall_r4, 4),
                "recall_at_10": round(overall_r10, 4),
                "mrr": round(overall_mrr, 4),
                "ndcg_at_4": round(overall_ndcg, 4),
                "citation_accuracy": round(overall_cite, 4),
                "citation_precision": round(overall_cite_p, 4),
                "citation_recall": round(overall_cite_r, 4),
                "citation_f1": round(overall_cite_f1, 4),
                "faithfulness": round(overall_faith, 4),
                "q_rag": round(q_rag, 4),
                "abstention": abstention_conf,
            },
            "latency": {
                "avg_ms": round(avg_latency, 2),
                "p50_ms": round(p50_latency, 2),
                "p95_ms": round(p95_latency, 2),
            },
            "categories": categories,
            "item_results": results,
        }

    def compare_modes(
        self,
        modes: list[str] | None = None,
        limit: int | None = None,
        skip_generation: bool = True,
    ) -> dict[str, Any]:
        """Evaluate and compare multiple retrieval modes side-by-side."""
        active_modes = modes or ["vector_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"]
        logger.info("Executing Benchmark comparison across modes: {}", active_modes)

        comparison_results: dict[str, Any] = {}
        for m in active_modes:
            mode_summary = self.evaluate_mode(mode=m, limit=limit, skip_generation=skip_generation)
            comparison_results[m] = mode_summary

        return {
            "evaluated_modes": active_modes,
            "dataset_size": limit if limit is not None else len(self.dataset),
            "modes": comparison_results,
        }
