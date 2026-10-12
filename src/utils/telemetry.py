"""Query observability spans, token counter, model pricing engine, and metric aggregations (L3)."""

from __future__ import annotations

import math
import time
from typing import Any
import uuid

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.database.models import QueryTelemetry


class TelemetrySpan:
    """Context manager and timer measuring stage execution duration in milliseconds."""

    def __init__(self, name: str | None = None):
        self.name = name
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> TelemetrySpan:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed_ms = round((time.perf_counter() - self.start_time) * 1000.0, 3)


class TelemetryCollector:
    """Aggregates pipeline stage latencies, token consumption, and cost accounting."""

    def __init__(self, request_id: str | None = None):
        self.request_id: str = request_id or str(uuid.uuid4())
        self.embed_latency_ms: float = 0.0
        self.search_latency_ms: float = 0.0
        self.rerank_latency_ms: float = 0.0
        self.llm_latency_ms: float = 0.0
        self.total_latency_ms: float = 0.0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.estimated_cost_usd: float = 0.0
        self.retrieved_count: int = 0
        self.top_rerank_score: float | None = None
        self.abstained: bool = False
        self.abstention_reason: str | None = None
        self._start_total: float = time.perf_counter()

    def start_span(self, name: str) -> TelemetrySpan:
        """Create a timer span for a named pipeline stage."""
        return TelemetrySpan(name)

    def record_stage(self, stage: str, elapsed_ms: float) -> None:
        """Record stage execution duration in milliseconds."""
        if stage == "embed":
            self.embed_latency_ms = round(elapsed_ms, 3)
        elif stage == "search":
            self.search_latency_ms = round(elapsed_ms, 3)
        elif stage == "rerank":
            self.rerank_latency_ms = round(elapsed_ms, 3)
        elif stage == "llm":
            self.llm_latency_ms = round(elapsed_ms, 3)
        elif stage == "total":
            self.total_latency_ms = round(elapsed_ms, 3)

    def finalize_total(self) -> float:
        """Finalize total end-to-end execution latency in milliseconds."""
        if self.total_latency_ms == 0.0:
            self.total_latency_ms = round((time.perf_counter() - self._start_total) * 1000.0, 3)
        return self.total_latency_ms

    def to_dict(self) -> dict[str, Any]:
        """Format a clean dictionary representation for API responses and persistence."""
        self.finalize_total()
        return {
            "request_id": self.request_id,
            "embed_latency_ms": self.embed_latency_ms,
            "search_latency_ms": self.search_latency_ms,
            "rerank_latency_ms": self.rerank_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "retrieved_count": self.retrieved_count,
            "top_rerank_score": self.top_rerank_score,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
        }


def count_tokens(text: str) -> int:
    """Fast, deterministic token estimation (~4 chars/token heuristic with word calibration)."""
    if not text:
        return 0
    clean_text = text.strip()
    if not clean_text:
        return 0
    char_tokens = max(1.0, len(clean_text) / 4.0)
    word_count = len(clean_text.split())
    word_tokens = max(1.0, word_count * 1.33)
    estimate = int(round((char_tokens * 0.6) + (word_tokens * 0.4)))
    return max(1, estimate)


def calculate_query_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate estimated USD query cost using settings.model_pricing_catalog."""
    catalog = settings.model_pricing_catalog
    pricing = catalog.get(model) or catalog.get(provider)
    if not pricing:
        # Case-insensitive lookup
        for key, p in catalog.items():
            if key.lower() == model.lower() or key.lower() == provider.lower():
                pricing = p
                break
    if not pricing:
        return 0.0

    prompt_rate = pricing.get("prompt_per_1m", 0.0) / 1_000_000.0
    completion_rate = pricing.get("completion_per_1m", 0.0) / 1_000_000.0
    cost = (prompt_tokens * prompt_rate) + (completion_tokens * completion_rate)
    return round(cost, 8)


def record_query_telemetry(
    session_factory: Any,
    telemetry_data: dict[str, Any],
) -> str:
    """Safely commit a QueryTelemetry record to metadata.db without failing queries."""
    if not settings.telemetry_enabled:
        return ""
    record_id = telemetry_data.get("id") or str(uuid.uuid4())
    try:
        with session_factory() as session:
            record = QueryTelemetry(
                id=record_id,
                user_id=telemetry_data.get("user_id"),
                org_id=telemetry_data.get("org_id", "default_org"),
                query_text=telemetry_data.get("query_text", ""),
                retrieval_mode=telemetry_data.get("retrieval_mode", "hybrid_rerank"),
                llm_provider=telemetry_data.get("llm_provider", "lmstudio"),
                llm_model=str(telemetry_data.get("llm_model", "gpt-oss-20b")),
                embed_latency_ms=float(telemetry_data.get("embed_latency_ms", 0.0)),
                search_latency_ms=float(telemetry_data.get("search_latency_ms", 0.0)),
                rerank_latency_ms=float(telemetry_data.get("rerank_latency_ms", 0.0)),
                llm_latency_ms=float(telemetry_data.get("llm_latency_ms", 0.0)),
                total_latency_ms=float(telemetry_data.get("total_latency_ms", 0.0)),
                retrieved_count=int(telemetry_data.get("retrieved_count", 0)),
                top_rerank_score=(
                    float(telemetry_data["top_rerank_score"])
                    if telemetry_data.get("top_rerank_score") is not None
                    else None
                ),
                abstained=bool(telemetry_data.get("abstained", False)),
                abstention_reason=telemetry_data.get("abstention_reason"),
                prompt_tokens=int(telemetry_data.get("prompt_tokens", 0)),
                completion_tokens=int(telemetry_data.get("completion_tokens", 0)),
                estimated_cost_usd=float(telemetry_data.get("estimated_cost_usd", 0.0)),
                cited_doc_ids=telemetry_data.get("cited_doc_ids", []),
            )
            session.add(record)
            session.commit()
            return record.id
    except Exception as exc:
        logger.warning(f"Telemetry persistence caught safely: {exc}")
        return ""


def _compute_percentiles(values: list[float]) -> dict[str, float]:
    """Calculate p50, p90, p95, p99, and avg for a list of values."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _p(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_vals[int(k)])
        return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))

    return {
        "p50": round(_p(0.50), 2),
        "p90": round(_p(0.90), 2),
        "p95": round(_p(0.95), 2),
        "p99": round(_p(0.99), 2),
        "avg": round(sum(sorted_vals) / n, 2),
    }


def get_aggregated_metrics(session: Session, org_id: str | None = None) -> dict[str, Any]:
    """Compute aggregate observability, latency percentiles, and cost telemetry."""
    stmt = select(QueryTelemetry)
    if org_id:
        stmt = stmt.where(QueryTelemetry.org_id == org_id)

    records = list(session.scalars(stmt).all())
    total_queries = len(records)
    if total_queries == 0:
        return {
            "total_queries": 0,
            "abstained_queries": 0,
            "abstention_rate": 0.0,
            "latencies": {
                "total": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0},
                "embed": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0},
                "search": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0},
                "rerank": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0},
                "llm": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0},
            },
            "tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "costs": {
                "total_estimated_cost_usd": 0.0,
            },
            "by_mode": {},
            "by_provider": {},
        }

    abstained_count = sum(1 for r in records if r.abstained)
    abstention_rate = round(abstained_count / total_queries, 4)

    total_prompt_tokens = sum(r.prompt_tokens for r in records)
    total_completion_tokens = sum(r.completion_tokens for r in records)
    total_estimated_cost_usd = round(sum(r.estimated_cost_usd for r in records), 6)

    by_mode: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    for r in records:
        by_mode[r.retrieval_mode] = by_mode.get(r.retrieval_mode, 0) + 1
        by_provider[r.llm_provider] = by_provider.get(r.llm_provider, 0) + 1

    return {
        "total_queries": total_queries,
        "abstained_queries": abstained_count,
        "abstention_rate": abstention_rate,
        "latencies": {
            "total": _compute_percentiles([r.total_latency_ms for r in records]),
            "embed": _compute_percentiles([r.embed_latency_ms for r in records]),
            "search": _compute_percentiles([r.search_latency_ms for r in records]),
            "rerank": _compute_percentiles([r.rerank_latency_ms for r in records]),
            "llm": _compute_percentiles([r.llm_latency_ms for r in records]),
        },
        "tokens": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "costs": {
            "total_estimated_cost_usd": total_estimated_cost_usd,
        },
        "by_mode": by_mode,
        "by_provider": by_provider,
    }
