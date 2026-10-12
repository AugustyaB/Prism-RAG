"""Phase 2.5 Automated Pytest Suite — Observability, Cost Tracking, Rate Limits & CI/CD (Sec 4 & 6).

Empirically validates:
1. X-Request-ID Distributed Tracing: Auto-generation, header propagation & echo across endpoints.
2. Sliding-Window Rate Limiting: 429 Too Many Requests, Retry-After header, window reset & bypass endpoints.
3. Stage-Level Latency Spans: Timers for embed, search, rerank, and llm in query responses.
4. Token Counter & Cost Calculation: Formula accuracy, edge cases (empty text, case-insensitive model catalog).
5. QueryTelemetry Persistence & Fallback: Safe DB writes without disrupting query execution.
6. Multi-Tenant Aggregated Metrics: /api/metrics calculations (p50/p90/p95/p99/avg) and RBAC tenant isolation.
7. Resilience & Provider Failure Handling: Graceful handling of LLM errors and Tier-1 short-circuiting.
8. Infrastructure Configuration Integrity: Validates Dockerfile, docker-compose.yml, and .github/workflows/ci.yml.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.api.server import app, rate_limiter
from src.auth.security import create_access_token, seed_demo_accounts
from src.config import settings
from src.database.models import Base, Organization, QueryTelemetry, User
from src.database.relational import SessionLocal, init_db
from src.utils.telemetry import (
    TelemetryCollector,
    TelemetrySpan,
    _compute_percentiles,
    calculate_query_cost,
    count_tokens,
    get_aggregated_metrics,
    record_query_telemetry,
)


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Ensure relational database and demo accounts are initialized."""
    init_db()
    seed_demo_accounts()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    rate_limiter.reset()
    return TestClient(app)


def get_auth_token(client: TestClient, email: str = "admin@default.org", pwd: str = "adminpassword123") -> str:
    """Helper to fetch a valid Bearer token."""
    resp = client.post("/api/auth/login", json={"email": email, "password": pwd})
    assert resp.status_code == 200, f"Login failed for {email}: {resp.text}"
    return resp.json()["access_token"]


# ==============================================================================
# 1. X-Request-ID Correlation & Tracing
# ==============================================================================


def test_x_request_id_correlation_tracing(client: TestClient):
    """Verify that requests auto-generate an X-Request-ID if missing and preserve client-supplied IDs."""
    # 1. Auto-generated request ID when client sends none
    resp1 = client.get("/health")
    assert resp1.status_code == 200
    assert "X-Request-ID" in resp1.headers
    generated_id = resp1.headers["X-Request-ID"]
    assert len(generated_id) >= 16

    # 2. Preserved custom client request ID
    custom_id = "test-custom-span-corr-9988"
    resp2 = client.get("/health", headers={"X-Request-ID": custom_id})
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Request-ID") == custom_id

    # 3. Preserved across authenticated routes
    token = get_auth_token(client)
    custom_me_id = f"trace-auth-me-{uuid.uuid4()}"
    resp3 = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}", "X-Request-ID": custom_me_id},
    )
    assert resp3.status_code == 200
    assert resp3.headers.get("X-Request-ID") == custom_me_id

    # 4. Preserved across 404 error routes
    resp_404 = client.get("/non_existent_route", headers={"X-Request-ID": "err-trace-404"})
    assert resp_404.status_code == 404
    assert resp_404.headers.get("X-Request-ID") == "err-trace-404"


# ==============================================================================
# 2. Sliding-Window Rate Limiting & Bypass Rules
# ==============================================================================


def test_sliding_window_rate_limiter_enforcement_and_bypass(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Verify rate limiter blocks bursts exceeding limit with 429 and Retry-After, while bypassing health/docs."""
    rate_limiter.reset()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_requests_per_minute", 5)

    token = get_auth_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    rate_limiter.reset()

    # 1. Send requests up to limit
    for _ in range(5):
        resp = client.get("/api/auth/me", headers=headers)
        assert resp.status_code == 200

    # 2. Next request must be rate-limited (HTTP 429)
    rate_limited_resp = client.get("/api/auth/me", headers=headers)
    assert rate_limited_resp.status_code == 429
    assert "Retry-After" in rate_limited_resp.headers
    retry_after = int(rate_limited_resp.headers["Retry-After"])
    assert retry_after > 0
    assert "Rate limit exceeded" in rate_limited_resp.json()["detail"]

    # 3. Health check and docs must bypass rate limiting even when client is blocked
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200

    # 4. Rate limiter reset clears block
    rate_limiter.reset()
    unblocked_resp = client.get("/api/auth/me", headers=headers)
    assert unblocked_resp.status_code == 200

    # 5. When rate_limit_enabled is False, bursts pass freely
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    for _ in range(10):
        resp = client.get("/api/auth/me", headers=headers)
        assert resp.status_code == 200


# ==============================================================================
# 3. Stage-Level Latency Spans & Telemetry Collector Primitives
# ==============================================================================


def test_telemetry_collector_and_span_primitives():
    """Verify TelemetrySpan context timer and TelemetryCollector aggregation behavior."""
    collector = TelemetryCollector(request_id="custom-req-001")
    assert collector.request_id == "custom-req-001"

    # Timer span
    with collector.start_span("embed") as span:
        time.sleep(0.01)  # 10ms
    assert span.elapsed_ms >= 5.0
    collector.record_stage("embed", span.elapsed_ms)
    assert collector.embed_latency_ms == span.elapsed_ms

    collector.record_stage("search", 15.5)
    collector.record_stage("rerank", 45.2)
    collector.record_stage("llm", 120.8)

    collector.prompt_tokens = 150
    collector.completion_tokens = 50
    collector.estimated_cost_usd = 0.000123
    collector.retrieved_count = 4
    collector.top_rerank_score = 0.85

    payload = collector.to_dict()
    assert payload["request_id"] == "custom-req-001"
    assert payload["embed_latency_ms"] >= 5.0
    assert payload["search_latency_ms"] == 15.5
    assert payload["rerank_latency_ms"] == 45.2
    assert payload["llm_latency_ms"] == 120.8
    assert payload["total_latency_ms"] > 0
    assert payload["prompt_tokens"] == 150
    assert payload["completion_tokens"] == 50
    assert payload["total_tokens"] == 200
    assert payload["estimated_cost_usd"] == 0.000123
    assert payload["retrieved_count"] == 4
    assert payload["top_rerank_score"] == 0.85
    assert payload["abstained"] is False


# ==============================================================================
# 4. Token Counting & Cost Estimation Mathematical Engine
# ==============================================================================


def test_token_counter_and_cost_estimation_math():
    """Verify count_tokens heuristics and calculate_query_cost pricing lookups."""
    # 1. Token counter edge cases
    assert count_tokens("") == 0
    assert count_tokens("   \n\t  ") == 0
    assert count_tokens("a") == 1
    assert count_tokens("hello world") >= 2

    long_text = "Enterprise architecture requires multi-tenant data isolation and low-latency retrieval." * 10
    tokens = count_tokens(long_text)
    assert 150 <= tokens <= 300

    # 2. Local model pricing (zero USD)
    assert calculate_query_cost("lmstudio", "gpt-oss-20b", prompt_tokens=500, completion_tokens=200) == 0.0
    assert calculate_query_cost("ollama", "llama3", prompt_tokens=1000, completion_tokens=500) == 0.0

    # 3. Hosted model pricing
    # gemini-2.5-flash: prompt $0.075 / 1M, completion $0.30 / 1M
    cost_gemini = calculate_query_cost("gemini", "gemini-2.5-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert round(cost_gemini, 3) == 0.375

    # Case-insensitive lookup
    cost_case = calculate_query_cost("GEMINI", "GEMINI-2.5-FLASH", prompt_tokens=1_000_000, completion_tokens=0)
    assert round(cost_case, 3) == 0.075

    # 4. Unknown model fallback
    assert calculate_query_cost("unknown_provider", "non_existent_model", prompt_tokens=100, completion_tokens=50) == 0.0


# ==============================================================================
# 5. QueryTelemetry Database Persistence & Safe Failure Fallback
# ==============================================================================


def test_query_telemetry_db_persistence_and_failure_fallback():
    """Verify QueryTelemetry records commit cleanly and handles DB errors gracefully."""
    rec_id = str(uuid.uuid4())
    data = {
        "id": rec_id,
        "user_id": "usr_test_1",
        "org_id": "test_org_persist",
        "query_text": "Sample query for telemetry test",
        "retrieval_mode": "hybrid_rerank",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash",
        "embed_latency_ms": 12.3,
        "search_latency_ms": 8.1,
        "rerank_latency_ms": 35.6,
        "llm_latency_ms": 180.2,
        "total_latency_ms": 236.2,
        "retrieved_count": 3,
        "top_rerank_score": 0.912,
        "abstained": False,
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "estimated_cost_usd": 0.000022,
        "cited_doc_ids": ["doc_hash_1", "doc_hash_2"],
    }

    # Successful commit
    res_id = record_query_telemetry(SessionLocal, data)
    assert res_id == rec_id

    # Verify query in DB
    with SessionLocal() as session:
        stored = session.scalar(select(QueryTelemetry).where(QueryTelemetry.id == rec_id))
        assert stored is not None
        assert stored.query_text == "Sample query for telemetry test"
        assert stored.total_latency_ms == 236.2
        assert stored.cited_doc_ids == ["doc_hash_1", "doc_hash_2"]

    # Resilience: corrupted session factory does not raise, returns empty string
    broken_factory = MagicMock(side_effect=RuntimeError("Simulated DB lock failure"))
    fallback_res = record_query_telemetry(broken_factory, data)
    assert fallback_res == ""


# ==============================================================================
# 6. Aggregated Metrics Math & Multi-Tenant Isolation
# ==============================================================================


def test_percentile_math_and_empty_metrics():
    """Verify percentile calculation math and zero-query edge case handling."""
    # Empty list
    empty_p = _compute_percentiles([])
    assert empty_p == {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}

    # Single value
    single_p = _compute_percentiles([100.0])
    assert single_p["p50"] == 100.0
    assert single_p["avg"] == 100.0

    # 10 values [10, 20, ..., 100]
    vals = [float(i * 10) for i in range(1, 11)]
    p = _compute_percentiles(vals)
    assert p["p50"] == 55.0
    assert p["avg"] == 55.0
    assert p["p90"] == 91.0

    # Empty DB metrics
    dummy_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=dummy_engine)
    MemSession = sessionmaker(bind=dummy_engine)
    with MemSession() as session:
        m = get_aggregated_metrics(session, org_id="empty_org")
        assert m["total_queries"] == 0
        assert m["abstained_queries"] == 0
        assert m["abstention_rate"] == 0.0
        assert m["latencies"]["total"]["p50"] == 0.0
        assert m["tokens"]["total_tokens"] == 0


def test_get_metrics_endpoint_rbac_tenant_isolation(client: TestClient):
    """Verify /api/metrics aggregates records and strictly enforces tenant boundaries."""
    # Seed telemetry records for two distinct orgs
    unique_suffix = uuid.uuid4().hex[:6]
    org1 = f"metrics_org_alpha_{unique_suffix}"
    org2 = f"metrics_org_beta_{unique_suffix}"

    with SessionLocal() as session:
        session.add(Organization(id=org1, name=f"Alpha Metrics Org {unique_suffix}"))
        session.add(Organization(id=org2, name=f"Beta Metrics Org {unique_suffix}"))
        session.commit()

    # Record 2 queries for Org Alpha, 1 query for Org Beta
    record_query_telemetry(SessionLocal, {
        "id": str(uuid.uuid4()), "org_id": org1, "query_text": "q1",
        "retrieval_mode": "hybrid_rerank", "llm_provider": "gemini",
        "embed_latency_ms": 10.0, "search_latency_ms": 15.0, "rerank_latency_ms": 20.0,
        "llm_latency_ms": 100.0, "total_latency_ms": 145.0, "prompt_tokens": 100,
        "completion_tokens": 50, "estimated_cost_usd": 0.00005, "abstained": False,
    })
    record_query_telemetry(SessionLocal, {
        "id": str(uuid.uuid4()), "org_id": org1, "query_text": "q2",
        "retrieval_mode": "vector_only", "llm_provider": "gemini",
        "embed_latency_ms": 10.0, "search_latency_ms": 10.0, "rerank_latency_ms": 0.0,
        "llm_latency_ms": 0.0, "total_latency_ms": 20.0, "prompt_tokens": 0,
        "completion_tokens": 0, "estimated_cost_usd": 0.0, "abstained": True,
    })
    record_query_telemetry(SessionLocal, {
        "id": str(uuid.uuid4()), "org_id": org2, "query_text": "q3",
        "retrieval_mode": "hybrid_rrf", "llm_provider": "lmstudio",
        "embed_latency_ms": 12.0, "search_latency_ms": 18.0, "rerank_latency_ms": 0.0,
        "llm_latency_ms": 80.0, "total_latency_ms": 110.0, "prompt_tokens": 80,
        "completion_tokens": 30, "estimated_cost_usd": 0.0, "abstained": False,
    })

    # Test as ADMIN of default_org:
    admin_token = get_auth_token(client, "admin@default.org", "adminpassword123")
    # ADMIN filtering by Org Alpha
    resp_alpha = client.get(f"/api/metrics?org_id={org1}", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp_alpha.status_code == 200
    data_alpha = resp_alpha.json()
    assert data_alpha["total_queries"] == 2
    assert data_alpha["abstained_queries"] == 1
    assert data_alpha["abstention_rate"] == 0.5
    assert data_alpha["tokens"]["prompt_tokens"] == 100

    # Test as VIEWER of tenant_b: cannot see Org Alpha even if requested
    viewer_token = get_auth_token(client, "viewer@tenant_b.org", "tenantbviewer123")
    resp_tamper = client.get(f"/api/metrics?org_id={org1}", headers={"Authorization": f"Bearer {viewer_token}"})
    assert resp_tamper.status_code == 200
    tamper_data = resp_tamper.json()
    # Must be forced to tenant_b, not org1
    assert tamper_data["total_queries"] == 0 or all(
        tamper_data["total_queries"] != data_alpha["total_queries"],
    )


# ==============================================================================
# 7. LLM Provider Failure & Tier-1 Short-Circuiting Resilience
# ==============================================================================


def test_generator_failure_and_abstention_resilience(client: TestClient):
    """Verify that LLM generation errors return clean responses and Tier-1 abstention skips LLM."""
    token = get_auth_token(client)
    headers = {"Authorization": f"Bearer {token}", "X-Request-ID": "test-resilience-span"}

    # 1. Tier-1 Abstention on trap question: short-circuits LLM, records telemetry with 0 prompt/completion tokens
    trap_resp = client.post(
        "/api/query",
        json={"query": "What is the orbital period of Saturn's ring Titan?", "mode": "hybrid_rerank"},
        headers=headers,
    )
    assert trap_resp.status_code == 200
    trap_data = trap_resp.json()
    assert trap_data["abstained"] is True
    assert trap_data["request_id"] == "test-resilience-span"
    assert trap_data["telemetry"]["prompt_tokens"] == 0
    assert trap_data["telemetry"]["completion_tokens"] == 0
    assert trap_data["telemetry"]["llm_latency_ms"] == 0.0

    # 2. Generator Exception Handling
    mock_generator = MagicMock()
    mock_generator.generate.side_effect = RuntimeError("Simulated remote LLM API timeout")
    mock_generator.model_name = "mock-failing-llm"

    with patch("src.api.server.get_generator", return_value=mock_generator):
        with patch("src.retrieval.retriever.Retriever.retrieve", return_value={
            "context": "Sample available context snippet.",
            "results": [{"text": "Sample available context snippet.", "score": 0.88, "metadata": {"page_number": 1}}],
            "abstained": False,
            "abstention_reason": None,
            "top_score": 0.88,
        }):
            gen_err_resp = client.post(
                "/api/query",
                json={"query": "Valid query text", "mode": "hybrid_rerank"},
                headers=headers,
            )
            assert gen_err_resp.status_code == 200
            err_data = gen_err_resp.json()
            assert "Error generating response" in err_data["answer"]
            assert err_data["telemetry"]["completion_tokens"] == 0


# ==============================================================================
# 8. Infrastructure & CI Configuration File Integrity
# ==============================================================================


def test_docker_and_ci_configuration_integrity():
    """Verify that Dockerfile, docker-compose.yml, and .github/workflows/ci.yml exist and contain required directives."""
    root = Path(__file__).resolve().parent.parent

    # 1. Dockerfile
    dockerfile = root / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile must exist at repository root"
    content = dockerfile.read_text(encoding="utf-8")
    assert "FROM python:3.13" in content
    assert "EXPOSE 8000" in content
    assert "HEALTHCHECK" in content
    assert "/health" in content
    assert "uvicorn" in content

    # 2. docker-compose.yml
    compose_file = root / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml must exist at repository root"
    compose_content = compose_file.read_text(encoding="utf-8")
    assert "qdrant:" in compose_content
    assert "rag-api:" in compose_content
    assert "8000:8000" in compose_content
    assert "6333:6333" in compose_content

    # 3. GitHub Actions CI
    ci_file = root / ".github" / "workflows" / "ci.yml"
    assert ci_file.exists(), ".github/workflows/ci.yml must exist"
    ci_content = ci_file.read_text(encoding="utf-8")
    assert "name: CI Pipeline" in ci_content
    assert "python scripts/check_architecture_layers.py" in ci_content
    assert "pytest tests/ -v" in ci_content
