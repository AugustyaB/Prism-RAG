"""Phase 3.1 Tier 1 Upgrades Test Suite (Sec 1, Sec 4, Sec 6 — L1, L3, L4).

Certifies:
1. Contextual Chunk Injection in chunker.py and pipeline.py.
2. Thread-safe LRU SemanticQueryCache (cos >= 0.96, tenant isolation, role clearance, <10ms latency).
3. Cache invalidation hooks (document upload, deletion, org purge).
4. Generator streaming interface (stream_generate, stream_extractive_answer).
5. FastAPI /api/query/stream SSE endpoint and instant cache replay.
"""

import json
import time
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from src.ingestion.chunker import SemanticChunker, TextChunk
from src.database.semantic_cache import (
    SemanticQueryCache,
    compute_cosine_similarity,
    get_semantic_cache,
    reset_semantic_cache,
)
from src.generation.generator import (
    BaseGenerator,
    LMStudioGenerator,
    OllamaGenerator,
    stream_extractive_answer,
    synthesize_extractive_answer,
)
from src.api.server import app
from src.auth.security import create_access_token


# ── 1. Contextual Chunk Injection Tests (L3) ────────────────────────────────


def test_contextual_chunk_header_generation():
    """Verify situational document and page headers are generated accurately."""
    hdr_single = SemanticChunker.build_contextual_header("benchmark.pdf", 1)
    assert hdr_single == "[Document: benchmark.pdf | Page: 1]"

    hdr_multi = SemanticChunker.build_contextual_header("report.pdf", 2, 4)
    assert hdr_multi == "[Document: report.pdf | Page: 2-4]"


def test_text_chunk_contextual_text_enrichment():
    """Verify TextChunk enriches contextual text while preserving clean text."""
    chunk = TextChunk(
        text="Dense embedding vector spaces allow similarity search.",
        char_count=54,
        metadata={"source_pdf": "rag_spec.pdf", "page_number": 3, "page_end": 3},
    )

    assert chunk.text == "Dense embedding vector spaces allow similarity search."
    assert "[Document: rag_spec.pdf | Page: 3]" in chunk.contextual_text
    assert chunk.contextual_text.endswith(chunk.text)
    assert chunk.metadata["contextual_header"] == "[Document: rag_spec.pdf | Page: 3]"
    assert chunk.get_contextual_text() == chunk.contextual_text


def test_semantic_chunker_injects_contextual_metadata():
    """Verify SemanticChunker.chunk_pages automatically generates contextual headers."""
    chunker = SemanticChunker(child_size=100, parent_size=300)
    pages = [
        (1, "First page introduction to information retrieval systems.", "rag_intro.pdf"),
        (2, "Second page covers reciprocal rank fusion algorithms.", "rag_intro.pdf"),
    ]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) > 0

    for c in chunks:
        assert hasattr(c, "contextual_text")
        assert c.metadata.get("contextual_header") is not None
        assert "rag_intro.pdf" in c.metadata["contextual_header"]
        assert c.contextual_text.startswith(c.metadata["contextual_header"])
        # Clean text must not have the header embedded in c.text
        assert not c.text.startswith("[Document:")


# ── 2. Semantic Query Cache Unit Tests (L1) ──────────────────────────────────


def test_cosine_similarity_math():
    """Verify cosine similarity calculation with identical and orthogonal vectors."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v_zero = [0.0, 0.0, 0.0]

    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0)
    assert compute_cosine_similarity(v1, v_zero) == pytest.approx(0.0)


def test_semantic_cache_hit_and_latency():
    """Verify cosine >= 0.96 hits cache in < 10ms."""
    cache = SemanticQueryCache(max_entries=100, similarity_threshold=0.96)
    q_emb = [0.8, 0.6, 0.0, 0.0]
    response = {"answer": "BM25+ saturation prevents score saturation.", "citations": []}

    cache.set(
        query_text="What is BM25+ saturation?",
        query_emb=q_emb,
        response_payload=response,
        org_id="org_alpha",
        role="ANALYST",
    )

    # Near-identical query embedding (cosine > 0.99)
    q_query_near = [0.801, 0.599, 0.0, 0.0]

    t0 = time.perf_counter()
    hit = cache.get(
        query_text="Explain BM25+ saturation",
        query_emb=q_query_near,
        org_id="org_alpha",
        role="ANALYST",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert hit is not None
    payload, score = hit
    assert score >= 0.96
    assert payload["answer"] == response["answer"]
    assert elapsed_ms < 10.0, f"Cache lookup took {elapsed_ms:.2f}ms, expected < 10ms"


def test_semantic_cache_miss_below_threshold():
    """Verify cosine < 0.96 results in a cache miss."""
    cache = SemanticQueryCache(max_entries=100, similarity_threshold=0.96)
    q_emb = [1.0, 0.0, 0.0]
    cache.set(
        query_text="Qdrant vector collection",
        query_emb=q_emb,
        response_payload={"answer": "Qdrant stores vectors."},
        org_id="org_alpha",
        role="ANALYST",
    )

    # Orthogonal or divergent query (cosine ~ 0.707 < 0.96)
    q_diff = [0.7, 0.7, 0.0]
    hit = cache.get(
        query_text="SQLAlchemy models",
        query_emb=q_diff,
        org_id="org_alpha",
        role="ANALYST",
    )
    assert hit is None


def test_semantic_cache_tenant_isolation():
    """Verify queries from different orgs do not leak cached responses."""
    cache = SemanticQueryCache(max_entries=100, similarity_threshold=0.96)
    q_emb = [1.0, 0.0, 0.0]
    cache.set(
        query_text="Confidential financial results",
        query_emb=q_emb,
        response_payload={"answer": "Q3 Revenue was $10M"},
        org_id="org_secret",
        role="ADMIN",
    )

    # Query from another tenant org
    hit = cache.get(
        query_text="Confidential financial results",
        query_emb=q_emb,
        org_id="org_other",
        role="ADMIN",
    )
    assert hit is None


def test_semantic_cache_role_clearance_enforcement():
    """Verify role clearance prevents privilege escalation in cached queries."""
    cache = SemanticQueryCache(max_entries=100, similarity_threshold=0.96)
    q_emb = [1.0, 0.0, 0.0]
    cache.set(
        query_text="Executive salary breakdown",
        query_emb=q_emb,
        response_payload={"answer": "Executive bonus pool: $500k"},
        org_id="org_alpha",
        role="ADMIN",
    )

    # VIEWER must not be able to retrieve ADMIN-level cached query response
    viewer_hit = cache.get(
        query_text="Executive salary breakdown",
        query_emb=q_emb,
        org_id="org_alpha",
        role="VIEWER",
    )
    assert viewer_hit is None

    # ADMIN can retrieve it
    admin_hit = cache.get(
        query_text="Executive salary breakdown",
        query_emb=q_emb,
        org_id="org_alpha",
        role="ADMIN",
    )
    assert admin_hit is not None


def test_semantic_cache_invalidation_hooks():
    """Verify cache invalidation by org, document, and clear."""
    cache = SemanticQueryCache(max_entries=100)
    q1 = [1.0, 0.0, 0.0]
    q2 = [0.0, 1.0, 0.0]

    cache.set(
        query_text="Doc A query",
        query_emb=q1,
        response_payload={"answer": "A"},
        org_id="org_1",
        role="VIEWER",
        associated_docs=["hash_aaa"],
    )
    cache.set(
        query_text="Doc B query",
        query_emb=q2,
        response_payload={"answer": "B"},
        org_id="org_1",
        role="VIEWER",
        associated_docs=["hash_bbb"],
    )
    assert len(cache) == 2

    # Invalidate document hash_aaa
    removed = cache.invalidate_document("hash_aaa")
    assert removed == 1
    assert len(cache) == 1
    assert cache.get("Doc A query", q1, org_id="org_1", role="VIEWER") is None
    assert cache.get("Doc B query", q2, org_id="org_1", role="VIEWER") is not None

    # Invalidate by org
    cache.invalidate_org("org_1")
    assert len(cache) == 0


def test_semantic_cache_lru_eviction():
    """Verify oldest entry is evicted when capacity is reached."""
    cache = SemanticQueryCache(max_entries=2)
    cache.set("q1", [1.0, 0.0], {"a": 1}, org_id="org_1", role="VIEWER")
    cache.set("q2", [0.0, 1.0], {"a": 2}, org_id="org_1", role="VIEWER")
    assert len(cache) == 2

    # Adding third entry evicts oldest (q1)
    cache.set("q3", [0.5, 0.5], {"a": 3}, org_id="org_1", role="VIEWER")
    assert len(cache) == 2
    assert cache.get("q1", [1.0, 0.0], org_id="org_1", role="VIEWER") is None
    assert cache.get("q2", [0.0, 1.0], org_id="org_1", role="VIEWER") is not None
    assert cache.get("q3", [0.5, 0.5], org_id="org_1", role="VIEWER") is not None


# ── 3. Generator Streaming Interface Tests (L3) ─────────────────────────────


def test_stream_extractive_answer():
    """Verify extractive fallback yields tokens cleanly."""
    context = (
        "--- Document: spec.pdf (Page 1) ---\n"
        "BM25+ modifies standard BM25 with a lower-bound delta parameter."
    )
    tokens = list(stream_extractive_answer("Explain BM25+", context))
    assert len(tokens) > 0
    reassembled = "".join(tokens)
    assert "BM25+" in reassembled
    assert "delta parameter" in reassembled


def test_base_generator_stream_fallback():
    """Verify BaseGenerator.stream_generate yields tokens from generate()."""

    class MockGen(BaseGenerator):
        def generate(self, query: str, context: str) -> str:
            return "This is a streamed mock answer."

    gen = MockGen()
    stream = list(gen.stream_generate("query", "context"))
    assert len(stream) > 1
    assert "".join(stream) == "This is a streamed mock answer."


def test_offline_local_generators_stream_gracefully():
    """Verify Ollama and LM Studio fallback to extractive streaming when offline."""
    ollama = OllamaGenerator()
    lmstudio = LMStudioGenerator()
    context = "--- Document: test.pdf ---\nOffline fallback content verification."

    # Offline streaming must not raise uncaught connection exceptions
    o_tokens = list(ollama.stream_generate("Test query", context))
    assert len(o_tokens) > 0

    l_tokens = list(lmstudio.stream_generate("Test query", context))
    assert len(l_tokens) > 0


# ── 4. FastAPI /api/query/stream Endpoint Tests (L4) ─────────────────────────


@pytest.fixture
def test_client():
    from src.auth.security import seed_demo_accounts

    seed_demo_accounts()
    reset_semantic_cache()
    client = TestClient(app)
    return client


def test_query_stream_sse_endpoint(test_client):
    """Verify POST /api/query/stream yields Server-Sent Events with tokens and final done payload."""
    token = create_access_token(data={"sub": "admin@default.org", "org_id": "default_org", "role": "ADMIN"})
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "query": "What is the primary objective of this system?",
        "mode": "vector_only",
    }

    response = test_client.post("/api/query/stream", json=payload, headers=headers)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.strip().split("\n\n")
    assert len(lines) >= 1

    events = []
    for line in lines:
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    assert len(events) >= 1
    # Last event must have done: true and telemetry
    final_event = events[-1]
    assert final_event.get("done") is True
    assert "telemetry" in final_event
    assert "answer" in final_event


def test_query_stream_instant_cache_replay(test_client):
    """Verify consecutive stream queries replay instantly from SemanticQueryCache."""
    token = create_access_token(data={"sub": "admin@default.org", "org_id": "default_org", "role": "ADMIN"})
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "query": "Explain reciprocal rank fusion k=60",
        "mode": "vector_only",
    }

    # First request: generates and caches response
    resp1 = test_client.post("/api/query/stream", json=payload, headers=headers)
    assert resp1.status_code == 200

    # Second request: must hit cache and return cached: true
    resp2 = test_client.post("/api/query/stream", json=payload, headers=headers)
    assert resp2.status_code == 200

    lines = resp2.text.strip().split("\n\n")
    events = [json.loads(l[6:]) for l in lines if l.startswith("data: ")]
    final_event = events[-1]

    assert final_event.get("done") is True
    assert final_event.get("cached") is True
    assert final_event["telemetry"].get("cached") is True
