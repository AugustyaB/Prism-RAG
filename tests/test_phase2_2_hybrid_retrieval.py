"""Phase 2.2 QA Test Suite: Cross-Page Parent-Child Chunking, Hybrid BM25+RRF, Parent Dedup & Two-Tier Abstention."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.api.server as server_module
import src.database.relational as relational_module
import src.jobs.ingestion_worker as worker_module
from src.config import settings
from src.database.models import Base, Organization
from src.database.vector_store import RRF_K, VectorStore
from src.embeddings.embedder import BGE_QUERY_PREFIX, LocalEmbedder
from src.embeddings.sparse_bm25 import SparseBM25Encoder
from src.ingestion.chunker import SemanticChunker, TextChunk
from src.ingestion.classifier import PageType
from src.ingestion.pipeline import IngestionPipeline, IngestionResult
from src.jobs.ingestion_worker import IngestionJobManager
from src.retrieval.retriever import Retriever


class HybridTestEmbedder:
    """Deterministic 384-d embedder & domain-aware Cross-Encoder simulator for QA testing."""

    def __init__(self) -> None:
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.embed_query_calls: list[str] = []
        self.embed_batch_calls: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            return []
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(384)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            return []
        self.embed_query_calls.append(query)
        prefixed = f"{BGE_QUERY_PREFIX}{query.strip()}"
        return self.embed_text(prefixed)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls.append(list(texts))
        return [self.embed_text(t) for t in texts]

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Simulate Cross-Encoder logits: negative (< -1.5) for out-of-domain queries, positive for matches."""
        enc = SparseBM25Encoder()
        q_tokens = set(enc.tokenize(query))
        scores: list[float] = []
        for t in texts:
            t_tokens = set(enc.tokenize(t))
            overlap = len(q_tokens & t_tokens)
            if overlap == 0:
                scores.append(-4.25)  # Out-of-domain logit below min_rerank_score (-1.5)
            else:
                scores.append(1.5 + float(overlap) * 1.2)
        return scores


@pytest.fixture()
def hybrid_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up an isolated Qdrant v2 hybrid store, SQLite DB, Retriever, and FastAPI TestClient."""
    db_file = tmp_path / "hybrid_meta.db"
    qdrant_dir = tmp_path / "qdrant_v2"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_file.as_posix()}")
    monkeypatch.setattr(settings, "vector_db_dir", qdrant_dir)
    monkeypatch.setattr(settings, "pdf_upload_dir", upload_dir)
    monkeypatch.setattr(settings, "min_rerank_score", -1.5)
    monkeypatch.setattr(settings, "auth_enabled", False)

    test_engine = create_engine(
        f"sqlite:///{db_file.as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    TestSessionLocal = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )

    monkeypatch.setattr(relational_module, "engine", test_engine)
    monkeypatch.setattr(relational_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(worker_module, "SessionLocal", TestSessionLocal)

    Base.metadata.create_all(bind=test_engine)
    with TestSessionLocal() as session:
        session.add(Organization(id="default_org", name="Default Enterprise Organization"))
        session.commit()

    embedder = HybridTestEmbedder()
    vector_store = VectorStore(embedder)
    retriever = Retriever(vector_store)
    pipeline = IngestionPipeline()
    job_manager = IngestionJobManager(pipeline, vector_store)

    monkeypatch.setattr(server_module, "embedder", embedder)
    monkeypatch.setattr(server_module, "vector_store", vector_store)
    monkeypatch.setattr(server_module, "retriever", retriever)
    monkeypatch.setattr(server_module, "ingestion_pipeline", pipeline)
    monkeypatch.setattr(server_module, "job_manager", job_manager)

    client = TestClient(server_module.app)
    yield {
        "tmp_path": tmp_path,
        "embedder": embedder,
        "vector_store": vector_store,
        "retriever": retriever,
        "client": client,
        "job_manager": job_manager,
    }

    job_manager.executor.shutdown(wait=True)
    try:
        vector_store.client.close()
    except Exception:
        pass
    test_engine.dispose()


# ============================================================================
# 1. CROSS-PAGE PARENT-CHILD CHUNKING & BOUNDARY INTEGRITY
# ============================================================================


def test_cross_page_parent_child_chunking_and_boundary_integrity():
    """Verify sentences crossing page breaks stay unified in parent_text with page_number=1, page_end=2."""
    chunker = SemanticChunker(
        parent_size=1200,
        parent_overlap=150,
        child_size=300,
        child_overlap=50,
    )

    page_1_text = (
        "1.1 ENTERPRISE SECURITY PROTOCOL\n\n"
        "All cryptographic keys inside the production cluster must be rotated every 90 days "
        "in accordance with policy POL-SEC-104.2. Furthermore, any automated worker node "
        "that encounters a hardware HSM fault across the primary page boundary must immediately"
    )
    page_2_text = (
        "trigger emergency failover routine _hsm_failover_dispatch within 500 milliseconds "
        "and emit diagnostic alert ERR-HSM-9001 to the central telemetry collector. "
        "Subsequent audit logs are archived in immutable cold storage for seven years."
    )

    chunks = chunker.chunk_pages(
        [
            (1, page_1_text, "security_spec.pdf"),
            (2, page_2_text, "security_spec.pdf"),
        ]
    )

    assert len(chunks) >= 2
    first_parent_text = chunks[0].metadata["parent_text"]

    # Cross-page sentence must be unbroken inside the unified parent block
    assert "boundary must immediately trigger emergency failover routine _hsm_failover_dispatch" in first_parent_text
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["page_end"] == 2
    assert chunks[0].metadata["parent_id"] == "security_spec.pdf::parent_0"
    assert chunks[0].metadata.get("section_header") == "1.1 ENTERPRISE SECURITY PROTOCOL"

    for idx, ch in enumerate(chunks):
        assert ch.metadata["chunk_index"] == idx
        assert ch.char_count <= 300
        assert len(ch.metadata["parent_text"]) >= ch.char_count


# ============================================================================
# 2. BGE v1.5 QUERY INSTRUCTION PREFIX & SPARSE BM25 ENCODER
# ============================================================================


def test_bge_query_instruction_prefix_and_empty_guards():
    """Verify LocalEmbedder.embed_query prepends BGE_QUERY_PREFIX only to queries, not document batches."""
    embedder = LocalEmbedder()

    with patch.object(embedder, "embed_text", return_value=[0.1] * 384) as spy_embed_text:
        res = embedder.embed_query("how does RRF fusion work?")
        assert len(res) == 384
        spy_embed_text.assert_called_once_with(
            f"{BGE_QUERY_PREFIX}how does RRF fusion work?"
        )

    # Already-prefixed query must not be double-prefixed
    with patch.object(embedder, "embed_text", return_value=[0.1] * 384) as spy_embed_text:
        embedder.embed_query(f"{BGE_QUERY_PREFIX}already prefixed")
        spy_embed_text.assert_called_once_with(f"{BGE_QUERY_PREFIX}already prefixed")

    # Blank/whitespace query returns [] without calling model
    assert embedder.embed_query("") == []
    assert embedder.embed_query("   ") == []


def test_sparse_bm25_encoder_identifiers_saturation_and_stopword_fallback():
    """Verify SparseBM25Encoder preserves technical codes, saturates TF weights, and handles stopword-only queries."""
    encoder = SparseBM25Encoder(k1=1.2, b=0.75, avg_dl=50.0)

    tokens = encoder.tokenize("Error ERR-CV-4092 in _render_page_to_image violates POL-SEC-104.2")
    assert "err-cv-4092" in tokens
    assert "4092" in tokens
    assert "_render_page_to_image" in tokens
    assert "render" in tokens
    assert "pol-sec-104.2" in tokens

    # BM25 saturation check: 10x term repetition stays strictly bounded below k1 + 1 (2.2)
    _, vals_single = encoder.encode_document("err-cv-4092")
    _, vals_repeated = encoder.encode_document(" ".join(["err-cv-4092"] * 10))
    assert vals_repeated[0] > vals_single[0]
    assert vals_repeated[0] < 2.2

    # Stopword-only query fallback: must not degenerate to empty vector ([], [])
    sw_indices, sw_values = encoder.encode_query("what is the")
    assert len(sw_indices) > 0
    assert len(sw_values) == len(sw_indices)


# ============================================================================
# 3. HYBRID SEARCH (4 MODES), RRF (k=60) & EXACT IDENTIFIER RECALL
# ============================================================================


def test_hybrid_search_all_four_modes_and_exact_identifier_recall(hybrid_env: dict):
    """Verify bm25_only, hybrid_rrf, and hybrid_rerank rank exact error code ERR-CV-4092 #1."""
    vector_store: VectorStore = hybrid_env["vector_store"]
    chunker = SemanticChunker()

    pages = [
        (
            1,
            "System Architecture Overview: The ingestion worker pool processes PDF pages asynchronously "
            "using PyMuPDF and stores metadata inside SQLAlchemy relational tables.",
            "ops_manual.pdf",
        ),
        (
            2,
            " Critical Diagnostic Reference: When subsystem code ERR-CV-4092 triggers during EasyOCR "
            "pixmap allocation, the worker invokes _render_page_to_image at half DPI per policy POL-SEC-104.2.",
            "ops_manual.pdf",
        ),
        (
            3,
            "Deployment Guidelines: Production containers run behind uvicorn workers with health checks "
            "and automated Qdrant collection backups every twenty-four hours.",
            "ops_manual.pdf",
        ),
    ]
    chunks = chunker.chunk_pages(pages)
    ing_result = IngestionResult(
        source_pdf="ops_manual.pdf",
        file_hash="hash_ops_manual_v2",
        total_pages=3,
        classification_summary={PageType.TEXT: 3, PageType.IMAGE: 0, PageType.MIXED: 0},
        total_chunks=len(chunks),
        chunks=chunks,
    )
    vector_store.add_document(ing_result)

    for mode in ("bm25_only", "hybrid_rrf", "hybrid_rerank"):
        hits = vector_store.search("ERR-CV-4092 POL-SEC-104.2", initial_k=10, final_k=3, mode=mode)
        assert len(hits) >= 1, f"No results returned for mode={mode}"
        assert "ERR-CV-4092" in hits[0]["parent_text"], f"Mode {mode} failed to rank ERR-CV-4092 #1"

    # Also verify vector_only mode executes cleanly and returns valid schema
    vec_hits = vector_store.search("ingestion worker pool PyMuPDF", initial_k=10, final_k=3, mode="vector_only")
    assert len(vec_hits) >= 1
    assert "parent_id" in vec_hits[0]
    assert "parent_text" in vec_hits[0]


# ============================================================================
# 4. PARENT BLOCK DEDUPLICATION
# ============================================================================


def test_parent_deduplication_eliminates_sibling_child_redundancy(hybrid_env: dict):
    """Verify multiple matching child chunks sharing the same parent_id are deduplicated in search results."""
    vector_store: VectorStore = hybrid_env["vector_store"]

    shared_parent_id = "dedup_doc.pdf::parent_0"
    shared_parent_text = (
        "Parent Block Zero: Quantum telemetry calibration requires token QTM-8899 across all "
        "four child segments so every child chunk matches QTM-8899 with high BM25 and dense scores."
    )

    sibling_chunks = [
        TextChunk(
            text=f"Child segment {i} discussing QTM-8899 quantum telemetry calibration.",
            char_count=68,
            metadata={
                "source_pdf": "dedup_doc.pdf",
                "page_number": 1,
                "page_end": 2,
                "chunk_index": i,
                "parent_id": shared_parent_id,
                "parent_text": shared_parent_text,
            },
        )
        for i in range(4)
    ]

    ing_result = IngestionResult(
        source_pdf="dedup_doc.pdf",
        file_hash="hash_dedup_doc",
        total_pages=2,
        classification_summary={PageType.TEXT: 2, PageType.IMAGE: 0, PageType.MIXED: 0},
        total_chunks=len(sibling_chunks),
        chunks=sibling_chunks,
    )
    vector_store.add_document(ing_result)

    results = vector_store.search("QTM-8899 quantum telemetry", initial_k=10, final_k=4, mode="hybrid_rerank")
    parent_ids = [r["parent_id"] for r in results]
    assert parent_ids.count(shared_parent_id) == 1, f"Duplicate parent_id found in results: {parent_ids}"


# ============================================================================
# 5. TWO-TIER ABSTENTION GATE (EMPTY_RETRIEVAL, LOW_SCORE & LLM_ABSTAIN)
# ============================================================================


def test_two_tier_abstention_gate_and_api_short_circuit(hybrid_env: dict):
    """Verify Pre-LLM (EMPTY_RETRIEVAL, LOW_SCORE) and Post-LLM (LLM_ABSTAIN) abstention gates."""
    client: TestClient = hybrid_env["client"]
    vector_store: VectorStore = hybrid_env["vector_store"]

    # 1. Tier 1a: Empty Vector DB triggers EMPTY_RETRIEVAL without calling LLM
    mock_gen = MagicMock()
    with patch("src.api.server.get_generator", return_value=mock_gen):
        empty_resp = client.post(
            "/api/query",
            json={"query": "What is policy POL-SEC-104.2?", "mode": "hybrid_rerank"},
        )
    assert empty_resp.status_code == 200
    empty_data = empty_resp.json()
    assert empty_data["abstained"] is True
    assert empty_data["abstention_reason"] == "EMPTY_RETRIEVAL"
    assert empty_data["citations"] == []
    mock_gen.generate.assert_not_called()

    # Index a technical document into Qdrant
    chunk = TextChunk(
        text="Policy POL-SEC-104.2 mandates hardware key rotation every 90 days.",
        char_count=66,
        metadata={
            "source_pdf": "policy.pdf",
            "page_number": 4,
            "page_end": 5,
            "chunk_index": 0,
            "parent_id": "policy.pdf::parent_0",
            "parent_text": "Policy POL-SEC-104.2 mandates hardware key rotation every 90 days across pages 4-5.",
        },
    )
    vector_store.add_document(
        IngestionResult(
            source_pdf="policy.pdf",
            file_hash="hash_policy_pdf",
            total_pages=5,
            classification_summary={PageType.TEXT: 5, PageType.IMAGE: 0, PageType.MIXED: 0},
            total_chunks=1,
            chunks=[chunk],
        )
    )

    # 2. Tier 1b: Out-of-domain trap question triggers LOW_SCORE Pre-LLM Abstention and skips LLM
    mock_gen.reset_mock()
    with patch("src.api.server.get_generator", return_value=mock_gen):
        trap_resp = client.post(
            "/api/query",
            json={
                "query": "What is the orbital velocity of Jupiter's moon Europa?",
                "mode": "hybrid_rerank",
            },
        )
    assert trap_resp.status_code == 200
    trap_data = trap_resp.json()
    assert trap_data["abstained"] is True
    assert trap_data["abstention_reason"] == "LOW_SCORE"
    assert trap_data["citations"] == []
    assert trap_data["top_score"] < settings.min_rerank_score
    mock_gen.generate.assert_not_called()

    # 3. In-domain query passes Tier 1, invokes LLM, and formats cross-page range (Page: 4-5)
    mock_gen.reset_mock()
    mock_gen.generate.return_value = "Keys must be rotated every 90 days per POL-SEC-104.2 [policy.pdf, Page 4-5]."
    with patch("src.api.server.get_generator", return_value=mock_gen):
        ok_resp = client.post(
            "/api/query",
            json={"query": "How often does POL-SEC-104.2 require key rotation?", "mode": "hybrid_rerank"},
        )
    assert ok_resp.status_code == 200
    ok_data = ok_resp.json()
    assert ok_data["abstained"] is False
    assert ok_data["abstention_reason"] is None
    assert len(ok_data["citations"]) == 1
    assert ok_data["citations"][0]["page"] == 4
    assert ok_data["citations"][0]["page_end"] == 5
    assert ok_data["citations"][0]["parent_id"] == "policy.pdf::parent_0"
    mock_gen.generate.assert_called_once()
    passed_context = mock_gen.generate.call_args[0][1]
    assert "(Page: 4-5)" in passed_context

    # 4. Tier 2: Post-LLM Abstention Gate when LLM responds that it cannot answer from context
    mock_gen.reset_mock()
    mock_gen.generate.return_value = "I cannot answer this based on the provided documents."
    with patch("src.api.server.get_generator", return_value=mock_gen):
        tier2_resp = client.post(
            "/api/query",
            json={"query": "What is the CEO's home address in POL-SEC-104.2?", "mode": "hybrid_rerank"},
        )
    assert tier2_resp.status_code == 200
    tier2_data = tier2_resp.json()
    assert tier2_data["abstained"] is True
    assert tier2_data["abstention_reason"] == "LLM_ABSTAIN"
    assert tier2_data["citations"] == []


# ============================================================================
# 6. QA EDGE CASES: LEGACY SCHEMA MIGRATION, RERANKER FALLBACK & API VALIDATION
# ============================================================================


def test_legacy_collection_migration_reranker_fallback_and_query_validation(hybrid_env: dict):
    """Verify legacy unnamed Qdrant schema auto-migration, reranker failure fallback, and 400 validations."""
    client: TestClient = hybrid_env["client"]
    vector_store: VectorStore = hybrid_env["vector_store"]
    embedder: HybridTestEmbedder = hybrid_env["embedder"]

    # 1. Legacy v1 collection schema (unnamed vector) is automatically upgraded to v2 named hybrid schema
    legacy_col = "legacy_v1_upgrade_test"
    vector_store.client.create_collection(
        collection_name=legacy_col,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    vector_store._ensure_hybrid_collection(legacy_col, 384)
    col_info = vector_store.client.get_collection(legacy_col)
    assert isinstance(col_info.config.params.vectors, dict)
    assert "dense" in col_info.config.params.vectors

    # 2. Index a chunk and simulate Cross-Encoder exception -> falls back to Stage-1 RRF scores
    chunk = TextChunk(
        text="Fallback test chunk with identifier ERR-FB-777.",
        char_count=47,
        metadata={
            "source_pdf": "fallback.pdf",
            "page_number": 1,
            "page_end": 1,
            "chunk_index": 0,
            "parent_id": "fallback.pdf::parent_0",
            "parent_text": "Fallback test chunk with identifier ERR-FB-777.",
        },
    )
    vector_store.add_document(
        IngestionResult(
            source_pdf="fallback.pdf",
            file_hash="hash_fallback",
            total_pages=1,
            classification_summary={PageType.TEXT: 1, PageType.IMAGE: 0, PageType.MIXED: 0},
            total_chunks=1,
            chunks=[chunk],
        )
    )

    with patch.object(embedder, "rerank", side_effect=RuntimeError("CUDA out of memory")):
        fb_hits = vector_store.search("ERR-FB-777", mode="hybrid_rerank")
        assert len(fb_hits) == 1
        assert fb_hits[0]["score"] == pytest.approx(2.0 / (RRF_K + 1.0), rel=1e-3)

    # 3. Invalid retrieval mode raises 400 Bad Request on /api/query
    bad_mode_resp = client.post(
        "/api/query",
        json={"query": "ERR-FB-777", "mode": "non_existent_mode"},
    )
    assert bad_mode_resp.status_code == 400
    assert "Unsupported retrieval mode" in bad_mode_resp.json()["detail"]

    # 4. Blank / whitespace-only query raises 400 Bad Request on /api/query
    blank_resp = client.post(
        "/api/query",
        json={"query": "   ", "mode": "hybrid_rerank"},
    )
    assert blank_resp.status_code == 400
    assert "empty" in blank_resp.json()["detail"].lower()
