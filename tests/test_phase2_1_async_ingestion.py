"""Phase 2.1 QA Test Suite: Core Refactor, Relational DB, Async Ingestion, Fallbacks & Edge Cases."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import src.api.server as server_module
import src.database.relational as relational_module
import src.jobs.ingestion_worker as worker_module
from src.config import settings
from src.database.models import Base, DocumentRecord, IngestionJob, Organization
from src.database.vector_store import VectorStore
from src.ingestion.ocr import OcrEngine, OcrTimeoutError
from src.ingestion.pdf_parser import PdfParser
from src.ingestion.pipeline import IngestionPipeline
from src.jobs.ingestion_worker import IngestionJobManager
from src.retrieval.retriever import Retriever


class DeterministicEmbedder:
    """Fast deterministic 384-d embedder and reranker for isolated QA & CI testing."""

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            return []
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(384)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        q_words = set(query.lower().split())
        scores: list[float] = []
        for t in texts:
            t_words = set(t.lower().split())
            overlap = len(q_words & t_words)
            scores.append(float(overlap))
        return scores


def create_sample_pdf(path: Path, num_pages: int = 3, password: str | None = None) -> Path:
    """Generate a multi-page synthetic PDF with rich text for chunking and ingestion tests."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        body = (
            f"Enterprise RAG Architecture Specification — Page {i + 1}\n\n"
            f"Section {i + 1}.1: Asynchronous Ingestion & SHA-256 Deduplication.\n"
            "When a document is uploaded through POST /api/upload, the system computes a "
            "cryptographic SHA-256 digest of the binary payload before scheduling OCR or "
            "vector indexing on the bounded ThreadPoolExecutor worker pool.\n\n"
            f"Section {i + 1}.2: Relational Persistence & Qdrant Vector Rollback.\n"
            "All job transitions (PENDING -> PROCESSING -> COMPLETED or FAILED) are persisted "
            "in SQLAlchemy metadata tables while dense chunk embeddings are stored in Qdrant."
        )
        page.insert_text((50, 72), body, fontsize=11)

    if password:
        doc.save(
            str(path),
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
        )
    else:
        doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Configure an isolated SQLite DB, Qdrant directory, upload directory, and FastAPI services."""
    db_file = tmp_path / "test_metadata.db"
    qdrant_dir = tmp_path / "qdrant_test"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_file.as_posix()}")
    monkeypatch.setattr(settings, "vector_db_dir", qdrant_dir)
    monkeypatch.setattr(settings, "pdf_upload_dir", upload_dir)
    monkeypatch.setattr(settings, "max_job_retries", 2)

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

    embedder = DeterministicEmbedder()
    vector_store = VectorStore(embedder)
    pipeline = IngestionPipeline()
    job_manager = IngestionJobManager(pipeline, vector_store)
    retriever = Retriever(vector_store)

    monkeypatch.setattr(server_module, "embedder", embedder)
    monkeypatch.setattr(server_module, "vector_store", vector_store)
    monkeypatch.setattr(server_module, "ingestion_pipeline", pipeline)
    monkeypatch.setattr(server_module, "job_manager", job_manager)
    monkeypatch.setattr(server_module, "retriever", retriever)

    client = TestClient(server_module.app)
    yield {
        "tmp_path": tmp_path,
        "client": client,
        "session_factory": TestSessionLocal,
        "vector_store": vector_store,
        "pipeline": pipeline,
        "job_manager": job_manager,
    }

    job_manager.executor.shutdown(wait=True)
    try:
        vector_store.client.close()
    except Exception:
        pass
    test_engine.dispose()


# ============================================================================
# 1. WAVE 1 CORE REFACTOR TESTS: Single fitz.open, Progress Callback & OCR Timeout
# ============================================================================


def test_single_fitz_open_and_progress_callback(tmp_path: Path):
    """Verify IngestionPipeline.process_pdf opens the PDF strictly once and emits stage callbacks."""
    pdf_path = create_sample_pdf(tmp_path / "single_open.pdf", num_pages=4)
    pipeline = IngestionPipeline()
    progress_events: list[tuple[int, int, str]] = []

    real_fitz_open = fitz.open
    with patch("src.ingestion.pipeline.fitz.open", wraps=real_fitz_open) as spy_open:
        result = pipeline.process_pdf(
            pdf_path,
            progress_callback=lambda done, total, stage: progress_events.append((done, total, stage)),
        )
        assert spy_open.call_count == 1, f"Expected fitz.open called once, got {spy_open.call_count}"

    assert result.total_pages == 4
    assert result.total_chunks > 0
    assert result.errors == []
    assert progress_events[0] == (0, 4, "PARSING")
    assert progress_events[-1] == (4, 4, "CHUNKING")
    assert any(done == 4 and stage == "PARSING" for done, _, stage in progress_events)


def test_ocr_timeout_enforcement_and_graceful_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify OcrEngine enforces ocr_timeout_seconds on both full-page and MIXED region OCR."""
    monkeypatch.setattr(settings, "ocr_timeout_seconds", 1)
    ocr = OcrEngine(gpu=False)

    class SlowReader:
        def readtext(self, img_array, detail=0):
            time.sleep(2.5)
            return ["never reached"]

    ocr._reader = SlowReader()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 1. Direct timeout wrapper raises OcrTimeoutError
    with pytest.raises(OcrTimeoutError, match="OCR exceeded timeout"):
        ocr._readtext_with_timeout(dummy_img)

    # 2. Full-page extract_text catches OcrTimeoutError and returns "" fallback
    pdf_path = create_sample_pdf(tmp_path / "ocr_test.pdf", num_pages=1)
    doc = fitz.open(str(pdf_path))
    try:
        extracted = ocr.extract_text(doc[0], page_num=1)
        assert extracted == ""
    finally:
        doc.close()

    # 3. MIXED page extract_from_regions enforces timeout per image region and preserves existing_text
    mixed_pdf_path = tmp_path / "mixed_ocr_timeout.pdf"
    mixed_doc = fitz.open()
    mpage = mixed_doc.new_page()
    mpage.insert_text((50, 50), "Native PDF text preserved on timeout.")
    # Insert a 100x100 PNG image onto the page
    from io import BytesIO
    from PIL import Image

    img_buf = BytesIO()
    Image.fromarray(np.full((100, 100, 3), 200, dtype=np.uint8)).save(img_buf, format="PNG")
    mpage.insert_image(fitz.Rect(50, 100, 150, 200), stream=img_buf.getvalue())
    mixed_doc.save(str(mixed_pdf_path))
    mixed_doc.close()

    doc2 = fitz.open(str(mixed_pdf_path))
    try:
        preserved = ocr.extract_from_regions(
            doc2[0], page_num=1, existing_text="Native PDF text preserved on timeout."
        )
        assert preserved == "Native PDF text preserved on timeout."
    finally:
        doc2.close()


# ============================================================================
# 2. WAVE 2 & 3 API TESTS: 202 Upload (<500ms), Polling, Dedup & Document CRUD
# ============================================================================


def test_non_blocking_upload_polling_dedup_and_crud(isolated_env: dict):
    """End-to-end verification of 202 upload (<500ms), job polling, SHA-256 dedup, and DELETE CRUD."""
    client: TestClient = isolated_env["client"]
    tmp_path: Path = isolated_env["tmp_path"]
    vector_store: VectorStore = isolated_env["vector_store"]

    pdf_path = create_sample_pdf(tmp_path / "enterprise_handbook.pdf", num_pages=3)
    expected_hash = PdfParser.compute_file_hash(pdf_path)

    # 1. POST /api/upload returns 202 Accepted in < 500ms
    t0 = time.perf_counter()
    with open(pdf_path, "rb") as f:
        resp = client.post(
            "/api/upload",
            files={"file": ("enterprise_handbook.pdf", f, "application/pdf")},
            data={"force_reindex": "false", "visibility": "ORG"},
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 202, resp.text
    assert elapsed_ms < 500.0, f"Upload took {elapsed_ms:.1f}ms (exceeded 500ms non-blocking SLA)"
    upload_data = resp.json()
    job_id = upload_data["job_id"]
    assert upload_data["file_hash"] == expected_hash
    assert upload_data["duplicate"] is False

    # 2. Poll GET /api/jobs/{job_id} until COMPLETED
    job_status = None
    for _ in range(60):
        poll_resp = client.get(f"/api/jobs/{job_id}")
        assert poll_resp.status_code == 200
        job_status = poll_resp.json()
        if job_status["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    assert job_status is not None
    assert job_status["status"] == "COMPLETED", f"Job failed: {job_status}"
    assert job_status["current_stage"] == "DONE"
    assert job_status["processed_pages"] == 3
    assert job_status["total_pages"] == 3
    assert job_status["chunks_created"] > 0

    # 3. Re-uploading identical PDF with force_reindex=False short-circuits via SHA-256 dedup
    with open(pdf_path, "rb") as f:
        dup_resp = client.post(
            "/api/upload",
            files={"file": ("enterprise_handbook_copy.pdf", f, "application/pdf")},
            data={"force_reindex": "false"},
        )
    assert dup_resp.status_code == 202
    dup_data = dup_resp.json()
    assert dup_data["duplicate"] is True
    assert dup_data["status"] == "COMPLETED"
    assert dup_data["chunks_created"] == job_status["chunks_created"]
    assert dup_data["duration_ms"] == 0.0

    # 4. GET /api/jobs lists both jobs
    jobs_resp = client.get("/api/jobs")
    assert jobs_resp.status_code == 200
    assert len(jobs_resp.json()["jobs"]) >= 2

    # 5. GET /api/documents lists the active document
    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    docs = docs_resp.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["file_hash"] == expected_hash
    assert docs[0]["total_pages"] == 3
    assert docs[0]["total_chunks"] == job_status["chunks_created"]

    # Verify chunks exist in Qdrant before deletion
    col_name = vector_store.collection_name
    assert vector_store.client.count(collection_name=col_name).count == job_status["chunks_created"]

    # 6. DELETE /api/documents/{file_hash} removes vectors from Qdrant and marks DELETED
    del_resp = client.delete(f"/api/documents/{expected_hash}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "DELETED"
    assert vector_store.client.count(collection_name=col_name).count == 0

    # Verify GET /api/documents is now empty
    assert client.get("/api/documents").json()["documents"] == []

    # 7. Re-uploading after deletion re-indexes cleanly
    with open(pdf_path, "rb") as f:
        reupload_resp = client.post(
            "/api/upload",
            files={"file": ("enterprise_handbook.pdf", f, "application/pdf")},
            data={"force_reindex": "false"},
        )
    assert reupload_resp.status_code == 202
    assert reupload_resp.json()["duplicate"] is False


# ============================================================================
# 3. QA RESILIENCE, FALLBACK & ERROR CONDITION TESTS
# ============================================================================


def test_interrupted_jobs_and_orphan_documents_recovery_on_startup(isolated_env: dict):
    """Verify recover_interrupted_jobs marks stale PENDING/PROCESSING jobs and orphan docs FAILED."""
    session_factory = isolated_env["session_factory"]
    job_manager: IngestionJobManager = isolated_env["job_manager"]

    with session_factory() as session:
        stale_doc = DocumentRecord(
            id="doc_stale_1",
            file_hash="abc123stalehash",
            filename="crashed_upload.pdf",
            org_id="default_org",
            status="PROCESSING",
        )
        stale_job_1 = IngestionJob(
            id="job_stale_pending",
            document_id="doc_stale_1",
            file_hash="abc123stalehash",
            filename="crashed_upload.pdf",
            org_id="default_org",
            status="PENDING",
            current_stage="QUEUED",
        )
        stale_job_2 = IngestionJob(
            id="job_stale_processing",
            document_id="doc_stale_1",
            file_hash="abc123stalehash",
            filename="crashed_upload.pdf",
            org_id="default_org",
            status="PROCESSING",
            current_stage="OCR",
        )
        session.add_all([stale_doc, stale_job_1, stale_job_2])
        session.commit()

    recovered_count = job_manager.recover_interrupted_jobs()
    assert recovered_count == 2

    with session_factory() as session:
        j1 = session.get(IngestionJob, "job_stale_pending")
        j2 = session.get(IngestionJob, "job_stale_processing")
        doc = session.get(DocumentRecord, "doc_stale_1")
        assert j1.status == "FAILED"
        assert j1.error_message == "Interrupted by server restart"
        assert j2.status == "FAILED"
        assert doc.status == "FAILED", "Orphaned PROCESSING DocumentRecord must also be marked FAILED"


def test_transient_retry_recovery_and_exhausted_retry_qdrant_rollback(
    isolated_env: dict, monkeypatch: pytest.MonkeyPatch
):
    """Verify worker retries transient errors and rolls back Qdrant chunks on persistent failure."""
    tmp_path: Path = isolated_env["tmp_path"]
    job_manager: IngestionJobManager = isolated_env["job_manager"]
    vector_store: VectorStore = isolated_env["vector_store"]
    session_factory = isolated_env["session_factory"]

    monkeypatch.setattr(settings, "max_job_retries", 2)

    # Scenario A: Transient error on Attempt 1, succeeds on Attempt 2
    pdf_a = create_sample_pdf(tmp_path / "transient_ok.pdf", num_pages=2)
    real_add = vector_store.add_document
    call_counter = {"count": 0}

    def flaky_add_document(result):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            raise RuntimeError("Simulated transient Qdrant timeout")
        return real_add(result)

    with patch.object(vector_store, "add_document", side_effect=flaky_add_document):
        sub_a = job_manager.submit_job(pdf_a, "transient_ok.pdf")
        job_a_id = sub_a["job_id"]
        for _ in range(60):
            ja = job_manager.get_job(job_a_id)
            if ja and ja["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.1)

    ja = job_manager.get_job(job_a_id)
    assert ja["status"] == "COMPLETED"
    assert ja["retry_count"] == 1
    assert ja["error_message"] is None

    # Scenario B: Persistent failure across all attempts triggers Qdrant rollback and FAILED status
    pdf_b = create_sample_pdf(tmp_path / "persistent_fail.pdf", num_pages=2)
    hash_b = PdfParser.compute_file_hash(pdf_b)

    with (
        patch.object(
            vector_store,
            "add_document",
            side_effect=RuntimeError("Persistent storage disk full"),
        ),
        patch.object(vector_store, "delete_document", wraps=vector_store.delete_document) as spy_rollback,
    ):
        sub_b = job_manager.submit_job(pdf_b, "persistent_fail.pdf")
        job_b_id = sub_b["job_id"]
        for _ in range(60):
            jb = job_manager.get_job(job_b_id)
            if jb and jb["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.1)

        assert spy_rollback.call_count >= 1
        spy_rollback.assert_called_with(hash_b)

    jb = job_manager.get_job(job_b_id)
    assert jb["status"] == "FAILED"
    assert jb["current_stage"] == "FAILED"
    assert jb["retry_count"] == 2
    assert "Persistent storage disk full" in jb["error_message"]

    with session_factory() as session:
        doc_b = session.scalars(
            select(DocumentRecord).where(DocumentRecord.file_hash == hash_b)
        ).first()
        assert doc_b is not None
        assert doc_b.status == "FAILED"


def test_encrypted_pdf_fails_gracefully_without_active_zero_chunk_record(
    isolated_env: dict, monkeypatch: pytest.MonkeyPatch
):
    """Verify password-protected PDFs fail cleanly with descriptive error instead of active 0-chunk record."""
    tmp_path: Path = isolated_env["tmp_path"]
    job_manager: IngestionJobManager = isolated_env["job_manager"]
    monkeypatch.setattr(settings, "max_job_retries", 0)

    enc_pdf = create_sample_pdf(tmp_path / "secret.pdf", num_pages=2, password="super-secret-password")
    sub = job_manager.submit_job(enc_pdf, "secret.pdf")
    job_id = sub["job_id"]

    for _ in range(40):
        job = job_manager.get_job(job_id)
        if job and job["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    job = job_manager.get_job(job_id)
    assert job["status"] == "FAILED"
    assert "password-protected" in (job["error_message"] or "")
    assert job_manager.list_documents() == []


def test_api_error_boundaries_and_invalid_inputs(isolated_env: dict):
    """Verify API 400/404 error responses for non-PDF uploads, empty files, unknown IDs, and bad providers."""
    client: TestClient = isolated_env["client"]

    # 1. Non-PDF file extension rejected with 400
    bad_ext = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert bad_ext.status_code == 400
    assert "Only PDF files" in bad_ext.json()["detail"]

    # 2. Empty 0-byte PDF file rejected with 400
    empty_pdf = client.post(
        "/api/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert empty_pdf.status_code == 400
    assert "empty" in empty_pdf.json()["detail"].lower()

    # 3. Unknown job_id returns 404
    missing_job = client.get("/api/jobs/non-existent-job-id")
    assert missing_job.status_code == 404

    # 4. Deleting non-existent file_hash returns 404
    missing_doc = client.delete("/api/documents/00000000000000000000000000000000")
    assert missing_doc.status_code == 404

    # 5. Querying with unknown LLM provider falls back to LMStudio & empty DB returns safe no-context response
    fallback_query = client.post(
        "/api/query",
        json={"query": "What is the architecture?", "provider": "invalid_provider_xyz"},
    )
    assert fallback_query.status_code == 200
    assert fallback_query.json()["citations"] == []
    assert "No matching context" in fallback_query.json()["answer"]

    # 6. Querying with unconfigured Gemini API key returns 400 Bad Request
    settings.gemini_api_key = ""
    bad_gemini_query = client.post(
        "/api/query",
        json={"query": "What is the architecture?", "provider": "gemini"},
    )
    assert bad_gemini_query.status_code == 400
    assert "Generator init failed" in bad_gemini_query.json()["detail"]


def test_force_reindex_and_corrupt_pdf_handling(isolated_env: dict, monkeypatch: pytest.MonkeyPatch):
    """Verify force_reindex=True reuses DocumentRecord without duplicates, and corrupt PDF bytes fail cleanly."""
    client: TestClient = isolated_env["client"]
    tmp_path: Path = isolated_env["tmp_path"]
    monkeypatch.setattr(settings, "max_job_retries", 0)

    # 1. Ingest a valid PDF first
    pdf_path = create_sample_pdf(tmp_path / "reindex_target.pdf", num_pages=2)
    with open(pdf_path, "rb") as f:
        r1 = client.post(
            "/api/upload",
            files={"file": ("reindex_target.pdf", f, "application/pdf")},
            data={"force_reindex": "false"},
        )
    assert r1.status_code == 202
    j1_id = r1.json()["job_id"]
    doc_id_1 = r1.json()["document_id"]

    for _ in range(40):
        if client.get(f"/api/jobs/{j1_id}").json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    # 2. Re-upload with force_reindex=True -> bypasses duplicate check and re-indexes same DocumentRecord
    with open(pdf_path, "rb") as f:
        r2 = client.post(
            "/api/upload",
            files={"file": ("reindex_target.pdf", f, "application/pdf")},
            data={"force_reindex": "true", "visibility": "PRIVATE"},
        )
    assert r2.status_code == 202
    assert r2.json()["duplicate"] is False
    assert r2.json()["document_id"] == doc_id_1
    j2_id = r2.json()["job_id"]

    for _ in range(40):
        j2_state = client.get(f"/api/jobs/{j2_id}").json()
        if j2_state["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    assert j2_state["status"] == "COMPLETED"
    docs = client.get("/api/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id_1
    assert docs[0]["visibility"] == "PRIVATE"

    # 3. Upload a corrupted non-PDF payload with .pdf extension -> 202 Accepted then transitions to FAILED
    corrupt_resp = client.post(
        "/api/upload",
        files={"file": ("corrupted_payload.pdf", b"NOT_A_VALID_PDF_BINARY_HEADER", "application/pdf")},
    )
    assert corrupt_resp.status_code == 202
    corrupt_job_id = corrupt_resp.json()["job_id"]

    for _ in range(40):
        c_state = client.get(f"/api/jobs/{corrupt_job_id}").json()
        if c_state["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    assert c_state["status"] == "FAILED"
    assert "Failed to open PDF" in (c_state["error_message"] or "")

