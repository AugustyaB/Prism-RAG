"""Phase 2.8 QA Test Suite: Ingestion Concurrency Engine, Rate Limit Decoupling,
Multi-File Queue & Auto-Ingest Watcher (Sec 4, Sec 5, Sec 6; L1, L2, L4).

Empirically validates:
1. Rate Limit Decoupling for Job Polling:
   - GET /api/jobs/{id} and GET /api/jobs are completely exempt from rate limiting
     (can be polled 120+ times consecutively without receiving HTTP 429).
2. Burst Rate Limiting on Standard Endpoints:
   - POST /api/query still triggers HTTP 429 with Retry-After when rate limit is exceeded.
3. Strict Sequential Ingestion Concurrency (Concurrency = 1):
   - Multiple simultaneous uploads line up in an orderly queue.
   - Job 1 is PROCESSING, Jobs 2 and 3 are QUEUED/PENDING with queue_position 2 and 3.
   - Zero concurrent GPU/OCR clashes (maximum concurrent worker count is strictly 1).
   - Dynamic queue_position updates in real time as jobs complete.
4. AutoIngestWatcher Folder Scanning & Automatic Ingestion:
   - Placing a valid PDF into data/auto_ingest/ automatically moves it to uploads/
     and queues it for ingestion with org_id="default_org".
   - Non-PDF files, hidden files, and empty (0-byte) files are safely ignored.
5. Zero Live Database Pollution:
   - Tests execute in isolated SQLite databases without polluting data/metadata.db.
"""

from __future__ import annotations

import fitz
import pytest
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import src.api.server as server_module
import src.database.relational as relational_module
import src.jobs.ingestion_worker as worker_module
import src.auth.security as security_module
from src.config import settings
from src.database.models import Base, DocumentRecord, IngestionJob, Organization
from src.jobs.ingestion_worker import IngestionJobManager, AutoIngestWatcher
from src.api.server import app, rate_limiter


def create_dummy_pdf(file_path: Path, title: str = "Test Doc", pages: int = 2) -> Path:
    """Create a minimal synthetic PDF file for testing."""
    doc = fitz.open()
    for p in range(pages):
        page = doc.new_page()
        page.insert_text(
            (50, 72),
            f"Title: {title}\nPage: {p + 1}\nContent: Synthetic benchmark test payload for concurrency verification.",
            fontsize=12,
        )
    doc.save(str(file_path))
    doc.close()
    return file_path


class MockChunk:
    def __init__(self):
        self.metadata = {}


class MockIngestionResult:
    def __init__(self, total_pages: int = 1, total_chunks: int = 1):
        self.source_pdf = "test.pdf"
        self.file_hash = "mockhash"
        self.total_pages = total_pages
        self.classification_summary = {}
        self.total_chunks = total_chunks
        self.chunks = [MockChunk() for _ in range(total_chunks)]
        self.errors = []


@pytest.fixture()
def isolated_phase28_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide an isolated database, directories, and sessionmaker for Phase 2.8 testing."""
    test_db = tmp_path / "test_phase28.db"
    test_uploads = tmp_path / "uploads"
    test_auto_ingest = tmp_path / "auto_ingest"
    test_extracted = tmp_path / "extracted"
    test_vectordb = tmp_path / "vectordb"

    test_uploads.mkdir(parents=True, exist_ok=True)
    test_auto_ingest.mkdir(parents=True, exist_ok=True)
    test_extracted.mkdir(parents=True, exist_ok=True)
    test_vectordb.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{test_db.as_posix()}")
    monkeypatch.setattr(settings, "pdf_upload_dir", test_uploads)
    monkeypatch.setattr(settings, "auto_ingest_dir", test_auto_ingest)
    monkeypatch.setattr(settings, "extracted_dir", test_extracted)
    monkeypatch.setattr(settings, "vector_db_dir", test_vectordb)
    monkeypatch.setattr(settings, "auth_enabled", False)

    test_engine = create_engine(
        f"sqlite:///{test_db.as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    TestSessionLocal = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )

    monkeypatch.setattr(relational_module, "engine", test_engine)
    monkeypatch.setattr(relational_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(worker_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(server_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(security_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(security_module, "sync_benchmark_catalog", lambda *a, **kw: 0)
    monkeypatch.setattr(worker_module, "sync_benchmark_catalog", lambda *a, **kw: 0)

    Base.metadata.create_all(bind=test_engine)
    with TestSessionLocal() as session:
        session.add(Organization(id="default_org", name="Default Enterprise Org"))
        session.commit()

    rate_limiter.reset()

    yield {
        "engine": test_engine,
        "session_maker": TestSessionLocal,
        "uploads": test_uploads,
        "auto_ingest": test_auto_ingest,
    }

    rate_limiter.reset()


def test_jobs_endpoint_rate_limit_exemption(isolated_phase28_env, monkeypatch):
    """Empirically verify that GET /api/jobs/{id} and /api/jobs are exempt from rate limiting.

    Even when polled 120+ times consecutively, the requests must never return HTTP 429.
    """
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_requests_per_minute", 20)
    rate_limiter.reset()

    client = TestClient(app)

    # Poll /api/jobs 120 times consecutively
    statuses = []
    for _ in range(120):
        resp = client.get("/api/jobs")
        statuses.append(resp.status_code)

    assert all(code == 200 for code in statuses), (
        f"Expected all 120 GET /api/jobs requests to return 200 OK, got: {set(statuses)}"
    )

    # Poll a specific job ID 120 times (returns 404 since job does not exist, but NEVER 429)
    dummy_job_id = "00000000-0000-0000-0000-000000000000"
    job_statuses = []
    for _ in range(120):
        resp = client.get(f"/api/jobs/{dummy_job_id}")
        job_statuses.append(resp.status_code)

    assert 429 not in job_statuses, "Rate limiting unexpectedly throttled GET /api/jobs/{id}"
    assert all(code == 404 for code in job_statuses), (
        f"Expected 404 for non-existent job ID, got: {set(job_statuses)}"
    )


def test_query_rate_limit_enforced_with_retry_after(isolated_phase28_env, monkeypatch):
    """Empirically verify that heavy non-exempt endpoints like POST /api/query enforce rate limits.

    Bursts exceeding the configured rate_limit_requests_per_minute trigger HTTP 429 with Retry-After.
    """
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    # Set limit low to trigger within a short burst
    monkeypatch.setattr(settings, "rate_limit_requests_per_minute", 5)
    rate_limiter.reset()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "context": "sample context",
        "results": [],
        "citations": [],
        "top_score": 0.8,
        "abstained": False,
        "abstention_reason": None,
    }
    monkeypatch.setattr(server_module, "retriever", mock_retriever)

    mock_gen = MagicMock()
    mock_gen.generate_with_telemetry.return_value = ("Sample answer", 10, 5)
    mock_gen.model_name = "test-model"
    monkeypatch.setattr(server_module, "get_generator", lambda *a, **kw: mock_gen)

    client = TestClient(app)

    responses = []
    for i in range(10):
        resp = client.post("/api/query", json={"query": f"Test burst query {i}", "mode": "dense"})
        responses.append(resp)

    # The first 5 requests should not be 429; the 6th onwards MUST be 429
    throttled = [r for r in responses if r.status_code == 429]
    assert len(throttled) > 0, "Expected rate limiter to throttle requests beyond limit"

    first_429 = throttled[0]
    assert first_429.status_code == 429
    assert "Retry-After" in first_429.headers
    retry_after = int(first_429.headers["Retry-After"])
    assert retry_after >= 1, f"Expected Retry-After header >= 1s, got {retry_after}"
    body = first_429.json()
    assert "detail" in body
    assert "Rate limit exceeded" in body["detail"]


def test_sequential_ingestion_concurrency_and_queue_positions(isolated_phase28_env, monkeypatch):
    """Empirically verify strict sequential ingestion concurrency (Concurrency = 1) and dynamic queue positions.

    When 3 simultaneous PDFs are submitted:
    - Job 1 enters PROCESSING (queue_position=1).
    - Job 2 enters PENDING (queue_position=2).
    - Job 3 enters PENDING (queue_position=3).
    - Zero concurrent GPU worker collisions (max active workers <= 1).
    - As Job 1 completes, Job 2 advances to queue_position=1, and Job 3 advances to queue_position=2.
    """
    uploads_dir = isolated_phase28_env["uploads"]
    pdf1 = create_dummy_pdf(uploads_dir / "doc1.pdf", title="Doc 1", pages=1)
    pdf2 = create_dummy_pdf(uploads_dir / "doc2.pdf", title="Doc 2", pages=1)
    pdf3 = create_dummy_pdf(uploads_dir / "doc3.pdf", title="Doc 3", pages=1)

    # Create coordination primitives
    job1_started = threading.Event()
    job1_release = threading.Event()
    job2_started = threading.Event()
    job2_release = threading.Event()

    active_workers = 0
    max_concurrent_workers = 0
    worker_lock = threading.Lock()

    def mock_process_pdf(file_path, progress_callback=None):
        nonlocal active_workers, max_concurrent_workers
        with worker_lock:
            active_workers += 1
            if active_workers > max_concurrent_workers:
                max_concurrent_workers = active_workers

        fname = Path(file_path).name
        if progress_callback:
            progress_callback(1, 1, "PARSING")

        if fname == "doc1.pdf":
            job1_started.set()
            job1_release.wait(timeout=5.0)
        elif fname == "doc2.pdf":
            job2_started.set()
            job2_release.wait(timeout=5.0)

        with worker_lock:
            active_workers -= 1

        return MockIngestionResult(total_pages=1, total_chunks=1)

    mock_pipeline = MagicMock()
    mock_pipeline.process_pdf.side_effect = mock_process_pdf

    mock_vector_store = MagicMock()
    mock_vector_store.add_document.return_value = None

    job_manager = IngestionJobManager(pipeline=mock_pipeline, vector_store=mock_vector_store)

    # 1. Submit Job 1
    job1_payload = job_manager.submit_job(
        file_path=pdf1,
        filename="doc1.pdf",
        org_id="default_org",
        visibility="ORG",
    )
    job1_id = job1_payload["job_id"]

    # Wait until Job 1 has actually entered the worker thread
    assert job1_started.wait(timeout=3.0), "Job 1 failed to start within timeout"

    # 2. While Job 1 is running, submit Job 2 and Job 3
    job2_payload = job_manager.submit_job(
        file_path=pdf2,
        filename="doc2.pdf",
        org_id="default_org",
        visibility="ORG",
    )
    job2_id = job2_payload["job_id"]

    job3_payload = job_manager.submit_job(
        file_path=pdf3,
        filename="doc3.pdf",
        org_id="default_org",
        visibility="ORG",
    )
    job3_id = job3_payload["job_id"]

    # Verify initial queue positions
    SessionLocal = isolated_phase28_env["session_maker"]
    with SessionLocal() as session:
        j1 = session.get(IngestionJob, job1_id)
        j2 = session.get(IngestionJob, job2_id)
        j3 = session.get(IngestionJob, job3_id)

        assert j1.status == "PROCESSING"
        assert j1.queue_position == 1

        assert j2.status == "PENDING"
        assert j2.queue_position == 2

        assert j3.status == "PENDING"
        assert j3.queue_position == 3

    # Release Job 1
    job1_release.set()

    # Wait for Job 2 to start processing
    assert job2_started.wait(timeout=3.0), "Job 2 failed to start processing"

    # Verify updated queue positions while Job 2 is running
    with SessionLocal() as session:
        j1 = session.get(IngestionJob, job1_id)
        j2 = session.get(IngestionJob, job2_id)
        j3 = session.get(IngestionJob, job3_id)

        assert j1.status == "COMPLETED"
        assert j1.queue_position is None

        assert j2.status == "PROCESSING"
        assert j2.queue_position == 1

        assert j3.status == "PENDING"
        assert j3.queue_position == 2

    # Release Job 2
    job2_release.set()

    # Wait for Job 3 to finish
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with SessionLocal() as session:
            j3 = session.get(IngestionJob, job3_id)
            if j3.status == "COMPLETED":
                break
        time.sleep(0.1)

    with SessionLocal() as session:
        j3 = session.get(IngestionJob, job3_id)
        assert j3.status == "COMPLETED"
        assert j3.queue_position is None

    # Crucial Assertion: Enforce concurrency = 1 (Zero concurrent GPU collisions)
    assert max_concurrent_workers == 1, (
        f"Concurrency violation! Max active workers was {max_concurrent_workers}, expected strictly 1"
    )

    job_manager.shutdown(wait=True)


def test_auto_ingest_watcher_detection_and_queueing(isolated_phase28_env, monkeypatch):
    """Empirically verify AutoIngestWatcher file drop detection, moving to upload_dir, and job queueing."""
    watch_dir = isolated_phase28_env["auto_ingest"]
    upload_dir = isolated_phase28_env["uploads"]

    # Mock job manager
    mock_job_manager = MagicMock()
    mock_job_manager.submit_job.return_value = {
        "job_id": "test-auto-job-uuid",
        "queue_position": 1,
        "status": "PENDING",
    }

    watcher = AutoIngestWatcher(
        job_manager=mock_job_manager,
        watch_dir=watch_dir,
        interval_seconds=0.1,
        upload_dir=upload_dir,
    )

    # Create a test PDF in watch_dir
    pdf_file = watch_dir / "auto_drop.pdf"
    create_dummy_pdf(pdf_file, title="Auto Drop Ingestion", pages=1)

    # Verify is_file_ready
    assert watcher.is_file_ready(pdf_file) is True

    # Run scan_and_ingest
    queued = watcher.scan_and_ingest()

    assert len(queued) == 1
    assert queued[0]["job_id"] == "test-auto-job-uuid"
    assert not pdf_file.exists(), "Original PDF was not removed from watch_dir"

    target_pdf = upload_dir / "auto_drop.pdf"
    assert target_pdf.exists(), "Target PDF was not moved to upload_dir"

    mock_job_manager.submit_job.assert_called_once_with(
        file_path=target_pdf,
        filename="auto_drop.pdf",
        org_id="default_org",
        visibility="ORG",
    )


def test_auto_ingest_watcher_safety_filters(isolated_phase28_env):
    """Empirically verify that AutoIngestWatcher skips non-PDF files, hidden files, and empty (0-byte) files."""
    watch_dir = isolated_phase28_env["auto_ingest"]
    upload_dir = isolated_phase28_env["uploads"]

    mock_job_manager = MagicMock()
    watcher = AutoIngestWatcher(
        job_manager=mock_job_manager,
        watch_dir=watch_dir,
        interval_seconds=0.1,
        upload_dir=upload_dir,
    )

    # 1. Non-pdf file
    txt_file = watch_dir / "notes.txt"
    txt_file.write_text("Hello world", encoding="utf-8")

    # 2. 0-byte PDF
    empty_pdf = watch_dir / "empty.pdf"
    empty_pdf.write_bytes(b"")

    # 3. Hidden PDF
    hidden_pdf = watch_dir / ".hidden.pdf"
    create_dummy_pdf(hidden_pdf, title="Hidden", pages=1)

    assert watcher.is_file_ready(empty_pdf) is False

    queued = watcher.scan_and_ingest()

    assert len(queued) == 0, f"Expected 0 files queued, got {len(queued)}"
    mock_job_manager.submit_job.assert_not_called()

    # Verify files remain in watch_dir and were not moved
    assert txt_file.exists()
    assert empty_pdf.exists()
    assert hidden_pdf.exists()


def test_live_database_remains_clean():
    """Verify that live data/metadata.db contains zero test records or temporary artifacts from this suite."""
    live_db_path = Path("./data/metadata.db")
    if not live_db_path.exists():
        return

    live_engine = create_engine(
        f"sqlite:///{live_db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    LiveSession = sessionmaker(bind=live_engine)

    with LiveSession() as session:
        live_docs = session.query(DocumentRecord).all()
        filenames = [d.filename for d in live_docs]
        for name in filenames:
            assert not name.startswith("doc"), f"Found test document in live database: {name}"
            assert not name.startswith("auto_drop"), f"Found test auto-ingest file in live DB: {name}"

