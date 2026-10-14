"""Database-backed background PDF ingestion worker pool with SHA-256 deduplication and retries (L2)."""

from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select

from src.config import settings
from src.auth.security import sync_benchmark_catalog
from src.database.models import DocumentRecord, IngestionJob
from src.database.relational import SessionLocal, init_db
from src.database.vector_store import VectorStore
from src.ingestion.pdf_parser import PdfParser
from src.ingestion.pipeline import IngestionPipeline


class IngestionJobManager:
    """Manages non-blocking PDF ingestion jobs using a bounded ThreadPoolExecutor and SQLAlchemy state."""

    def __init__(self, pipeline: IngestionPipeline, vector_store: VectorStore) -> None:
        self.pipeline = pipeline
        self.vector_store = vector_store
        # Strict sequential document ingestion concurrency (Concurrency = 1) for GPU VRAM stability
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rag_ingest_worker",
        )
        init_db()
        self.recover_interrupted_jobs()
        sync_benchmark_catalog()

    def _refresh_queue_positions_in_db(self, session) -> None:
        """Synchronize queue_position indices for active and waiting jobs in SQLite."""
        session.flush()
        # Clear queue_position for any jobs no longer active/pending
        finished_jobs = list(
            session.scalars(
                select(IngestionJob).where(
                    IngestionJob.status.not_in(["PROCESSING", "PENDING"]),
                    IngestionJob.queue_position.is_not(None),
                )
            ).all()
        )
        for fj in finished_jobs:
            fj.queue_position = None

        processing_jobs = list(
            session.scalars(
                select(IngestionJob)
                .where(IngestionJob.status == "PROCESSING")
                .order_by(IngestionJob.created_at.asc())
            ).all()
        )
        pending_jobs = list(
            session.scalars(
                select(IngestionJob)
                .where(IngestionJob.status == "PENDING")
                .order_by(IngestionJob.created_at.asc())
            ).all()
        )
        pos = 1
        for j in processing_jobs:
            j.queue_position = pos
            pos += 1
        for j in pending_jobs:
            j.queue_position = pos
            pos += 1


    def recover_interrupted_jobs(self) -> int:
        """Mark any stale PENDING/PROCESSING jobs and orphaned PROCESSING documents from an unclean shutdown as FAILED."""
        now = datetime.now(timezone.utc)
        recovered = 0
        with SessionLocal() as session:
            stmt = select(IngestionJob).where(IngestionJob.status.in_(["PENDING", "PROCESSING"]))
            stale_jobs = list(session.scalars(stmt).all())
            for job in stale_jobs:
                job.status = "FAILED"
                job.current_stage = "FAILED"
                job.queue_position = None
                job.error_message = "Interrupted by server restart"
                job.updated_at = now
                if job.document_id:
                    doc = session.get(DocumentRecord, job.document_id)
                    if doc is not None and doc.status == "PROCESSING":
                        doc.status = "FAILED"
                recovered += 1

            orphan_docs = list(
                session.scalars(
                    select(DocumentRecord).where(DocumentRecord.status == "PROCESSING")
                ).all()
            )
            for doc in orphan_docs:
                doc.status = "FAILED"

            if recovered > 0 or orphan_docs:
                self._refresh_queue_positions_in_db(session)
                session.commit()
                logger.warning("Recovered {} interrupted ingestion job(s) on startup", recovered)
        return recovered

    # Alias matching exact camelCase in ACTIVE_PLAN.md specification
    recover_Interrupted_jobs = recover_interrupted_jobs

    def submit_job(
        self,
        file_path: Path,
        filename: str,
        org_id: str = "default_org",
        visibility: str = "ORG",
        allowed_roles: list[str] | None = None,
        uploaded_by: str | None = None,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Submit a PDF for asynchronous ingestion or short-circuit if SHA-256 duplicate exists."""
        file_path = Path(file_path)
        file_hash = PdfParser.compute_file_hash(file_path)
        roles = allowed_roles or ["ADMIN", "MANAGER", "ANALYST", "VIEWER"]

        with SessionLocal() as session:
            stmt = (
                select(DocumentRecord)
                .where(
                    DocumentRecord.file_hash == file_hash,
                    DocumentRecord.org_id == org_id,
                    DocumentRecord.status == "ACTIVE",
                )
                .order_by(DocumentRecord.created_at.desc())
            )
            existing_doc = session.scalars(stmt).first()

            if existing_doc is not None and not force_reindex:
                dup_job = IngestionJob(
                    document_id=existing_doc.id,
                    file_hash=file_hash,
                    filename=filename,
                    org_id=org_id,
                    status="COMPLETED",
                    current_stage="DONE",
                    total_pages=existing_doc.total_pages,
                    processed_pages=existing_doc.total_pages,
                    chunks_created=existing_doc.total_chunks,
                    retry_count=0,
                    duration_ms=0.0,
                )
                session.add(dup_job)
                session.commit()
                session.refresh(dup_job)
                logger.info(
                    "Duplicate upload detected for {} (hash={}...), skipping OCR/indexing",
                    filename,
                    file_hash[:12],
                )
                payload = dup_job.to_dict()
                payload["duplicate"] = True
                payload["deduplicated"] = True
                payload["message"] = (
                    f"Duplicate document '{filename}' already indexed ({existing_doc.total_chunks} chunks)."
                )
                return payload

            if existing_doc is not None and force_reindex:
                doc_record = existing_doc
                doc_record.filename = filename
                doc_record.visibility = visibility
                doc_record.allowed_roles = roles
            else:
                doc_record = DocumentRecord(
                    file_hash=file_hash,
                    filename=filename,
                    org_id=org_id,
                    uploaded_by=uploaded_by,
                    visibility=visibility,
                    allowed_roles=roles,
                    total_pages=0,
                    total_chunks=0,
                    status="PROCESSING",
                )
                session.add(doc_record)
                session.flush()

            job = IngestionJob(
                document_id=doc_record.id,
                file_hash=file_hash,
                filename=filename,
                org_id=org_id,
                status="PENDING",
                current_stage="QUEUED",
                total_pages=0,
                processed_pages=0,
                chunks_created=0,
                retry_count=0,
            )
            session.add(job)
            session.flush()
            self._refresh_queue_positions_in_db(session)
            session.commit()
            session.refresh(job)
            job_id = job.id
            doc_id = doc_record.id
            payload = job.to_dict()

        self.executor.submit(self._run_job, job_id, doc_id, file_path, file_hash, force_reindex)

        payload["duplicate"] = False
        payload["deduplicated"] = False
        if payload.get("queue_position") and payload["queue_position"] > 1:
            payload["message"] = (
                f"Ingestion job queued for '{filename}' (position #{payload['queue_position']})."
            )
        else:
            payload["message"] = f"Ingestion job queued for '{filename}'."
        return payload

    def _update_progress(
        self,
        job_id: str,
        processed_pages: int,
        total_pages: int,
        current_stage: str,
    ) -> None:
        """Persist page-level progress from IngestionPipeline callback."""
        with SessionLocal() as session:
            job = session.get(IngestionJob, job_id)
            if job is not None:
                job.processed_pages = processed_pages
                job.total_pages = total_pages
                job.current_stage = current_stage
                job.updated_at = datetime.now(timezone.utc)
                session.commit()

    def _run_job(
        self,
        job_id: str,
        doc_id: str,
        file_path: Path,
        file_hash: str,
        force_reindex: bool = False,
    ) -> None:
        """Execute PDF parsing, OCR, chunking, and Qdrant indexing with automatic retries."""
        start_perf = time.perf_counter()
        max_retries = max(0, int(settings.max_job_retries))

        doc_org_id = "default_org"
        doc_visibility = "ORG"
        doc_roles = ["ADMIN", "MANAGER", "ANALYST", "VIEWER"]
        with SessionLocal() as session:
            job = session.get(IngestionJob, job_id)
            doc = session.get(DocumentRecord, doc_id)
            if doc is not None:
                doc_org_id = doc.org_id
                doc_visibility = doc.visibility
                doc_roles = list(doc.allowed_roles or doc_roles)
            if job is not None:
                job.status = "PROCESSING"
                job.current_stage = "PARSING"
                job.updated_at = datetime.now(timezone.utc)
                self._refresh_queue_positions_in_db(session)
                session.commit()

        def _callback(processed: int, total: int, stage: str) -> None:
            self._update_progress(job_id, processed, total, stage)

        for attempt in range(max_retries + 1):
            try:
                result = self.pipeline.process_pdf(file_path, progress_callback=_callback)
                if result.total_chunks == 0 and result.errors:
                    raise RuntimeError("; ".join(result.errors))

                # Attach RBAC metadata onto result and child chunks for Qdrant payload indexing
                setattr(result, "org_id", doc_org_id)
                setattr(result, "visibility", doc_visibility)
                setattr(result, "allowed_roles", doc_roles)
                for chunk in result.chunks:
                    chunk.metadata["org_id"] = doc_org_id
                    chunk.metadata["visibility"] = doc_visibility
                    chunk.metadata["allowed_roles"] = doc_roles

                self._update_progress(job_id, result.total_pages, result.total_pages, "INDEXING")
                if force_reindex:
                    self.vector_store.delete_document(file_hash, org_id=doc_org_id)
                self.vector_store.add_document(result)

                duration_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
                now = datetime.now(timezone.utc)

                with SessionLocal() as session:
                    job = session.get(IngestionJob, job_id)
                    doc = session.get(DocumentRecord, doc_id)
                    if doc is not None:
                        doc.total_pages = result.total_pages
                        doc.total_chunks = result.total_chunks
                        doc.status = "ACTIVE"
                    if job is not None:
                        job.status = "COMPLETED"
                        job.current_stage = "DONE"
                        job.queue_position = None
                        job.total_pages = result.total_pages
                        job.processed_pages = result.total_pages
                        job.chunks_created = result.total_chunks
                        job.duration_ms = duration_ms
                        job.error_message = None
                        job.updated_at = now
                    self._refresh_queue_positions_in_db(session)
                    session.commit()

                logger.info(
                    "Job {} COMPLETED in {:.1f}ms ({} pages, {} chunks)",
                    job_id,
                    duration_ms,
                    result.total_pages,
                    result.total_chunks,
                )
                return

            except Exception as exc:
                logger.error(
                    "Ingestion job {} failed on attempt {}/{}: {}",
                    job_id,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                if attempt < max_retries:
                    with SessionLocal() as session:
                        job = session.get(IngestionJob, job_id)
                        if job is not None:
                            job.retry_count = attempt + 1
                            job.error_message = f"Transient failure (attempt {attempt + 1}): {exc}"
                            job.updated_at = datetime.now(timezone.utc)
                            session.commit()
                    time.sleep(0.25)
                else:
                    try:
                        self.vector_store.delete_document(file_hash)
                    except Exception as cleanup_exc:
                        logger.warning("Cleanup rollback failed for {}: {}", file_hash, cleanup_exc)

                    duration_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
                    now = datetime.now(timezone.utc)
                    with SessionLocal() as session:
                        job = session.get(IngestionJob, job_id)
                        doc = session.get(DocumentRecord, doc_id)
                        if doc is not None:
                            doc.status = "FAILED"
                            doc.total_chunks = 0
                        if job is not None:
                            job.status = "FAILED"
                            job.current_stage = "FAILED"
                            job.queue_position = None
                            job.retry_count = attempt
                            job.error_message = str(exc)
                            job.duration_ms = duration_ms
                            job.updated_at = now
                        self._refresh_queue_positions_in_db(session)
                        session.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve a single IngestionJob record by ID with active queue position."""
        with SessionLocal() as session:
            job = session.get(IngestionJob, job_id)
            if job is None:
                return None
            if job.status in ("PENDING", "PROCESSING"):
                self._refresh_queue_positions_in_db(session)
                session.commit()
                session.refresh(job)
            return job.to_dict()

    def list_jobs(self, limit: int = 50, org_id: str | None = None) -> list[dict[str, Any]]:
        """List recent ingestion jobs ordered newest first with active queue positions."""
        with SessionLocal() as session:
            self._refresh_queue_positions_in_db(session)
            session.commit()
            stmt = select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(limit)
            if org_id:
                stmt = stmt.where(IngestionJob.org_id == org_id)
            return [j.to_dict() for j in session.scalars(stmt).all()]

    def shutdown(self, wait: bool = False) -> None:
        """Cleanly shut down the ingestion worker thread pool executor."""
        self.executor.shutdown(wait=wait)

    def list_documents(self, org_id: str | None = None) -> list[dict[str, Any]]:
        """List all ACTIVE DocumentRecord entries ordered newest first."""
        with SessionLocal() as session:
            stmt = (
                select(DocumentRecord)
                .where(DocumentRecord.status == "ACTIVE")
                .order_by(DocumentRecord.created_at.desc())
            )
            if org_id:
                stmt = stmt.where(DocumentRecord.org_id == org_id)
            return [d.to_dict() for d in session.scalars(stmt).all()]

    def delete_document(self, file_hash: str, org_id: str | None = None) -> dict[str, Any] | None:
        """Delete document chunks from Qdrant and mark DocumentRecord status as DELETED."""
        with SessionLocal() as session:
            stmt = select(DocumentRecord).where(
                DocumentRecord.file_hash == file_hash,
                DocumentRecord.status == "ACTIVE",
            )
            if org_id:
                stmt = stmt.where(DocumentRecord.org_id == org_id)
            docs = list(session.scalars(stmt).all())
            if not docs:
                return None

            self.vector_store.delete_document(file_hash, org_id=org_id)
            for doc in docs:
                doc.status = "DELETED"
            session.commit()
            return {
                "file_hash": file_hash,
                "deleted_records": len(docs),
                "status": "DELETED",
            }


class AutoIngestWatcher:
    """Background daemon thread monitoring a drop directory for automatic PDF ingestion."""

    def __init__(
        self,
        job_manager: IngestionJobManager,
        watch_dir: Path | None = None,
        interval_seconds: float | None = None,
        upload_dir: Path | None = None,
    ) -> None:
        self.job_manager = job_manager
        self.watch_dir = Path(watch_dir or settings.auto_ingest_dir)
        self.interval = float(interval_seconds or settings.auto_ingest_interval_seconds)
        self.upload_dir = Path(upload_dir or settings.pdf_upload_dir)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def is_file_ready(self, filepath: Path) -> bool:
        """Verify that a discovered file has finished writing and can be read safely."""
        if not filepath.is_file():
            return False
        try:
            size1 = filepath.stat().st_size
            if size1 == 0:
                return False
            time.sleep(0.1)
            size2 = filepath.stat().st_size
            if size1 != size2:
                return False
            with open(filepath, "rb") as f:
                f.read(min(size2, 1024))
            return True
        except (OSError, PermissionError):
            return False

    def scan_and_ingest(self) -> list[dict[str, Any]]:
        """Scan watch_dir for ready PDFs, move to upload_dir, and queue for ingestion."""
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        queued_jobs: list[dict[str, Any]] = []

        try:
            candidates = [
                p
                for p in self.watch_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
            ]
        except OSError as exc:
            logger.warning("AutoIngestWatcher failed listing {}: {}", self.watch_dir, exc)
            return queued_jobs

        for pdf_path in candidates:
            if not self.is_file_ready(pdf_path):
                continue

            original_name = pdf_path.name
            target_path = self.upload_dir / original_name
            if target_path.exists():
                target_path = self.upload_dir / f"{pdf_path.stem}_{int(time.time())}{pdf_path.suffix}"

            try:
                shutil.move(str(pdf_path), str(target_path))
                job_payload = self.job_manager.submit_job(
                    file_path=target_path,
                    filename=original_name,
                    org_id="default_org",
                    visibility="ORG",
                )
                queued_jobs.append(job_payload)
                logger.info(
                    "AutoIngestWatcher ingested '{}' (job_id={}, queue_position={})",
                    original_name,
                    job_payload.get("job_id"),
                    job_payload.get("queue_position"),
                )
            except Exception as exc:
                logger.error("AutoIngestWatcher failed processing '{}': {}", original_name, exc)

        return queued_jobs

    def start(self) -> None:
        """Start the background daemon thread if not running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="auto_ingest_watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "AutoIngestWatcher started (watching '{}' every {}s)",
            self.watch_dir,
            self.interval,
        )

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the watcher thread to stop and wait for it to join."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("AutoIngestWatcher stopped")
        self._thread = None

    def _run_loop(self) -> None:
        """Periodic scan loop."""
        while not self._stop_event.is_set():
            try:
                self.scan_and_ingest()
            except Exception as exc:
                logger.error("AutoIngestWatcher error in loop: {}", exc)
            self._stop_event.wait(timeout=self.interval)
