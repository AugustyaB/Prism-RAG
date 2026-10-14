"""FastAPI Gateway exposing non-blocking 202 PDF upload, job polling, document CRUD, RAG query, and telemetry metrics (L4)."""

from __future__ import annotations

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
import uuid

import requests
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.agents.memory import ConversationMemoryManager
from src.agents.state_machine import AgenticRAGPipeline
from src.agents.tools import execute_tool
from src.api.auth_routes import router as auth_router
from src.auth.dependencies import build_qdrant_rbac_filter, get_current_user
from src.auth.security import seed_demo_accounts
from src.config import settings
from src.database.models import QueryTelemetry, User
from src.database.relational import SessionLocal, get_db, init_db
from src.database.vector_store import VectorStore
from src.embeddings.embedder import LocalEmbedder
from src.generation.generator import get_generator, synthesize_extractive_answer
from src.ingestion.pipeline import IngestionPipeline
from src.jobs.ingestion_worker import AutoIngestWatcher, IngestionJobManager
from src.retrieval.retriever import Retriever
from src.utils.telemetry import (
    TelemetryCollector,
    TelemetrySpan,
    calculate_query_cost,
    count_tokens,
    get_aggregated_metrics,
    record_query_telemetry,
)

# Global singletons
embedder: LocalEmbedder | None = None
vector_store: VectorStore | None = None
retriever: Retriever | None = None
ingestion_pipeline: IngestionPipeline | None = None
job_manager: IngestionJobManager | None = None
auto_ingest_watcher: AutoIngestWatcher | None = None
memory_manager: ConversationMemoryManager | None = None
agentic_pipeline: AgenticRAGPipeline | None = None


def ensure_services_initialized() -> None:
    """Lazily initialize backend singletons if not already initialized by lifespan."""
    global embedder, vector_store, retriever, ingestion_pipeline, job_manager, auto_ingest_watcher, memory_manager, agentic_pipeline
    if job_manager is not None:
        return
    settings.ensure_dirs()
    init_db()
    seed_demo_accounts()
    embedder = LocalEmbedder()
    vector_store = VectorStore(embedder)
    retriever = Retriever(vector_store)
    ingestion_pipeline = IngestionPipeline()
    job_manager = IngestionJobManager(ingestion_pipeline, vector_store)
    auto_ingest_watcher = AutoIngestWatcher(job_manager)
    memory_manager = ConversationMemoryManager(session_factory=SessionLocal)
    agentic_pipeline = AgenticRAGPipeline(
        retriever=retriever,
        vector_store=vector_store,
        memory_manager=memory_manager,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize relational DB, demo accounts, vector store, embedder, and async worker pool on startup."""
    ensure_services_initialized()
    if auto_ingest_watcher is not None:
        auto_ingest_watcher.start()
    try:
        yield
    finally:
        if auto_ingest_watcher is not None:
            auto_ingest_watcher.stop()
        if job_manager is not None:
            job_manager.shutdown(wait=False)


# ── Correlation ID & Rate Limiting Infrastructure ───────────────────────────


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attaches or preserves X-Request-ID, injects into Loguru context, and echoes in response headers."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        with logger.contextualize(request_id=req_id):
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding window rate limiter tracking request timestamps per client."""

    def __init__(self, window_seconds: float = 60.0):
        self.window = window_seconds
        self.history: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_key: str, limit: int | None = None) -> tuple[bool, int]:
        """Check if request is within allowed rate. Returns (is_allowed, retry_after_seconds)."""
        now = time.time()
        cutoff = now - self.window
        active_limit = limit if limit is not None else settings.rate_limit_requests_per_minute

        with self._lock:
            timestamps = self.history.get(client_key, [])
            valid_timestamps = [t for t in timestamps if t > cutoff]
            if len(valid_timestamps) >= active_limit:
                oldest = valid_timestamps[0]
                retry_after = max(1, int(math.ceil(self.window - (now - oldest))))
                self.history[client_key] = valid_timestamps
                return False, retry_after

            valid_timestamps.append(now)
            self.history[client_key] = valid_timestamps
            return True, 0

    def reset(self) -> None:
        """Clear rate limiter history (useful for test resets)."""
        with self._lock:
            self.history.clear()


rate_limiter = SlidingWindowRateLimiter(window_seconds=60.0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces sliding-window rate limit returning 429 Too Many Requests when threshold exceeded."""

    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path
        if (
            path.startswith("/docs")
            or path.startswith("/redoc")
            or path == "/openapi.json"
            or path == "/health"
            or path.startswith("/api/jobs")
        ):
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        allowed, retry_after = rate_limiter.is_allowed(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Try again in {retry_after} seconds."},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)


app = FastAPI(title="ResinRAG Enterprise API", version="3.0.0", lifespan=lifespan)

# Add Middleware Stack
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*", "X-Request-ID"],
)

# Mount Authentication & RBAC Router
app.include_router(auth_router)


# ── Health & Diagnostics ───────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Liveness probe for Docker container and orchestrator health checks."""
    return {"status": "ok", "version": "3.0.0"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return inline SVG favicon to prevent browser 404 errors."""
    svg_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>'
        '</svg>'
    )
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/api/config/llm-status")
def get_llm_status(
    x_gemini_api_key: str | None = Header(default=None, alias="X-Gemini-API-Key"),
):
    """Return availability and connectivity status of supported LLM backends."""
    import requests

    has_gemini = bool(x_gemini_api_key or settings.gemini_api_key)

    lmstudio_alive = False
    try:
        base = settings.lmstudio_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        resp = requests.get(f"{base}/models", timeout=(0.1, 0.4))
        lmstudio_alive = (resp.status_code == 200)
    except Exception:
        lmstudio_alive = False

    ollama_alive = False
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=(0.1, 0.4))
        ollama_alive = (resp.status_code == 200)
    except Exception:
        ollama_alive = False

    return {
        "active_provider": settings.llm_provider,
        "providers": {
            "lmstudio": {
                "available": lmstudio_alive,
                "url": settings.lmstudio_base_url,
                "model": settings.lmstudio_model,
            },
            "ollama": {
                "available": ollama_alive,
                "url": settings.ollama_base_url,
                "model": settings.ollama_model,
            },
            "gemini": {
                "available": has_gemini,
                "configured": has_gemini,
                "model": settings.gemini_model,
            },
        },
    }


class TestStorageRequest(BaseModel):
    cloud_qdrant_url: str | None = None
    cloud_qdrant_api_key: str | None = None
    cloud_database_url: str | None = None


class StorageModeRequest(BaseModel):
    storage_mode: str = Field(..., description="Target storage mode: 'local' or 'cloud'")
    cloud_qdrant_url: str | None = None
    cloud_qdrant_api_key: str | None = None
    cloud_database_url: str | None = None


@app.get("/api/config/storage-status")
def get_storage_status() -> dict[str, Any]:
    """Return active storage mode, configured targets (masked), and health state."""
    is_cloud = (settings.storage_mode == "cloud")

    masked_qdrant = None
    if settings.cloud_qdrant_url:
        u = settings.cloud_qdrant_url
        masked_qdrant = u[:12] + "..." + u[-8:] if len(u) > 20 else u

    masked_db = None
    if settings.cloud_database_url:
        d = settings.cloud_database_url
        masked_db = d[:12] + "..." + d[-8:] if len(d) > 20 else d

    return {
        "storage_mode": settings.storage_mode,
        "is_cloud": is_cloud,
        "local": {
            "database_url": settings.database_url,
            "vector_db_dir": str(settings.vector_db_dir),
            "status": "ready",
        },
        "cloud": {
            "qdrant_url_configured": bool(settings.cloud_qdrant_url),
            "qdrant_url_masked": masked_qdrant,
            "qdrant_api_key_configured": bool(settings.cloud_qdrant_api_key),
            "database_url_configured": bool(settings.cloud_database_url),
            "database_url_masked": masked_db,
            "status": "configured" if (settings.cloud_qdrant_url or settings.cloud_database_url) else "unconfigured",
        },
    }


@app.post("/api/config/test-storage-connection")
def test_storage_connection(req: TestStorageRequest) -> dict[str, Any]:
    """Validate external cloud storage connection with timeout and health report."""
    target_qdrant_url = req.cloud_qdrant_url or settings.cloud_qdrant_url
    target_qdrant_key = req.cloud_qdrant_api_key or settings.cloud_qdrant_api_key
    target_db_url = req.cloud_database_url or settings.cloud_database_url

    results: dict[str, Any] = {}

    if target_qdrant_url:
        try:
            from qdrant_client import QdrantClient
            qc = QdrantClient(url=target_qdrant_url, api_key=target_qdrant_key, timeout=3)
            collections = qc.get_collections()
            results["qdrant"] = {
                "status": "success",
                "message": f"Successfully connected to Qdrant Cloud ({len(collections.collections)} collections found).",
            }
        except Exception as exc:
            results["qdrant"] = {
                "status": "error",
                "message": f"Qdrant connection failed: {str(exc)}",
            }

    if target_db_url:
        try:
            from sqlalchemy import create_engine, text
            test_engine = create_engine(
                target_db_url,
                connect_args={"connect_timeout": 3} if "postgres" in target_db_url else {},
            )
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            results["database"] = {
                "status": "success",
                "message": "Successfully connected to remote relational database.",
            }
        except Exception as exc:
            results["database"] = {
                "status": "error",
                "message": f"Database connection failed: {str(exc)}",
            }

    if not target_qdrant_url and not target_db_url:
        return {
            "status": "error",
            "message": "No cloud storage parameters provided to test.",
            "results": {},
        }

    overall_success = all(r.get("status") == "success" for r in results.values())
    return {
        "status": "success" if overall_success else "error",
        "message": (
            "Cloud storage connection test passed."
            if overall_success
            else "One or more cloud storage connections failed."
        ),
        "results": results,
    }


@app.post("/api/config/storage-mode")
def set_storage_mode(req: StorageModeRequest) -> dict[str, Any]:
    """Switch active storage mode ('local' vs 'cloud') with validation."""
    mode = req.storage_mode.strip().lower()
    if mode not in {"local", "cloud"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid storage mode. Must be 'local' or 'cloud'.",
        )

    if mode == "cloud":
        if req.cloud_qdrant_url:
            settings.cloud_qdrant_url = req.cloud_qdrant_url.strip()
        if req.cloud_qdrant_api_key is not None:
            settings.cloud_qdrant_api_key = req.cloud_qdrant_api_key.strip()
        if req.cloud_database_url:
            settings.cloud_database_url = req.cloud_database_url.strip()

    settings.storage_mode = mode
    logger.info("Storage mode updated to '{}'", mode)
    return {
        "status": "success",
        "storage_mode": settings.storage_mode,
        "message": f"Storage mode successfully updated to '{mode}'.",
    }


class CreateSessionRequest(BaseModel):
    title: str = "New Conversation"


class QueryRequest(BaseModel):
    query: str
    provider: str | None = None
    mode: str = "hybrid_rerank"
    api_key: str | None = None
    session_id: str | None = None


def _record_telemetry_collector(
    req: QueryRequest,
    user: User | None,
    provider: str,
    generator: Any,
    telemetry: dict[str, Any],
    cited_doc_ids: list[str],
) -> None:
    """Record query execution telemetry to relational database using TelemetryCollector output."""
    raw_model = getattr(generator, "model_name", None)
    model_name = raw_model if isinstance(raw_model, str) else provider

    telemetry_data = {
        "user_id": user.id if user else None,
        "org_id": user.org_id if user else "default_org",
        "query_text": req.query,
        "retrieval_mode": req.mode,
        "llm_provider": provider,
        "llm_model": model_name,
        "embed_latency_ms": telemetry.get("embed_latency_ms", 0.0),
        "search_latency_ms": telemetry.get("search_latency_ms", 0.0),
        "rerank_latency_ms": telemetry.get("rerank_latency_ms", 0.0),
        "llm_latency_ms": telemetry.get("llm_latency_ms", 0.0),
        "total_latency_ms": telemetry.get("total_latency_ms", 0.0),
        "retrieved_count": telemetry.get("retrieved_count", 0),
        "top_rerank_score": telemetry.get("top_rerank_score"),
        "abstained": telemetry.get("abstained", False),
        "abstention_reason": telemetry.get("abstention_reason"),
        "prompt_tokens": telemetry.get("prompt_tokens", 0),
        "completion_tokens": telemetry.get("completion_tokens", 0),
        "estimated_cost_usd": telemetry.get("estimated_cost_usd", 0.0),
        "cited_doc_ids": cited_doc_ids,
    }
    record_query_telemetry(SessionLocal, telemetry_data)


# ── PDF Upload & Job Polling ────────────────────────────────────────────────


@app.post("/api/upload", status_code=202)
async def upload_pdf(
    file: UploadFile = File(...),
    force_reindex: bool = Form(False),
    visibility: str = Form("ORG"),
    allowed_roles: str = Form("ADMIN,MANAGER,ANALYST,VIEWER"),
    current_user: User = Depends(get_current_user),
):
    """Accept a PDF upload, persist to disk, and enqueue an asynchronous ingestion job scoped to tenant."""
    ensure_services_initialized()

    # RBAC upload permission: ADMIN and MANAGER only
    if current_user.role not in {"ADMIN", "MANAGER"}:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: uploading documents requires ADMIN or MANAGER role.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    settings.pdf_upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    file_path = settings.pdf_upload_dir / safe_name

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file_path.stat().st_size == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

    parsed_roles = [r.strip().upper() for r in allowed_roles.split(",") if r.strip()]
    if not parsed_roles:
        parsed_roles = ["ADMIN", "MANAGER", "ANALYST", "VIEWER"]

    try:
        job_info = await run_in_threadpool(
            job_manager.submit_job,
            file_path=file_path,
            filename=safe_name,
            org_id=current_user.org_id,
            visibility=visibility.strip().upper(),
            allowed_roles=parsed_roles,
            uploaded_by=current_user.id,
            force_reindex=force_reindex,
        )
        return JSONResponse(status_code=202, content=job_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
async def get_ingestion_job(job_id: str):
    """Poll real-time status and page-level progress for an ingestion job."""
    ensure_services_initialized()
    job = await run_in_threadpool(job_manager.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/api/jobs")
async def list_ingestion_jobs(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """List recent asynchronous ingestion jobs scoped to current organization."""
    ensure_services_initialized()
    jobs = await run_in_threadpool(job_manager.list_jobs, limit=limit, org_id=current_user.org_id)
    return {"jobs": jobs}


@app.get("/api/documents")
async def list_documents(current_user: User = Depends(get_current_user)):
    """List all active ingested documents strictly scoped to current user's organization."""
    ensure_services_initialized()
    docs = await run_in_threadpool(job_manager.list_documents, org_id=current_user.org_id)
    return {"documents": docs}


@app.delete("/api/documents/{file_hash}")
async def delete_document(
    file_hash: str,
    current_user: User = Depends(get_current_user),
):
    """Remove all vectors for a document from Qdrant and mark record DELETED (ADMIN only)."""
    ensure_services_initialized()
    if current_user.role != "ADMIN":
        raise HTTPException(
            status_code=403,
            detail="Forbidden: deleting documents requires ADMIN role.",
        )
    deleted = await run_in_threadpool(
        job_manager.delete_document, file_hash, org_id=current_user.org_id
    )
    if deleted is None:
        raise HTTPException(
            status_code=404,
            detail=f"Active document with file_hash '{file_hash}' not found for organization '{current_user.org_id}'.",
        )
    return deleted


# ── Query Pipeline with Telemetry Spans & Token Accounting ─────────────────


def _execute_query_sync(
    req: QueryRequest,
    user: User | None = None,
    request_id: str | None = None,
    header_api_key: str | None = None,
) -> dict:
    """Run synchronous hybrid retrieval + RBAC filter + Stage Spans + Abstention Gate + LLM generation."""
    ensure_services_initialized()
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    effective_api_key = req.api_key or header_api_key
    provider = req.provider or settings.llm_provider
    try:
        generator = get_generator(provider, api_key=effective_api_key)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Generator init failed: {exc}"
        ) from exc

    collector = TelemetryCollector(request_id=request_id)
    rbac_filter = build_qdrant_rbac_filter(user) if user else None

    # Conversational memory window and reasoning trace
    conv_context = ""
    history_window: list[dict[str, Any]] = []
    reasoning_trace: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    if req.session_id and memory_manager:
        sess = memory_manager.get_session(req.session_id)
        if not sess:
            raise HTTPException(status_code=404, detail=f"Chat session '{req.session_id}' not found.")
        if user and sess.get("org_id") != user.org_id:
            raise HTTPException(status_code=403, detail="Forbidden: session belongs to another organization.")
        conv_context, history_window = memory_manager.get_context_window(req.session_id, k=3)
        if conv_context:
            reasoning_trace.append(
                {
                    "step": "memory",
                    "title": "Loaded Conversational Memory",
                    "detail": f"Active sliding window contains {len(history_window)} turns.",
                }
            )

    # Intent-based Tool Execution
    calc_match = re.search(
        r"(?:calculate|compute|what is|how much is)\s+([\d\.\s\+\-\*\/\(\)\%]+)",
        req.query,
        re.IGNORECASE,
    )
    if calc_match and any(op in calc_match.group(1) for op in ("+", "-", "*", "/")):
        expr = calc_match.group(1).strip().rstrip("?.")
        tc = {"tool": "deterministic_calculator", "arguments": {"expression": expr}}
        tool_calls.append(tc)
        res = execute_tool("deterministic_calculator", {"expression": expr})
        tool_results.append(res)
        reasoning_trace.append(
            {
                "step": "tool",
                "title": "Deterministic Calculator Tool",
                "detail": f"Evaluated '{expr}' = {res.get('result')}",
            }
        )

    summary_match = re.search(
        r"(?:summarize|summary of|overview of)\s+([a-zA-Z0-9_\-\.]+\.pdf)",
        req.query,
        re.IGNORECASE,
    )
    if summary_match:
        doc_target = summary_match.group(1).strip()
        tc = {"tool": "document_summary", "arguments": {"filename": doc_target}}
        tool_calls.append(tc)
        res = execute_tool("document_summary", {"filename": doc_target}, vector_store=vector_store)
        tool_results.append(res)
        reasoning_trace.append(
            {
                "step": "tool",
                "title": "Document Summary Tool",
                "detail": f"Extracted cover & abstract for '{doc_target}'",
            }
        )

    try:
        outcome = retriever.retrieve(
            req.query,
            initial_k=20,
            final_k=4,
            mode=req.mode,
            query_filter=rbac_filter,
            user=user,
            collector=collector,
            enable_multi_query=(req.mode in ("multi_query", "hybrid_rerank") and len(req.query) > 25),
            enable_crag=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    top_score = outcome.get("top_score")
    collector.retrieved_count = len(outcome.get("results", []))
    collector.top_rerank_score = top_score

    mq_meta = outcome.get("multi_query", {})
    if mq_meta.get("decomposed"):
        sub_q = mq_meta.get("sub_queries", [])
        reasoning_trace.append(
            {
                "step": "multi_query",
                "title": f"Decomposed into {len(sub_q)} Perspectives",
                "detail": "; ".join(sub_q),
            }
        )

    top_score_str = f"{top_score:.3f}" if top_score is not None else "N/A"
    reasoning_trace.append(
        {
            "step": "retrieval",
            "title": f"Retrieved {len(outcome.get('results', []))} Chunks (Score: {top_score_str})",
            "detail": f"Retrieved using mode '{req.mode}'.",
        }
    )

    crag = outcome.get("crag", {})
    if crag:
        grade = crag.get("grade", "high").capitalize()
        action = crag.get("action", "proceed")
        reasoning_trace.append(
            {
                "step": "crag",
                "title": f"CRAG Self-Reflection ({grade})",
                "detail": f"Action: {action}",
            }
        )

    if outcome["abstained"]:
        reason = outcome["abstention_reason"]
        collector.abstained = True
        collector.abstention_reason = reason
        collector.finalize_total()

        if reason in {"RBAC_EMPTY_RETRIEVAL", "EMPTY_RETRIEVAL"}:
            answer_msg = (
                "I cannot answer this based on the provided documents. "
                "No matching context found in the vector database."
            )
        else:
            answer_msg = "I cannot answer this based on the provided documents."

        telemetry_dict = collector.to_dict()
        _record_telemetry_collector(req, user, provider, generator, telemetry_dict, [])

        if req.session_id and memory_manager:
            try:
                memory_manager.append_turn(
                    session_id=req.session_id,
                    user_text=req.query,
                    assistant_text=answer_msg,
                    citations=[],
                    tokens=0,
                    latency_ms=collector.total_latency_ms,
                )
                reasoning_trace.append(
                    {
                        "step": "memory_persist",
                        "title": "Conversational Memory Persisted",
                        "detail": f"Saved abstained conversational turn to session '{req.session_id}'.",
                    }
                )
            except Exception as exc:
                logger.warning("Failed to persist abstained turn: {}", exc)

        return {
            "answer": answer_msg,
            "citations": [],
            "abstained": True,
            "abstention_reason": reason,
            "mode": req.mode,
            "top_score": top_score,
            "request_id": collector.request_id,
            "telemetry": telemetry_dict,
            "provider_status": "live",
            "provider_fallback": False,
            "session_id": req.session_id,
            "reasoning_trace": reasoning_trace,
            "tool_calls": tool_calls,
        }

    # Assemble composite context
    context_parts: list[str] = []
    if conv_context:
        context_parts.append(f"--- Conversation History ---\n{conv_context}")
    for tr in tool_results:
        if tr.get("status") == "success" and "result" in tr:
            context_parts.append(f"--- Calculator Tool Output ---\n{tr['expression']} = {tr['result']}")
        elif tr.get("status") == "success" and "summary_excerpt" in tr:
            context_parts.append(f"--- Document Summary ({tr.get('filename')}) ---\n{tr['summary_excerpt']}")
    if outcome.get("context"):
        context_parts.append(outcome["context"])

    synthesis_context = "\n\n".join(context_parts) if context_parts else outcome.get("context", "")

    # Generation stage with timing, token counting, and offline fallback
    t_llm_start = time.perf_counter()
    provider_status = "live"
    provider_fallback = False
    try:
        gen_result = None
        if hasattr(generator, "generate_with_telemetry"):
            try:
                raw_res = generator.generate_with_telemetry(req.query, synthesis_context)
                if isinstance(raw_res, tuple) and len(raw_res) == 3 and isinstance(raw_res[0], str):
                    gen_result = raw_res
            except Exception:
                gen_result = None

        if gen_result is not None:
            answer, prompt_tokens, completion_tokens = gen_result
        else:
            answer = generator.generate(req.query, synthesis_context)
            prompt_tokens = count_tokens(synthesis_context + " " + req.query)
            completion_tokens = count_tokens(answer)

        # Check for offline local provider connection failure strings
        if (
            "Error: LM Studio is not running" in answer
            or "Error connecting to" in answer
            or "Local LLM timed out" in answer
        ):
            has_gemini_key = bool(effective_api_key or settings.gemini_api_key)
            if has_gemini_key and provider != "gemini":
                logger.warning("Local provider '{}' is offline; automatically falling over to Gemini cloud LLM.", provider)
                try:
                    gem_gen = get_generator("gemini", api_key=effective_api_key)
                    if hasattr(gem_gen, "generate_with_telemetry"):
                        answer, prompt_tokens, completion_tokens = gem_gen.generate_with_telemetry(req.query, synthesis_context)
                    else:
                        answer = gem_gen.generate(req.query, synthesis_context)
                        prompt_tokens = count_tokens(synthesis_context + " " + req.query)
                        completion_tokens = count_tokens(answer)
                    provider = "gemini"
                    provider_status = "live"
                    provider_fallback = True
                except Exception as fallback_exc:
                    logger.error("Gemini automatic fallback failed: {}", fallback_exc)
                    answer = synthesize_extractive_answer(req.query, synthesis_context)
                    provider_status = "degraded_extractive"
                    provider_fallback = True
                    completion_tokens = count_tokens(answer)
            else:
                logger.warning("Local provider '{}' is offline; synthesizing grounded extractive fallback.", provider)
                answer = synthesize_extractive_answer(req.query, synthesis_context)
                provider_status = "degraded_extractive"
                provider_fallback = True
                completion_tokens = count_tokens(answer)
    except (ConnectionError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        has_gemini_key = bool(effective_api_key or settings.gemini_api_key)
        if has_gemini_key and provider != "gemini":
            logger.warning("Connection error to provider '{}' ({}); automatically falling over to Gemini cloud LLM.", provider, exc)
            try:
                gem_gen = get_generator("gemini", api_key=effective_api_key)
                if hasattr(gem_gen, "generate_with_telemetry"):
                    answer, prompt_tokens, completion_tokens = gem_gen.generate_with_telemetry(req.query, synthesis_context)
                else:
                    answer = gem_gen.generate(req.query, synthesis_context)
                    prompt_tokens = count_tokens(synthesis_context + " " + req.query)
                    completion_tokens = count_tokens(answer)
                provider = "gemini"
                provider_status = "live"
                provider_fallback = True
            except Exception as fallback_exc:
                logger.error("Gemini automatic fallback failed: {}", fallback_exc)
                answer = synthesize_extractive_answer(req.query, synthesis_context)
                provider_status = "degraded_extractive"
                provider_fallback = True
                prompt_tokens = count_tokens(synthesis_context + " " + req.query)
                completion_tokens = count_tokens(answer)
        else:
            logger.warning(f"Connection error to provider '{provider}' ({exc}); synthesizing grounded extractive fallback.")
            answer = synthesize_extractive_answer(req.query, synthesis_context)
            provider_status = "degraded_extractive"
            provider_fallback = True
            prompt_tokens = count_tokens(synthesis_context + " " + req.query)
            completion_tokens = count_tokens(answer)
    except Exception as exc:
        logger.error(f"Generation error: {exc}")
        answer = f"Error generating response: {exc}"
        prompt_tokens = count_tokens(synthesis_context + " " + req.query)
        completion_tokens = 0

    collector.record_stage("llm", (time.perf_counter() - t_llm_start) * 1000.0)
    collector.prompt_tokens = prompt_tokens
    collector.completion_tokens = completion_tokens
    model_name = getattr(generator, "model_name", provider)
    if not isinstance(model_name, str):
        model_name = provider
    collector.estimated_cost_usd = calculate_query_cost(
        provider, model_name, prompt_tokens, completion_tokens
    )

    # Tier-2 Post-LLM Abstention Gate
    if "cannot answer this based on the provided documents" in answer.lower():
        collector.abstained = True
        collector.abstention_reason = "LLM_ABSTAIN"
        collector.finalize_total()
        telemetry_dict = collector.to_dict()
        _record_telemetry_collector(req, user, provider, generator, telemetry_dict, [])

        if req.session_id and memory_manager:
            try:
                memory_manager.append_turn(
                    session_id=req.session_id,
                    user_text=req.query,
                    assistant_text=answer,
                    citations=[],
                    tokens=prompt_tokens + completion_tokens,
                    latency_ms=collector.total_latency_ms,
                )
                reasoning_trace.append(
                    {
                        "step": "memory_persist",
                        "title": "Conversational Memory Persisted",
                        "detail": f"Saved abstained conversational turn to session '{req.session_id}'.",
                    }
                )
            except Exception as exc:
                logger.warning("Failed to persist abstained turn: {}", exc)

        return {
            "answer": answer,
            "citations": [],
            "abstained": True,
            "abstention_reason": "LLM_ABSTAIN",
            "mode": req.mode,
            "top_score": top_score,
            "request_id": collector.request_id,
            "telemetry": telemetry_dict,
            "provider_status": provider_status,
            "provider_fallback": provider_fallback,
            "session_id": req.session_id,
            "reasoning_trace": reasoning_trace,
            "tool_calls": tool_calls,
        }

    citations = []
    cited_doc_ids: list[str] = []
    for hit in outcome["results"]:
        meta = hit.get("metadata", {})
        page_start = meta.get("page_number", "?")
        page_end = meta.get("page_end", page_start)
        doc_id = meta.get("file_hash") or meta.get("source_pdf", "Unknown")
        cited_doc_ids.append(str(doc_id))
        raw_snippet = hit.get("text") or ""
        clean_snippet = raw_snippet.replace("\ufffd", " ").strip()[:240]
        citations.append(
            {
                "pdf": meta.get("source_pdf", "Unknown"),
                "page": page_start,
                "page_end": page_end,
                "parent_id": hit.get("parent_id") or meta.get("parent_id"),
                "score": round(hit.get("score", 0.0), 3),
                "snippet": clean_snippet,
            }
        )

    collector.finalize_total()
    telemetry_dict = collector.to_dict()
    _record_telemetry_collector(req, user, provider, generator, telemetry_dict, cited_doc_ids)

    # Persist conversational turn to relational database if session_id is active
    if req.session_id and memory_manager:
        try:
            memory_manager.append_turn(
                session_id=req.session_id,
                user_text=req.query,
                assistant_text=answer,
                citations=citations,
                tokens=prompt_tokens + completion_tokens,
                latency_ms=collector.total_latency_ms,
            )
            reasoning_trace.append(
                {
                    "step": "memory_persist",
                    "title": "Conversational Memory Persisted",
                    "detail": f"Saved conversational turn to session '{req.session_id}'.",
                }
            )
        except Exception as exc:
            logger.warning("Failed to persist turn to session '{}': {}", req.session_id, exc)

    reasoning_trace.append(
        {
            "step": "synthesis",
            "title": "Grounded Synthesis",
            "detail": f"Generated answer citing {len(citations)} document blocks ({provider_status}).",
        }
    )

    return {
        "answer": answer,
        "citations": citations,
        "abstained": False,
        "abstention_reason": None,
        "mode": req.mode,
        "top_score": top_score,
        "request_id": collector.request_id,
        "telemetry": telemetry_dict,
        "provider_status": provider_status,
        "provider_fallback": provider_fallback,
        "session_id": req.session_id,
        "reasoning_trace": reasoning_trace,
        "tool_calls": tool_calls,
    }


@app.post("/api/query")
async def query_pipeline(
    req: QueryRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Execute grounded RAG query on threadpool scoped by pre-retrieval RBAC filter with stage telemetry."""
    req_id = (
        getattr(request.state, "request_id", None)
        or request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    header_key = request.headers.get("X-Gemini-API-Key")
    return await run_in_threadpool(_execute_query_sync, req, current_user, req_id, header_key)


# ── Multi-Turn Conversational Session Endpoints ────────────────────────────


@app.post("/api/chat/sessions")
async def create_chat_session(
    body: CreateSessionRequest | None = None,
    current_user: User = Depends(get_current_user),
):
    """Create a new conversational session for the authenticated user."""
    ensure_services_initialized()
    title = body.title if body and body.title else "New Conversation"
    session_data = await run_in_threadpool(
        memory_manager.create_session,
        user_id=current_user.id,
        org_id=current_user.org_id,
        title=title,
    )
    return session_data


@app.get("/api/chat/sessions")
async def list_chat_sessions(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """List conversational sessions belonging to current user in current organization."""
    ensure_services_initialized()
    sessions = await run_in_threadpool(
        memory_manager.list_sessions,
        user_id=current_user.id,
        org_id=current_user.org_id,
        limit=limit,
    )
    return {"sessions": sessions}


@app.get("/api/chat/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
):
    """Retrieve full chronological message history for a specific chat session."""
    ensure_services_initialized()
    session_record = await run_in_threadpool(memory_manager.get_session, session_id)
    if not session_record:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    if session_record["org_id"] != current_user.org_id:
        raise HTTPException(status_code=403, detail="Forbidden: cannot access sessions from another tenant.")

    messages = await run_in_threadpool(memory_manager.get_messages, session_id, limit=limit)
    return {"session": session_record, "messages": messages}


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Delete a chat session and all associated turns."""
    ensure_services_initialized()
    session_record = await run_in_threadpool(memory_manager.get_session, session_id)
    if not session_record:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    if session_record["org_id"] != current_user.org_id:
        raise HTTPException(status_code=403, detail="Forbidden: cannot delete sessions from another tenant.")

    deleted = await run_in_threadpool(memory_manager.delete_session, session_id)
    return {"deleted": deleted, "session_id": session_id}


# ── Observability & Telemetry Metrics Endpoint ─────────────────────────────


@app.get("/api/metrics")
async def get_metrics(
    org_id: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return aggregated latency percentiles, token usage, and cost telemetry."""
    target_org_id = org_id
    if current_user.role != "ADMIN":
        target_org_id = current_user.org_id

    return await run_in_threadpool(get_aggregated_metrics, db, target_org_id)


@app.delete("/api/metrics")
async def reset_metrics(
    org_id: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Clear QueryTelemetry records for the specified organization (ADMIN role required)."""
    if current_user.role != "ADMIN":
        raise HTTPException(
            status_code=403,
            detail="Forbidden: resetting telemetry metrics requires ADMIN role.",
        )

    target_org_id = org_id or current_user.org_id

    def _clear_sync(sess: Session, org_filter: str | None) -> int:
        query = sess.query(QueryTelemetry)
        if org_filter:
            query = query.filter(QueryTelemetry.org_id == org_filter)
        count = query.delete(synchronize_session=False)
        sess.commit()
        return count

    deleted_count = await run_in_threadpool(_clear_sync, db, target_org_id)
    return {
        "status": "cleared",
        "org_id": target_org_id,
        "deleted_count": deleted_count,
    }


# ── Static Web UI ──────────────────────────────────────────────────────────

os.makedirs("src/api/static", exist_ok=True)
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
