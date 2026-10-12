"""FastAPI Gateway exposing non-blocking 202 PDF upload, job polling, document CRUD, RAG query, and telemetry metrics (L4)."""

from __future__ import annotations

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.api.auth_routes import router as auth_router
from src.auth.dependencies import build_qdrant_rbac_filter, get_current_user
from src.auth.security import seed_demo_accounts
from src.config import settings
from src.database.models import QueryTelemetry, User
from src.database.relational import SessionLocal, get_db, init_db
from src.database.vector_store import VectorStore
from src.embeddings.embedder import LocalEmbedder
from src.generation.generator import get_generator
from src.ingestion.pipeline import IngestionPipeline
from src.jobs.ingestion_worker import IngestionJobManager
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


def ensure_services_initialized() -> None:
    """Lazily initialize backend singletons if not already initialized by lifespan."""
    global embedder, vector_store, retriever, ingestion_pipeline, job_manager
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize relational DB, demo accounts, vector store, embedder, and async worker pool on startup."""
    ensure_services_initialized()
    yield


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


app = FastAPI(title="Enterprise RAG Pipeline API", version="2.5.0", lifespan=lifespan)

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
    return {"status": "ok", "version": "2.5.0"}


class QueryRequest(BaseModel):
    query: str
    provider: str | None = None
    mode: str = "hybrid_rerank"


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
) -> dict:
    """Run synchronous hybrid retrieval + RBAC filter + Stage Spans + Abstention Gate + LLM generation."""
    ensure_services_initialized()
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    provider = req.provider or settings.llm_provider
    try:
        generator = get_generator(provider)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Generator init failed: {exc}"
        ) from exc

    collector = TelemetryCollector(request_id=request_id)
    rbac_filter = build_qdrant_rbac_filter(user) if user else None

    try:
        outcome = retriever.retrieve(
            req.query,
            initial_k=20,
            final_k=4,
            mode=req.mode,
            query_filter=rbac_filter,
            user=user,
            collector=collector,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    top_score = outcome.get("top_score")
    collector.retrieved_count = len(outcome.get("results", []))
    collector.top_rerank_score = top_score

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

        return {
            "answer": answer_msg,
            "citations": [],
            "abstained": True,
            "abstention_reason": reason,
            "mode": req.mode,
            "top_score": top_score,
            "request_id": collector.request_id,
            "telemetry": telemetry_dict,
        }

    # Generation stage with timing and token counting
    t_llm_start = time.perf_counter()
    try:
        gen_result = None
        if hasattr(generator, "generate_with_telemetry"):
            try:
                raw_res = generator.generate_with_telemetry(req.query, outcome["context"])
                if isinstance(raw_res, tuple) and len(raw_res) == 3 and isinstance(raw_res[0], str):
                    gen_result = raw_res
            except Exception:
                gen_result = None

        if gen_result is not None:
            answer, prompt_tokens, completion_tokens = gen_result
        else:
            answer = generator.generate(req.query, outcome["context"])
            prompt_tokens = count_tokens(outcome["context"] + " " + req.query)
            completion_tokens = count_tokens(answer)
    except Exception as exc:
        logger.error(f"Generation error: {exc}")
        answer = "Error generating response."
        prompt_tokens = count_tokens(req.query)
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
        return {
            "answer": answer,
            "citations": [],
            "abstained": True,
            "abstention_reason": "LLM_ABSTAIN",
            "mode": req.mode,
            "top_score": top_score,
            "request_id": collector.request_id,
            "telemetry": telemetry_dict,
        }

    citations = []
    cited_doc_ids: list[str] = []
    for hit in outcome["results"]:
        meta = hit.get("metadata", {})
        page_start = meta.get("page_number", "?")
        page_end = meta.get("page_end", page_start)
        doc_id = meta.get("file_hash") or meta.get("source_pdf", "Unknown")
        cited_doc_ids.append(str(doc_id))
        citations.append(
            {
                "pdf": meta.get("source_pdf", "Unknown"),
                "page": page_start,
                "page_end": page_end,
                "parent_id": hit.get("parent_id") or meta.get("parent_id"),
                "score": round(hit.get("score", 0.0), 3),
                "snippet": (hit.get("text") or "")[:240],
            }
        )

    collector.finalize_total()
    telemetry_dict = collector.to_dict()
    _record_telemetry_collector(req, user, provider, generator, telemetry_dict, cited_doc_ids)

    return {
        "answer": answer,
        "citations": citations,
        "abstained": False,
        "abstention_reason": None,
        "mode": req.mode,
        "top_score": top_score,
        "request_id": collector.request_id,
        "telemetry": telemetry_dict,
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
    return await run_in_threadpool(_execute_query_sync, req, current_user, req_id)


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


# ── Static Web UI ──────────────────────────────────────────────────────────

os.makedirs("src/api/static", exist_ok=True)
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
