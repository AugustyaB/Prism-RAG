"""FastAPI Gateway exposing non-blocking 202 PDF upload, job polling, document CRUD, and RAG query endpoints (L4)."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil
import time
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.api.auth_routes import router as auth_router
from src.auth.dependencies import build_qdrant_rbac_filter, get_current_user
from src.auth.security import seed_demo_accounts
from src.config import settings
from src.database.models import QueryTelemetry, User
from src.database.relational import SessionLocal, init_db
from src.database.vector_store import VectorStore
from src.embeddings.embedder import LocalEmbedder
from src.generation.generator import get_generator
from src.ingestion.pipeline import IngestionPipeline
from src.jobs.ingestion_worker import IngestionJobManager
from src.retrieval.retriever import Retriever

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


app = FastAPI(title="Enterprise RAG Pipeline API", version="2.4.0", lifespan=lifespan)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount Authentication & RBAC Router
app.include_router(auth_router)


class QueryRequest(BaseModel):
    query: str
    provider: str | None = None
    mode: str = "hybrid_rerank"


def _record_telemetry(
    req: QueryRequest,
    user: User | None,
    provider: str,
    generator: Any,
    outcome: dict[str, Any],
    total_latency_ms: float,
    abstained: bool | None = None,
    reason: str | None = None,
) -> None:
    """Record query execution telemetry to relational database."""
    try:
        user_id = user.id if user else None
        org_id = user.org_id if user else "default_org"
        is_abstained = abstained if abstained is not None else outcome.get("abstained", False)
        abstention_reason = reason if reason is not None else outcome.get("abstention_reason")

        raw_model = getattr(generator, "model_name", None)
        model_name = raw_model if isinstance(raw_model, str) else provider

        with SessionLocal() as db_session:
            record = QueryTelemetry(
                user_id=user_id,
                org_id=org_id,
                query_text=req.query,
                retrieval_mode=req.mode,
                llm_provider=provider,
                llm_model=model_name,
                total_latency_ms=total_latency_ms,
                retrieved_count=len(outcome.get("results", [])),
                top_rerank_score=outcome.get("top_score"),
                abstained=is_abstained,
                abstention_reason=abstention_reason,
            )
            db_session.add(record)
            db_session.commit()
    except Exception as exc:
        logger.warning("Query telemetry recording skipped: {}", exc)


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
    if current_user.role.upper() not in {"ADMIN", "MANAGER"}:
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
    if current_user.role.upper() != "ADMIN":
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


def _execute_query_sync(req: QueryRequest, user: User | None = None) -> dict:
    """Run synchronous PyTorch hybrid retrieval + Pre-Retrieval RBAC + Two-Tier Abstention Gate + LLM generation."""
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

    start_total = time.perf_counter()
    rbac_filter = build_qdrant_rbac_filter(user) if user else None

    try:
        outcome = retriever.retrieve(
            req.query,
            initial_k=20,
            final_k=4,
            mode=req.mode,
            query_filter=rbac_filter,
            user=user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if outcome["abstained"]:
        reason = outcome["abstention_reason"]
        if reason in {"RBAC_EMPTY_RETRIEVAL", "EMPTY_RETRIEVAL"}:
            answer_msg = (
                "I cannot answer this based on the provided documents. "
                "No matching context found in the vector database."
            )
        else:
            answer_msg = "I cannot answer this based on the provided documents."

        total_ms = round((time.perf_counter() - start_total) * 1000.0, 2)
        _record_telemetry(req, user, provider, generator, outcome, total_ms)
        return {
            "answer": answer_msg,
            "citations": [],
            "abstained": True,
            "abstention_reason": reason,
            "mode": req.mode,
            "top_score": outcome.get("top_score"),
        }

    answer = generator.generate(req.query, outcome["context"])

    # Tier-2 Post-LLM Abstention Gate
    if "cannot answer this based on the provided documents" in answer.lower():
        total_ms = round((time.perf_counter() - start_total) * 1000.0, 2)
        _record_telemetry(
            req,
            user,
            provider,
            generator,
            outcome,
            total_ms,
            abstained=True,
            reason="LLM_ABSTAIN",
        )
        return {
            "answer": answer,
            "citations": [],
            "abstained": True,
            "abstention_reason": "LLM_ABSTAIN",
            "mode": req.mode,
            "top_score": outcome.get("top_score"),
        }

    citations = []
    for hit in outcome["results"]:
        meta = hit.get("metadata", {})
        page_start = meta.get("page_number", "?")
        page_end = meta.get("page_end", page_start)
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

    total_ms = round((time.perf_counter() - start_total) * 1000.0, 2)
    _record_telemetry(req, user, provider, generator, outcome, total_ms)

    return {
        "answer": answer,
        "citations": citations,
        "abstained": False,
        "abstention_reason": None,
        "mode": req.mode,
        "top_score": outcome.get("top_score"),
    }


@app.post("/api/query")
async def query_pipeline(
    req: QueryRequest,
    current_user: User = Depends(get_current_user),
):
    """Execute grounded RAG query on the threadpool scoped by pre-retrieval RBAC filter."""
    return await run_in_threadpool(_execute_query_sync, req, current_user)


# Create static directory if it doesn't exist so mounting won't crash
os.makedirs("src/api/static", exist_ok=True)
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
