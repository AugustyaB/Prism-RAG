"""FastAPI Gateway exposing non-blocking 202 PDF upload, job polling, document CRUD, and RAG query endpoints (L4)."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings
from src.database.relational import init_db
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
    embedder = LocalEmbedder()
    vector_store = VectorStore(embedder)
    retriever = Retriever(vector_store)
    ingestion_pipeline = IngestionPipeline()
    job_manager = IngestionJobManager(ingestion_pipeline, vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize relational DB, vector store, embedder, and async job worker pool on startup."""
    ensure_services_initialized()
    yield


app = FastAPI(title="Enterprise RAG Pipeline API", version="2.1.0", lifespan=lifespan)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    provider: str | None = None
    mode: str = "hybrid_rerank"


@app.post("/api/upload", status_code=202)
async def upload_pdf(
    file: UploadFile = File(...),
    force_reindex: bool = Form(False),
    visibility: str = Form("ORG"),
):
    """Accept a PDF upload, persist to disk, and enqueue an asynchronous ingestion job (202 Accepted)."""
    ensure_services_initialized()
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

    try:
        job_info = await run_in_threadpool(
            job_manager.submit_job,
            file_path=file_path,
            filename=safe_name,
            visibility=visibility,
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
async def list_ingestion_jobs(limit: int = 50):
    """List recent asynchronous ingestion jobs."""
    ensure_services_initialized()
    jobs = await run_in_threadpool(job_manager.list_jobs, limit)
    return {"jobs": jobs}


@app.get("/api/documents")
async def list_documents():
    """List all active ingested documents in the relational catalog."""
    ensure_services_initialized()
    docs = await run_in_threadpool(job_manager.list_documents)
    return {"documents": docs}


@app.delete("/api/documents/{file_hash}")
async def delete_document(file_hash: str):
    """Remove all vectors for a document from Qdrant and mark its record DELETED."""
    ensure_services_initialized()
    deleted = await run_in_threadpool(job_manager.delete_document, file_hash)
    if deleted is None:
        raise HTTPException(
            status_code=404,
            detail=f"Active document with file_hash '{file_hash}' not found.",
        )
    return deleted


def _execute_query_sync(req: QueryRequest) -> dict:
    """Run synchronous PyTorch hybrid retrieval + Two-Tier Abstention Gate + LLM generation."""
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

    try:
        outcome = retriever.retrieve(
            req.query,
            initial_k=20,
            final_k=4,
            mode=req.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if outcome["abstained"]:
        reason = outcome["abstention_reason"]
        answer_msg = (
            "I cannot answer this based on the provided documents. No matching context found in the vector database."
            if reason == "EMPTY_RETRIEVAL"
            else "I cannot answer this based on the provided documents."
        )
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

    return {
        "answer": answer,
        "citations": citations,
        "abstained": False,
        "abstention_reason": None,
        "mode": req.mode,
        "top_score": outcome.get("top_score"),
    }


@app.post("/api/query")
async def query_pipeline(req: QueryRequest):
    """Execute grounded RAG query on the threadpool without blocking job status polling."""
    return await run_in_threadpool(_execute_query_sync, req)


# Create static directory if it doesn't exist so mounting won't crash
os.makedirs("src/api/static", exist_ok=True)
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
