#!/usr/bin/env python
"""
============================================================================
Project-RAG_LLM — Model Pipeline & Multi-Scenario Test Utility
============================================================================

Tests:
1. LLM Provider Health: checks LM Studio, Ollama, and Google Gemini.
2. Auto-Fallback: if local LLM is offline, automatically switches to Gemini with API key.
3. Recent Document Ingestion & Verification: loads recent PDFs from Downloads folder.
4. Comprehensive Retrieval & Generation Scenarios:
   - Scenario A: High-Level Document Executive Summary & Problem Scope.
   - Scenario B: Key Results, Experimental Findings & Methodology.
   - Scenario C: Specific Technical Details / Domain Inquiries.
5. Telemetry & Scoring Scorecard.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger
import requests

from src.config import settings
from src.database.relational import SessionLocal, init_db
from src.database.models import DocumentRecord, User
from src.auth.security import seed_demo_accounts
from src.embeddings.embedder import LocalEmbedder
from src.database.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import get_generator, GeminiGenerator, LMStudioGenerator, OllamaGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.jobs.ingestion_worker import IngestionJobManager


def check_providers() -> dict[str, dict[str, Any]]:
    """Test connectivity for LM Studio, Ollama, and Gemini."""
    status: dict[str, dict[str, Any]] = {}

    # 1. LM Studio
    lm_alive = False
    lm_err = ""
    try:
        base = settings.lmstudio_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        resp = requests.get(f"{base}/models", timeout=(0.15, 0.5))
        lm_alive = (resp.status_code == 200)
    except Exception as e:
        lm_err = str(e)
    status["lmstudio"] = {
        "alive": lm_alive,
        "endpoint": settings.lmstudio_base_url,
        "model": settings.lmstudio_model,
        "error": lm_err,
    }

    # 2. Ollama
    ol_alive = False
    ol_err = ""
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=(0.15, 0.5))
        ol_alive = (resp.status_code == 200)
    except Exception as e:
        ol_err = str(e)
    status["ollama"] = {
        "alive": ol_alive,
        "endpoint": settings.ollama_base_url,
        "model": settings.ollama_model,
        "error": ol_err,
    }

    # 3. Gemini
    gem_alive = False
    gem_err = ""
    if settings.gemini_api_key:
        try:
            from google import genai
            client = genai.Client(api_key=settings.gemini_api_key)
            test_resp = client.models.generate_content(
                model=settings.gemini_model,
                contents="Ping. Respond with 'PONG'."
            )
            gem_alive = bool(test_resp.text and "PONG" in test_resp.text.upper())
        except Exception as e:
            gem_err = str(e)
    else:
        gem_err = "No GEMINI_API_KEY set in .env"

    status["gemini"] = {
        "alive": gem_alive,
        "endpoint": "https://generativelanguage.googleapis.com",
        "model": settings.gemini_model,
        "error": gem_err,
    }

    return status


def get_recent_downloads_pdfs(limit: int = 5) -> list[Path]:
    """Find recent PDF files in the user's Downloads directory."""
    user_home = Path.home()
    downloads_dir = user_home / "Downloads"
    if not downloads_dir.exists():
        return []
    pdfs = list(downloads_dir.glob("*.pdf"))
    pdfs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pdfs[:limit]


def ensure_document_ingested(pdf_path: Path, job_manager: IngestionJobManager) -> DocumentRecord:
    """Ensure the target PDF is ingested in both SQLite metadata and Qdrant vector store."""
    with SessionLocal() as db:
        existing = db.query(DocumentRecord).filter(DocumentRecord.filename == pdf_path.name).first()
        if existing and existing.total_chunks > 0:
            logger.info("Document '{}' already indexed ({} pages, {} chunks).", pdf_path.name, existing.total_pages, existing.total_chunks)
            return existing

    logger.info("Ingesting recent document: '{}' ({} MB)...", pdf_path.name, round(pdf_path.stat().st_size / (1024 * 1024), 2))
    job = job_manager.submit_job(
        file_path=pdf_path,
        filename=pdf_path.name,
        org_id="default_org",
        visibility="ORG",
        allowed_roles=["ADMIN", "MANAGER", "ANALYST", "VIEWER"],
        uploaded_by="system_tester",
    )

    # Poll until completed
    while True:
        info = job_manager.get_job(job["job_id"])
        if info["status"] in {"COMPLETED", "FAILED"}:
            break
        time.sleep(1.0)

    if info["status"] == "FAILED":
        raise RuntimeError(f"Ingestion failed for {pdf_path.name}: {info.get('error_message')}")

    with SessionLocal() as db:
        doc = db.query(DocumentRecord).filter(DocumentRecord.filename == pdf_path.name).first()
        return doc


def run_test_scenario(
    scenario_title: str,
    query: str,
    retriever: Retriever,
    generator: Any,
    provider_name: str,
    admin_user: User,
) -> dict[str, Any]:
    """Execute a single retrieval + generation scenario and print structured results."""
    print("\n" + "=" * 80)
    print(f"[SCENARIO] {scenario_title}")
    print(f"[QUERY]    \"{query}\"")
    print("=" * 80)

    t_ret_start = time.perf_counter()
    outcome = retriever.retrieve(
        query=query,
        initial_k=15,
        final_k=4,
        mode="hybrid_rerank",
        user=admin_user,
    )
    t_ret_ms = (time.perf_counter() - t_ret_start) * 1000.0

    print(f"\n[RETRIEVAL METRICS]")
    print(f"  * Top Rerank Score : {outcome.get('top_score', 0.0):.4f}")
    print(f"  * Chunks Retained  : {len(outcome.get('results', []))}")
    print(f"  * Retrieval Latency: {t_ret_ms:.1f} ms")
    print(f"  * Abstained        : {outcome.get('abstained', False)}")

    if outcome.get("abstained"):
        print(f"\n[WARN] Abstention Triggered: {outcome.get('abstention_reason')}")
        return {
            "answer": "Abstained",
            "retrieval_ms": t_ret_ms,
            "llm_ms": 0.0,
            "abstained": True,
        }

    # Print context snippets
    print("\n[CONTEXT SOURCES]")
    for idx, hit in enumerate(outcome.get("results", []), start=1):
        meta = hit.get("metadata", {})
        p_start = meta.get("page_start", "?")
        p_end = meta.get("page_end", p_start)
        page_str = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"
        snippet = (hit.get("text") or "")[:120].replace("\n", " ")
        print(f"  [{idx}] {meta.get('filename')} ({page_str}) - Score: {hit.get('score', 0.0):.4f}")
        print(f"      Excerpt: \"{snippet}...\"")

    # LLM Generation
    t_llm_start = time.perf_counter()
    answer = generator.generate(query, outcome["context"])
    t_llm_ms = (time.perf_counter() - t_llm_start) * 1000.0

    print(f"\n[{provider_name.upper()} RESPONSE] ({t_llm_ms:.1f} ms):")
    print("-" * 80)
    print(answer)
    print("-" * 80)

    return {
        "answer": answer,
        "retrieval_ms": t_ret_ms,
        "llm_ms": t_llm_ms,
        "top_score": outcome.get("top_score"),
        "abstained": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Test RAG Model Pipeline with Local and Cloud LLMs.")
    parser.add_argument("--pdf", type=str, help="Specific PDF file to test with (default: latest in Downloads).")
    parser.add_argument("--provider", type=str, choices=["auto", "lmstudio", "ollama", "gemini"], default="auto", help="LLM Provider to use.")
    args = parser.parse_args()

    print("=" * 80)
    print("Project-RAG_LLM -- Model Pipeline & Document Testing Suite")
    print("=" * 80)

    # 1. Initialize Relational DB & Services
    settings.ensure_dirs()
    init_db()
    seed_demo_accounts()

    with SessionLocal() as db:
        admin_user = db.query(User).filter(User.role == "ADMIN", User.org_id == "default_org").first()
        if not admin_user:
            raise RuntimeError("Default admin user not found.")

    embedder = LocalEmbedder()
    vector_store = VectorStore(embedder)
    retriever = Retriever(vector_store)
    pipeline = IngestionPipeline()
    job_manager = IngestionJobManager(pipeline, vector_store)

    # 2. Check Provider Connectivity
    print("\n[INFO] Checking LLM Provider Connectivity...")
    prov_status = check_providers()
    print("-" * 80)
    for p_name, p_info in prov_status.items():
        state_icon = "[ONLINE]" if p_info["alive"] else "[OFFLINE]"
        print(f"  * {p_name.upper():<10}: {state_icon} | Model: {p_info['model']} | Endpoint: {p_info['endpoint']}")
        if not p_info["alive"] and p_info["error"]:
            print(f"    -> Error detail: {p_info['error']}")
    print("-" * 80)

    # 3. Determine Effective Generator
    chosen_provider = args.provider
    if chosen_provider == "auto":
        if prov_status["lmstudio"]["alive"]:
            chosen_provider = "lmstudio"
        elif prov_status["ollama"]["alive"]:
            chosen_provider = "ollama"
        elif prov_status["gemini"]["alive"]:
            chosen_provider = "gemini"
        else:
            chosen_provider = "gemini"

    print(f"\n[INFO] Selected Active Provider: {chosen_provider.upper()}")
    if chosen_provider == "gemini" and not prov_status["lmstudio"]["alive"]:
        print("  [Notice] Local LM Studio is not running on port 1234. Automatically routing through Cloud Gemini with your configured API key.")

    generator = get_generator(chosen_provider, api_key=settings.gemini_api_key)

    # 4. Identify Document to Test
    target_pdf: Path | None = None
    if args.pdf:
        target_pdf = Path(args.pdf)
    else:
        recent_pdfs = get_recent_downloads_pdfs(5)
        if recent_pdfs:
            print("\n[RECENT DOWNLOADS] PDFs Found in Downloads folder:")
            for i, p in enumerate(recent_pdfs, start=1):
                size_mb = round(p.stat().st_size / (1024 * 1024), 2)
                mtime_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))
                print(f"  [{i}] {p.name:<60} ({size_mb} MB, {mtime_str})")
            target_pdf = recent_pdfs[0]
        else:
            # Fallback to catalog upload
            uploads = list(settings.pdf_upload_dir.glob("*.pdf"))
            if uploads:
                target_pdf = uploads[0]

    if not target_pdf or not target_pdf.exists():
        print("[ERROR] No PDF found to test. Please specify --pdf <path>.")
        sys.exit(1)

    print(f"\n[TARGET DOCUMENT] {target_pdf.name}")
    doc_record = ensure_document_ingested(target_pdf, job_manager)

    # 5. Execute Scenarios
    scenarios = [
        (
            "Document Executive Summary & Problem Scope",
            f"What is '{target_pdf.name}' about? Provide a detailed executive summary of the document, explaining its main objectives, context, and problem statement."
        ),
        (
            "Methodology, Key Results & Technical Findings",
            f"What are the main results, technical methodology, experimental findings, or conclusions detailed in '{target_pdf.name}'?"
        ),
        (
            "Specific Technical Details & Formulations",
            f"List key technical details, formulas, quantitative models, datasets, or specific guidelines referenced in '{target_pdf.name}'."
        )
    ]

    results_summary = []
    for title, q in scenarios:
        res = run_test_scenario(title, q, retriever, generator, chosen_provider, admin_user)
        results_summary.append((title, res))

    # 6. Final Summary Scorecard
    print("\n" + "=" * 80)
    print("Testing Pipeline Summary Scorecard")
    print("=" * 80)
    print(f"Document        : {target_pdf.name} ({doc_record.total_pages} pages, {doc_record.total_chunks} chunks)")
    print(f"Active Provider : {chosen_provider.upper()} ({prov_status.get(chosen_provider, {}).get('model', '')})")
    for title, res in results_summary:
        status_tag = "[PASSED]" if not res.get("abstained") and len(res.get("answer", "")) > 100 else "[CHECK]"
        ret_ms = res.get("retrieval_ms", 0.0)
        llm_ms = res.get("llm_ms", 0.0)
        print(f"  * {title:<45}: {status_tag} (Retrieval: {ret_ms:.0f}ms | Generation: {llm_ms:.0f}ms)")
    print("=" * 80)
    print("Testing completed successfully.")



if __name__ == "__main__":
    main()
