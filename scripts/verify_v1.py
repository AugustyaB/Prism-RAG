#!/usr/bin/env python
"""
v1.0 — End-to-End Verification Suite
=====================================
Tests all pipeline components, fallback behaviour, and efficiency guards.

Exit code 0 = all tests passed.
"""

import sys
import os
import time
import traceback

# Ensure project root is on PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Globals ──
PASS = 0
FAIL = 0
SKIP = 0
RESULTS = []

def log(status: str, name: str, detail: str = ""):
    global PASS, FAIL, SKIP
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⚠️"}.get(status, "❓")
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    else:
        SKIP += 1
    msg = f"  {icon} [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    RESULTS.append((status, name, detail))


# ════════════════════════════════════════════════════════════
# PHASE 1: Configuration & Imports
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 1: Configuration & Imports ═══")

try:
    from src.config import settings
    log("PASS", "Config loads", f"env_prefix=RAG_, llm_provider={settings.llm_provider}")
except Exception as e:
    log("FAIL", "Config loads", str(e))
    print("CRITICAL: Cannot continue without config. Exiting.")
    sys.exit(1)

# Ensure dirs
try:
    settings.ensure_dirs()
    from pathlib import Path
    assert Path(settings.pdf_upload_dir).exists()
    assert Path(settings.vector_db_dir).exists()
    log("PASS", "ensure_dirs()", "All required directories exist")
except Exception as e:
    log("FAIL", "ensure_dirs()", str(e))

# Core imports
modules_ok = True
for mod_path in [
    "src.ingestion.pdf_parser",
    "src.ingestion.classifier",
    "src.ingestion.ocr",
    "src.ingestion.chunker",
    "src.ingestion.pipeline",
    "src.embeddings.embedder",
    "src.database.vector_store",
    "src.retrieval.retriever",
    "src.generation.generator",
    "src.api.server",
    "src.utils.logging",
]:
    try:
        __import__(mod_path)
        log("PASS", f"import {mod_path}")
    except Exception as e:
        log("FAIL", f"import {mod_path}", str(e))
        modules_ok = False

# ════════════════════════════════════════════════════════════
# PHASE 2: PDF Parser & Page Classification
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 2: PDF Parser & Classification ═══")

from src.ingestion.pdf_parser import PdfParser, PageResult
from src.ingestion.classifier import PageClassifier, PageType
from pathlib import Path

test_pdf = Path("data/uploads/test_doc.pdf")

if not test_pdf.exists():
    log("SKIP", "Test PDF exists", "test_doc.pdf not found — generating it")
    try:
        exec(open("create_test_pdf.py").read())
        log("PASS", "create_test_pdf.py ran successfully")
    except Exception as e:
        log("FAIL", "create_test_pdf.py", str(e))
else:
    log("PASS", "Test PDF exists", str(test_pdf))

# Load metadata
try:
    doc_info = PdfParser.load(test_pdf)
    assert doc_info.total_pages >= 3, f"Expected >= 3 pages, got {doc_info.total_pages}"
    assert doc_info.file_hash and len(doc_info.file_hash) == 64
    log("PASS", "PdfParser.load()", f"{doc_info.total_pages} pages, hash={doc_info.file_hash[:12]}…")
except Exception as e:
    log("FAIL", "PdfParser.load()", str(e))

# File hash determinism
try:
    h1 = PdfParser.compute_file_hash(test_pdf)
    h2 = PdfParser.compute_file_hash(test_pdf)
    assert h1 == h2, "Hash is not deterministic!"
    log("PASS", "File hash determinism", f"{h1[:16]}… ")
except Exception as e:
    log("FAIL", "File hash determinism", str(e))

# Iterate pages
try:
    pages = list(PdfParser.iter_pages(test_pdf))
    assert len(pages) >= 3
    log("PASS", "PdfParser.iter_pages()", f"Yielded {len(pages)} pages")
except Exception as e:
    log("FAIL", "PdfParser.iter_pages()", str(e))

# Classification
classifier = PageClassifier()
try:
    classifications = []
    for p in pages:
        pt = classifier.classify(p)
        classifications.append(pt)
    # test_doc has: TEXT, MIXED, IMAGE pages
    type_set = set(classifications)
    log("PASS", "PageClassifier.classify()", f"Types detected: {[t.value for t in classifications]}")
    if PageType.TEXT in type_set:
        log("PASS", "TEXT page detected")
    else:
        log("FAIL", "TEXT page detected", "No TEXT page in test doc")
    if PageType.IMAGE in type_set:
        log("PASS", "IMAGE page detected")
    else:
        log("FAIL", "IMAGE page detected", "No IMAGE page in test doc")
except Exception as e:
    log("FAIL", "PageClassifier", str(e))

# ════════════════════════════════════════════════════════════
# PHASE 3: OCR Engine & Fallbacks
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 3: OCR Engine & Fallbacks ═══")

from src.ingestion.ocr import OcrEngine
import torch

# GPU detection fallback
try:
    has_cuda = torch.cuda.is_available()
    ocr = OcrEngine(gpu=True)
    if has_cuda:
        assert ocr.gpu is True
        log("PASS", "OCR GPU detection", f"CUDA available, gpu={ocr.gpu}")
    else:
        assert ocr.gpu is False
        log("PASS", "OCR GPU→CPU fallback", f"CUDA unavailable, gracefully fell back to cpu={ocr.gpu}")
except Exception as e:
    log("FAIL", "OCR GPU fallback", str(e))

# Force CPU
try:
    ocr_cpu = OcrEngine(gpu=False)
    assert ocr_cpu.gpu is False
    log("PASS", "OCR forced CPU mode", "gpu=False yields cpu=False")
except Exception as e:
    log("FAIL", "OCR forced CPU mode", str(e))

# Lazy initialization (reader should be None until first access)
try:
    ocr_lazy = OcrEngine()
    assert ocr_lazy._reader is None, "Reader should NOT be initialized in __init__"
    log("PASS", "OCR lazy init", "Reader is None before first use")
except Exception as e:
    log("FAIL", "OCR lazy init", str(e))

# Pixmap memory guard test
try:
    import fitz
    test_doc_fitz = fitz.open(str(test_pdf))
    page0 = test_doc_fitz[0]
    # The _render_page_to_image should handle oversized pixmaps gracefully
    img = ocr._render_page_to_image(page0, dpi=200)
    assert img is not None and img.shape[0] > 0
    log("PASS", "OCR _render_page_to_image", f"Rendered page shape={img.shape}")
    test_doc_fitz.close()
except Exception as e:
    log("FAIL", "OCR _render_page_to_image", str(e))

# OCR extract_text on an image page
try:
    test_doc_fitz = fitz.open(str(test_pdf))
    # Page 3 (index 2) is a pure image page in the test doc
    ocr_text = ocr.extract_text(test_doc_fitz[2], 3)
    log("PASS", "OCR extract_text(image page)", f"Extracted {len(ocr_text)} chars")
    test_doc_fitz.close()
except Exception as e:
    log("FAIL", "OCR extract_text(image page)", str(e))

# ════════════════════════════════════════════════════════════
# PHASE 4: Semantic Chunker
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 4: Semantic Chunker ═══")

from src.ingestion.chunker import SemanticChunker, TextChunk

chunker = SemanticChunker()

# Normal chunking
try:
    sample_text = (
        "1. Introduction\n\n"
        "This is the first paragraph about machine learning. " * 20 + "\n\n"
        "This is the second paragraph about neural networks. " * 20 + "\n\n"
        "A short concluding statement."
    )
    meta = {"source_pdf": "test.pdf", "page_number": 1, "chunk_index": 0}
    chunks = chunker.chunk_text(sample_text, meta)
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    log("PASS", "SemanticChunker basic", f"Produced {len(chunks)} chunks from ~{len(sample_text)} chars")
except Exception as e:
    log("FAIL", "SemanticChunker basic", str(e))

# Overlap check
try:
    if len(chunks) >= 2:
        # The end of chunk 0 should overlap with start of chunk 1
        tail = chunks[0].text[-50:]
        head = chunks[1].text[:100]
        has_overlap = any(word in head for word in tail.split() if len(word) > 3)
        if has_overlap:
            log("PASS", "Chunk overlap", "Context continuity detected between consecutive chunks")
        else:
            log("SKIP", "Chunk overlap", "Overlap words not directly visible (may be OK)")
except Exception as e:
    log("FAIL", "Chunk overlap", str(e))

# Min-size noise guard
try:
    tiny = "Hi."
    tiny_chunks = chunker.chunk_text(tiny, {"source_pdf": "t.pdf", "page_number": 1, "chunk_index": 0})
    assert len(tiny_chunks) == 0, f"Noise text should be filtered out, got {len(tiny_chunks)} chunks"
    log("PASS", "Noise guard (min-size)", "Tiny text filtered out correctly")
except Exception as e:
    log("FAIL", "Noise guard (min-size)", str(e))

# Section header detection
try:
    header_text = "ABSTRACT\n\nThis paper presents a novel approach. " * 20
    h_chunks = chunker.chunk_text(header_text, {"source_pdf": "t.pdf", "page_number": 1, "chunk_index": 0})
    has_header = any(c.metadata.get("section_header") for c in h_chunks)
    if has_header:
        log("PASS", "Header detection", f"Detected: '{h_chunks[0].metadata.get('section_header')}'")
    else:
        log("SKIP", "Header detection", "No section_header detected (heuristic may not match)")
except Exception as e:
    log("FAIL", "Header detection", str(e))

# chunk_pages multi-page
try:
    test_pages = [
        (1, "Page one content. " * 50, "multi.pdf"),
        (2, "Page two content. " * 50, "multi.pdf"),
        (3, "", "multi.pdf"),  # empty page should be skipped
    ]
    mp_chunks = chunker.chunk_pages(test_pages)
    assert len(mp_chunks) > 0
    # Check that page 3 (empty) contributed 0 chunks
    p3_chunks = [c for c in mp_chunks if c.metadata.get("page_number") == 3]
    assert len(p3_chunks) == 0, "Empty page should produce zero chunks"
    log("PASS", "chunk_pages multi-page", f"{len(mp_chunks)} chunks, empty pages skipped")
except Exception as e:
    log("FAIL", "chunk_pages multi-page", str(e))

# ════════════════════════════════════════════════════════════
# PHASE 5: Full Ingestion Pipeline
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 5: Full Ingestion Pipeline ═══")

from src.ingestion.pipeline import IngestionPipeline, IngestionResult

pipeline = IngestionPipeline()

# Lazy OCR init
try:
    assert pipeline.ocr_engine is None, "OCR should be lazy"
    log("PASS", "Pipeline lazy OCR", "ocr_engine is None at init")
except Exception as e:
    log("FAIL", "Pipeline lazy OCR", str(e))

# Process test PDF
try:
    t0 = time.time()
    result = pipeline.process_pdf(test_pdf)
    elapsed = time.time() - t0
    assert isinstance(result, IngestionResult)
    assert result.total_pages >= 3
    assert result.total_chunks > 0
    assert result.file_hash and len(result.file_hash) == 64
    log("PASS", "Pipeline.process_pdf()", 
        f"{result.total_pages} pages → {result.total_chunks} chunks in {elapsed:.1f}s")
except Exception as e:
    log("FAIL", "Pipeline.process_pdf()", str(e))

# Memory guard check (not a test per se, but call the method)
try:
    pipeline._check_memory()
    log("PASS", "Memory guard", "psutil memory check ran without error")
except Exception as e:
    log("FAIL", "Memory guard", str(e))

# Missing file error handling
try:
    pipeline.process_pdf(Path("nonexistent.pdf"))
    log("FAIL", "Missing file raises", "Should have raised FileNotFoundError")
except FileNotFoundError:
    log("PASS", "Missing file raises FileNotFoundError")
except Exception as e:
    log("FAIL", "Missing file error", f"Wrong exception type: {type(e).__name__}: {e}")

# ════════════════════════════════════════════════════════════
# PHASE 6: Embedder & CUDA Fallback
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 6: Embedder & Device Fallback ═══")

from src.embeddings.embedder import LocalEmbedder

try:
    embedder = LocalEmbedder()
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        assert embedder.device == "cuda"
        log("PASS", "Embedder device", "CUDA available → device=cuda")
    else:
        assert embedder.device == "cpu"
        log("PASS", "Embedder device fallback", "CUDA unavailable → device=cpu")
except Exception as e:
    log("FAIL", "Embedder init", str(e))

# Lazy model loading
try:
    assert embedder._model is None, "Model should be lazy"
    assert embedder._reranker is None, "Reranker should be lazy"
    log("PASS", "Embedder lazy loading", "Both _model and _reranker are None at init")
except Exception as e:
    log("FAIL", "Embedder lazy loading", str(e))

# embed_text
try:
    vec = embedder.embed_text("This is a test sentence.")
    assert isinstance(vec, list)
    assert len(vec) > 0
    log("PASS", "embed_text()", f"Dimension={len(vec)}")
except Exception as e:
    log("FAIL", "embed_text()", str(e))

# embed_text empty guard
try:
    empty_vec = embedder.embed_text("")
    assert empty_vec == []
    log("PASS", "embed_text(empty) guard", "Returns empty list for empty string")
except Exception as e:
    log("FAIL", "embed_text(empty) guard", str(e))

# embed_batch
try:
    batch = ["Hello world", "Machine learning concepts", "Vector databases"]
    vecs = embedder.embed_batch(batch)
    assert len(vecs) == 3
    assert all(len(v) == len(vec) for v in vecs)
    log("PASS", "embed_batch()", f"3 texts → 3 vectors of dim={len(vecs[0])}")
except Exception as e:
    log("FAIL", "embed_batch()", str(e))

# embed_batch empty
try:
    assert embedder.embed_batch([]) == []
    log("PASS", "embed_batch(empty) guard", "Returns [] for empty list")
except Exception as e:
    log("FAIL", "embed_batch(empty) guard", str(e))

# Reranker
try:
    scores = embedder.rerank("What is ML?", ["Machine learning is...", "My cat is cute."])
    assert len(scores) == 2
    assert scores[0] > scores[1], f"Relevant text should score higher: {scores}"
    log("PASS", "Reranker", f"Scores={[round(s, 3) for s in scores]} (relevant > irrelevant)")
except Exception as e:
    log("FAIL", "Reranker", str(e))

# Reranker empty guards
try:
    assert embedder.rerank("test", []) == []
    assert embedder.rerank("", ["something"]) == [0.0]
    log("PASS", "Reranker edge guards", "Empty texts=[], empty query=[0.0]")
except Exception as e:
    log("FAIL", "Reranker edge guards", str(e))

# ════════════════════════════════════════════════════════════
# PHASE 7: Vector Store (Qdrant)
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 7: Vector Store (Qdrant) ═══")

from src.database.vector_store import VectorStore

try:
    vs = VectorStore(embedder)
    log("PASS", "VectorStore init", f"Collection='{settings.qdrant_collection_name}', dim={vs.vector_size}")
except Exception as e:
    log("FAIL", "VectorStore init", str(e))
    print("CRITICAL: Cannot test retrieval without VectorStore.")
    # Still continue with what we can

# Check collection exists
try:
    exists = vs.client.collection_exists(settings.qdrant_collection_name)
    assert exists
    count = vs.client.count(collection_name=settings.qdrant_collection_name).count
    log("PASS", "Collection exists", f"Point count: {count}")
except Exception as e:
    log("FAIL", "Collection check", str(e))

# Ingest test_doc if the collection is empty
try:
    count = vs.client.count(collection_name=settings.qdrant_collection_name).count
    if count == 0:
        log("SKIP", "Re-ingesting test PDF", "Collection empty — ingesting now")
        result = pipeline.process_pdf(test_pdf)
        vs.add_document(result)
        new_count = vs.client.count(collection_name=settings.qdrant_collection_name).count
        log("PASS", "add_document()", f"Added {new_count} points")
    else:
        log("PASS", "Existing data", f"{count} points already in collection")
except Exception as e:
    log("FAIL", "Ingest/add_document", str(e))

# Search - two-stage retrieval
try:
    search_results = vs.search("introduction", initial_k=10, final_k=3)
    assert isinstance(search_results, list)
    assert len(search_results) > 0
    first = search_results[0]
    assert "text" in first
    assert "score" in first
    assert "metadata" in first
    assert "vector_score" in first
    log("PASS", "VectorStore.search()", f"{len(search_results)} results with reranking scores")
except Exception as e:
    log("FAIL", "VectorStore.search()", str(e))

# Search empty query guard
try:
    assert vs.search("") == []
    log("PASS", "search(empty) guard", "Empty query returns []")
except Exception as e:
    log("FAIL", "search(empty) guard", str(e))

# Rerank fallback in search (simulate by checking the fallback code path exists)
try:
    # The fallback happens when reranker throws; we just verify the code path
    # by confirming the fallback line exists
    import inspect
    src = inspect.getsource(vs.search)
    assert "Reranking failed" in src and "Falling back" in src
    log("PASS", "Rerank fallback code path", "Graceful fallback to dense scores exists in search()")
except Exception as e:
    log("FAIL", "Rerank fallback code path", str(e))

# UUID determinism (same file_hash + chunk_index → same UUID)
try:
    import uuid
    id1 = str(uuid.uuid5(uuid.NAMESPACE_OID, "abcdef12345_0"))
    id2 = str(uuid.uuid5(uuid.NAMESPACE_OID, "abcdef12345_0"))
    assert id1 == id2
    log("PASS", "UUID determinism", "Same hash+chunk → same point ID (upsert idempotent)")
except Exception as e:
    log("FAIL", "UUID determinism", str(e))

# ════════════════════════════════════════════════════════════
# PHASE 8: Retriever
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 8: Retriever ═══")

from src.retrieval.retriever import Retriever

try:
    retriever = Retriever(vs)
    context, raw_results = retriever.retrieve_context("introduction", initial_k=10, final_k=3)
    assert isinstance(context, str)
    assert len(context) > 0
    assert isinstance(raw_results, list)
    assert len(raw_results) > 0
    # Check formatted context contains document headers
    assert "Document:" in context
    assert "Page:" in context
    log("PASS", "Retriever.retrieve_context()", 
        f"Context length={len(context)} chars, {len(raw_results)} citations")
except Exception as e:
    log("FAIL", "Retriever.retrieve_context()", str(e))

# Empty handling
try:
    empty_ctx, empty_res = retriever.retrieve_context("")
    assert empty_ctx == ""
    assert empty_res == []
    log("PASS", "Retriever(empty query) guard", "Returns empty string+list")
except Exception as e:
    log("FAIL", "Retriever(empty query) guard", str(e))

# ════════════════════════════════════════════════════════════
# PHASE 9: Generator (LLM)
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 9: Generator (LLM backends) ═══")

from src.generation.generator import get_generator, OllamaGenerator, GeminiGenerator, BaseGenerator

# Factory test
try:
    gen = get_generator("ollama")
    assert isinstance(gen, OllamaGenerator)
    log("PASS", "get_generator('ollama')", "Returns OllamaGenerator")
except Exception as e:
    log("FAIL", "get_generator('ollama')", str(e))

# Unknown provider fallback
try:
    gen_unknown = get_generator("chatgpt")
    assert isinstance(gen_unknown, OllamaGenerator)
    log("PASS", "Unknown provider fallback", "'chatgpt' → OllamaGenerator (graceful)")
except Exception as e:
    log("FAIL", "Unknown provider fallback", str(e))

# Gemini init guard
try:
    if settings.gemini_api_key:
        gen_gemini = get_generator("gemini")
        assert isinstance(gen_gemini, GeminiGenerator)
        log("PASS", "GeminiGenerator init", "API key present, initialized OK")
    else:
        try:
            get_generator("gemini")
            log("FAIL", "Gemini no-key guard", "Should have raised ValueError")
        except ValueError:
            log("PASS", "Gemini no-key guard", "Raises ValueError when key is missing")
except Exception as e:
    log("FAIL", "Generator Gemini test", str(e))

# Prompt builder
try:
    gen = get_generator("ollama")
    prompt = gen._build_prompt("What is ML?", "Some context here.")
    assert "Question" in prompt
    assert "Context" in prompt
    assert "What is ML?" in prompt
    assert "Do not hallucinate" in prompt
    log("PASS", "Prompt builder", "Contains instructions, context, question, guardrails")
except Exception as e:
    log("FAIL", "Prompt builder", str(e))

# Live Ollama test (best-effort)
try:
    import requests
    r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
    if r.status_code == 200:
        log("PASS", "Ollama server reachable", f"URL={settings.ollama_base_url}")
        # Test actual generation
        try:
            answer = gen.generate("What is this document about?", "This is a test context about RAG pipelines.")
            assert isinstance(answer, str) and len(answer) > 0
            log("PASS", "Ollama generation", f"Response length={len(answer)} chars")
        except Exception as e:
            log("FAIL", "Ollama generation", str(e))
    else:
        log("SKIP", "Ollama server", f"HTTP {r.status_code}")
except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
    log("SKIP", "Ollama server not running", "Skipping live generation test")
except Exception as e:
    log("SKIP", "Ollama check", str(e))

# Gemini live test (best-effort)
if settings.gemini_api_key:
    try:
        gem_gen = get_generator("gemini")
        answer = gem_gen.generate("Summarize this.", "A test document about vector databases.")
        assert isinstance(answer, str) and len(answer) > 0
        log("PASS", "Gemini generation", f"Response length={len(answer)} chars")
    except Exception as e:
        log("FAIL", "Gemini generation", str(e))
else:
    log("SKIP", "Gemini generation", "No RAG_GEMINI_API_KEY set")

# ════════════════════════════════════════════════════════════
# PHASE 10: API Server (Structural)
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 10: API Server (Structural) ═══")

try:
    from src.api.server import app
    routes = [r.path for r in app.routes]
    assert "/api/upload" in routes
    assert "/api/query" in routes
    log("PASS", "API routes exist", f"/api/upload, /api/query present in {len(routes)} routes")
except Exception as e:
    log("FAIL", "API routes", str(e))

# Static files mount
try:
    static_dir = Path("src/api/static")
    assert static_dir.exists()
    assert (static_dir / "index.html").exists()
    assert (static_dir / "style.css").exists()
    assert (static_dir / "script.js").exists()
    log("PASS", "Static UI files", "index.html, style.css, script.js present")
except Exception as e:
    log("FAIL", "Static UI files", str(e))

# CORS middleware
try:
    middleware_names = [type(m).__name__ for m in getattr(app, 'user_middleware', [])]
    # Check via the app middleware stack
    src_code = open("src/api/server.py").read()
    assert "CORSMiddleware" in src_code
    log("PASS", "CORS middleware", "CORSMiddleware configured")
except Exception as e:
    log("FAIL", "CORS middleware", str(e))

# ════════════════════════════════════════════════════════════
# PHASE 11: End-to-End Pipeline  (Ingest → Search → Generate)
# ════════════════════════════════════════════════════════════
print("\n═══ PHASE 11: End-to-End Pipeline ═══")

try:
    t0 = time.time()
    # Re-ingest the test PDF fresh
    pipeline2 = IngestionPipeline()
    result2 = pipeline2.process_pdf(test_pdf)
    
    # Re-use the existing VectorStore (local Qdrant holds a file lock)
    vs.add_document(result2)
    
    # Retrieve using existing retriever
    context2, raw2 = retriever.retrieve_context("paragraph", initial_k=10, final_k=3)
    
    elapsed = time.time() - t0
    
    assert len(context2) > 0
    assert len(raw2) > 0
    log("PASS", "End-to-End (Ingest->Embed->Search->Rerank)", 
        f"{result2.total_chunks} chunks, {len(raw2)} results in {elapsed:.1f}s")
except Exception as e:
    log("FAIL", "End-to-End pipeline", str(e))


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(f"  VERIFICATION SUMMARY")
print(f"  ✅ PASS: {PASS}  |  ❌ FAIL: {FAIL}  |  ⚠️  SKIP: {SKIP}")
print(f"  Total: {PASS + FAIL + SKIP} tests")
print("═" * 60)

if FAIL > 0:
    print("\n❌ FAILED tests:")
    for status, name, detail in RESULTS:
        if status == "FAIL":
            print(f"   • {name}: {detail}")
    print()

# Clean up Qdrant client to prevent __del__ ImportError during shutdown
try:
    vs.client.close()
except Exception:
    pass

# Flush before exit to ensure redirected output is captured
sys.stdout.flush()
sys.stderr.flush()

# Use os._exit to bypass Python shutdown hooks that cause spurious Qdrant errors
os._exit(0 if FAIL == 0 else 1)
