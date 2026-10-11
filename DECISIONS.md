# DECISION.md — Architecture & Technical Decision Log

> **Single Source of Truth** for major technical, architectural, code approach, and library selection decisions made during the project lifecycle.

---

## Decision Log (Tier 1 Read: Lines 1–26)

| Date | ID | Category | Summary | Status |
|------|----|----------|---------|--------|
| 2026-04-12 | ADR-001 | Architecture & Ingestion | Streaming `PyMuPDF` Parser, 3-Way Page Classifier (`TEXT`/`IMAGE`/`MIXED`) & Lazy GPU `EasyOCR` | Accepted |
| 2026-04-12 | ADR-002 | Chunking & Storage | Hierarchical `SemanticChunker` & Persistent Local `Qdrant` with Deterministic `UUIDv5` IDs | Accepted |
| 2026-04-12 | ADR-003 | Retrieval & Reranking | Two-Stage Retrieval (`BAAI/bge-small-en-v1.5` Bi-Encoder + `bge-reranker-v2-m3` Cross-Encoder) | Accepted |
| 2026-04-22 | ADR-004 | LLM Generation | Multi-Provider LLM Strategy Pattern (`LMStudio`, `Ollama`, `Gemini`) & Grounded Prompt Isolation | Accepted |
| 2026-10-12 | ADR-005 | Stack & Persistence | Unified Python `FastAPI` Backend + Dual Storage (`Qdrant` + `SQLAlchemy` SQLite/PostgreSQL) | Accepted |
| 2026-10-12 | ADR-006 | Hybrid Search | Native Qdrant Named Vectors (`dense` + `sparse` BM25 IDF) & Reciprocal Rank Fusion ($k=60$) | Accepted |
| 2026-10-12 | ADR-007 | Chunking & Safety | Cross-Page Parent-Child Chunking & Two-Tier Abstention Gate (Pre-LLM Rerank Threshold) | Accepted |
| 2026-10-12 | ADR-008 | Security & Async Jobs | Multi-Tenant JWT Auth, Pre-Retrieval Qdrant RBAC Filters & DB-Backed Async Worker Pool | Accepted |
| 2026-10-12 | ADR-009 | Evaluation & Telemetry | 100-QA Benchmark Suite (3 PDFs + Hybrid Evaluator) & Stage-Level Latency/Token/Cost Telemetry | Accepted |
| 2026-10-12 | ADR-010 | Governance & Layers | Multi-Chat Role System, 3-Tier Context Fencing, 5-Layer × 6-Section Matrix & AST Verifier | Accepted |

---

## Detailed Architectural Decision Records (Tier 2 Reference: Line 27+)

### ADR-001: Streaming `PyMuPDF` Parser, 3-Way Page Classifier (`TEXT`/`IMAGE`/`MIXED`) & Lazy GPU `EasyOCR`

- **Date**: 2026-04-12 (Updated 2026-10-12 for v2.1 single-handle refactor)
- **Status**: Accepted
- **Category**: Architecture, Ingestion & Library Selection (`Sec 5`, `L3`)

#### 1. Context & Problem Statement
Ingesting massive technical PDFs (1,000–2,000+ pages) on a single workstation (16GB RAM, RTX 4070 8GB VRAM) presents three major failure modes:
1. **RAM Exhaustion**: Loading an entire 1,500-page PDF and its rasterized page bitmaps into memory simultaneously crashes the process.
2. **OCR Bottleneck**: Running full-page Optical Character Recognition (OCR) on every page of a 1,000-page textbook takes 20–30 minutes, even when 85% of the pages contain clean, digital text streams.
3. **Oversized Page OOMs & Tiny Icon Noise**: High-resolution architectural diagrams or poster pages rendered at 200 DPI can exceed 100MB per bitmap, while running region OCR on embedded 16×16 pixel UI icons wastes GPU cycles and injects garbage tokens.

#### 2. Decision Made
1. **Streaming Page Parser (`src/ingestion/pdf_parser.py`)**:
   - Compute a streaming `SHA-256` file digest in `8192`-byte chunks (`compute_file_hash`) for document deduplication before parsing.
   - Iterate through pages lazily one at a time via PyMuPDF (`fitz`), isolating each page inside a `try...except` block so a corrupted page never aborts the entire document.
   - *(v2.1 Refactor)*: Open the `fitz.Document` handle **once** per ingestion run in `IngestionPipeline.process_pdf()` and pass the open handle directly to page extraction and OCR methods.
2. **Adaptive 3-Way Page Classifier (`src/ingestion/classifier.py`)**:
   - Inspect `text_length = len(raw_text.strip())` and `image_count = len(page.get_images(full=True))` against `settings.text_threshold = 50` characters:
     - **`PageType.TEXT`** (`text_length >= 50` and `image_count == 0`): Route directly to PyMuPDF native text extraction (~1ms/page, zero GPU usage).
     - **`PageType.IMAGE`** (`text_length < 50`): Treat as a scanned page; rasterize at `ocr_dpi = 200` (`zoom = 200 / 72`) and run full-page GPU `EasyOCR`.
     - **`PageType.MIXED`** (`text_length >= 50` and `image_count > 0`): Retain native PyMuPDF text **and** surgically extract embedded images by PDF cross-reference ID (`xref`), skipping decorative icons (`width < 50` or `height < 50` px), and append OCR text under `[OCR from images]`.
3. **Lazy GPU `EasyOCR` & Recursive Pixmap Memory Guard (`src/ingestion/ocr.py`)**:
   - Defer `easyocr.Reader` initialization until the first `IMAGE` or `MIXED` page is encountered (0 MB VRAM if a PDF is pure digital text).
   - Automatically detect `torch.cuda.is_available()` and fall back to CPU if CUDA is unavailable.
   - Before converting a rendered page pixmap to a NumPy array, check `pixmap_mb = len(pixmap.samples) / (1024 * 1024)`. If `pixmap_mb > settings.ocr_max_pixmap_mb` (50 MB), free the pixmap and recursively re-render at `dpi // 2`.
   - Enforce a per-page timeout (`settings.ocr_timeout_seconds = 30`) so a pathological scan never hangs an ingestion worker indefinitely.

#### 3. Library Selection Rationale
- **`PyMuPDF` (`fitz`) vs. `pypdf` / `pdfplumber` / `pdfminer.six`**: PyMuPDF is written in C (`MuPDF`) and extracts text and XREF image streams 10–25× faster than pure-Python parsers while providing built-in DPI matrix rasterization (`page.get_pixmap`) without requiring external `poppler` system binaries on Windows.
- **`EasyOCR` vs. `Tesseract` / `PaddleOCR`**: EasyOCR integrates natively with PyTorch CUDA (`torch.cuda`), requires no external Windows installer binaries, and handles noisy multi-font technical diagrams reliably.

#### 4. Trade-offs & Impact
- **Positive**: Reduces ingestion time by **50–70%** on hybrid technical documents and ~98% on digital PDFs; bounds peak RAM usage via lazy iteration and the 50MB pixmap guard.
- **Trade-off**: `EasyOCR` initialization adds a ~2–3s cold-start penalty on the first scanned page of a worker's lifecycle.

---

### ADR-002: Hierarchical `SemanticChunker` & Persistent Local `Qdrant` with Deterministic `UUIDv5` IDs

- **Date**: 2026-04-12
- **Status**: Accepted (Extended by ADR-006 & ADR-007 in v2.2)
- **Category**: Chunking, Vector Database & Data Model (`Sec 1, Sec 5`, `L1, L3`)

#### 1. Context & Problem Statement
1. **Naive Fixed-Character Chunking**: Splitting text blindly every $N$ characters slices sentences and words in half, destroying semantic coherence and degrading embedding quality.
2. **Duplicate Vector Pollution**: In many RAG prototypes, re-running ingestion on an already-indexed PDF generates random UUIDs for each chunk, creating duplicate points in the vector store that skew top-$K$ retrieval results.

#### 2. Decision Made
1. **Structure-Aware `SemanticChunker` (`src/ingestion/chunker.py`)**:
   - **Primary Boundary (Paragraphs)**: Normalize line endings and split on double newlines (`\n{2,}`), accumulating paragraphs up to `max_chunk_size = 1000` characters (while merging fragments smaller than `min_chunk_size = 100`).
   - **Secondary Boundary (Sentence Fallback)**: When a single dense paragraph exceeds `max_chunk_size`, split at sentence boundaries using `(?<=[.!?])\s+(?=[A-Z])`.
   - **Word-Safe Rolling Overlap (`overlap_size = 100`)**: Carry over the trailing 100 characters of the finalized chunk into the next chunk, snapping forward to the nearest whitespace (`overlap_text.find(' ')`) so words are never cleaved mid-token.
   - **Heuristic Section Header Detection (`_attempt_header_detection`)**: Detect short (`<60` chars), unpunctuated all-caps or numbered headings (`^\d+(\.\d+)*\s+[A-Z]`) and attach `section_header` to chunk metadata.
2. **Persistent Local `Qdrant` & Deterministic `UUIDv5` Point IDs (`src/database/vector_store.py`)**:
   - Persist vectors on disk at `data/vectordb` via `QdrantClient(path=...)`.
   - Generate deterministic point IDs from the document's `SHA-256` hash and chunk index:
     ```python
     point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{file_hash}_{chunk_idx}"))
     ```
   - Use `client.upsert()` so re-ingesting the same document overwrites existing points idempotently.

#### 3. Library Selection Rationale
- **`Qdrant` vs. `FAISS` / `ChromaDB`**:
  - `FAISS` is a pure vector index with no built-in payload storage, metadata filtering, or deletions by payload condition.
  - `ChromaDB` had SQLite concurrency and sparse-vector limitations.
  - `Qdrant` provides a fast Rust-backed engine, local embedded disk mode *and* Docker server mode with identical APIs, rich boolean payload filtering (`must`/`should`/`MatchAny`), and native support for both **Dense** and **Sparse (BM25/IDF)** vectors in a single collection.

---

### ADR-003: Two-Stage Retrieval (`BAAI/bge-small-en-v1.5` Bi-Encoder + `bge-reranker-v2-m3` Cross-Encoder)

- **Date**: 2026-04-12
- **Status**: Accepted
- **Category**: Retrieval & Reranking (`Sec 1`, `L1, L3`)

#### 1. Context & Problem Statement
Single-stage bi-encoder vector search compresses an entire chunk and a user query independently into fixed 384-dimensional vectors. Because there is no token-by-token cross-attention between query tokens and document tokens, bi-encoders frequently rank superficially similar passages above passages containing the exact answer.

#### 2. Decision Made
Implement a **Two-Stage Retrieval Funnel** in `LocalEmbedder` (`src/embeddings/embedder.py`) and `VectorStore.search` (`src/database/vector_store.py`):
1. **Stage 1 — Fast Bi-Encoder Candidate Recall (`initial_k = 20`)**:
   - Encode chunks and queries with **`BAAI/bge-small-en-v1.5`** (`384`-d vectors, $L_2$-normalized so Cosine Similarity reduces to inner product).
   - Retrieve a broad candidate pool (`initial_k = 20`) from Qdrant in `<15ms`.
2. **Stage 2 — High-Precision Cross-Encoder Reranking (`final_k = 4`)**:
   - Pass all 20 `[query, chunk_text]` pairs into **`BAAI/bge-reranker-v2-m3`** (`sentence_transformers.CrossEncoder` running on CUDA).
   - Sort candidates by their Cross-Encoder relevance score descending and pass only the top `final_k = 4` chunks to the LLM.

#### 3. Library & Model Selection Rationale
- **`BAAI/bge-small-en-v1.5` (33M params, 384-d)**: Top-tier MTEB retrieval accuracy per MB of VRAM (~130MB on disk), allowing batch size 64 embedding on an 8GB RTX 4070 alongside EasyOCR and the reranker.
- **`BAAI/bge-reranker-v2-m3`**: Multilingual/technical cross-encoder that evaluates full self-attention across query + passage tokens simultaneously, dramatically boosting top-4 precision.

---

### ADR-004: Multi-Provider LLM Strategy Pattern (`LMStudio`, `Ollama`, `Gemini`) & Grounded Prompt Isolation

- **Date**: 2026-04-22
- **Status**: Accepted
- **Category**: LLM Generation (`Sec 1, Sec 6`, `L3`)

#### 1. Context & Problem Statement
Users need the flexibility to run 100% offline/private inference using local models (via LM Studio or Ollama) or switch on-the-fly to a fast cloud API (`Gemini 2.5 Flash`) without changing retrieval code. Additionally, the LLM must be prevented from using pre-trained world knowledge when the uploaded documents do not support an answer.

#### 2. Decision Made
1. **Strategy Pattern (`src/generation/generator.py`)**:
   - Define abstract `BaseGenerator` with concrete implementations:
     - `LMStudioGenerator`: OpenAI-compatible `POST /v1/chat/completions` client (`gpt-oss-20b`, `temperature=0.3`).
     - `OllamaGenerator`: Local `POST /api/generate` client (`llama3`).
     - `GeminiGenerator`: Cloud `google-genai` client (`gemini-2.5-flash`).
   - Instantiate dynamically per request via `get_generator(provider)`.
2. **Strict Context Isolation & Source Formatting (`src/retrieval/retriever.py`)**:
   - Format retrieved chunks with explicit provenance headers (`--- Document: {source} (Page: {page}) ---`) and instruct the LLM in `_build_prompt()` to rely **exclusively** on the provided context or explicitly abstain (`"I cannot answer this based on the provided documents."`).

---

### ADR-005: Unified Python `FastAPI` Backend + Dual Storage (`Qdrant` + `SQLAlchemy` SQLite/PostgreSQL)

- **Date**: 2026-10-12
- **Status**: Accepted
- **Category**: Architecture & Persistence (`Sec 3, Sec 4, Sec 5, Sec 6`, `L1, L4`)

#### 1. Context & Problem Statement
Upgrading `Project-RAG_LLM` to an enterprise-grade v2.0 platform requires persistent relational entities (Organizations, Users, Document ACLs, Asynchronous Ingestion Jobs, and Query Telemetry) alongside vector/hybrid search. We evaluated whether to split the system into a Java Spring Boot gateway + Python AI microservice vs. keeping a unified Python FastAPI architecture, and whether to use `pgvector` vs. `Qdrant + SQLAlchemy`.

#### 2. Decision Made
1. **Unified Python (`FastAPI`) Stack**: Keep 100% of the backend in Python (`FastAPI`), executing a **Backend-First** roadmap (`Phases 2.1–2.5`) followed by the **Interactive Demo Web Studio** (`Phase 2.6`).
2. **Dual Storage Architecture (`Qdrant` + `SQLAlchemy`)**:
   - Retain **Qdrant** (`data/vectordb`) as the specialized high-performance vector + sparse BM25 search engine.
   - Add **SQLAlchemy 2.0 ORM** (`src/database/models.py`, `src/database/relational.py`) defaulting to local **SQLite** (`sqlite:///./data/metadata.db`) for zero-config Windows execution and switchable to **PostgreSQL** via `RAG_DATABASE_URL` in Docker/production.

#### 3. Rationale & Alternatives Considered
- **Unified Python vs. Polyglot Spring Boot + Python**: Splitting into two languages duplicates DTO schemas, requires maintaining two runtimes, and diverts engineering time into boilerplate HTTP proxying instead of deep RAG capabilities (Hybrid RRF, Evaluation Benchmarks, Pre-Retrieval RBAC, and Telemetry).
- **`Qdrant + SQLAlchemy` vs. Postgres-only (`pgvector`)**: Preserves our existing Qdrant investment while leveraging Qdrant's native sparse IDF vectors and payload filter indexes without requiring a local PostgreSQL C-extension build on Windows 11.

---

### ADR-006: Native Qdrant Named Vectors (`dense` + `sparse` BM25 IDF) & Reciprocal Rank Fusion ($k=60$)

- **Date**: 2026-10-12
- **Status**: Accepted (Target: Phase 2.2)
- **Category**: Hybrid Search & Retrieval Math (`Sec 1`, `L1, L3`)

#### 1. Context & Problem Statement
1. **Lexical Blindness of Dense Embeddings**: Dense vectors (`bge-small-en-v1.5`) struggle when users search for exact error codes (`ERR-CV-4092`), policy clauses (`POL-SEC-104.2`), or function names (`_render_page_to_image`).
2. **Missing BGE Query Prefix**: `BAAI/bge-small-en-v1.5` was trained with an asymmetric query instruction prefix (`"Represent this sentence for searching relevant passages: "`) that was omitted in v1.0.
3. **Security Hazard of External BM25 Indexes**: If BM25 runs in an external Python `rank_bm25` array outside Qdrant, multi-tenant RBAC filters (`org_id`, `allowed_roles`) cannot be applied atomically during candidate retrieval.

#### 2. Decision Made
1. **New Qdrant Collection (`rag_documents_v2`) with Named Vectors**:
   - `"dense"`: `384`-d Cosine vector (`BAAI/bge-small-en-v1.5`), adding `embed_query()` with the BGE v1.5 query instruction prefix while keeping `embed_batch()` raw for document chunks.
   - `"sparse"`: Qdrant native sparse vector configured with `models.SparseVectorParams(modifier=models.Modifier.IDF)`. `src/embeddings/sparse_bm25.py` tokenizes text (preserving alphanumeric codes, hyphenated IDs, and `snake_case` symbols) into deterministic 32-bit term hashes and term-frequency weights, while Qdrant computes Inverse Document Frequency (IDF) dynamically across the collection.
2. **Reciprocal Rank Fusion (`RRF`, $k=60$)**:
   - Execute Dense and Sparse queries in Qdrant (both enforcing identical RBAC payload filters) and merge their ranked candidate lists using parameter-free rank fusion:
     $$\text{RRF}(d) = \frac{1}{60 + r_{\text{dense}}(d)} + \frac{1}{60 + r_{\text{sparse}}(d)}$$
3. **Multi-Mode Benchmarking Support**:
   - Expose `mode="vector_only" | "bm25_only" | "hybrid_rrf" | "hybrid_rerank"` on `VectorStore.search()` and `Retriever.retrieve_context()` so the Evaluation Suite can benchmark each stage's exact contribution.

---

### ADR-007: Cross-Page Parent-Child Chunking & Two-Tier Abstention Gate

- **Date**: 2026-10-12
- **Status**: Accepted (Target: Phase 2.2)
- **Category**: Chunking & Hallucination Prevention (`Sec 1, Sec 2`, `L3`)

#### 1. Context & Problem Statement
1. **Page-Boundary Severing & Size Dilemma**: In v1.0, `SemanticChunker.chunk_pages()` flushed chunks at the end of every page, slicing cross-page paragraphs in half. Furthermore, embedding large `1000-char` chunks dilutes vector specificity, whereas passing tiny `300-char` chunks to the LLM starves it of surrounding context.
2. **Unnecessary LLM Calls on Out-of-Domain Queries**: In v1.0, even when all retrieved chunks had negative Cross-Encoder logits (completely unrelated to the user's question), the pipeline still sent them to the LLM, wasting latency/tokens and risking hallucinated answers.

#### 2. Decision Made
1. **Cross-Page Parent-Child Chunking (`src/ingestion/chunker.py`)**:
   - Stitch page text streams continuously while maintaining a character-offset-to-page map so chunks spanning a page break record accurate `page_number` (start) and `page_end` metadata.
   - Generate **`~1200-char` Parent Context Blocks** (`parent_id`, `parent_text`) and subdivide each parent into **`~300-char` Child Chunks** (`50-char` overlap).
   - Embed and index the **Child Chunks** in Qdrant (for sharp dense + BM25 matching), storing `parent_id` and `parent_text` in the payload. During retrieval, deduplicate hits by `parent_id` (keeping the max child score) and pass the full `parent_text` to the Cross-Encoder and LLM.
2. **Two-Tier Abstention Gate (`src/retrieval/retriever.py`)**:
   - **Tier 1 (Pre-LLM Reranker Confidence Gate)**: If the top Cross-Encoder score falls below `settings.min_rerank_score` (default `-1.5` logit), short-circuit **before** calling the LLM and return `"I cannot answer this based on the provided documents."` with `abstained=True, abstention_reason="LOW_SCORE"`.
   - **Tier 2 (LLM Groundedness Gate)**: If candidates pass Tier 1 but still lack the specific answer, the LLM abstains via strict prompt instructions (`abstention_reason="LLM_ABSTAIN"`).

---

### ADR-008: Multi-Tenant JWT Auth, Pre-Retrieval Qdrant RBAC Filters & DB-Backed Async Worker Pool

- **Date**: 2026-10-12
- **Status**: Accepted (Target: Phase 2.1 for Async Jobs, Phase 2.4 for JWT/RBAC)
- **Category**: Security & Background Processing (`Sec 3, Sec 5`, `L2, L4`)

#### 1. Context & Problem Statement
1. **Blocking HTTP Uploads**: Running PDF parsing, GPU OCR, and embedding synchronously inside `POST /api/upload` blocks HTTP connections for minutes and loses state on client disconnect.
2. **Cross-Tenant Data Leakage**: In a multi-user environment, filtering retrieved chunks *after* search or via LLM prompts is insecure and leaks top-$K$ slots to unauthorized documents.

#### 2. Decision Made
1. **Self-Contained DB Async Job Queue (`src/jobs/ingestion_worker.py`)**:
   - `POST /api/upload` computes the `SHA-256` hash immediately, short-circuits duplicates (`force_reindex=False`), creates an `IngestionJob` (`PENDING`) in `metadata.db`, and dispatches execution to an in-process `ThreadPoolExecutor(max_workers=2)` returning `202 Accepted` in `<50ms`.
   - Workers update live page progress (`PARSING` $\to$ `OCR` $\to$ `CHUNKING` $\to$ `INDEXING` $\to$ `COMPLETED`/`FAILED`), retry transient errors up to `max_job_retries=2`, roll back partial Qdrant points on terminal failure, and recover interrupted jobs on server startup.
   - *Why over Redis/Celery*: Works 100% natively on Windows 11 with zero external message broker processes while preserving full database state durability.
2. **Multi-Tenant JWT Auth & Pre-Retrieval Qdrant RBAC (`src/auth/`)**:
   - Authenticate users via stateless **JWT Bearer tokens** carrying `{sub: user_id, org_id, role}` across a 4-tier role hierarchy (`ADMIN`, `MANAGER`, `ANALYST`, `VIEWER`).
   - Tag every Qdrant chunk payload with `org_id`, `visibility` (`PUBLIC | ORG | RESTRICTED`), and `allowed_roles`.
   - Inject a mandatory `models.Filter` into every Qdrant dense and sparse query so chunks outside the user's organization or role are **never read from the index**.

---

### ADR-009: 100-QA Benchmark Suite (3 PDFs + Hybrid Evaluator) & Stage-Level Telemetry

- **Date**: 2026-10-12
- **Status**: Accepted (Target: Phase 2.3 & Phase 2.5)
- **Category**: Evaluation & Observability (`Sec 2, Sec 4`, `L3, L4`)

#### 1. Context & Problem Statement
Claiming a RAG upgrade "improves quality" without quantitative evidence is unscientific. Similarly, operating an AI backend without knowing per-stage latency (`embed` vs. `search` vs. `rerank` vs. `LLM`) or token cost makes performance tuning impossible.

#### 2. Decision Made
1. **Reproducible 3-PDF Benchmark & 100-QA Dataset (`src/evaluation/`)**:
   - Include a programmatic generator (`scripts/generate_benchmark_pdfs.py`) producing 3 canonical PDFs (Technical Manual with exact error codes, Enterprise Policy doc with clause IDs, and Mixed/Scanned OCR doc) alongside `data/eval/benchmark_dataset.json` (40 semantic, 30 lexical/code, 15 OCR, 15 unanswerable trap questions).
   - Build `evaluate.py` to compute deterministic retrieval metrics (**Recall@4, Recall@10, HitRate@4, MRR**) and configurable LLM-as-a-Judge generation metrics (**Faithfulness, Citation Accuracy, Abstention Accuracy**) across all 4 retrieval modes.
2. **Stage-Level Telemetry & Cost Accounting (`src/utils/telemetry.py`)**:
   - Track `X-Request-ID`, `embed_latency_ms`, `search_latency_ms`, `rerank_latency_ms`, `llm_latency_ms`, `total_latency_ms`, `prompt_tokens`, `completion_tokens`, and `estimated_cost_usd` in `QueryTelemetry`, exposed via `GET /api/metrics` (`p50`/`p95` latencies, abstention rate, and cost totals).

---

### ADR-010: Multi-Chat Role System, 3-Tier Context Fencing, 5-Layer × 6-Section Matrix & AST Verifier

- **Date**: 2026-10-12
- **Status**: Accepted
- **Category**: Governance & Architecture (`GEMINI.md`, `FLOW.md`, `scripts/check_architecture_layers.py`)

#### 1. Context & Problem Statement
Complex multi-phase engineering degrades when a single chat session accumulates massive context history or mixes planning, backend implementation, UI styling, and QA auditing in a single turn.

#### 2. Decision Made
Adopt the **Ledgerly Multi-Chat Implementation System**:
1. **3-Tier Context Fencing (`GEMINI.md`)**: Keep pre-task reads under ~500 tokens by reading only `.gsd/PULSE.md` (last 3 lines), `.gsd/ACTIVE_PLAN.md`, `.gsd/STATE.md` (lines 1–28), `.gsd/ROADMAP.md`, and `.gsd/DECISION.md` (lines 1–26).
2. **Role Allocation & Single-Wave Firewall**: Split execution across `PM`, `Backend`, `Frontend`, and `QA` using native GSD XML `<wave>` blocks in `.gsd/ACTIVE_PLAN.md` and zero-prompt file handoffs in `.gsd/PULSE.md`. Each agent executes at most **one Wave per user turn** and halts immediately.
3. **5-Layer × 6-Section Matrix & Automated AST Verifier (`FLOW.md` + `scripts/check_architecture_layers.py`)**: Map every source file to Layers `L1`–`L5B` and Sections `Sec 1`–`Sec 6`, verifying 100% file registration, zero syntax errors, and downward-only imports (`L5 -> L4 -> L3 -> L2 -> L1`) before any wave is marked complete.
