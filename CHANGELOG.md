# Changelog — Project-RAG_LLM

All notable changes, architectural upgrades, and milestone completions for `Project-RAG_LLM` are documented in this file.

## [v3.0.0] — 2026-10-14 (Milestone v3.0 Production Platform Complete & Certified)

### Added
- **Native Agentic Intelligence & Pure-Python Finite State Machine (`Phase 2.10`, `src/agents/*`, `src/retrieval/*`)**:
  - **Conversational Session Memory (`src/database/models.py`, `src/agents/memory.py`)**: Persistent SQLite/SQLAlchemy `ChatSession` and `ChatMessage` models with a 3-turn sliding window and background rolling summarization for long-running dialogues.
  - **Multi-Query Decomposition & Expansion (`src/retrieval/multi_query.py`)**: Fast heuristic-triggered sub-query generator producing 3 distinct retrieval angles merged via native Qdrant Reciprocal Rank Fusion ($k=60$).
  - **Corrective RAG (CRAG) & Self-Reflection Engine (`src/retrieval/crag.py`)**: Two-tier confidence grader (`HIGH`, `MEDIUM`, `LOW`) with a strictly bounded 1-retry query reformulation loop, preventing infinite agentic cycles.
  - **Curated 4-Tool Dispatcher (`src/agents/tools.py`)**: First-principles tool registry providing `document_search`, `document_summary`, AST-safe `deterministic_calculator` (zero `eval`/`exec`), and `document_comparison`.
  - **Pure-Python FSM State Machine (`src/agents/state_machine.py`)**: Typed state loop (`Load Memory` $\to$ `Decompose` $\to$ `Retrieve` $\to$ `Grade` $\to$ `Tool` $\to$ `Synthesize`) operating without heavy third-party framework overhead (LangChain/LangGraph-free).
- **Multi-Turn Conversational Web Studio & Reasoning Accordion (`Phase 2.10`, `src/api/static/*`)**:
  - Full multi-turn chat stream rendering user and assistant Obsidian/Emerald message bubbles with live response typing.
  - Expandable Agent Reasoning Disclosure Accordion (`#reasoning-trace`) exposing decomposition angles, confidence grades, tool execution chips, and grounded synthesis steps.
  - Persistent session management with `ChatSessionController`, active session indicators, and 1-click "New Session" action.
  - Enforced strict **Titanium Obsidian & Emerald** design system (0% blue/purple, 0% emojis, 100% inline SVG vectors).
- **Enterprise Identity Management & Cloud Storage Configuration (`Phase 2.9`, `src/api/auth_routes.py`, `src/api/static/*`)**:
  - Enterprise Login Gate (`#login-modal`) supporting email/password authentication with 1-click demo persona quick-fill and remember-me persistence.
  - Dual-mode password recovery (secure SMTP email token dispatch with Admin token fallback).
  - Top-right authenticated profile avatar, user details drawer, and sliding RBAC clearances drawer.
  - Cloud storage configuration toggle supporting Supabase and Qdrant Cloud API keys.
- **Ingestion Concurrency Engine & Auto-Watcher (`Phase 2.8`, `src/jobs/*`, `src/api/static/*`)**:
  - Sequential ingestion queue worker (`Concurrency=1`) guaranteeing zero CUDA VRAM thrashing and deterministic queue lifecycle (`PENDING` $\to$ `PROCESSING` $\to$ `COMPLETED` / `FAILED`).
  - Background folder watcher automatically indexing PDFs dropped into `data/auto_ingest/`.
  - Multi-file drag-and-drop UI upload queue with real-time per-file progress bars.
  - Rate limiter exemption for `/api/jobs` and increased default threshold to 180 req/min.

### Verified & Certified
- **89/89 Automated Pytest Suite Passing (100%)** across all layers (`L1–L5B`) and product sections (`Sec 1–6`).
- **41/41 Active Files Passing Architecture Layer Boundary & Visual Contract Checks (0 violations)** in `scripts/check_architecture_layers.py`.
- **Database & Payload Hygiene Certified**: Zero test fixture leakage into `data/metadata.db`; clean demo baseline intact.

---

### Added
- **Top Persona Switcher Ribbon Scroll Chevrons & Compact Density (`src/api/static/*`)**:
  - Compacted persona buttons (`padding: 4px 9px`, `font-size: 0.72rem`) so all 6 personas (`Admin (Default)` through `Viewer (Tenant B)`) fit side-by-side on desktop (`1440×900`) and laptop (`1280×800`) displays.
  - Added flanking `<` (`#persona-scroll-left`) and `>` (`#persona-scroll-right`) inline SVG chevron buttons enabling single-click smooth horizontal scrolling on narrower viewports.
- **Above-the-Fold Sidebar Layout, Dark Checkboxes & Responsive Viewports (`src/api/static/style.css`)**:
  - Compacted vertical spacing in `#sidebar` and capped `#doc-list` (`max-height: 185px; overflow-y: auto`) so all 4 sidebar panels (Ingestion, Catalog, Retrieval Modes, and Model Engine with Cloud Gemini API key) fit 100% above the fold on desktop and tablet.
  - Aligned `@media (max-width: 1024px)` side-by-side tablet layout and `@media (max-width: 640px)` mobile layout so the bottom `#query-form` chat input is fully visible without clipping.
  - Replaced bright-white browser default checkboxes with custom dark obsidian boxes and Emerald SVG checkmarks.
- **Rich Extractive Markdown Formatting & Fast-Fail Offline Local LLM (`src/generation/generator.py`, `src/api/static/script.js`)**:
  - Upgraded `formatMarkdownToHtml()` to render styled Emerald `<blockquote class="md-quote">`, `<em>`, `<code>`, and `<ul><li>` bullet lists.
  - Stripped trailing `---` context block header artifacts and removed duplicate inline offline notice text.
  - Applied `timeout=(0.8, 180)` tuple connect timeouts to `LMStudioGenerator` and `OllamaGenerator`, slashing offline local LLM TCP connection failover latency from `4,058ms` down to **`251.6ms` (16× speedup)**.
- **Admin DELETE /api/metrics Telemetry Reset Endpoint & Clean Metrics UI (`src/api/server.py`, `src/api/static/*`)**:
  - Added `DELETE /api/metrics` endpoint (ADMIN only) and `#reset-metrics-btn` in the Observability Drawer header to reset telemetry with one click.
  - Formatted latency percentiles with explicit `ms` units (e.g., `15.7ms P50`).
  - Purged 145 mocked unit-test rows and leftover test upload PDFs from production storage, and isolated all pytest suites to `tmp_path`.

### Verified
- **46/46 Automated Pytest Suite Passing (100%)** in 54.43s.
- **35/35 Active Files Passing Static Architecture Boundary & Visual Contract Checks (0 violations)** in `scripts/check_architecture_layers.py`.
- **Playwright Visual & Responsive Audit (`tests/playwright_audit/run_studio_audit.py`)**: 0 console errors, 0 page errors, 0 network errors, 0 visual oddities across 4 viewports (`1440×900`, `1280×800`, `768×1024`, `375×812`).

---

## [v2.7.0] — 2026-10-12 (Milestone v2.0 Complete & Stabilized)

### Added
- **Dynamic Cloud Gemini API Key Configuration & Status Endpoint (`src/generation/generator.py`, `src/api/server.py`)**:
  - Direct UI / header runtime injection via `X-Gemini-API-Key` and `QueryRequest.api_key`, enabling instant Gemini activation without `.env` edits or server restarts.
  - New `GET /api/config/llm-status` route reporting connectivity and configuration status across `lmstudio`, `ollama`, and `gemini`.
- **Offline Grounded Extractive Fallback Synthesis (`src/api/server.py`, `src/generation/generator.py`)**:
  - Built graceful degraded fallback mode (`provider_status="degraded_extractive"`, `provider_fallback=True`) synthesizing structured direct context extracts when local LLM is offline or unreachable.
- **Web Studio UI Stabilization & Favicon (`src/api/static/*`, `src/api/server.py`)**:
  - Horizontal scrolling support on `.persona-switcher` with slim Titanium Obsidian & Emerald scrollbars, ensuring all 6 personas across `default_org` and `tenant_b` are accessible on all screen widths.
  - Sidebar Cloud Gemini API Key configuration drawer with masked input and `localStorage` persistence.
  - Bound and flip logic for `.citation-tooltip` preventing screen-edge clipping, and "Clear Chat" action.
  - Inline Emerald SVG `/favicon.ico` route returning 200 OK to eliminate browser console 404 errors.

### Fixed & Hardened
- **Relational Database Hygiene & Test Isolation (`tests/`, `data/metadata.db`)**:
  - Isolated test fixtures across `test_phase2_4_rbac_security.py`, `test_phase2_5_observability_resilience.py`, and `test_web_studio_ui.py` to temporary SQLite databases, guaranteeing that zero `test_*.pdf` rows leak into `data/metadata.db`.
  - Added empirical verification in `test_live_database_remains_clean_after_suite` checking the live disk database after full regression runs.
  - Auto-synchronized canonical benchmark PDFs (`compliance_policy.pdf`, `scanned_operations_log.pdf`, `tech_manual.pdf`) into `metadata.db` and Qdrant payloads with `default_org` and all 4 roles allowed.
- **Sub-Threshold Rerank Noise Filtering (`src/retrieval/retriever.py`)**:
  - Filtered candidate chunks where rerank score falls below `settings.min_rerank_score` while retaining top-1 candidate, eliminating low-confidence noise citations.
  - Sanitized unicode replacement characters (`\ufffd` and stray `?` symbols) in retrieved snippets.
- **QA Verification & Layer Boundaries**:
  - 46/46 automated `pytest` regression tests passing (100%).
  - 35/35 files verified with zero AST layer or visual violations in `scripts/check_architecture_layers.py`.

---

## [v2.6.0] — 2026-10-12 (Milestone v2.0 Complete)

### Added
- **Interactive Enterprise Demo Web Studio (`src/api/static/index.html`, `src/api/static/style.css`, `src/api/static/script.js`)**:
  - **Titanium Obsidian & Emerald Design Language (`L5B`)**: Obsidian `#09090b`/`#121316`/`#18181b` surfaces, Emerald `#10b981` primary accents, Warm Amber `#f59e0b`, Bronze `#d97706`, and Crimson `#ef4444` status indicators with strictly 0% blue/purple hex codes and 0% emojis (100% inline SVG vectors).
  - **Multi-Tenant Persona Switcher & JWT State Machine (`L5A`)**: One-click switching across 6 pre-seeded accounts (`Admin`, `Manager`, `Analyst`, `Viewer` in `default_org` & `tenant_b`) with auto-seed fallback (`POST /api/auth/seed`), active user/role badge, and dynamic RBAC upload/delete permission guards.
  - **Asynchronous 202 Ingestion Queue Polling & Document Catalog**: Drag-and-drop PDF upload with visibility (`ORG`, `PUBLIC`, `RESTRICTED`) and role ACL selectors, 750ms bounded `/api/jobs/{job_id}` progress polling, SHA-256 deduplication detection, and tenant-scoped `/api/documents` catalog with `ADMIN` deletion.
  - **4-Mode Retrieval Selector, Two-Tier Abstention & Citation Tooltips**: Interactive mode switcher (`hybrid_rerank`, `hybrid_rrf`, `vector_only`, `bm25_only`), model engine toggle (`lmstudio` vs. `gemini`), styled Two-Tier Abstention Gate banners (`LOW_CONFIDENCE`, `RBAC_EMPTY_RETRIEVAL`, `LLM_ABSTAIN`), and verbatim citation chips with cross-page labels, parent chunk IDs, rerank scores, and hover snippet tooltips.
  - **Per-Query Telemetry Ribbon & Slide-Out Observability Drawer**: Per-message stage latency chips (`Total`, `Embed`, `Search`, `Rerank`, `LLM` ms), token counters, USD cost badges, and live tenant-scoped `GET /api/metrics` slide-out drawer (`p50`/`p90`/`p95`/`avg` percentiles, token totals, USD spend, and abstention rate).
- **QA Verification & Static Contract Hardening**:
  - Built 5-test automated Web Studio UI & API integration suite in `tests/test_web_studio_ui.py`.
  - Upgraded `scripts/check_architecture_layers.py` to statically enforce Visual Rule 1 (0% blue/purple hexes) and Visual Rule 2 (0% emojis) across `L5A`/`L5B` in addition to CSP zero-inline-handler checks.
  - Hardened `IngestionJobManager.submit_job` (`duplicate`/`deduplicated` keys), `script.js` bounded polling & HTTP 429/non-JSON error resilience, and `test_phase2_5_observability_resilience.py` tenant isolation assertion.
  - **43/43 full regression tests passing (100%)** across all 6 phases (Phase 2.1 through 2.6) and **35/35 active files verified (0 violations)**.
  - Final Milestone `v2.0` audit certification logged in `docs/QA_AUDIT_REPORTS.md`.

---

## [v2.5.0] — 2026-10-12

### Added
- **Observability Spans & Distributed Tracing (`src/utils/telemetry.py`, `src/api/server.py`)**:
  - `CorrelationIdMiddleware` with incoming header preservation and auto-generated `X-Request-ID` UUIDv4 propagation.
  - Stage-level execution timers via `TelemetrySpan` tracking `embed_latency_ms`, `search_latency_ms`, `rerank_latency_ms`, `llm_latency_ms`, and `total_latency_ms`.
  - Offline deterministic token estimation `count_tokens` (~4 characters per token with word-boundary calibration).
  - USD query cost estimation engine `calculate_query_cost` with case-insensitive model catalog lookups and zero-cost local model support.
- **Sliding-Window Rate Limiting (`src/api/server.py`)**:
  - Thread-safe `SlidingWindowRateLimiter` returning HTTP `429 Too Many Requests` with `Retry-After: <seconds>` headers.
  - Clean bypass rules for health checks (`/health`), OpenAPI schemas (`/openapi.json`), and interactive documentation (`/docs`, `/redoc`).
- **Multi-Tenant Metrics Endpoint (`GET /api/metrics`)**:
  - Protected endpoint aggregating total queries, abstentions, latency percentiles (`p50`, `p90`, `p95`, `p99`, `avg`), token usage, and costs.
  - RBAC tenant isolation: `ADMIN` cross-org inspection vs. strict tenant bounding for non-admin accounts.
- **Containerization & CI/CD Pipeline**:
  - Multi-stage production `Dockerfile` with system libraries, health check, and ASGI server.
  - `docker-compose.yml` deploying `rag-api` and `qdrant` with persistent data volumes.
  - `.github/workflows/ci.yml` running automated layer boundary enforcement and pytest test suites on Python 3.13.
- **QA Verification & Environment Hardening**:
  - Hardened `OcrEngine` against Windows Smart App Control `c10_cuda.dll` blocks with automatic CPU fallback.
  - Hardened generator invocation in `server.py` against mock return value unpacking failures.
  - Built comprehensive 9-test suite in `tests/test_phase2_5_observability_resilience.py`.
  - 38/38 full regression tests passing across all 5 phases (Phase 2.1 through 2.5).
  - 35/35 active files verified with 0 AST layer violations in `scripts/check_architecture_layers.py`.
  - Formal audit certification report logged in `docs/QA_AUDIT_REPORTS.md`.

---

## [v2.4.0] — 2026-10-12

### Added
- **Cryptographic Security & JWT Primitives (`src/auth/`)**:
  - Standard library PBKDF2-HMAC-SHA256 password hashing with 16-byte random salt and timing-safe verification (`src/auth/security.py`).
  - PyJWT access token generation and claims validation (`sub`, `org_id`, `role`, `exp`, `iat`).
  - Idempotent demo account seeder initializing `default_org` and `tenant_b` with 4-tier RBAC accounts (`ADMIN`, `MANAGER`, `ANALYST`, `VIEWER`).
- **FastAPI Auth Dependencies & Pre-Retrieval RBAC Filters (`src/auth/dependencies.py`)**:
  - `get_current_user` OAuth2 Bearer token extractor and session validator.
  - `require_role()` role-hierarchy authorization dependency gate.
  - `build_qdrant_rbac_filter()` generating mandatory Qdrant pre-retrieval filters with multi-tenant org isolation and role visibility ACLs.
- **Authentication & RBAC REST APIs (`src/api/auth_routes.py`)**:
  - `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`, and `POST /api/auth/seed-demo-accounts`.
  - Integrated router into FastAPI gateway in `src/api/server.py`.
- **Multi-Tenant Document & Vector Isolation Hardening**:
  - Hardened `VectorStore.combine_filters()` with nested `models.Filter(must=[base, rbac])` preventing boolean `should` leakage.
  - Hardened `VectorStore.delete_document()` and `IngestionJobManager` with explicit `org_id` partition scoping.
  - Secured `/api/upload`, `GET /api/documents`, `DELETE /api/documents/{file_hash}`, and `POST /api/query` routes with JWT authentication and tenant boundaries.
- **QA Verification & Security Leakage Test Suite**:
  - 7/7 automated security and leakage tests passing in `tests/test_phase2_4_rbac_security.py` (cross-tenant zero data leakage verified).
  - Full regression suite passing 29/29 tests across all 4 phases.
  - 34/34 active files verified with 0 AST layer violations in `scripts/check_architecture_layers.py`.
  - Official audit certification report logged in `docs/QA_AUDIT_REPORTS.md`.

---

## [v2.3.0] — 2026-10-12

### Added
- **3-PDF Programmatic Benchmark Suite (`scripts/generate_benchmark_pdfs.py`)**:
  - `tech_manual.pdf` (4 pages): AetherDB cluster administration guide with ports (`9092`, `9093`, `8080`), flags (`--max-connections-per-node=1024`, `--raft-heartbeat-ms=250`), and hex errors (`ERR_BUFFER_OVERFLOW_0x7A`, `ERR_RAFT_SPLIT_BRAIN_0x9F`).
  - `compliance_policy.pdf` (4 pages): Governance framework with numbered clauses (`SEC-4.1.2`, `RET-7.3.1`), retention schedules, and cross-page text.
  - `scanned_operations_log.pdf` (3 pages): Incident logs with incident IDs (`INC-8821`, `INC-9943`) and simulated scanned image page for OCR testing.
- **100-QA Ground-Truth Dataset (`data/eval/benchmark_dataset.json`)**:
  - Curated 100 QA items: 40 Semantic, 30 Lexical / exact codes, 15 OCR table / scans, and 15 Unanswerable trap questions (`should_abstain = true`).
- **EvaluationEngine & Multi-Metric Scoring (`src/evaluation/evaluator.py`)**:
  - Recall@4, Recall@10, Mean Reciprocal Rank (MRR), Citation Accuracy (doc + page overlap), and deterministic Lexical Answer Faithfulness.
  - Abstention confusion matrix: True Positives, False Positives, Precision, Recall, and F1 score.
- **CLI Benchmark Runner & Reports (`evaluate.py`, `docs/BENCHMARK_REPORT.md`)**:
  - Command-line runner printing ASCII scorecards and exporting comprehensive Markdown benchmark reports comparing `vector_only`, `bm25_only`, `hybrid_rrf`, and `hybrid_rerank`.
  - Calibrated `min_rerank_score = 0.15` yielding 88.2% Abstention F1 and 0.898 MRR on RTX 4070.
- **QA Verification & Hardening**:
  - 22/22 combined regression tests passing across Phases 2.1, 2.2, and 2.3 in `tests/test_phase2_3_evaluation.py`.
  - 30/30 active files verified with 0 violations in `scripts/check_architecture_layers.py`.
  - Official audit report logged in `docs/QA_AUDIT_REPORTS.md`.

---

## [v2.2.0] — 2026-10-12

### Added
- **Cross-Page Parent-Child Chunking (`src/ingestion/chunker.py`)**:
  - Continuous multi-page character stream prevents splitting sentences and paragraphs across page boundaries.
  - Builds rich ~1200-char Parent blocks with section header detection and page ranges (`page_number` to `page_end`).
  - Subdivides into ~300-char Child chunks tagged with deterministic `parent_id`, `parent_text`, and ACL fields.
- **Sparse BM25 Encoder & BGE Query Prefix (`src/embeddings/`)**:
  - Built `SparseBM25Encoder` in `src/embeddings/sparse_bm25.py` preserving technical codes (`ERR-CV-4092`, `POL-SEC-104.2`, `_render_page_to_image`) with BM25 term-frequency saturation and stopword-only fallback.
  - Added asymmetric `BGE_QUERY_PREFIX` in `LocalEmbedder.embed_query` for BGE v1.5 query-side instruction tuning.
- **Qdrant Named Vectors & Hybrid Search (`src/database/vector_store.py`)**:
  - Upgraded to `rag_documents_v2` with named `"dense"` (384-d Cosine) and `"sparse"` (Qdrant `Modifier.IDF`) vectors.
  - Implemented 4 retrieval modes: `vector_only`, `bm25_only`, `hybrid_rrf`, and `hybrid_rerank`.
  - Reciprocal Rank Fusion (`k=60`) fusing dense and sparse ranks with tie-break stability.
  - Parent Block Deduplication retaining the highest-scoring child per parent block.
- **Two-Tier Abstention Gate (`src/retrieval/retriever.py`, `src/api/server.py`, `query.py`)**:
  - Tier-1 Pre-LLM Abstention: short-circuits on `EMPTY_RETRIEVAL` or when top rerank score < `min_rerank_score` (`LOW_SCORE`), skipping 100% of LLM compute and latency.
  - Tier-2 Post-LLM Abstention: detects model uncertainty (`LLM_ABSTAIN`) and suppresses false citations.
  - Added `-m / --mode` CLI flag to `query.py` and `mode` parameter to `POST /api/query`.
- **QA Verification & Hardening**:
  - 15/15 combined unit and API tests passing in `tests/test_phase2_1_async_ingestion.py` and `tests/test_phase2_2_hybrid_retrieval.py`.
  - 28/28 active files verified with 0 violations in `scripts/check_architecture_layers.py`.
  - Official audit report logged in `docs/QA_AUDIT_REPORTS.md`.

---

## [v2.1.0] — 2026-10-12

### Added
- **Core Refactor & Ingestion Resiliency (`src/ingestion/`)**:
  - Eliminated triple `fitz.open()` bug; unified parser and classifier under a single document handle.
  - Enforced `settings.ocr_timeout_seconds` on full-page and `MIXED` region EasyOCR.
  - Added live `progress_callback` reporting `(processed_pages, total_pages, stage)`.
- **Relational Metadata Engine & ORM Models (`src/database/`)**:
  - Created SQLAlchemy 2.0 ORM models for `Organization`, `User`, `DocumentRecord`, `IngestionJob`, and `QueryTelemetry`.
  - Initialized SQLite/PostgreSQL engine with multi-threaded compatibility.
- **Asynchronous Ingestion Worker Pool & CRUD API (`src/jobs/`, `src/api/`)**:
  - Built `IngestionJobManager` with bounded threadpool, retries, SHA-256 deduplication, and Qdrant rollback.
  - Exposed non-blocking HTTP `202 Accepted` `/api/upload`, `/api/jobs/{job_id}`, `/api/jobs`, `/api/documents`, and `DELETE /api/documents/{file_hash}`.

---

## [v2.0.0-planning] — 2026-10-12

### Added
- **Multi-Chat Implementation System (`ADR-009`, `ADR-010`)**:
  - Installed `GEMINI.md` Mission Control with 3-Tier Context Fencing (<500 tokens pre-task read), `<95`-token Intent/Pushback header, Multi-Chat Role Allocation (`PM`, `Backend`, `Frontend`, `QA`), Single-Wave Halt Firewall, and Zero-Prompt File-Driven Handoffs via `.gsd/PULSE.md`.
  - Created `FLOW.md` (and `.gsd/FLOW.md`) defining the canonical **5-Layer (`L1`–`L5B`) × 6-Section (`Sec 1`–`Sec 6`)** architecture matrix, end-to-end Mermaid sequence diagrams, and module manifest.
  - Created `DECISIONS.md` (and `.gsd/DECISION.md`) documenting `ADR-001` through `ADR-010` across PDF parsing, 3-way page classification, GPU EasyOCR memory guards, semantic/parent-child chunking, Qdrant UUIDv5 storage, Hybrid BM25+RRF search, Cross-Encoder reranking, Two-Tier Abstention, Multi-Tenant JWT/RBAC, Async Ingestion jobs, and Automated Evaluation.
  - Built `scripts/check_architecture_layers.py` static AST & layer boundary verifier (`23/23 active files passing, 0 violations`).
  - Scoped **Phase 2.1 (`v2.1.0`): Core Refactor, Relational DB & Asynchronous Ingestion** into `.gsd/ACTIVE_PLAN.md`.

---

## [v1.0.0] — 2026-04-22

### Added
- **Phase 1 — Adaptive PDF Ingestion Pipeline (`src/ingestion/`)**:
  - `PdfParser`: Streaming 8KB SHA-256 hashing and lazy page-by-page PyMuPDF extraction with per-page fault isolation.
  - `PageClassifier`: 3-way routing (`TEXT`, `IMAGE`, `MIXED`) based on `text_threshold=50` chars and embedded image counts.
  - `OcrEngine`: Lazy GPU-accelerated `EasyOCR` with 200 DPI rendering, recursive 50MB pixmap halving memory guard, and surgical XREF region OCR skipping icons `<50x50px`.
  - `SemanticChunker`: Paragraph-first and sentence-fallback chunking with word-safe rolling overlap (`100` chars) and heuristic section header detection.
- **Phase 2 — Local GPU Embeddings & Persistent Qdrant Store (`src/embeddings/`, `src/database/`)**:
  - `LocalEmbedder`: Lazy GPU loading for `BAAI/bge-small-en-v1.5` (384-d $L_2$-normalized embeddings) and `BAAI/bge-reranker-v2-m3` (`CrossEncoder`).
  - `VectorStore`: Persistent disk-backed Qdrant collection (`data/vectordb`) with deterministic `UUIDv5` chunk IDs and two-stage retrieval (`query_points` + Cross-Encoder reranking).
- **Phase 3 — Multi-Provider LLM Generation & CLI (`src/retrieval/`, `src/generation/`, `ingest.py`, `query.py`)**:
  - `Retriever`: Context block formatter with source PDF and page number provenance headers.
  - `BaseGenerator`: Pluggable strategy supporting `LMStudioGenerator` (`gpt-oss-20b`), `OllamaGenerator` (`llama3`), and `GeminiGenerator` (`gemini-2.5-flash`).
- **Phase 4 — FastAPI Backend & Glassmorphic Web UI (`src/api/`)**:
  - `server.py`: `/api/upload` and `/api/query` REST endpoints with static file hosting.
  - `src/api/static/`: Dark-mode glassmorphic Web UI (`index.html`, `style.css`, `script.js`) with drag-and-drop PDF upload, LLM provider toggle, and source citation chips.
