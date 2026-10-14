# Project-RAG_LLM — Enterprise Knowledge Intelligence & Agentic RAG Platform

An enterprise-grade, self-hosted, GPU-accelerated **Agentic Retrieval-Augmented Generation (RAG)** platform built with **Python, FastAPI, Qdrant Hybrid Search, SQLAlchemy, PyMuPDF, EasyOCR, and SentenceTransformers**.

Designed to ingest massive, heterogeneous enterprise documents (technical manuals, financial reports, compliance standards, and scanned forms), index them with **Parent-Child Chunking, Dense + Sparse BM25 Hybrid Search & Cross-Encoder Reranking**, operate a **Pure-Python Agentic State Machine (Multi-Query RRF, CRAG Self-Reflection & Tool Dispatcher)**, manage **Persistent Conversational Sessions**, enforce **Multi-Tenant JWT Authentication & Pre-Retrieval RBAC**, and provide an interactive **Multi-Turn Web Studio UI** adhering strictly to the **Titanium Obsidian & Emerald** design system.

---

## Core Architecture & Features

### 1. Native Agentic Intelligence & Tool Dispatcher (`v3.0`, `Sec 1, Sec 2, Sec 6`)
* **Pure-Python FSM State Machine (`src/agents/state_machine.py`)**: Finite state execution loop (`Load Memory` $\to$ `Decompose` $\to$ `Retrieve` $\to$ `Grade` $\to$ `Tool` $\to$ `Synthesize`) operating without heavy third-party framework overhead (LangChain/LangGraph-free).
* **Conversational Session Memory (`src/agents/memory.py`)**: Persistent SQLite/SQLAlchemy `ChatSession` and `ChatMessage` models with a 3-turn sliding window and rolling background summarization for coherent multi-turn context.
* **Multi-Query Decomposition & Expansion (`src/retrieval/multi_query.py`)**: Fast heuristic sub-query expansion producing 3 distinct retrieval angles merged via native Qdrant Reciprocal Rank Fusion ($k=60$).
* **Corrective RAG (CRAG) & Self-Reflection (`src/retrieval/crag.py`)**: Evaluates retrieved passage confidence (`HIGH`, `MEDIUM`, `LOW`) with a strictly bounded 1-retry query reformulation loop, preventing hallucinations and infinite loops.
* **Curated 4-Tool Dispatcher (`src/agents/tools.py`)**: First-principles tool execution providing:
  * `document_search`: Targeted hybrid vector + BM25 keyword search.
  * `document_summary`: Comprehensive multi-chunk executive summarizer.
  * `deterministic_calculator`: Safe AST arithmetic evaluation engine (zero `eval`/`exec`).
  * `document_comparison`: Side-by-side comparative synthesis across documents.

### 2. Multi-Turn Conversational Web Studio (`v3.0`, `Sec 6`)
* **Titanium Obsidian & Emerald Design**: Built with a strict visual contract—strictly 0% blue/purple colors and 0% emojis (100% inline SVG vectors).
* **Multi-Turn Thread & Reasoning Trace Accordion**: Full interactive chat stream rendering user and assistant Obsidian/Emerald cards with live typing effect and an expandable `#reasoning-trace` disclosure exposing query decomposition angles, confidence grades, tool execution chips, and citations.
* **Session Switcher & Profile Drawer**: 1-click "New Session" creation, session history dropdown, and top-right authenticated user profile avatar with sliding RBAC clearances drawer.

### 3. Adaptive PDF Ingestion & Concurrency Queue (`Sec 4, Sec 5`)
* **Streaming SHA-256 Fingerprinting (`PdfParser`)**: Hashes files in 8KB chunks for instant deduplication and idempotent `UUIDv5` Qdrant indexing.
* **3-Way Page Classifier (`PageClassifier`)**: Routes pages based on extractable text and embedded object counts (`PageType.TEXT` via C-level PyMuPDF, `PageType.IMAGE` via GPU-accelerated EasyOCR with 50MB pixmap guard, and `PageType.MIXED` with surgical XREF OCR).
* **Sequential Concurrency Queue (`IngestionJobManager`)**: Non-blocking `202 Accepted` background upload queue with strict `Concurrency = 1` preventing GPU CUDA out-of-memory thrashing.
* **Auto-Ingest Folder Watcher (`AutoIngestWatcher`)**: Background daemon monitoring `data/auto_ingest/` for automated directory ingestion.

### 4. Hybrid Search, RRF & Cross-Encoder Reranking (`Sec 1`)
* **Cross-Page Parent-Child Chunking (`SemanticChunker`)**: Indexes granular child chunks (`~300 chars`) linked to broad cross-page parent contexts (`~1200 chars`).
* **Two-Stage Hybrid Retrieval (`VectorStore` & `Retriever`)**:
  * **Stage 1 (Recall)**: Dense BGE embeddings (`BAAI/bge-small-en-v1.5`, 384-d Cosine) and sparse BM25 IDF vectors fused via **Reciprocal Rank Fusion ($\text{RRF}, k=60$)**.
  * **Stage 2 (Precision)**: Deep cross-encoding via **`BAAI/bge-reranker-v2-m3`** on CUDA.
* **Two-Tier Abstention Gate**: Short-circuits when reranker confidence falls below threshold, saving latency and token costs.

### 5. Multi-Tenant RBAC, Enterprise Identity & Observability (`Sec 2, Sec 3, Sec 4`)
* **Enterprise Login Gate & Dual-Mode Password Recovery (`v2.9`)**: Email/password authentication, 1-click demo persona switcher, and secure SMTP / Admin-token password reset.
* **Pre-Retrieval Document RBAC**: Enforces Organization, Role (`ADMIN`, `MANAGER`, `ANALYST`, `VIEWER`), and Visibility (`PUBLIC`, `ORG`, `RESTRICTED`) filters inside Qdrant before vector search.
* **Observability & Cost Telemetry**: Tracks `X-Request-ID`, stage latencies (`embed`, `search`, `rerank`, `llm`), p50/p95 metrics, token usage, and USD accounting.
* **Automated 100-Question Evaluation Suite**: Comprehensive benchmark evaluating Recall@K, MRR, Answer Faithfulness, and Citation Accuracy.

---

## Documentation Map

* **Architecture & 5-Layer × 6-Section Matrix**: [FLOW.md](./FLOW.md)
* **Architectural Decision Records (ADRs 001–010)**: [DECISIONS.md](./DECISIONS.md) / [.gsd/DECISION.md](./.gsd/DECISION.md)
* **Project Specification**: [.gsd/SPEC.md](./.gsd/SPEC.md)
* **Milestone Roadmap**: [.gsd/ROADMAP.md](./.gsd/ROADMAP.md)
* **Active Phase XML Plan**: [.gsd/ACTIVE_PLAN.md](./.gsd/ACTIVE_PLAN.md)
* **Project State & Pulse Log**: [.gsd/STATE.md](./.gsd/STATE.md) & [.gsd/PULSE.md](./.gsd/PULSE.md)

---

## Quick Start

### 1. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run Architecture & Layer Boundary Check
```powershell
.\.venv\Scripts\python.exe scripts/check_architecture_layers.py
```

### 3. Start FastAPI Server & Web Studio
```powershell
.\.venv\Scripts\python.exe -m src.api.server
```
Open **`http://127.0.0.1:8000`** in your browser (Swagger API docs at **`http://127.0.0.1:8000/docs`**).

### 4. CLI Ingestion & Querying
```powershell
# Ingest a PDF or directory of PDFs into Qdrant
.\.venv\Scripts\python.exe ingest.py data\raw\sample.pdf --save

# Interactive RAG Query REPL
.\.venv\Scripts\python.exe query.py

# Single query with provider override (lmstudio | ollama | gemini)
.\.venv\Scripts\python.exe query.py -q "How does the page classifier work?" -p gemini
```