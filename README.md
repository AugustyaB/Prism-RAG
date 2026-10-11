# Project-RAG_LLM — Enterprise Knowledge Intelligence & RAG Platform

An enterprise-grade, self-hosted, GPU-accelerated **Retrieval-Augmented Generation (RAG)** platform built with **Python, FastAPI, Qdrant, SQLAlchemy, PyMuPDF, EasyOCR, and SentenceTransformers**.

Designed to ingest massive, heterogeneous PDFs (1,000+ page technical manuals, enterprise policy docs, and scanned layouts), index them with **Dense + Sparse BM25 Hybrid Search & Cross-Encoder Reranking**, enforce **Multi-Tenant JWT Authentication & Pre-Retrieval RBAC**, track **Stage-Level Latency & Token/Cost Telemetry**, and verify retrieval/generation quality via an **Automated 100-Question Evaluation Benchmark**.

---

## Core Architecture & Features

### 1. Adaptive PDF Ingestion & GPU OCR (`Sec 5`)
* **Streaming SHA-256 Fingerprinting (`PdfParser`)**: Hashes files in 8KB chunks for instant deduplication and idempotent `UUIDv5` Qdrant indexing.
* **3-Way Page Classifier (`PageClassifier`)**: Routes each page based on extractable text (`text_threshold = 50` chars) and embedded object counts:
  * `PageType.TEXT`: Direct C-level PyMuPDF extraction (~1ms/page, zero GPU usage).
  * `PageType.IMAGE`: Full-page 200 DPI rasterization + GPU-accelerated **EasyOCR** with a recursive **50MB pixmap halving guard** and per-page timeout.
  * `PageType.MIXED`: Retains native digital text and surgically OCRs embedded XREF images (filtering out decorative icons `<50x50px`).
* **Asynchronous Background Ingestion (`v2.1`)**: Non-blocking `202 Accepted` uploads backed by SQLAlchemy (`data/metadata.db`) and a background worker pool tracking `PENDING` $\to$ `PROCESSING` $\to$ `COMPLETED` / `FAILED`.

### 2. Hybrid Search, RRF & Cross-Encoder Reranking (`Sec 1`)
* **Cross-Page Parent-Child Chunking (`SemanticChunker`)**: Splits documents across paragraph and sentence boundaries with word-safe rolling overlap, indexing sharp child chunks (`~300 chars`) linked to cross-page parent contexts (`~1200 chars`).
* **Two-Stage Hybrid Retrieval (`VectorStore` & `LocalEmbedder`)**:
  * **Stage 1 (Candidate Recall)**: Combines dense semantic vectors (`BAAI/bge-small-en-v1.5`, 384-d Cosine with BGE query prefix) and sparse lexical BM25 vectors (`SparseVectorParams(modifier=IDF)`) via **Reciprocal Rank Fusion ($\text{RRF}, k=60$)**.
  * **Stage 2 (Precision Reranking)**: Scores candidate passages jointly with the query using **`BAAI/bge-reranker-v2-m3`** (`CrossEncoder` on CUDA).
* **Two-Tier Abstention Gate**: Short-circuits before calling the LLM when reranker confidence falls below `min_rerank_score` (saving latency and token cost), backed by strict grounded prompt instructions.

### 3. Multi-Provider LLM Generation (`Sec 1, Sec 6`)
* Pluggable `BaseGenerator` strategy supporting:
  * **LM Studio (`LMStudioGenerator`)**: Local OpenAI-compatible inference (`gpt-oss-20b`).
  * **Ollama (`OllamaGenerator`)**: Local inference (`llama3`).
  * **Google Gemini (`GeminiGenerator`)**: Cloud fallback (`gemini-2.5-flash`).

### 4. Automated Evaluation, Multi-Tenant RBAC & Observability (`Sec 2, Sec 3, Sec 4`)
* **Automated 100-Question Evaluation Suite (`v2.3`)**: Benchmarks `vector_only`, `bm25_only`, `hybrid_rrf`, and `hybrid_rerank` across **Recall@K, MRR, Answer Faithfulness, Citation Accuracy, and Abstention Quality**.
* **Multi-Tenant JWT Auth & Pre-Retrieval RBAC (`v2.4`)**: Enforces `Organization`, `Role` (`ADMIN`, `MANAGER`, `ANALYST`, `VIEWER`), and `Visibility` (`PUBLIC`, `ORG`, `RESTRICTED`) filters directly inside Qdrant before retrieved text reaches the LLM.
* **Observability & Cost Tracking (`v2.5`)**: Tracks `X-Request-ID`, per-stage latency spans (`embed`, `search`, `rerank`, `llm`), p50/p95 latencies, token usage, and estimated USD cost.

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