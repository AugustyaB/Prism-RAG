---
trigger: always_on
description: "Automatic Two-Tier Code Review Protocol combining Python AST/Layer Boundary Verification with Antigravity Semantic Review. Automatically executed at the end of every phase before documentation sync."
---

# Code Review Protocol (AST/Layer Check + Antigravity Semantic Review + Pytest)

This rule defines the automated code review protocol for **`Project-RAG_LLM`**. It combines deterministic static analysis (`scripts/check_architecture_layers.py`) with deep semantic analysis and `pytest` verification.

---

## 1. Two-Tier Review Execution Strategy

### Tier 1: Deterministic Static & Layer Boundary Analysis
1. **Run Architecture Layer & AST Syntax Verifier**:
   ```powershell
   .\.venv\Scripts\python.exe scripts/check_architecture_layers.py
   ```
   *Strict Requirement:* Must pass with **0 violations** and **100% file registration**.

### Tier 2: Antigravity Semantic Review
Inspect all modified files in the active Wave/Phase across:
1. **Multi-Tenant Security & RBAC Isolation**:
   - Ensure document permissions (`org_id`, `visibility`, `allowed_roles`) are enforced **pre-retrieval** inside Qdrant payload filters before any chunk text reaches the LLM.
   - Verify password hashes and JWT tokens never leak in API responses or logs.
2. **Hardware, Memory & Async Resilience**:
   - Ensure PDF handles (`fitz.Document`) are always closed in `finally` blocks.
   - Verify synchronous PyTorch/EasyOCR/LLM calls never block FastAPI's main asyncio event loop during async job status polling.
   - Verify failed background jobs cleanly roll back partial Qdrant points (`delete_document(file_hash)`).
3. **RAG Retrieval & Citation Fidelity**:
   - Ensure BGE v1.5 query instruction prefixes are applied to search queries (`embed_query`) and omitted from document passages (`embed_batch`).
   - Verify parent-child deduplication prevents sending duplicate parent blocks to the LLM.

### Tier 3: Automated Pytest Verification
Run the phase's automated `pytest` suite and capture empirical output before marking any Wave/Phase complete.
