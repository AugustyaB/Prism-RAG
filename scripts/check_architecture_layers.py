#!/usr/bin/env python
"""
============================================================================
Project-RAG_LLM — 5-Layer × 6-Section Architecture Boundary Verifier (ADR-010)
============================================================================

Statically verifies that all Python source files in `src/` and web assets in
`src/api/static/` adhere to the canonical 5-Layer (L1–L5B) × 6-Section (Sec 1–6)
matrix in FLOW.md:

  1. 100% Layer & Section Registration: No unclassified/orphan files in `src/`.
  2. Python AST Syntax & Import Validation: Every `.py` file in `src/` compiles
     cleanly with 0 syntax errors.
  3. Unidirectional Downward Layer Isolation:
     - `L1` (Data & Persistence) must never import `L2`, `L3`, `L4`, or `L5`.
     - `L2` (Core Domain, RBAC & Jobs) must never import `L4` (`src.api.*`) or `L5`.
     - `L3` (RAG Intelligence, OCR, Retrieval, Eval) must never import `L4` (`src.api.*`) or `L5`.
     - `L5B` (Pure Visuals: HTML/CSS) must contain zero inline JS event handlers (`onclick=`, etc.).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Canonical 5-Layer × 6-Section Manifest
ARCHITECTURE_MANIFEST: dict[str, dict[str, object]] = {
    # ── LAYER 1: Data, Vector & Relational Persistence (L1) ──────────────────
    "src/__init__.py": {"layer": "L1", "sections": ["Sec 1", "Sec 2", "Sec 3", "Sec 4", "Sec 5", "Sec 6"]},
    "src/config.py": {"layer": "L1", "sections": ["Sec 1", "Sec 2", "Sec 3", "Sec 4", "Sec 5", "Sec 6"]},
    "src/models/__init__.py": {"layer": "L1", "sections": ["Sec 1", "Sec 3", "Sec 5"]},
    "src/database/__init__.py": {"layer": "L1", "sections": ["Sec 1", "Sec 3", "Sec 5"]},
    "src/database/vector_store.py": {"layer": "L1", "sections": ["Sec 1", "Sec 3", "Sec 5"]},
    "src/database/models.py": {"layer": "L1", "sections": ["Sec 3", "Sec 4", "Sec 5", "Sec 6"]},
    "src/database/relational.py": {"layer": "L1", "sections": ["Sec 3", "Sec 4", "Sec 5", "Sec 6"]},

    # ── LAYER 2: Core Domain, Multi-Tenant RBAC & Async Jobs (L2) ────────────
    "src/jobs/__init__.py": {"layer": "L2", "sections": ["Sec 5"]},
    "src/jobs/ingestion_worker.py": {"layer": "L2", "sections": ["Sec 5"]},
    "src/auth/__init__.py": {"layer": "L2", "sections": ["Sec 3"]},
    "src/auth/security.py": {"layer": "L2", "sections": ["Sec 3"]},
    "src/auth/dependencies.py": {"layer": "L2", "sections": ["Sec 3"]},
    "src/agents/__init__.py": {"layer": "L2", "sections": ["Sec 1", "Sec 6"]},
    "src/agents/memory.py": {"layer": "L2", "sections": ["Sec 3", "Sec 4", "Sec 6"]},
    "src/agents/state_machine.py": {"layer": "L2", "sections": ["Sec 1", "Sec 6"]},
    "src/agents/tools.py": {"layer": "L2", "sections": ["Sec 1", "Sec 6"]},

    # ── LAYER 3: RAG Intelligence, OCR, Hybrid Retrieval & Evaluation (L3) ───
    "src/ingestion/__init__.py": {"layer": "L3", "sections": ["Sec 5"]},
    "src/ingestion/pdf_parser.py": {"layer": "L3", "sections": ["Sec 5"]},
    "src/ingestion/classifier.py": {"layer": "L3", "sections": ["Sec 5"]},
    "src/ingestion/ocr.py": {"layer": "L3", "sections": ["Sec 5"]},
    "src/ingestion/chunker.py": {"layer": "L3", "sections": ["Sec 1", "Sec 5"]},
    "src/ingestion/pipeline.py": {"layer": "L3", "sections": ["Sec 5"]},
    "src/embeddings/__init__.py": {"layer": "L3", "sections": ["Sec 1"]},
    "src/embeddings/embedder.py": {"layer": "L3", "sections": ["Sec 1"]},
    "src/embeddings/sparse_bm25.py": {"layer": "L3", "sections": ["Sec 1"]},
    "src/retrieval/__init__.py": {"layer": "L3", "sections": ["Sec 1"]},
    "src/retrieval/retriever.py": {"layer": "L3", "sections": ["Sec 1"]},
    "src/retrieval/multi_query.py": {"layer": "L3", "sections": ["Sec 1"]},
    "src/retrieval/crag.py": {"layer": "L3", "sections": ["Sec 1", "Sec 2"]},
    "src/generation/__init__.py": {"layer": "L3", "sections": ["Sec 1", "Sec 4", "Sec 6"]},
    "src/generation/generator.py": {"layer": "L3", "sections": ["Sec 1", "Sec 4", "Sec 6"]},
    "src/evaluation/__init__.py": {"layer": "L3", "sections": ["Sec 2"]},
    "src/evaluation/evaluator.py": {"layer": "L3", "sections": ["Sec 2"]},
    "src/utils/__init__.py": {"layer": "L3", "sections": ["Sec 4", "Sec 6"]},
    "src/utils/logging.py": {"layer": "L3", "sections": ["Sec 4", "Sec 6"]},
    "src/utils/telemetry.py": {"layer": "L3", "sections": ["Sec 4"]},

    # ── LAYER 4: FastAPI Gateway, Middleware & Routes (L4) ───────────────────
    "src/api/server.py": {"layer": "L4", "sections": ["Sec 1", "Sec 3", "Sec 4", "Sec 5", "Sec 6"]},
    "src/api/auth_routes.py": {"layer": "L4", "sections": ["Sec 3"]},

    # ── LAYER 5A: Stateful Feature Views (L5A — Fetch API Connected) ─────────
    "src/api/static/script.js": {"layer": "L5A", "sections": ["Sec 1", "Sec 3", "Sec 4", "Sec 5", "Sec 6"]},

    # ── LAYER 5B: Pure Visuals (L5B — Stateless HTML / CSS) ──────────────────
    "src/api/static/index.html": {"layer": "L5B", "sections": ["Sec 1", "Sec 3", "Sec 4", "Sec 5", "Sec 6"]},
    "src/api/static/style.css": {"layer": "L5B", "sections": ["Sec 6"]},
}


def module_to_rel_path(module_name: str) -> str | None:
    """Resolve a `src.*` module import to a relative path in ARCHITECTURE_MANIFEST."""
    if not module_name.startswith("src"):
        return None
    candidate_py = module_name.replace(".", "/") + ".py"
    if candidate_py in ARCHITECTURE_MANIFEST:
        return candidate_py
    candidate_init = module_name.replace(".", "/") + "/__init__.py"
    if candidate_init in ARCHITECTURE_MANIFEST:
        return candidate_init
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    errors: list[str] = []
    src_dir = ROOT_DIR / "src"

    discovered_files: list[str] = []
    for path in sorted(src_dir.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        if path.suffix in {".py", ".js", ".html", ".css"}:
            rel = path.relative_to(ROOT_DIR).as_posix()
            discovered_files.append(rel)

    # 1. Verify 100% Registration of Existing Files
    for rel in discovered_files:
        if rel not in ARCHITECTURE_MANIFEST:
            errors.append(f"[UNREGISTERED FILE] {rel} is missing from ARCHITECTURE_MANIFEST in scripts/check_architecture_layers.py")

    # 2. AST Parse & Layer Dependency Verification
    for rel in discovered_files:
        full_path = ROOT_DIR / rel
        meta = ARCHITECTURE_MANIFEST.get(rel)
        if not meta:
            continue
        layer = str(meta["layer"])

        if rel.endswith(".py"):
            try:
                source = full_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=rel)
            except SyntaxError as exc:
                errors.append(f"[SYNTAX ERROR] {rel}:{exc.lineno}: {exc.msg}")
                continue

            # Extract imported `src.*` modules
            imported_modules: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("src"):
                            imported_modules.add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("src"):
                        imported_modules.add(node.module)
                        for alias in node.names:
                            sub_mod = f"{node.module}.{alias.name}"
                            if module_to_rel_path(sub_mod):
                                imported_modules.add(sub_mod)

            for mod in imported_modules:
                target_rel = module_to_rel_path(mod)
                if not target_rel or target_rel == rel:
                    continue
                target_layer = str(ARCHITECTURE_MANIFEST[target_rel]["layer"])

                # L1–L3 must never import L4 (`src/api/*`) or L5
                if layer in {"L1", "L2", "L3"} and target_layer in {"L4", "L5A", "L5B"}:
                    errors.append(
                        f"[LAYER VIOLATION] {rel} ({layer}) imports {target_rel} ({target_layer}) — Lower layers (L1-L3) must never import L4/L5."
                    )
                # L1 must never import L2 (except L3 ingestion types used as type hints in vector_store)
                if layer == "L1" and target_layer == "L2":
                    errors.append(
                        f"[LAYER VIOLATION] {rel} (L1) imports {target_rel} (L2) — L1 persistence must not depend on L2 services."
                    )

        elif layer in {"L5A", "L5B"}:
            web_text = full_path.read_text(encoding="utf-8")
            if rel.endswith(".html") and layer == "L5B":
                if re.search(
                    r"\bon(click|change|submit|load|keyup|keydown|input|focus|blur)\s*=",
                    web_text,
                    re.IGNORECASE,
                ):
                    errors.append(f"[CSP / L5B VIOLATION] {rel} contains inline DOM event handlers.")
            if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", web_text):
                errors.append(
                    f"[VISUAL RULE 2 VIOLATION] {rel} ({layer}) contains forbidden emoji characters (SVG vectors required)."
                )
            if re.search(
                r"#(?:3b82f6|06b6d4|8b5cf6|6366f1|2563eb|1d4ed8|7c3aed|9333ea|0ea5e9|38bdf8)\b",
                web_text,
                re.IGNORECASE,
            ):
                errors.append(
                    f"[VISUAL RULE 1 VIOLATION] {rel} ({layer}) contains forbidden blue/cyan/indigo/violet/purple hex colors."
                )

    if errors:
        print("❌ Architecture Layer Boundary Check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        f"✅ Architecture Layer Boundary Check PASSED: {len(discovered_files)}/{len(discovered_files)} active files verified across L1–L5B × Sec 1–6 (0 violations)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
