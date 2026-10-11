# Project-RAG_LLM — Mission Control

Enterprise-grade, self-hosted, GPU-accelerated RAG platform (Python, FastAPI, Qdrant Hybrid Search, SQLAlchemy, EasyOCR, Cross-Encoder Reranking & Automated Evaluation).

## 1. 3-Tier Context Fencing (Header-First Token Discipline)
To keep pre-task context analysis under ~500 tokens while preserving 100% native GSD alignment:

- **🟢 Tier 1: Active Task Fence (Read at Task/Wave Start):**
  1. [`.gsd/PULSE.md`](./.gsd/PULSE.md) — Last 3 lines only.
  2. [`.gsd/ACTIVE_PLAN.md`](./.gsd/ACTIVE_PLAN.md) — Current Phase native GSD XML `<wave>` & `<task>` blocks (overwritten per phase).
  3. [`.gsd/STATE.md`](./.gsd/STATE.md) — Lines 1–28 (Current position & active milestone).
  4. [`.gsd/ROADMAP.md`](./.gsd/ROADMAP.md) — Active Milestone section only.
  5. [`.gsd/DECISION.md`](./.gsd/DECISION.md) — **Lines 1–26 only** (Master Decision Table).
- **🟡 Tier 2: On-Demand Reference Fence (Read ONLY if Needed by Task):**
  - [`FLOW.md`](./FLOW.md) (**Lines 1–42 only** for the 5-Layer × 6-Section Matrix), specific ADR bodies in [`.gsd/DECISION.md`](./.gsd/DECISION.md) (Line 27+), [`.gsd/SPEC.md`](./.gsd/SPEC.md), [`requirements.txt`](./requirements.txt), and targeted code snippets (`StartLine`–`EndLine`).
- **🔴 Tier 3: Checkpoint & Version-End Fence (Strictly Forbidden During Normal Tasks):**
  - Accessed **only** at declared QA Checkpoints or Version Release boundaries: [`docs/protocols/code_review.md`](./docs/protocols/code_review.md), [`docs/protocols/version_phase_lifecycle.md`](./docs/protocols/version_phase_lifecycle.md), [`.gsd/archive/PULSE_ARCHIVE.md`](./.gsd/archive/PULSE_ARCHIVE.md), `CHANGELOG.md`, `README.md`.

> [!IMPORTANT]
> **File-Driven Handoff vs. Single-Wave Halt:** When the user triggers a session (e.g., `"proceed"`, `"continue"`, `/resume`), the agent reads `ASSIGN:` / `NEXT:` in `PULSE.md` and the matching `<wave status="pending">` in `ACTIVE_PLAN.md` to understand and execute its assigned Wave directly from the files—**no copy-pasted handoff prompt is needed**. However, an agent may execute at most **ONE Wave per user turn** and MUST HALT immediately after logging completion.

## 2. Intent, Assumptions & Constructive Pushback Protocol (<95 Tokens)
Every response MUST begin with this compact header block (<95 tokens total):
```markdown
> **Intent [HIGH|MED|LOW]:** <1-line Rephrase & Respond (RaR) summary of user's core goal>
> **Assumptions & Suggestions:** <Stated default assumptions + proactive technical pushback/simpler alternatives>
```
- **Proactive Pushback Mandate:** Unless the user explicitly tags an instruction with **`[RULE]`**, agents MUST NOT blindly comply. Evaluate ideas for token waste, over-engineering, or architectural friction and proactively suggest superior alternatives.
- **1-Question Abstention Gate:** On critical architectural ambiguity where guessing risks rework, halt and ask **1 concise question** (<40 tokens) before acting.

## 3. Multi-Chat Role Allocation & Native GSD XML Waves
Work is divided across dedicated chat sessions using [`.gsd/ACTIVE_PLAN.md`](./.gsd/ACTIVE_PLAN.md):
- **Product Manager / Mission Control (`PM`)**: Scopes `v2.x` phases into `.gsd/ACTIVE_PLAN.md` using native GSD XML (`<wave id="N" role="..." section="Sec X" layers="LY" status="pending">`, `<task type="auto">`, `<files>` with exact line ranges, `<action>`, `<verify>`, `<done>`). **Halts after planning.**
- **Backend / AI Engineer (`Backend`)**: Layers `L1` (Data & Qdrant/SQLAlchemy Persistence), `L2` (Core Domain CRUD, Auth/RBAC & Async Jobs), `L3` (RAG Intelligence: Ingestion, OCR, Chunking, Hybrid Search, Reranking, LLM Generation & Evaluation), `L4` (FastAPI Gateway, Middleware & CLI Entrypoints).
- **Frontend Engineer (`Frontend`)**: Layer `L5A` (Stateful API-Connected Views) and Layer `L5B` (Pure Visuals / Stateless HTML & CSS).
- **QA Auditor (`QA`)**: Automated `pytest` verification, AST/Layer boundary validation (`.venv\Scripts\python.exe scripts/check_architecture_layers.py`), RBAC security leakage tests, and RAG Evaluation Benchmark sign-off.

**Work Package (Wave) Boundary & 20–30 Min Execution Burst Protocol:**
- **Zero-Exploration Execution:** Engineering/QA roles use the pre-targeted `<files>` and line pointers in `.gsd/ACTIVE_PLAN.md` to jump straight to execution without exploratory file-reading waste.
- **Continuous In-Wave Velocity:** Once triggered, the assigned role executes all `<task>` items in its assigned `<wave>` continuously to completion (20–30 min sprint), filtering CLI outputs and offloading deep research to ephemeral subagents.
- **Role Halt Mandate:** Upon completing and empirically verifying the assigned Wave, the agent marks `<wave status="completed">` in `ACTIVE_PLAN.md`, logs to `PULSE.md`, states which Role chat is next, and **MUST HALT IMMEDIATELY** (see ADR-006, ADR-009).

## 4. Zero-Prompt File-Driven Handoff & Pruning Rule
When completing any Wave/Task, append a single line to [`.gsd/PULSE.md`](./.gsd/PULSE.md) (**no `PROMPT:` field**):
```text
[YYYY-MM-DD HH:MM] | DONE: <what was executed> | ASSIGN: <PM | Backend | Frontend | QA> | NEXT: <Wave ID & short task pointer in .gsd/ACTIVE_PLAN.md>
```
- **No Copy-Paste Prompts:** Do NOT generate or output copy-paste prompts. The next assigned chat will derive its instructions automatically from `PULSE.md` (`ASSIGN` + `NEXT`) and `.gsd/ACTIVE_PLAN.md` when the user opens that chat.
- **Writer-Never-Chains Guard:** The agent that appends a `DONE:` line to `PULSE.md` is **categorically forbidden** from executing the `NEXT:` Wave in the same turn. Report the assigned next role to the user and **HALT EXECUTION**.
- **Pulse Pruning Policy:** Keep [`.gsd/PULSE.md`](./.gsd/PULSE.md) to a rolling window of the **last 3 active entries**; archive older entries to [`.gsd/archive/PULSE_ARCHIVE.md`](./.gsd/archive/PULSE_ARCHIVE.md).

## 5. Single-Wave Firewall & Feature Scoping (ADR-007, ADR-008, ADR-009)
1. **One Wave Per Turn:** Completing Wave X never authorizes executing Wave X+1 in the same turn. Each Wave boundary is a hard stop.
2. **Depth vs. Breadth (DFS vs. BFS) Matrix:** Evaluate every new feature against: (1) **Depth (DFS)** retrieval precision/resilience, (2) **Breadth (BFS)** API/UI touchpoints, (3) **Complexity Budget** schema/maintainability, and (4) **Performance Impact** GPU VRAM/RAM/latency bounds.

## 6. 5-Layer × 6-Section Architecture & Boundary Enforcement (ADR-010)
All code and `<wave>` plans map to [`FLOW.md`](./FLOW.md) (**Lines 1–42**):
- **5 Horizontal Layers:** `L1` (Data, Vector & Relational Persistence) $\leftarrow$ `L2` (Core Domain, Auth/RBAC & Async Jobs) $\leftarrow$ `L3` (RAG Intelligence, OCR, Hybrid Search, Reranking, Generation & Evaluation) $\leftarrow$ `L4` (FastAPI Gateway, Middleware & CLI) $\leftarrow$ `L5A` (Stateful Views) & `L5B` (Pure Visuals / Stateless UI).
- **6 Product Sections:** `Sec 1` (Hybrid Search & Reranking), `Sec 2` (Automated RAG Evaluation), `Sec 3` (JWT Auth & Document RBAC), `Sec 4` (Observability & Cost Telemetry), `Sec 5` (Adaptive PDF Ingestion & Async Jobs), `Sec 6` (API Gateway, Resilience & Web Studio).
- **Automated Boundary Contract (`scripts/check_architecture_layers.py`):** Every `.py` file in `src/` and `.js`/`.html`/`.css` file in `src/api/static/` must be registered in the layer manifest (`.venv\Scripts\python.exe scripts/check_architecture_layers.py`). `L1–L3` files must never import `L4` (`server.py`) or `L5` UI.
