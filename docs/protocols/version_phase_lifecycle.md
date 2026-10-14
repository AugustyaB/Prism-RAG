# Version & Phase Lifecycle Protocol — Project-RAG_LLM

> **Universal Development Flow:**
> **Planning (`PM`) ➔ Wave Implementation (`Backend`/`Frontend`) ➔ Auditing (`QA`) ➔ Documentation Sync ➔ Local Git Commit ➔ Single-Wave Halt**

---

## 1. Multi-Chat Wave Lifecycle

Each phase (`v2.1` – `v2.6`) is decomposed into atomic `<wave>` blocks inside [`.gsd/ACTIVE_PLAN.md`](../../.gsd/ACTIVE_PLAN.md):

1. **Tier 1 Context Fence Read (<500 Tokens)**:
   - At the start of any turn, read only:
     - `.gsd/PULSE.md` (last 3 lines)
     - `.gsd/ACTIVE_PLAN.md` (current phase XML waves)
     - `.gsd/STATE.md` (lines 1–28)
     - `.gsd/ROADMAP.md` (active milestone table)
     - `.gsd/DECISION.md` (lines 1–26)
2. **Wave Execution**:
   - Execute only the single assigned `<wave status="pending">` matching `NEXT:` in `.gsd/PULSE.md`.
   - Run the `<verify>` commands and `scripts/check_architecture_layers.py`.
3. **Wave Completion & Mandatory Halt**:
   - Mark `<wave status="completed">` in `.gsd/ACTIVE_PLAN.md`.
   - Append a single completion line to `.gsd/PULSE.md` (keeping at most 3 active entries; archive older lines to `.gsd/archive/PULSE_ARCHIVE.md`).
   - State which role chat is next (`PM`, `Backend`, `Frontend`, or `QA`) and **HALT IMMEDIATELY**.
4. **Local Git Policy**:
   - Commit completed phases locally (`feat(phase-2.x): ...`). Do **not** push to a remote GitHub origin until the user explicitly requests publishing the repository.
