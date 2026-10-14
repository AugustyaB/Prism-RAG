"""Pure-Python Agentic RAG Finite State Machine & Pipeline Coordinator (L2)."""

from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from loguru import logger

from src.agents.memory import ConversationMemoryManager
from src.agents.tools import execute_tool
from src.config import settings
from src.database.vector_store import VectorStore
from src.generation.generator import get_generator, synthesize_extractive_answer
from src.retrieval.retriever import Retriever
from src.utils.telemetry import count_tokens


class RAGState(TypedDict, total=False):
    """Execution state passed through the Agentic RAG State Machine."""

    query: str
    session_id: str | None
    user_id: str | None
    org_id: str
    mode: str
    provider: str | None
    api_key: str | None
    conversation_context: str
    history_messages: list[dict[str, Any]]
    sub_queries: list[str]
    retrieved_chunks: list[dict[str, Any]]
    context_str: str
    top_score: float | None
    crag_decision: dict[str, Any]
    multi_query_meta: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    reasoning_trace: list[dict[str, Any]]
    final_answer: str
    citations: list[dict[str, Any]]
    abstained: bool
    abstention_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    provider_status: str
    provider_fallback: bool


class AgenticRAGPipeline:
    """Pure-Python FSM coordinating Memory -> Multi-Query -> CRAG -> Tool Dispatch -> Synthesis -> Persistence."""

    def __init__(
        self,
        retriever: Retriever,
        vector_store: VectorStore,
        memory_manager: ConversationMemoryManager | None = None,
        generator: Any = None,
    ) -> None:
        self.retriever = retriever
        self.vector_store = vector_store
        self.memory_manager = memory_manager or ConversationMemoryManager()
        self.generator = generator

    def run(
        self,
        query: str,
        session_id: str | None = None,
        user: Any = None,
        mode: str = "hybrid_rerank",
        provider: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Execute the agentic state machine pipeline end-to-end."""
        t_start = time.perf_counter()
        state: RAGState = {
            "query": query.strip(),
            "session_id": session_id,
            "user_id": getattr(user, "id", None) if user else None,
            "org_id": getattr(user, "org_id", "default_org") if user else "default_org",
            "mode": mode,
            "provider": provider,
            "api_key": api_key,
            "conversation_context": "",
            "history_messages": [],
            "sub_queries": [],
            "retrieved_chunks": [],
            "context_str": "",
            "top_score": None,
            "crag_decision": {},
            "multi_query_meta": {},
            "tool_calls": [],
            "tool_results": [],
            "reasoning_trace": [],
            "final_answer": "",
            "citations": [],
            "abstained": False,
            "abstention_reason": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "duration_ms": 0.0,
            "provider_status": "live",
            "provider_fallback": False,
        }

        # 1. State: Memory Load
        self._step_load_memory(state)

        # 2. State: Multi-Query Expansion, Hybrid Retrieval & CRAG Grading
        self._step_retrieval_and_crag(state, user)

        # 3. State: Curated Tool Dispatching
        self._step_tool_dispatch(state, user)

        # 4. State: Grounded Synthesis
        self._step_grounded_synthesis(state)

        # 5. State: Memory Persistence
        state["duration_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
        self._step_persist_memory(state)

        return state

    def _step_load_memory(self, state: RAGState) -> None:
        """Step 1: Load prior conversational memory turns and rolling summary."""
        session_id = state.get("session_id")
        if not session_id or not self.memory_manager:
            return

        ctx_str, window = self.memory_manager.get_context_window(session_id, k=3)
        state["conversation_context"] = ctx_str
        state["history_messages"] = window
        if ctx_str:
            state["reasoning_trace"].append(
                {
                    "step": "memory_load",
                    "title": "Loaded Conversational Memory",
                    "detail": f"Active sliding window contains {len(window)} turns; contextual summary attached.",
                }
            )

    def _step_retrieval_and_crag(self, state: RAGState, user: Any = None) -> None:
        """Step 2: Execute Multi-Query Expansion, Hybrid Retrieval, and CRAG Self-Reflection."""
        query = state["query"]
        mode = state.get("mode", "hybrid_rerank")

        outcome = self.retriever.retrieve(
            query=query,
            mode=mode,
            user=user,
            enable_multi_query=True,
            enable_crag=True,
        )

        state["retrieved_chunks"] = outcome.get("results", [])
        state["context_str"] = outcome.get("context", "")
        state["top_score"] = outcome.get("top_score")
        state["abstained"] = outcome.get("abstained", False)
        state["abstention_reason"] = outcome.get("abstention_reason")
        state["crag_decision"] = outcome.get("crag", {})
        state["multi_query_meta"] = outcome.get("multi_query", {})

        mq_meta = state["multi_query_meta"]
        if mq_meta.get("decomposed"):
            sub_q = mq_meta.get("sub_queries", [])
            state["sub_queries"] = sub_q
            state["reasoning_trace"].append(
                {
                    "step": "multi_query",
                    "title": f"Decomposed into {len(sub_q)} Perspectives",
                    "detail": "; ".join(sub_q),
                }
            )

        top_score_str = f"{state['top_score']:.3f}" if state["top_score"] is not None else "N/A"
        state["reasoning_trace"].append(
            {
                "step": "retrieval",
                "title": f"Retrieved {len(state['retrieved_chunks'])} Chunks (Score: {top_score_str})",
                "detail": f"Retrieved using mode '{mode}' across knowledge base.",
            }
        )

        crag = state["crag_decision"]
        grade = crag.get("grade", "high").capitalize()
        action = crag.get("action", "proceed")
        reformulated = crag.get("reformulated_query")
        crag_detail = f"CRAG Grade: {grade} (Action: {action})"
        if reformulated:
            crag_detail += f" -> Reformulated: '{reformulated}'"
        state["reasoning_trace"].append(
            {
                "step": "crag_reflection",
                "title": f"CRAG Self-Reflection ({grade})",
                "detail": crag_detail,
            }
        )

    def _step_tool_dispatch(self, state: RAGState, user: Any = None) -> None:
        """Step 3: Detect tool calling intent and execute curated tools."""
        query = state["query"]
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        # Intent 1: Arithmetic Evaluation
        calc_match = re.search(
            r"(?:calculate|compute|what is|how much is)\s+([\d\.\s\+\-\*\/\(\)\%]+)",
            query,
            re.IGNORECASE,
        )
        if calc_match and any(op in calc_match.group(1) for op in ("+", "-", "*", "/")):
            expr = calc_match.group(1).strip().rstrip("?.")
            call_info = {"tool": "deterministic_calculator", "arguments": {"expression": expr}}
            tool_calls.append(call_info)
            res = execute_tool("deterministic_calculator", {"expression": expr})
            tool_results.append(res)
            state["reasoning_trace"].append(
                {
                    "step": "tool_dispatch",
                    "title": "Deterministic Calculator Tool",
                    "detail": f"Evaluated '{expr}' = {res.get('result')}",
                }
            )

        # Intent 2: Document Summary
        summary_match = re.search(
            r"(?:summarize|summary of|overview of)\s+([a-zA-Z0-9_\-\.]+\.pdf)",
            query,
            re.IGNORECASE,
        )
        if summary_match:
            doc_target = summary_match.group(1).strip()
            call_info = {"tool": "document_summary", "arguments": {"filename": doc_target}}
            tool_calls.append(call_info)
            res = execute_tool("document_summary", {"filename": doc_target}, vector_store=self.vector_store)
            tool_results.append(res)
            state["reasoning_trace"].append(
                {
                    "step": "tool_dispatch",
                    "title": "Document Summary Tool",
                    "detail": f"Extracted cover & abstract for '{doc_target}'",
                }
            )

        state["tool_calls"] = tool_calls
        state["tool_results"] = tool_results

    def _step_grounded_synthesis(self, state: RAGState) -> None:
        """Step 4: Generate grounded answer using context, conversation history, and tool outputs."""
        if state["abstained"]:
            reason = state.get("abstention_reason")
            if reason in {"RBAC_EMPTY_RETRIEVAL", "EMPTY_RETRIEVAL"}:
                state["final_answer"] = (
                    "I cannot answer this based on the provided documents. "
                    "No matching context found in the vector database."
                )
            else:
                state["final_answer"] = "I cannot answer this based on the provided documents."
            state["citations"] = []
            return

        # Build citations list
        citations: list[dict[str, Any]] = []
        for hit in state["retrieved_chunks"]:
            meta = hit.get("metadata", {})
            page_start = meta.get("page_number", "?")
            page_end = meta.get("page_end", page_start)
            raw_snippet = hit.get("text") or ""
            clean_snippet = raw_snippet.replace("\ufffd", " ").strip()[:240]
            citations.append(
                {
                    "pdf": meta.get("source_pdf", "Unknown"),
                    "page": page_start,
                    "page_end": page_end,
                    "parent_id": hit.get("parent_id") or meta.get("parent_id"),
                    "score": round(float(hit.get("score", 0.0)), 3),
                    "snippet": clean_snippet,
                }
            )
        state["citations"] = citations

        # Assemble full context including prior turns and tool outputs
        context_parts: list[str] = []
        if state["conversation_context"]:
            context_parts.append(f"--- Conversation History ---\n{state['conversation_context']}")

        for res in state["tool_results"]:
            if res.get("status") == "success" and "result" in res:
                context_parts.append(f"--- Tool Output ({res.get('tool')}) ---\nCalculation Result: {res['result']}")
            elif res.get("status") == "success" and "summary_excerpt" in res:
                context_parts.append(f"--- Tool Output ({res.get('tool')}: {res.get('filename')}) ---\n{res['summary_excerpt']}")

        if state["context_str"]:
            context_parts.append(f"--- Retrieved Knowledge Base Documents ---\n{state['context_str']}")

        composite_context = "\n\n".join(context_parts)
        query = state["query"]

        # Call Generator
        provider = state.get("provider") or settings.llm_provider
        api_key = state.get("api_key")
        try:
            generator = self.generator or get_generator(provider, api_key=api_key)
            if hasattr(generator, "generate_with_telemetry"):
                try:
                    raw_res = generator.generate_with_telemetry(query, composite_context)
                    if isinstance(raw_res, tuple) and len(raw_res) == 3 and isinstance(raw_res[0], str):
                        answer, prompt_tokens, completion_tokens = raw_res
                    else:
                        answer = generator.generate(query, composite_context)
                        prompt_tokens = count_tokens(composite_context + " " + query)
                        completion_tokens = count_tokens(answer)
                except Exception:
                    answer = generator.generate(query, composite_context)
                    prompt_tokens = count_tokens(composite_context + " " + query)
                    completion_tokens = count_tokens(answer)
            else:
                answer = generator.generate(query, composite_context)
                prompt_tokens = count_tokens(composite_context + " " + query)
                completion_tokens = count_tokens(answer)

            # Check for offline failure strings
            if (
                "Error: LM Studio is not running" in answer
                or "Error connecting to" in answer
                or "Local LLM timed out" in answer
            ):
                has_gemini = bool(api_key or settings.gemini_api_key)
                if has_gemini and provider != "gemini":
                    gem_gen = get_generator("gemini", api_key=api_key)
                    answer = gem_gen.generate(query, composite_context)
                    state["provider_status"] = "live"
                    state["provider_fallback"] = True
                else:
                    answer = synthesize_extractive_answer(query, composite_context)
                    state["provider_status"] = "degraded_extractive"
                    state["provider_fallback"] = True
        except Exception as exc:
            logger.warning("Generator invocation exception ({}): falling back to extractive synthesis.", exc)
            answer = synthesize_extractive_answer(query, composite_context)
            state["provider_status"] = "degraded_extractive"
            state["provider_fallback"] = True
            prompt_tokens = count_tokens(composite_context + " " + query)
            completion_tokens = count_tokens(answer)

        state["final_answer"] = answer
        state["prompt_tokens"] = prompt_tokens
        state["completion_tokens"] = completion_tokens
        state["reasoning_trace"].append(
            {
                "step": "synthesis",
                "title": "Grounded Synthesis",
                "detail": f"Generated grounded response citing {len(citations)} document blocks ({state['provider_status']}).",
            }
        )

    def _step_persist_memory(self, state: RAGState) -> None:
        """Step 5: Persist turn in relational database if session_id is active."""
        session_id = state.get("session_id")
        if not session_id or not self.memory_manager:
            return

        try:
            self.memory_manager.append_turn(
                session_id=session_id,
                user_text=state["query"],
                assistant_text=state["final_answer"],
                citations=state["citations"],
                tokens=state["prompt_tokens"] + state["completion_tokens"],
                latency_ms=state["duration_ms"],
            )
            state["reasoning_trace"].append(
                {
                    "step": "memory_persist",
                    "title": "Conversational Memory Persisted",
                    "detail": f"Saved conversational turn to session '{session_id}'.",
                }
            )
        except Exception as exc:
            logger.warning("Failed to persist turn to session '{}': {}", session_id, exc)
