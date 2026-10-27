"""Phase 2.10 QA Automated Test Suite — Native Agentic Intelligence & Tool Dispatcher (Sec 1–6; L1–L5B).

Empirically validates:
1. Multi-Turn Conversational Session Memory:
   - Multi-turn retention across 5+ turns (6 turns / 12 messages).
   - Sliding-window context retrieval (k=3 window returning latest 3 turns).
   - Session listing scoped by user_id and tenant org_id.
   - Clean cascading deletion of session and all message turns.
2. Corrective RAG (CRAG) Grader & Single-Retry Loop Bound:
   - 3-tier confidence grading: High (>= 0.15 -> proceed), Marginal (0.05..0.15 -> reformulate), Low (< 0.05 -> abstain).
   - Strict loop bound guard: when loop_count >= 1, reformulation is refused (loop_count <= 1).
   - Entity-based heuristic and LLM-based query reformulation.
3. Curated Tool Registry & Safe AST Calculator:
   - Deterministic calculator evaluating arithmetic expressions via pure Python AST (zero eval/exec).
   - Security rejection of imports, arbitrary attribute access, function injection, and non-arithmetic syntax.
   - Document summary extraction from relational metadata and vector store.
   - Document comparison dual-retrieval comparative matrix.
   - Dynamic tool execution through execute_tool registry dispatcher.
4. Multi-Query Decomposition & RRF Fusion:
   - Query qualification heuristics (length > 25 chars or multi-part indicator words).
   - 3-perspective decomposition (problem overview, technical methodology, experimental results).
   - Reciprocal Rank Fusion (k=60) candidate score aggregation.
   - Cross-page parent deduplication preserving highest-scoring chunk per parent_id.
5. Pure-Python FSM State Machine:
   - AgenticRAGPipeline end-to-end coordination: Load Memory -> Decompose -> Retrieve -> Grade -> Tool -> Synthesize -> Persist.
   - Structured reasoning trace compilation.
6. Chat REST APIs & Tenant RBAC Isolation:
   - POST /api/chat/sessions, GET /api/chat/sessions, GET /api/chat/sessions/{id}/messages, DELETE /api/chat/sessions/{id}.
   - Multi-turn /api/query integration passing session_id and attaching conversation context.
   - Cross-tenant session access and deletion forbidden with 403 Forbidden.
7. Zero Live Database Pollution:
   - Confirms live data/metadata.db remains completely pristine and unpolluted.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.agents.memory import ConversationMemoryManager
from src.agents.state_machine import AgenticRAGPipeline, RAGState
from src.agents.tools import (
    TOOL_REGISTRY,
    _safe_eval_node,
    deterministic_calculator,
    document_comparison,
    document_search,
    document_summary,
    execute_tool,
    get_tool_schemas,
)
import src.agents.tools as tools_mod
import src.agents.memory as mem_mod
from src.api.server import app, ensure_services_initialized
import src.api.server as srv
from src.auth.security import create_access_token, seed_demo_accounts
import src.auth.security as sec
from src.config import settings
from src.database.models import Base, ChatMessage, ChatSession, DocumentRecord, Organization, User
import src.database.relational as rel
from src.database.relational import get_db
from src.database.vector_store import RRF_K
from src.retrieval.crag import CRAGGrader
from src.retrieval.multi_query import MultiQueryDecomposer
from src.retrieval.retriever import Retriever


# ── Fixtures & Isolation Setup ──────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def setup_phase210_test_environment(tmp_path_factory):
    """Ensure all test databases and demo accounts run inside an isolated temporary SQLite database."""
    test_db_dir = tmp_path_factory.mktemp("agentic_test_db")
    test_db_path = test_db_dir / "test_metadata.db"
    test_db_url = f"sqlite:///{test_db_path}"

    orig_url = settings.database_url
    orig_upload_dir = settings.pdf_upload_dir
    orig_engine = rel.engine
    orig_session = rel.SessionLocal
    orig_srv_memory_mgr = srv.memory_manager
    orig_srv_agentic_pipeline = srv.agentic_pipeline

    test_upload_dir = test_db_dir / "uploads"
    test_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.database_url = test_db_url
    settings.pdf_upload_dir = test_upload_dir

    test_engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    test_session_maker = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )

    rel.engine = test_engine
    rel.SessionLocal = test_session_maker
    srv.SessionLocal = test_session_maker
    sec.SessionLocal = test_session_maker
    tools_mod.SessionLocal = test_session_maker
    mem_mod.SessionLocal = test_session_maker

    rel.init_db()
    sec.seed_demo_accounts()

    # Create isolated memory manager and pipeline on server
    test_mem_mgr = ConversationMemoryManager(session_factory=test_session_maker)
    srv.memory_manager = test_mem_mgr
    ensure_services_initialized()
    srv.memory_manager = test_mem_mgr
    srv.agentic_pipeline = AgenticRAGPipeline(
        retriever=srv.retriever,
        vector_store=srv.vector_store,
        memory_manager=test_mem_mgr,
    )

    def override_get_db():
        db = test_session_maker()
        try:
            yield db
        finally:
            db.close()

    srv.app.dependency_overrides[get_db] = override_get_db

    yield {
        "engine": test_engine,
        "session_maker": test_session_maker,
        "memory_manager": test_mem_mgr,
    }

    srv.app.dependency_overrides.pop(get_db, None)
    settings.database_url = orig_url
    settings.pdf_upload_dir = orig_upload_dir
    rel.engine = orig_engine
    rel.SessionLocal = orig_session
    srv.SessionLocal = orig_session
    sec.SessionLocal = orig_session
    tools_mod.SessionLocal = orig_session
    mem_mod.SessionLocal = orig_session
    srv.memory_manager = orig_srv_memory_mgr
    srv.agentic_pipeline = orig_srv_agentic_pipeline


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def login_user(client: TestClient, email: str = "admin@default.org", password: str = "adminpassword123") -> dict[str, Any]:
    """Helper to authenticate and return auth payload with JWT token."""
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"Login failed for {email}: {resp.text}"
    return resp.json()


# ── 1. Multi-Turn Conversational Session Memory Tests ───────────────────────


def test_multi_turn_conversational_memory_retention_over_5_turns(setup_phase210_test_environment):
    """Verify conversational session memory records 5+ turns (6 turns) and preserves sliding window."""
    mgr: ConversationMemoryManager = setup_phase210_test_environment["memory_manager"]

    # Create session
    session = mgr.create_session(
        user_id="user_qa_tester",
        org_id="org_qa_corp",
        title="Architecture Discussion",
    )
    session_id = session["id"]
    assert session_id is not None
    assert session["title"] == "Architecture Discussion"

    # Add 6 conversational turns (12 messages)
    turns = [
        ("What is Reciprocal Rank Fusion?", "RRF is an algorithm combining rankings by summing 1/(k+rank)."),
        ("What constant k is standard?", "A smoothing constant of k=60 is standard in information retrieval."),
        ("Does our system support parent-child chunking?", "Yes, cross-page parent-child chunking stores small children with parent text."),
        ("How does CRAG improve retrieval?", "CRAG adds a self-reflection grading step and 1-shot reformulation."),
        ("What LLMs are supported?", "LM Studio, Ollama, and Google Gemini are supported providers."),
        ("Can we evaluate math expressions?", "Yes, the deterministic calculator evaluates math via pure AST."),
    ]

    for q, a in turns:
        mgr.append_turn(
            session_id=session_id,
            user_text=q,
            assistant_text=a,
            citations=[{"pdf": "sample.pdf", "page": 1, "score": 0.95}],
            tokens=45,
            latency_ms=120.5,
        )

    # 1. Verify all 12 messages are retained in chronological order
    all_messages = mgr.get_messages(session_id, limit=50)
    assert len(all_messages) == 12
    assert all_messages[0]["role"] == "user"
    assert all_messages[0]["content"] == "What is Reciprocal Rank Fusion?"
    assert all_messages[1]["role"] == "assistant"
    assert all_messages[11]["content"] == "Yes, the deterministic calculator evaluates math via pure AST."

    # 2. Verify sliding window with k=3 returns exactly the latest 3 turns (6 messages)
    context_str, window = mgr.get_context_window(session_id, k=3)
    assert len(window) == 6
    assert window[0]["role"] == "user"
    assert window[0]["content"] == "How does CRAG improve retrieval?"
    assert window[-1]["role"] == "assistant"
    assert "deterministic calculator" in window[-1]["content"]

    # Verify context_str formats the turns cleanly
    assert "User: How does CRAG improve retrieval?" in context_str
    assert "User: What is Reciprocal Rank Fusion?" not in context_str  # Trimmed by k=3

    # 3. Verify session listing with user/org filtering
    user_sessions = mgr.list_sessions(user_id="user_qa_tester", org_id="org_qa_corp")
    assert any(s["id"] == session_id for s in user_sessions)

    # Cross-tenant list must return empty
    other_tenant_sessions = mgr.list_sessions(user_id="user_qa_tester", org_id="other_org")
    assert not any(s["id"] == session_id for s in other_tenant_sessions)

    # 4. Verify cascade deletion
    deleted = mgr.delete_session(session_id)
    assert deleted is True
    assert mgr.get_session(session_id) is None
    assert len(mgr.get_messages(session_id)) == 0


# ── 2. CRAG Grader & Strict Single-Retry Loop Bound Tests ───────────────────


def test_crag_grader_confidence_bands_and_single_retry_bound():
    """Verify CRAG 3-tier grading logic and absolute loop_count <= 1 bound."""
    grader = CRAGGrader()

    # Case 1: High confidence (score >= 0.15) -> Action: proceed
    res_high = grader.grade(top_score=0.45, results=[{"text": "Sample hit"}], loop_count=0)
    assert res_high["grade"] == "high"
    assert res_high["action"] == "proceed"
    assert res_high["reason"] in ("HIGH_CONFIDENCE", "HIGH_CONFIDENCE_DECISIVE")
    assert res_high["confidence"] == 0.45

    # Case 2: Low confidence (score < 0.05) -> Action: abstain
    res_low = grader.grade(top_score=0.03, results=[{"text": "Irrelevant"}], loop_count=0)
    assert res_low["grade"] == "low"
    assert res_low["action"] == "abstain"
    assert res_low["reason"] == "LOW_SCORE"

    # Case 3: Empty results or None top_score -> Action: abstain
    res_empty = grader.grade(top_score=None, results=[], loop_count=0)
    assert res_empty["grade"] == "low"
    assert res_empty["action"] == "abstain"
    assert res_empty["reason"] == "EMPTY_RETRIEVAL"

    # Case 4: Marginal confidence (0.05 <= score < 0.15) with loop_count == 0 -> Action: reformulate
    res_marginal_0 = grader.grade(top_score=0.10, results=[{"text": "Partial match"}], loop_count=0)
    assert res_marginal_0["grade"] == "marginal"
    assert res_marginal_0["action"] == "reformulate"
    assert res_marginal_0["reason"] == "MARGINAL_CONFIDENCE_RETRY"
    assert res_marginal_0["loop_count"] == 0

    # Case 5: Marginal confidence with loop_count == 1 (Retry already attempted) -> MUST NOT REFORMULATE
    res_marginal_1 = grader.grade(top_score=0.10, results=[{"text": "Partial match"}], loop_count=1)
    assert res_marginal_1["grade"] == "marginal"
    assert res_marginal_1["action"] in {"proceed", "abstain"}
    assert res_marginal_1["action"] != "reformulate"  # Strict loop termination
    assert res_marginal_1["loop_count"] == 1

    # Case 6: Marginal with loop_count == 2 -> Also strictly barred from reformulation
    res_marginal_2 = grader.grade(top_score=0.10, results=[{"text": "Partial match"}], loop_count=2)
    assert res_marginal_2["action"] != "reformulate"


def test_crag_query_reformulation_llm_and_heuristic():
    """Verify CRAG query reformulation behavior for LLM and rule-based fallback."""
    mock_gen = MagicMock()
    mock_gen.generate.return_value = "Transformer Self-Attention Matrix Dimensions"

    grader = CRAGGrader(generator=mock_gen)
    reformulated = grader.reformulate_query("can you please explain the transformer self attention?")
    assert reformulated == "Transformer Self-Attention Matrix Dimensions"

    # Test heuristic fallback when generator throws
    failing_gen = MagicMock()
    failing_gen.generate.side_effect = RuntimeError("Offline LLM unavailable")
    fallback_grader = CRAGGrader(generator=failing_gen)

    heuristic_result = fallback_grader.reformulate_query("can you please tell me about Qdrant vector indexing?")
    assert "qdrant" in heuristic_result.lower()
    assert "vector" in heuristic_result.lower()
    assert "can you please" not in heuristic_result.lower()


# ── 3. Curated Tool Registry & Safe AST Calculator Tests ─────────────────────


def test_deterministic_calculator_evaluates_complex_arithmetic():
    """Verify safe AST arithmetic evaluation supporting operators and safe math functions."""
    cases = [
        ("2 + 2", 4),
        ("100 - (25 * 3)", 25),
        ("(15 + 5) / 4", 5.0),
        ("2 ** 8", 256),
        ("17 % 5", 2),
        ("abs(-150) + round(4.6)", 155),
        ("min(10, 20) + max(5, 15)", 25),
        ("-10 + 20", 10),
    ]
    for expr, expected in cases:
        res = deterministic_calculator(expr)
        assert res["status"] == "success", f"Failed on '{expr}': {res}"
        assert res["result"] == expected


def test_deterministic_calculator_rejects_arbitrary_code_and_syntax():
    """Verify dangerous AST nodes, code execution, and arbitrary calls are strictly blocked."""
    injections = [
        "__import__('os').system('dir')",
        "eval('1+1')",
        "exec('a = 1')",
        "open('data/metadata.db', 'r')",
        "(lambda: 42)()",
        "().__class__.__bases__[0]",
        "'hello' + ' world'",
        "sum([1, 2, 3])",  # Disallowed function
        "a = 5",  # Statement, not expression
    ]
    for attack in injections:
        res = deterministic_calculator(attack)
        assert res["status"] == "error", f"Should have failed on '{attack}': {res}"
        assert "error" in res


def test_tool_registry_and_execute_tool_dispatch():
    """Verify execute_tool correctly routes to registered tools and rejects unknown tools."""
    # 1. Test deterministic_calculator through execute_tool
    calc_res = execute_tool("deterministic_calculator", {"expression": "50 * 2 + 5"})
    assert calc_res["status"] == "success"
    assert calc_res["result"] == 105

    # 2. Test document_summary with mock vector store
    mock_vs = MagicMock()
    mock_vs.search.return_value = [{"text": "Abstract: Deep learning approaches to RAG.", "parent_text": None}]
    summary_res = execute_tool("document_summary", {"filename": "attention_is_all_you_need.pdf"}, vector_store=mock_vs)
    assert summary_res["status"] == "success"
    assert summary_res["filename"] == "attention_is_all_you_need.pdf"

    # 3. Test document_comparison with mock vector store
    mock_vs.search.side_effect = [
        [{"text": "Doc A discussion of attention."}],
        [{"text": "Doc B discussion of recurrence."}],
    ]
    comp_res = execute_tool(
        "document_comparison",
        {"doc_a": "doc1.pdf", "doc_b": "doc2.pdf", "topic": "architecture"},
        vector_store=mock_vs,
    )
    assert comp_res["status"] == "success"
    assert len(comp_res["doc_a_findings"]) == 1
    assert len(comp_res["doc_b_findings"]) == 1

    # 4. Test unknown tool
    bad_res = execute_tool("unauthorized_shell_tool", {"cmd": "whoami"})
    assert bad_res["status"] == "error"
    assert "not registered" in bad_res["error"].lower()

    # 5. Verify tool schemas
    schemas = get_tool_schemas()
    assert len(schemas) >= 4
    tool_names = [s["function"]["name"] for s in schemas]
    assert "deterministic_calculator" in tool_names
    assert "document_summary" in tool_names
    assert "document_search" in tool_names
    assert "document_comparison" in tool_names


# ── 4. Multi-Query RRF Decomposition & Parent Deduplication Tests ────────────


def test_multi_query_decomposer_heuristics_and_decomposition():
    """Verify multi-query qualification heuristics and 3-perspective breakdown."""
    mock_vs = MagicMock()
    decomposer = MultiQueryDecomposer(vector_store=mock_vs)

    # Heuristic checks
    assert decomposer.should_decompose("") is False
    assert decomposer.should_decompose("short") is False
    assert decomposer.should_decompose("Compare BERT versus GPT") is True
    assert decomposer.should_decompose("Explain the end-to-end architecture of reciprocal rank fusion") is True

    # Decomposition generating 3 queries
    perspectives = decomposer.decompose("Compare transformer attention with recurrent neural networks")
    assert len(perspectives) == 3
    assert all(isinstance(p, str) and len(p) > 10 for p in perspectives)


def test_multi_query_search_and_merge_rrf_with_parent_deduplication():
    """Verify parallel retrieval across sub-queries and RRF merge with parent chunk deduplication."""
    mock_vs = MagicMock()

    # Setup simulated responses for 3 sub-queries
    # Note: doc1 and doc2 share parent_id="parent_alpha"
    # doc3 and doc4 share parent_id="parent_beta"
    mock_vs.search.side_effect = [
        [
            {"id": "doc1", "text": "Child 1 of Alpha", "parent_id": "parent_alpha", "file_hash": "hash1"},
            {"id": "doc3", "text": "Child 1 of Beta", "parent_id": "parent_beta", "file_hash": "hash1"},
        ],
        [
            {"id": "doc2", "text": "Child 2 of Alpha", "parent_id": "parent_alpha", "file_hash": "hash1"},
            {"id": "doc5", "text": "Child 1 of Gamma", "parent_id": "parent_gamma", "file_hash": "hash1"},
        ],
        [
            {"id": "doc1", "text": "Child 1 of Alpha", "parent_id": "parent_alpha", "file_hash": "hash1"},
            {"id": "doc4", "text": "Child 2 of Beta", "parent_id": "parent_beta", "file_hash": "hash1"},
        ],
    ]

    decomposer = MultiQueryDecomposer(vector_store=mock_vs)
    merged_results, meta = decomposer.search_and_merge(
        query="Compare transformer attention with recurrent neural networks",
        initial_k=10,
        final_k=4,
    )

    assert meta["decomposed"] is True
    assert len(meta["sub_queries"]) == 3

    # Parent deduplication check:
    # There should only be ONE chunk from parent_alpha, ONE from parent_beta, etc.
    parents_seen = [r.get("parent_id") for r in merged_results]
    assert len(parents_seen) == len(set(parents_seen)), f"Duplicate parents found in results: {parents_seen}"

    # RRF score ordering check:
    # doc1 appeared in query 1 (rank 0) and query 3 (rank 0).
    # Its score should be highest: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
    assert merged_results[0]["id"] == "doc1"
    assert merged_results[0]["parent_id"] == "parent_alpha"
    assert round(merged_results[0]["score"], 4) == round(2.0 / (RRF_K + 1), 4)


# ── 5. Pure-Python FSM State Machine Tests ─────────────────────────────────


def test_agentic_rag_pipeline_fsm_end_to_end_flow(setup_phase210_test_environment):
    """Verify AgenticRAGPipeline coordinates Memory -> Retrieval -> Tool -> Synthesis -> Persistence."""
    mgr: ConversationMemoryManager = setup_phase210_test_environment["memory_manager"]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "c1",
                "text": "The project uses Python 3.13 and FastAPI.",
                "score": 0.88,
                "metadata": {"source_pdf": "spec.pdf", "page_number": 1, "parent_id": "p1"},
            }
        ],
        "context": "The project uses Python 3.13 and FastAPI.",
        "top_score": 0.88,
        "abstained": False,
        "abstention_reason": None,
        "crag": {"grade": "high", "action": "proceed"},
        "multi_query": {"decomposed": False},
    }

    mock_gen = MagicMock()
    mock_gen.generate.return_value = "The project runs on Python 3.13 with a calculated value of 100."

    pipeline = AgenticRAGPipeline(
        retriever=mock_retriever,
        vector_store=MagicMock(),
        memory_manager=mgr,
        generator=mock_gen,
    )

    # Create session
    sess = mgr.create_session(user_id="user_fsm", org_id="org_fsm", title="Pipeline Test")
    sess_id = sess["id"]

    # Query with arithmetic intent to trigger deterministic_calculator
    query = "What is 25 * 4 and what Python version is used?"
    user_mock = MagicMock(id="user_fsm", org_id="org_fsm")

    state = pipeline.run(query=query, session_id=sess_id, user=user_mock)

    # 1. State assertions
    assert state["query"] == query
    assert state["session_id"] == sess_id
    assert state["abstained"] is False
    assert len(state["citations"]) == 1
    assert state["citations"][0]["pdf"] == "spec.pdf"

    # 2. Tool calls verification
    assert len(state["tool_calls"]) == 1
    assert state["tool_calls"][0]["tool"] == "deterministic_calculator"
    assert state["tool_results"][0]["result"] == 100

    # 3. Reasoning trace verification
    steps = [step["step"] for step in state["reasoning_trace"]]
    assert "retrieval" in steps
    assert "crag_reflection" in steps
    assert "tool_dispatch" in steps
    assert "synthesis" in steps
    assert "memory_persist" in steps

    # 4. Relational persistence verification
    messages = mgr.get_messages(sess_id)
    assert len(messages) == 2  # 1 turn (User + Assistant)
    assert messages[0]["content"] == query
    assert "Python 3.13" in messages[1]["content"]


# ── 6. Chat REST APIs & Multi-Tenant RBAC Isolation Tests ───────────────────


def test_chat_sessions_rest_api_lifecycle_and_tenant_isolation(client: TestClient):
    """Verify chat session REST endpoints, multi-turn /api/query integration, and tenant boundaries."""
    # 1. Log in as Default Admin (Org 1) and Tenant B Admin (Org 2)
    acme_auth = login_user(client, "admin@default.org", "adminpassword123")
    acme_token = acme_auth["access_token"]
    acme_headers = {"Authorization": f"Bearer {acme_token}"}

    globex_auth = login_user(client, "admin@tenant_b.org", "tenantbadmin123")
    globex_token = globex_auth["access_token"]
    globex_headers = {"Authorization": f"Bearer {globex_token}"}

    # 2. Create session as Default Admin
    create_resp = client.post(
        "/api/chat/sessions",
        headers=acme_headers,
        json={"title": "Acme Confidential Q&A"},
    )
    assert create_resp.status_code == 200, create_resp.text
    session_data = create_resp.json()
    session_id = session_data["id"]
    assert session_data["title"] == "Acme Confidential Q&A"

    # 3. List sessions for Acme Admin
    list_resp = client.get("/api/chat/sessions", headers=acme_headers)
    assert list_resp.status_code == 200
    sessions = list_resp.json()["sessions"]
    assert any(s["id"] == session_id for s in sessions)

    # 4. Globex Admin listing must NOT see Acme's session
    globex_list = client.get("/api/chat/sessions", headers=globex_headers)
    assert globex_list.status_code == 200
    globex_sessions = globex_list.json()["sessions"]
    assert not any(s["id"] == session_id for s in globex_sessions)

    # 5. Post query linked to session_id as Acme Admin
    query_resp = client.post(
        "/api/query",
        headers=acme_headers,
        json={
            "query": "What is 12 * 12?",
            "mode": "hybrid_rerank",
            "session_id": session_id,
        },
    )
    assert query_resp.status_code == 200, query_resp.text
    q_data = query_resp.json()
    assert q_data["session_id"] == session_id
    assert "reasoning_trace" in q_data
    assert "tool_calls" in q_data

    # Verify messages saved to session
    msg_resp = client.get(f"/api/chat/sessions/{session_id}/messages", headers=acme_headers)
    assert msg_resp.status_code == 200
    messages = msg_resp.json()["messages"]
    assert len(messages) >= 2
    assert messages[0]["content"] == "What is 12 * 12?"

    # 6. Tenant Security: Globex Admin must be rejected with 403 Forbidden
    cross_tenant_get = client.get(f"/api/chat/sessions/{session_id}/messages", headers=globex_headers)
    assert cross_tenant_get.status_code == 403

    cross_tenant_delete = client.delete(f"/api/chat/sessions/{session_id}", headers=globex_headers)
    assert cross_tenant_delete.status_code == 403

    # 7. Acme Admin deletes session
    delete_resp = client.delete(f"/api/chat/sessions/{session_id}", headers=acme_headers)
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    # Subsequent fetch must return 404
    missing_resp = client.get(f"/api/chat/sessions/{session_id}/messages", headers=acme_headers)
    assert missing_resp.status_code == 404


# ── 7. Zero Live Database Pollution Test ────────────────────────────────────


def test_live_metadata_db_remains_unpolluted():
    """Verify that live data/metadata.db was never written to or polluted by Phase 2.10 tests."""
    live_db_path = Path("data/metadata.db")
    if not live_db_path.exists():
        pytest.skip("No live database exists at data/metadata.db.")

    live_engine = create_engine(f"sqlite:///{live_db_path}", connect_args={"check_same_thread": False})
    LiveSession = sessionmaker(bind=live_engine, autoflush=False, autocommit=False)

    with LiveSession() as session:
        # 1. Verify 0 test users leaked into live users table
        test_users = session.scalars(
            select(User).where(User.email.like("%test%") | User.email.like("%qa%"))
        ).all()
        assert len(test_users) == 0, f"Pollution: Test users leaked into live metadata.db: {test_users}"

        # 2. Verify only canonical demo accounts exist
        all_users = session.scalars(select(User)).all()
        allowed_emails = {
            "admin@default.org",
            "manager@default.org",
            "analyst@default.org",
            "viewer@default.org",
            "admin@tenant_b.org",
            "viewer@tenant_b.org",
        }
        for u in all_users:
            assert u.email in allowed_emails, f"Pollution: Unexpected user in live DB: {u.email}"

        # 3. Verify test chat sessions did not leak
        test_sessions = session.scalars(
            select(ChatSession).where(
                ChatSession.title.in_(["Architecture Discussion", "Pipeline Test", "Acme Confidential Q&A"])
                | ChatSession.user_id.in_(["user_qa_tester", "user_fsm"])
            )
        ).all()
        assert len(test_sessions) == 0, f"Pollution: Test chat sessions leaked into live metadata.db: {test_sessions}"

        # 4. Verify test chat messages did not leak
        test_messages = session.scalars(
            select(ChatMessage).where(
                ChatMessage.content.like("%Reciprocal Rank Fusion%")
                | ChatMessage.content.like("%12 * 12%")
                | ChatMessage.content.like("%25 * 4%")
            )
        ).all()
        assert len(test_messages) == 0, f"Pollution: Test chat messages leaked into live metadata.db: {test_messages}"

    live_engine.dispose()
