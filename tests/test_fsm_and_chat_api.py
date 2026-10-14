"""Unit and API tests for Phase 2.10 Wave 3: Pure-Python FSM, Tool Dispatcher & Chat REST APIs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.memory import ConversationMemoryManager
from src.agents.state_machine import AgenticRAGPipeline
from src.agents.tools import (
    TOOL_REGISTRY,
    deterministic_calculator,
    execute_tool,
    get_tool_schemas,
)
from src.api.server import app, ensure_services_initialized
from src.database.models import Base, Organization, User


# ── 1. Tool Dispatcher & Safe AST Calculator Tests ─────────────────────────


def test_deterministic_calculator_arithmetic():
    """Verify safe arithmetic evaluation without eval/exec."""
    res1 = deterministic_calculator("15 * 4 + 10")
    assert res1["status"] == "success"
    assert res1["result"] == 70

    res2 = deterministic_calculator("(100 - 25) / 5")
    assert res2["status"] == "success"
    assert res2["result"] == 15.0

    res3 = deterministic_calculator("abs(-50) + round(4.7)")
    assert res3["status"] == "success"
    assert res3["result"] == 55


def test_deterministic_calculator_security_rejections():
    """Verify injection attempts and unauthorized AST syntax are rejected."""
    # Attempt import
    bad1 = deterministic_calculator("__import__('os').system('dir')")
    assert bad1["status"] == "error"

    # Attempt attribute access
    bad2 = deterministic_calculator("(1).__class__")
    assert bad2["status"] == "error"

    # Attempt arbitrary function call
    bad3 = deterministic_calculator("eval('2+2')")
    assert bad3["status"] == "error"


def test_get_tool_schemas_validity():
    """Verify tool schemas follow OpenAI/Gemini function specification."""
    schemas = get_tool_schemas()
    assert len(schemas) >= 4
    names = [s["function"]["name"] for s in schemas]
    assert "deterministic_calculator" in names
    assert "document_search" in names
    assert "document_summary" in names
    assert "document_comparison" in names


# ── 2. AgenticRAGPipeline FSM Coordinator Tests ────────────────────────────


def test_agentic_rag_pipeline_fsm_execution():
    """Verify AgenticRAGPipeline state machine coordinates memory, retrieval, CRAG, tools, and synthesis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "fsm_test.db"
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        memory_mgr = ConversationMemoryManager(session_factory=Session)

        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = {
            "context": "--- Document: test.pdf (Page: 1) ---\nKey policy points.",
            "results": [{"text": "Key policy points.", "score": 0.42, "metadata": {"source_pdf": "test.pdf", "page_number": 1}}],
            "abstained": False,
            "abstention_reason": None,
            "top_score": 0.42,
            "crag": {"grade": "high", "action": "proceed"},
            "multi_query": {"decomposed": True, "sub_queries": ["sub1", "sub2", "sub3"]},
        }

        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Policy mandates 90 day keys."

        pipeline = AgenticRAGPipeline(
            retriever=mock_retriever,
            vector_store=mock_vs,
            memory_manager=memory_mgr,
            generator=mock_gen,
        )

        sess = memory_mgr.create_session(user_id="u1", org_id="o1", title="Initial Session")
        sess_id = sess["id"]

        state = pipeline.run(
            query="What is the policy rotation schedule? Also calculate 12 * 8",
            session_id=sess_id,
            mode="hybrid_rerank",
        )

        assert state["abstained"] is False
        assert state["final_answer"] == "Policy mandates 90 day keys."
        assert len(state["citations"]) == 1
        assert len(state["tool_calls"]) == 1
        assert state["tool_calls"][0]["tool"] == "deterministic_calculator"
        assert len(state["reasoning_trace"]) >= 4

        # Verify turn was persisted to conversational memory
        messages = memory_mgr.get_messages(sess_id)
        assert len(messages) == 2  # 1 user + 1 assistant
        assert messages[0]["content"].startswith("What is the policy")
        assert messages[1]["content"] == "Policy mandates 90 day keys."

        engine.dispose()


# ── 3. Chat Session REST API Endpoints ─────────────────────────────────────


@pytest.fixture
def api_client():
    """Create test client with initialized database services."""
    ensure_services_initialized()
    return TestClient(app)


def _login(client: TestClient, email: str = "admin@default.org", password: str = "adminpassword123") -> str:
    """Helper to authenticate and retrieve access token."""
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


def test_chat_session_lifecycle_and_query_persistence(api_client: TestClient):
    """Verify POST /api/chat/sessions, query with session_id, and GET /api/chat/sessions/{id}/messages."""
    token = _login(api_client)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create new chat session
    create_resp = api_client.post(
        "/api/chat/sessions",
        headers=headers,
        json={"title": "Deployment Architecture Discussion"},
    )
    assert create_resp.status_code == 200
    session_data = create_resp.json()
    assert session_data["title"] == "Deployment Architecture Discussion"
    session_id = session_data["id"]

    # 2. List sessions
    list_resp = api_client.get("/api/chat/sessions", headers=headers)
    assert list_resp.status_code == 200
    sessions = list_resp.json()["sessions"]
    assert any(s["id"] == session_id for s in sessions)

    # 3. Post query with session_id
    mock_gen = MagicMock()
    mock_gen.generate.return_value = "Kubernetes pods require resource requests and limits."
    mock_retrieval_outcome = {
        "results": [
            {
                "text": "Kubernetes pods require resource requests and limits.",
                "score": 0.92,
                "metadata": {"source_pdf": "k8s_guide.pdf", "page_number": 1, "parent_id": "p1"},
            }
        ],
        "context": "Kubernetes pods require resource requests and limits.",
        "abstained": False,
        "abstention_reason": None,
        "top_score": 0.92,
    }
    with patch("src.api.server.get_generator", return_value=mock_gen), \
         patch("src.api.server.retriever.retrieve", return_value=mock_retrieval_outcome):
        query_resp = api_client.post(
            "/api/query",
            headers=headers,
            json={
                "query": "How are Kubernetes pods configured? Also calculate 50 * 20",
                "session_id": session_id,
                "mode": "hybrid_rerank",
            },
        )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert q_data["session_id"] == session_id
    assert "reasoning_trace" in q_data
    assert "tool_calls" in q_data
    assert any(tc["tool"] == "deterministic_calculator" for tc in q_data["tool_calls"])

    # 4. Get session messages
    msg_resp = api_client.get(f"/api/chat/sessions/{session_id}/messages", headers=headers)
    assert msg_resp.status_code == 200
    msg_data = msg_resp.json()
    assert len(msg_data["messages"]) >= 2
    assert msg_data["session"]["id"] == session_id

    # 5. Delete session
    del_resp = api_client.delete(f"/api/chat/sessions/{session_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 6. Verify 404 after deletion
    get_after_del = api_client.get(f"/api/chat/sessions/{session_id}/messages", headers=headers)
    assert get_after_del.status_code == 404
