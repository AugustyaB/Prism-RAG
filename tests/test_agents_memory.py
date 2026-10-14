"""Unit tests for Phase 2.10 Wave 1: Conversational Session Memory & Sliding-Window Manager."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.memory import ConversationMemoryManager
from src.database.models import Base, ChatMessage, ChatSession


@pytest.fixture
def memory_manager():
    """Create an isolated SQLite database and return a ConversationMemoryManager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        mgr = ConversationMemoryManager(session_factory=Session)
        yield mgr
        engine.dispose()


def test_session_creation_and_retrieval(memory_manager: ConversationMemoryManager):
    """Verify session creation, title assignment, and retrieval."""
    sess = memory_manager.create_session(
        user_id="user_test_123",
        org_id="org_test_456",
        title="Project Discussion",
    )
    assert sess["id"] is not None
    assert sess["user_id"] == "user_test_123"
    assert sess["org_id"] == "org_test_456"
    assert sess["title"] == "Project Discussion"
    assert sess["summary"] is None

    fetched = memory_manager.get_session(sess["id"])
    assert fetched is not None
    assert fetched["id"] == sess["id"]
    assert fetched["title"] == "Project Discussion"


def test_sliding_window_context_k3(memory_manager: ConversationMemoryManager):
    """Verify sliding context window preserves exactly last k turns (2k messages)."""
    sess = memory_manager.create_session(user_id="u1", org_id="o1")
    sess_id = sess["id"]

    turns = [
        ("What is step 1?", "Step 1 is ingestion."),
        ("What is step 2?", "Step 2 is chunking."),
        ("What is step 3?", "Step 3 is embedding."),
    ]
    for q, a in turns:
        memory_manager.append_turn(sess_id, q, a)

    # 3 turns = 6 messages
    context_str, window = memory_manager.get_context_window(sess_id, k=3)
    assert len(window) == 6
    assert "User: What is step 1?" in context_str
    assert "Assistant: Step 3 is embedding." in context_str
    assert "[Previous Conversation Summary:" not in context_str


def test_background_summarization_threshold(memory_manager: ConversationMemoryManager):
    """Verify that exceeding threshold triggers background summarization and prepends to context."""
    sess = memory_manager.create_session(user_id="u1", org_id="o1")
    sess_id = sess["id"]

    # Append 4 turns (8 messages > 6 threshold)
    turns = [
        ("Query A: How to index PDFs?", "PDFs are converted to text and OCR."),
        ("Query B: What chunk size?", "Chunk size is 512 tokens."),
        ("Query C: What reranker?", "Cross-encoder bge-reranker-v2-m3."),
        ("Query D: What vector store?", "Qdrant with hybrid dense-sparse indexing."),
    ]
    for q, a in turns:
        memory_manager.append_turn(sess_id, q, a)

    # Check that summary was generated
    session_data = memory_manager.get_session(sess_id)
    assert session_data["summary"] is not None
    assert len(session_data["summary"]) > 0

    # Verify context window now prepends the summary
    context_str, window = memory_manager.get_context_window(sess_id, k=3)
    assert len(window) == 6  # Last 3 turns = 6 messages
    assert "[Previous Conversation Summary:" in context_str
    assert "Query D: What vector store?" in context_str


def test_auto_title_generation(memory_manager: ConversationMemoryManager):
    """Verify that default title 'New Conversation' is replaced by user query snippet."""
    sess = memory_manager.create_session(user_id="u1", org_id="o1", title="New Conversation")
    sess_id = sess["id"]

    memory_manager.append_turn(sess_id, "Explain the neural architecture of the cross-encoder", "It scores pairs.")
    updated = memory_manager.get_session(sess_id)
    assert updated["title"] != "New Conversation"
    assert "Explain the neural architecture" in updated["title"]


def test_session_deletion_cascades_messages(memory_manager: ConversationMemoryManager):
    """Verify deleting a session removes session and all its messages."""
    sess = memory_manager.create_session(user_id="u1", org_id="o1")
    sess_id = sess["id"]

    memory_manager.append_turn(sess_id, "Hello", "Hi there!")
    messages = memory_manager.get_messages(sess_id)
    assert len(messages) == 2

    deleted = memory_manager.delete_session(sess_id)
    assert deleted is True
    assert memory_manager.get_session(sess_id) is None
    assert len(memory_manager.get_messages(sess_id)) == 0
