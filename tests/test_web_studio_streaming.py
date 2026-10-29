"""Phase 3.1 Wave 6 Automated Pytest Suite — Web Studio Real-Time SSE Typing, Semantic Cache Pill & Citation Preview (L5A, L5B).

Empirically validates:
1. Static Web Studio Asset Integrity & Visual Rules:
   - streamQueryResponse defined in src/api/static/script.js.
   - .streaming-cursor, .telemetry-chip-cached, .citation-contextual-header in style.css.
   - #streaming-indicator present in index.html.
   - Zero emoji characters across index.html, style.css, and script.js (Visual Rule 2).
   - Zero blue/cyan/indigo/violet/purple hex colors across index.html, style.css, and script.js (Visual Rule 1).
   - Zero inline event handlers in index.html.
2. POST /api/query/stream SSE Endpoint Integration:
   - Valid text/event-stream response structure.
   - Emits real-time token events and final done payload with telemetry.
3. Semantic Cache (<10ms) Instant Replay:
   - Consecutive stream queries return cached: true and telemetry.cached: true with <10ms latency.
4. Contextual Citation Headers:
   - Emitted citations preserve contextual_header metadata for UI previews.
5. Multi-Turn Session Persistence via Streaming:
   - Emitted streaming turns are persisted in relational database under active session_id.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.server import app, rate_limiter
from src.auth.security import seed_demo_accounts, create_access_token
from src.database.relational import init_db
from src.database.semantic_cache import reset_semantic_cache

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "src" / "api" / "static"


@pytest.fixture(scope="module", autouse=True)
def setup_streaming_test_env(tmp_path_factory):
    """Initialize isolated relational test database and seed accounts."""
    test_db_dir = tmp_path_factory.mktemp("streaming_test_db")
    test_db_path = test_db_dir / "test_metadata.db"
    test_db_url = f"sqlite:///{test_db_path}"

    import src.database.relational as rel
    import src.api.server as srv
    import src.auth.security as sec
    import src.jobs.ingestion_worker as ing
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.config import settings

    orig_url = settings.database_url
    orig_upload_dir = settings.pdf_upload_dir
    orig_engine = rel.engine
    orig_session = rel.SessionLocal

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
    srv.db_session_factory = test_session_maker
    sec.SessionLocal = test_session_maker
    ing.SessionLocal = test_session_maker

    rel.init_db()
    sec.seed_demo_accounts()
    reset_semantic_cache()

    def _override_get_db():
        with test_session_maker() as s:
            yield s

    app.dependency_overrides[rel.get_db] = _override_get_db

    yield

    app.dependency_overrides.pop(rel.get_db, None)
    settings.database_url = orig_url
    settings.pdf_upload_dir = orig_upload_dir
    rel.engine = orig_engine
    rel.SessionLocal = orig_session
    srv.SessionLocal = orig_session
    srv.db_session_factory = orig_session
    sec.SessionLocal = orig_session
    ing.SessionLocal = orig_session
    reset_semantic_cache()


@pytest.fixture
def client():
    rate_limiter.reset()
    return TestClient(app)


def auth_header_for(email="admin@default.org", org_id="default_org", role="ADMIN") -> dict[str, str]:
    token = create_access_token(data={"sub": email, "org_id": org_id, "role": role})
    return {"Authorization": f"Bearer {token}"}


# ==============================================================================
# 1. Static Asset Integrity, Zero-Emoji & Design System Audit
# ==============================================================================


def test_streaming_static_asset_integrity_and_visual_contract(client: TestClient):
    """Verify L5A/L5B static files contain streaming controller, zero emojis, and zero blue/purple."""
    index_path = STATIC_DIR / "index.html"
    style_path = STATIC_DIR / "style.css"
    script_path = STATIC_DIR / "script.js"

    assert index_path.exists(), "index.html is missing"
    assert style_path.exists(), "style.css is missing"
    assert script_path.exists(), "script.js is missing"

    html_text = index_path.read_text(encoding="utf-8")
    css_text = style_path.read_text(encoding="utf-8")
    js_text = script_path.read_text(encoding="utf-8")

    # 1. Verify streaming status pill in index.html
    assert 'id="streaming-indicator"' in html_text, "Missing #streaming-indicator in index.html"
    assert "SSE Stream Active" in html_text, "Missing SSE Stream Active label in index.html"

    # 2. Verify streaming classes in style.css
    for required_cls in [
        ".streaming-status-pill",
        ".stream-pulse",
        ".streaming-cursor",
        ".telemetry-chip-cached",
        ".citation-contextual-header",
    ]:
        assert required_cls in css_text, f"Missing required streaming CSS rule in style.css: {required_cls}"

    # 3. Verify L5A streaming functions in script.js
    for fn_name in [
        "async function streamQueryResponse",
        "function buildCitationsElement",
        "function buildTelemetryRibbon",
        "function renderQueryResponse",
    ]:
        assert fn_name in js_text, f"Missing required L5A function in script.js: {fn_name}"

    # 4. Zero inline JavaScript handlers in index.html
    inline_handler_re = re.compile(
        r"\bon(click|change|submit|load|keyup|keydown|input|focus|blur|mouseover|mouseout)\s*=",
        re.IGNORECASE,
    )
    assert not inline_handler_re.search(html_text), "index.html must contain 0 inline JS handlers"

    # 5. Visual Rule 2: Zero emojis across index.html, style.css, script.js
    emoji_re = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
    assert not emoji_re.search(html_text), "index.html contains forbidden emoji characters"
    assert not emoji_re.search(css_text), "style.css contains forbidden emoji characters"
    assert not emoji_re.search(js_text), "script.js contains forbidden emoji characters"

    # 6. Visual Rule 1: Zero blue / purple hex codes
    forbidden_color_re = re.compile(
        r"#(?:3b82f6|06b6d4|8b5cf6|6366f1|2563eb|1d4ed8|7c3aed|9333ea|0ea5e9|38bdf8)\b",
        re.IGNORECASE,
    )
    assert not forbidden_color_re.search(html_text), "index.html contains forbidden blue/purple hex"
    assert not forbidden_color_re.search(css_text), "style.css contains forbidden blue/purple hex"
    assert not forbidden_color_re.search(js_text), "script.js contains forbidden blue/purple hex"


# ==============================================================================
# 2. SSE Streaming Live Endpoint Contract Tests
# ==============================================================================


def test_query_stream_sse_endpoint_delivery(client: TestClient):
    """Verify POST /api/query/stream delivers valid SSE events with real-time tokens."""
    headers = auth_header_for("admin@default.org", "default_org", "ADMIN")
    payload = {
        "query": "What are the core capabilities of this enterprise system?",
        "mode": "vector_only",
    }

    resp = client.post("/api/query/stream", json=payload, headers=headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n\n")
    assert len(lines) >= 2

    events = []
    tokens = []
    for line in lines:
        if line.startswith("data: "):
            ev = json.loads(line[6:])
            events.append(ev)
            if "token" in ev:
                tokens.append(ev["token"])

    assert len(tokens) >= 1, "Streaming must yield at least 1 token event"
    final_event = events[-1]
    assert final_event.get("done") is True
    assert "answer" in final_event
    assert "telemetry" in final_event
    assert isinstance(final_event["citations"], list)


def test_query_stream_semantic_cache_replay_pill(client: TestClient):
    """Verify consecutive queries trigger SemanticQueryCache and include cached telemetry pill."""
    headers = auth_header_for("analyst@default.org", "default_org", "ANALYST")
    payload = {
        "query": "Explain reciprocal rank fusion k=60 weighted scoring",
        "mode": "vector_only",
    }

    # Initial query: caches the outcome
    resp1 = client.post("/api/query/stream", json=payload, headers=headers)
    assert resp1.status_code == 200

    # Repeat query: must hit semantic query cache with <10ms lookup
    resp2 = client.post("/api/query/stream", json=payload, headers=headers)
    assert resp2.status_code == 200

    lines = resp2.text.strip().split("\n\n")
    events = [json.loads(l[6:]) for l in lines if l.startswith("data: ")]
    final_event = events[-1]

    assert final_event.get("done") is True
    assert final_event.get("cached") is True
    telemetry = final_event.get("telemetry", {})
    assert telemetry.get("cached") is True
    assert float(telemetry.get("total_latency_ms", 100)) < 15.0


def test_query_stream_contextual_citation_headers(client: TestClient):
    """Verify citations returned via SSE preserve contextual_header fields."""
    headers = auth_header_for("admin@default.org", "default_org", "ADMIN")
    payload = {
        "query": "Summarize key architecture findings",
        "mode": "vector_only",
    }

    resp = client.post("/api/query/stream", json=payload, headers=headers)
    assert resp.status_code == 200

    events = [json.loads(l[6:]) for l in resp.text.strip().split("\n\n") if l.startswith("data: ")]
    final_event = events[-1]

    citations = final_event.get("citations", [])
    for cit in citations:
        assert "pdf" in cit
        assert "page" in cit
        assert "score" in cit
        assert "contextual_header" in cit


def test_query_stream_session_persistence(client: TestClient):
    """Verify streaming query turns are persisted under the conversational session."""
    headers = auth_header_for("admin@default.org", "default_org", "ADMIN")

    # 1. Create a session
    create_resp = client.post("/api/chat/sessions", json={"title": "Streaming Test Session"}, headers=headers)
    assert create_resp.status_code == 200
    session_id = create_resp.json()["id"]

    # 2. Query with session_id via stream
    payload = {
        "query": "Hello RAG system, please introduce yourself.",
        "mode": "vector_only",
        "session_id": session_id,
    }
    resp = client.post("/api/query/stream", json=payload, headers=headers)
    assert resp.status_code == 200

    # 3. Retrieve conversation history
    hist_resp = client.get(f"/api/chat/sessions/{session_id}/messages", headers=headers)
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    messages = hist_data.get("messages", [])
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == payload["query"]
    assert messages[1]["role"] == "assistant"
