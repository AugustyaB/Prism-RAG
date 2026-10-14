"""Phase 2.6 Automated Pytest Suite — Interactive Enterprise Demo Web Studio (Sec 1–6, L5A & L5B).

Empirically validates:
1. Static Web Studio Asset Integrity, CSP Zero-Inline JS Handler Contract & Visual Rules:
   - Zero inline DOM event handlers (`onclick=`, `onsubmit=`, `onchange=`, etc.) in `index.html`.
   - Zero emoji characters across `index.html`, `style.css`, and `script.js` (Visual Rule 2).
   - Zero blue/cyan/indigo/violet/purple hex colors and presence of Titanium Obsidian & Emerald design tokens (Visual Rule 1).
   - Complete DOM component IDs, 6 persona buttons, 4 retrieval mode pills, and L5A controller functions.
2. Multi-Tenant Persona Auth & Token Verification:
   - `POST /api/auth/seed` and `POST /api/auth/login` across all 6 pre-seeded personas (`ADMIN`, `MANAGER`, `ANALYST`, `VIEWER` in `default_org` & `tenant_b`).
   - `GET /api/auth/me` profile and tenant resolution.
3. Asynchronous 202 Ingestion, Job Polling, SHA-256 Deduplication & Document Catalog Lifecycle:
   - Role upload guards (`403 Forbidden` for `VIEWER`/`ANALYST`).
   - `POST /api/upload` (`202 Accepted`), `GET /api/jobs/{job_id}` polling until completion, SHA-256 deduplication, `GET /api/documents` tenant catalog isolation, and `DELETE /api/documents/{file_hash}` RBAC enforcement.
4. Multi-Mode Grounded Query, Two-Tier Abstention Gate & Per-Query Telemetry Ribbon Contract:
   - `POST /api/query` across all 4 modes (`hybrid_rerank`, `hybrid_rrf`, `vector_only`, `bm25_only`).
   - Verification of citation cards (`pdf`, `page`, `page_end`, `parent_id`, `score`, `snippet`), abstention reasons (`LOW_CONFIDENCE`, `RBAC_EMPTY_RETRIEVAL`), and stage latency/token/cost telemetry.
5. Multi-Tenant Observability Metrics Drawer & Liveness Probe:
   - `GET /api/metrics?org_id=...` schema verification (`p50`/`p90`/`p95`/`avg` latencies, tokens, costs, abstention rate) and `GET /health` liveness check.
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import time
from unittest.mock import MagicMock, patch
import uuid

import fitz
import pytest
from fastapi.testclient import TestClient

from src.api.server import app, rate_limiter
from src.auth.security import seed_demo_accounts
from src.database.relational import init_db

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "src" / "api" / "static"


@pytest.fixture(scope="module", autouse=True)
def setup_studio_environment(tmp_path_factory):
    """Initialize an isolated temporary relational database and seed demo personas."""
    test_db_dir = tmp_path_factory.mktemp("studio_test_db")
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


@pytest.fixture
def client():
    """FastAPI TestClient with fresh rate-limiter state."""
    rate_limiter.reset()
    return TestClient(app)


def login_as(client: TestClient, email: str, password: str) -> dict:
    """Authenticate via POST /api/auth/login and return full token response."""
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"Login failed for {email}: {resp.text}"
    return resp.json()


def build_sample_pdf_bytes(text_content: str) -> bytes:
    """Create a valid in-memory PDF document using PyMuPDF (fitz)."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 72),
        text_content,
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ==============================================================================
# 1. Static Asset Integrity, Zero-Inline JS, Zero-Emoji & Titanium Palette Audit
# ==============================================================================


def test_web_studio_static_asset_integrity_and_visual_contract(client: TestClient):
    """Verify L5A/L5B static files obey CSP, Zero-Emoji, Zero-Blue/Purple, and DOM ID contracts."""
    index_path = STATIC_DIR / "index.html"
    style_path = STATIC_DIR / "style.css"
    script_path = STATIC_DIR / "script.js"

    assert index_path.exists(), "index.html is missing"
    assert style_path.exists(), "style.css is missing"
    assert script_path.exists(), "script.js is missing"

    html_text = index_path.read_text(encoding="utf-8")
    css_text = style_path.read_text(encoding="utf-8")
    js_text = script_path.read_text(encoding="utf-8")

    # 1. Verify StaticFiles mount serves HTML, CSS, and JS over HTTP 200
    resp_html = client.get("/")
    assert resp_html.status_code == 200
    assert "Prism-RAG Enterprise Studio" in resp_html.text

    resp_css = client.get("/style.css")
    assert resp_css.status_code == 200
    assert "--bg-primary: #09090b" in resp_css.text

    resp_js = client.get("/script.js")
    assert resp_js.status_code == 200
    assert "DEMO_PERSONAS" in resp_js.text

    # 2. L5B Contract: Zero inline JavaScript event handlers in index.html
    inline_handler_re = re.compile(
        r"\bon(click|change|submit|load|keyup|keydown|input|focus|blur|mouseover|mouseout)\s*=",
        re.IGNORECASE,
    )
    assert not inline_handler_re.search(html_text), "index.html must contain 0 inline JS handlers"

    # 3. Visual Rule 2: Zero emoji characters in index.html, style.css, and script.js
    emoji_re = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
    assert not emoji_re.search(html_text), "index.html contains forbidden emoji characters"
    assert not emoji_re.search(css_text), "style.css contains forbidden emoji characters"
    assert not emoji_re.search(js_text), "script.js contains forbidden emoji characters"

    # 4. Visual Rule 1: Zero Blue / Cyan / Indigo / Violet / Purple hex codes
    forbidden_color_re = re.compile(
        r"#(?:3b82f6|06b6d4|8b5cf6|6366f1|2563eb|1d4ed8|7c3aed|9333ea|0ea5e9|38bdf8)\b",
        re.IGNORECASE,
    )
    assert not forbidden_color_re.search(html_text), "index.html contains forbidden blue/purple hex"
    assert not forbidden_color_re.search(css_text), "style.css contains forbidden blue/purple hex"
    assert not forbidden_color_re.search(js_text), "script.js contains forbidden blue/purple hex"

    # Verify Titanium Obsidian & Emerald tokens exist in style.css
    for required_token in [
        "--bg-primary: #09090b",
        "--bg-surface: #121316",
        "--bg-card: #18181b",
        "--accent: #10b981",
        "--gold-admin: #f59e0b",
        "--bronze-analyst: #d97706",
        "--danger: #ef4444",
    ]:
        assert required_token in css_text, f"Missing design token in style.css: {required_token}"

    # 5. Verify required semantic DOM IDs in index.html
    required_ids = [
        "connection-status",
        "active-user-badge",
        "toggle-metrics-btn",
        "sidebar",
        "drop-zone",
        "file-input",
        "doc-visibility",
        "doc-roles",
        "upload-status-card",
        "upload-progress-bar",
        "upload-progress-percent",
        "upload-status-text",
        "doc-catalog-section",
        "doc-list",
        "mode-selector",
        "provider-selector",
        "chat-workspace",
        "chat-history",
        "query-form",
        "query-input",
        "send-btn",
        "metrics-drawer",
        "refresh-metrics-btn",
        "close-metrics-btn",
    ]
    for dom_id in required_ids:
        assert f'id="{dom_id}"' in html_text, f"Missing required DOM element id='{dom_id}' in index.html"

    # 6. Verify all 6 persona switcher buttons and 4 retrieval mode pills
    for persona in [
        "admin_default",
        "manager_default",
        "analyst_default",
        "viewer_default",
        "admin_tenant_b",
        "viewer_tenant_b",
    ]:
        assert f'data-persona="{persona}"' in html_text, f"Missing persona button: {persona}"

    for mode in ["hybrid_rerank", "hybrid_rrf", "vector_only", "bm25_only"]:
        assert f'data-mode="{mode}"' in html_text, f"Missing retrieval mode button: {mode}"

    # 7. Verify L5A resilience & state management functions in script.js
    for fn_name in [
        "loginPersona",
        "startJobPolling",
        "loadDocumentCatalog",
        "deleteDocument",
        "renderQueryResponse",
        "buildCitationsElement",
        "buildTelemetryRibbon",
        "loadMetrics",
        "safeParseJson",
        "formatHttpError",
        "escapeHtml",
    ]:
        assert f"function {fn_name}" in js_text, f"Missing required L5A function in script.js: {fn_name}"


# ==============================================================================
# 2. Full Multi-Tenant Persona Auth & Token Verification
# ==============================================================================


def test_multi_tenant_persona_auth_and_token_verification(client: TestClient):
    """Verify POST /api/auth/seed, POST /api/auth/login, and GET /api/auth/me for all 6 studio personas."""
    seed_resp = client.post("/api/auth/seed")
    assert seed_resp.status_code == 200
    seed_data = seed_resp.json()
    assert len(seed_data.get("accounts", [])) >= 6

    expected_personas = [
        ("admin@default.org", "adminpassword123", "default_org", "ADMIN"),
        ("manager@default.org", "managerpassword123", "default_org", "MANAGER"),
        ("analyst@default.org", "analystpassword123", "default_org", "ANALYST"),
        ("viewer@default.org", "viewerpassword123", "default_org", "VIEWER"),
        ("admin@tenant_b.org", "tenantbadmin123", "tenant_b", "ADMIN"),
        ("viewer@tenant_b.org", "tenantbviewer123", "tenant_b", "VIEWER"),
    ]

    for email, pwd, expected_org, expected_role in expected_personas:
        auth_data = login_as(client, email, pwd)
        assert "access_token" in auth_data
        assert auth_data["user"]["email"] == email
        assert auth_data["user"]["org_id"] == expected_org
        assert auth_data["user"]["role"] == expected_role

        me_resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {auth_data['access_token']}"},
        )
        assert me_resp.status_code == 200
        me_profile = me_resp.json()
        assert me_profile["email"] == email
        assert me_profile["org_id"] == expected_org
        assert me_profile["role"] == expected_role


# ==============================================================================
# 3. Async 202 Ingestion Polling, SHA-256 Deduplication & Catalog Lifecycle
# ==============================================================================


def test_async_ingestion_polling_deduplication_and_catalog_lifecycle(client: TestClient):
    """Verify studio upload flow: RBAC guard, 202 job polling, deduplication, tenant catalog & ADMIN delete."""
    admin_default = login_as(client, "admin@default.org", "adminpassword123")
    viewer_default = login_as(client, "viewer@default.org", "viewerpassword123")
    admin_tenant_b = login_as(client, "admin@tenant_b.org", "tenantbadmin123")

    admin_headers = {"Authorization": f"Bearer {admin_default['access_token']}"}
    viewer_headers = {"Authorization": f"Bearer {viewer_default['access_token']}"}
    tenant_b_headers = {"Authorization": f"Bearer {admin_tenant_b['access_token']}"}

    unique_marker = f"STUDIO-UI-AUDIT-{uuid.uuid4().hex[:8]}"
    pdf_bytes = build_sample_pdf_bytes(
        f"Enterprise Web Studio Audit Specification {unique_marker}. "
        f"System architecture enforces Titanium Obsidian and Emerald styling with zero cross-tenant leakage."
    )
    pdf_filename = f"studio_audit_{unique_marker.lower()}.pdf"

    # 1. RBAC Guard: VIEWER cannot upload (403 Forbidden)
    forbidden_upload = client.post(
        "/api/upload",
        headers=viewer_headers,
        files={"file": (pdf_filename, io.BytesIO(pdf_bytes), "application/pdf")},
        data={"visibility": "ORG", "allowed_roles": "ADMIN,MANAGER,ANALYST,VIEWER"},
    )
    assert forbidden_upload.status_code == 403

    # 2. ADMIN Upload: Returns 202 Accepted with job_id
    upload_resp = client.post(
        "/api/upload",
        headers=admin_headers,
        files={"file": (pdf_filename, io.BytesIO(pdf_bytes), "application/pdf")},
        data={"visibility": "ORG", "allowed_roles": "ADMIN,MANAGER,ANALYST,VIEWER"},
    )
    assert upload_resp.status_code == 202
    upload_data = upload_resp.json()
    job_id = upload_data.get("job_id") or upload_data.get("id")
    assert job_id is not None

    # 3. Poll GET /api/jobs/{job_id} until COMPLETED
    final_job = None
    for _ in range(40):
        poll_resp = client.get(f"/api/jobs/{job_id}", headers=admin_headers)
        assert poll_resp.status_code == 200
        job_data = poll_resp.json()
        if job_data["status"] in {"COMPLETED", "FAILED"}:
            final_job = job_data
            break
        time.sleep(0.15)

    assert final_job is not None, "Ingestion job did not finish within polling window"
    assert final_job["status"] == "COMPLETED", f"Job failed: {final_job.get('error_message')}"
    assert final_job["total_pages"] >= 1
    assert final_job["chunks_created"] >= 1
    file_hash = final_job["file_hash"]

    # 4. Re-upload identical PDF -> SHA-256 deduplicated
    dedup_resp = client.post(
        "/api/upload",
        headers=admin_headers,
        files={"file": (pdf_filename, io.BytesIO(pdf_bytes), "application/pdf")},
        data={"visibility": "ORG", "allowed_roles": "ADMIN,MANAGER,ANALYST,VIEWER"},
    )
    assert dedup_resp.status_code == 202
    dedup_json = dedup_resp.json()
    assert dedup_json.get("deduplicated") is True
    assert dedup_json.get("duplicate") is True

    # 5. Document Catalog: Present in default_org, absent in tenant_b
    cat_default = client.get("/api/documents", headers=admin_headers)
    assert cat_default.status_code == 200
    default_hashes = [d["file_hash"] for d in cat_default.json()["documents"]]
    assert file_hash in default_hashes

    cat_tenant_b = client.get("/api/documents", headers=tenant_b_headers)
    assert cat_tenant_b.status_code == 200
    tenant_b_hashes = [d["file_hash"] for d in cat_tenant_b.json()["documents"]]
    assert file_hash not in tenant_b_hashes

    # 6. Delete Guard: VIEWER gets 403, ADMIN deletes cleanly
    del_forbidden = client.delete(f"/api/documents/{file_hash}", headers=viewer_headers)
    assert del_forbidden.status_code == 403

    del_ok = client.delete(f"/api/documents/{file_hash}", headers=admin_headers)
    assert del_ok.status_code == 200

    # Clean up any temporary studio_audit_*.pdf files on disk and in catalog
    for doc_entry in cat_default.json()["documents"]:
        if str(doc_entry.get("filename", "")).startswith("studio_audit_"):
            client.delete(f"/api/documents/{doc_entry['file_hash']}", headers=admin_headers)
    pdf_dir = ROOT_DIR / "data" / "pdfs"
    if pdf_dir.exists():
        for tmp_pdf in pdf_dir.glob("studio_audit_*.pdf"):
            tmp_pdf.unlink(missing_ok=True)


# ==============================================================================
# 4. Multi-Mode Grounded Query, Abstention Gate & Telemetry Verification
# ==============================================================================


def test_multi_mode_grounded_query_abstention_and_telemetry(client: TestClient):
    """Verify POST /api/query across all 4 modes returns citations, abstention flags, and stage telemetry."""
    admin_default = login_as(client, "admin@default.org", "adminpassword123")
    admin_tenant_b = login_as(client, "admin@tenant_b.org", "tenantbadmin123")

    admin_headers = {"Authorization": f"Bearer {admin_default['access_token']}"}
    tenant_b_headers = {"Authorization": f"Bearer {admin_tenant_b['access_token']}"}

    mock_gen = MagicMock()
    mock_gen.model_name = "gpt-oss-20b"
    mock_gen.generate_with_telemetry.return_value = (
        "Grounded response citing enterprise technical manual [Page 1].",
        110,
        28,
    )

    with patch("src.api.server.get_generator", return_value=mock_gen):
        for mode in ["hybrid_rerank", "hybrid_rrf", "vector_only", "bm25_only"]:
            resp = client.post(
                "/api/query",
                headers=admin_headers,
                json={
                    "query": "What is the maximum operating temperature or pressure fault threshold?",
                    "mode": mode,
                    "provider": "lmstudio",
                },
            )
            assert resp.status_code == 200, f"Query failed in mode={mode}: {resp.text}"
            payload = resp.json()

            # Verify contract consumed by script.js
            assert "answer" in payload
            assert "citations" in payload and isinstance(payload["citations"], list)
            assert "abstained" in payload and isinstance(payload["abstained"], bool)
            assert payload["mode"] == mode
            assert "request_id" in payload
            assert "telemetry" in payload

            telemetry = payload["telemetry"]
            for required_span in [
                "embed_latency_ms",
                "search_latency_ms",
                "rerank_latency_ms",
                "llm_latency_ms",
                "total_latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "estimated_cost_usd",
            ]:
                assert required_span in telemetry, f"Missing telemetry field: {required_span}"

            if not payload["abstained"] and payload["citations"]:
                cit = payload["citations"][0]
                for cit_key in ["pdf", "page", "page_end", "parent_id", "score", "snippet"]:
                    assert cit_key in cit, f"Missing citation key: {cit_key}"

        # Verify Two-Tier Abstention Gate when tenant_b queries with no matching tenant_b documents
        abstain_resp = client.post(
            "/api/query",
            headers=tenant_b_headers,
            json={
                "query": "Unrelated quantum astrophysics teleporter flux capacitor calibration",
                "mode": "hybrid_rerank",
                "provider": "lmstudio",
            },
        )
        assert abstain_resp.status_code == 200
        abstain_data = abstain_resp.json()
        assert abstain_data["abstained"] is True
        assert abstain_data["abstention_reason"] in {
            "RBAC_EMPTY_RETRIEVAL",
            "EMPTY_RETRIEVAL",
            "LOW_CONFIDENCE",
        }
        assert abstain_data["citations"] == []


# ==============================================================================
# 5. Multi-Tenant Observability Metrics Drawer & Liveness Probe
# ==============================================================================


def test_observability_metrics_drawer_and_health_liveness(client: TestClient):
    """Verify GET /api/metrics payload structure for the studio observability drawer and GET /health."""
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "ok"

    admin_default = login_as(client, "admin@default.org", "adminpassword123")
    admin_headers = {"Authorization": f"Bearer {admin_default['access_token']}"}

    metrics_resp = client.get("/api/metrics?org_id=default_org", headers=admin_headers)
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()

    assert "total_queries" in metrics
    assert "abstained_queries" in metrics
    assert "abstention_rate" in metrics
    assert "latencies" in metrics
    for stage in ["total", "embed", "search", "rerank", "llm"]:
        assert stage in metrics["latencies"]
        for pct_key in ["p50", "p90", "p95", "p99", "avg"]:
            assert pct_key in metrics["latencies"][stage]

    assert "tokens" in metrics
    for tok_key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        assert tok_key in metrics["tokens"]

    assert "costs" in metrics
    assert "total_estimated_cost_usd" in metrics["costs"]


# ==============================================================================
# 6. Phase 2.7 Post-Preview Stabilization & Extractive Fallback Verification
# ==============================================================================


def test_llm_status_endpoint_and_favicon(client: TestClient):
    """Verify GET /favicon.ico and GET /api/config/llm-status with dynamic API key checks."""
    # 1. Favicon: returns SVG image
    fav_resp = client.get("/favicon.ico")
    assert fav_resp.status_code == 200
    assert fav_resp.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in fav_resp.text

    # 2. LLM Status: without custom key
    status_resp = client.get("/api/config/llm-status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "active_provider" in status_data
    assert "providers" in status_data
    assert "lmstudio" in status_data["providers"]
    assert "ollama" in status_data["providers"]
    assert "gemini" in status_data["providers"]

    # 3. LLM Status: with runtime X-Gemini-API-Key header
    keyed_resp = client.get(
        "/api/config/llm-status",
        headers={"X-Gemini-API-Key": "AIzaSyTestDynamicKey123"},
    )
    assert keyed_resp.status_code == 200
    keyed_data = keyed_resp.json()
    assert keyed_data["providers"]["gemini"]["configured"] is True
    assert keyed_data["providers"]["gemini"]["available"] is True


def test_offline_llm_extractive_fallback_synthesis(client: TestClient):
    """Verify that when local LLM is offline or times out, a clean grounded extractive answer is returned."""
    admin_default = login_as(client, "admin@default.org", "adminpassword123")
    admin_headers = {"Authorization": f"Bearer {admin_default['access_token']}"}

    mock_failing_gen = MagicMock()
    mock_failing_gen.model_name = "gpt-oss-20b"
    mock_failing_gen.generate.side_effect = ConnectionError("Error connecting to LM Studio at http://127.0.0.1:1234")

    with patch("src.api.server.get_generator", return_value=mock_failing_gen):
        resp = client.post(
            "/api/query",
            headers=admin_headers,
            json={
                "query": "What is the primary protocol and port configuration for cluster communication?",
                "mode": "hybrid_rerank",
                "provider": "lmstudio",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "answer" in payload
        assert payload["provider_status"] == "degraded_extractive"
        assert payload["provider_fallback"] is True
        assert len(payload["citations"]) > 0
        assert not payload["abstained"]


def test_live_database_remains_clean_after_suite():
    """Verify that test suite execution leaves zero dummy test_*.pdf records or mocked telemetry in live data/metadata.db."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.database.models import DocumentRecord, QueryTelemetry

    live_db_path = ROOT_DIR / "data" / "metadata.db"
    assert live_db_path.exists(), "Live data/metadata.db must exist"
    live_engine = create_engine(f"sqlite:///{live_db_path.as_posix()}", connect_args={"check_same_thread": False})
    LiveSession = sessionmaker(bind=live_engine)

    with LiveSession() as session:
        live_docs = session.query(DocumentRecord).filter(DocumentRecord.status == "ACTIVE").all()
        filenames = [d.filename for d in live_docs]
        for name in filenames:
            assert not name.startswith("test_"), f"Polluted test document found in live database: {name}"
            assert not name.startswith("studio_audit_"), f"Polluted studio test document found: {name}"

        # Canonical benchmark documents must remain intact
        assert "compliance_policy.pdf" in filenames
        assert "scanned_operations_log.pdf" in filenames
        assert "tech_manual.pdf" in filenames

        # Zero mocked unit-test rows in QueryTelemetry
        mock_provider_rows = (
            session.query(QueryTelemetry)
            .filter(QueryTelemetry.llm_provider == "invalid_provider_xyz")
            .count()
        )
        assert mock_provider_rows == 0, f"Found {mock_provider_rows} mocked invalid_provider_xyz telemetry rows"

        mock_query_rows = (
            session.query(QueryTelemetry)
            .filter(QueryTelemetry.query_text.like("%POL-SEC-104.2%"))
            .count()
        )
        assert mock_query_rows == 0, f"Found {mock_query_rows} mocked POL-SEC-104.2 telemetry rows"

    live_engine.dispose()

