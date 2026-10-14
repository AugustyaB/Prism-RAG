"""Phase 2.9 QA Automated Test Suite — Enterprise Identity Management & Cloud Storage (Sec 1, 3, 6; L1–L4).

Empirically validates:
1. Dual-Mode Password Reset Crypto:
   - Time-limited signed JWT token generation, claims verification, and expiration handling.
   - Master Admin Reset Token bypass for offline/self-hosted demo resilience.
   - Tampered/invalid/mismatched token rejection.
2. Password Recovery API Flow:
   - POST /api/auth/forgot-password: Token generation, demo token echo, and 404 for unknown email.
   - POST /api/auth/reset-password: Password update with PBKDF2 hashing, email mismatch guards,
     and re-authentication verification.
3. Team Member Management & RBAC Permissions:
   - GET /api/auth/users: Organization tenant boundary isolation; ADMIN and MANAGER permitted;
     ANALYST and VIEWER denied with 403 Forbidden.
   - PATCH /api/auth/users/{user_id}: ADMIN only; role validation; cross-tenant protection.
4. Cloud Database Storage Configuration & Connection Verification:
   - GET /api/config/storage-status: Active mode, endpoint masking, and health status.
   - POST /api/config/test-storage-connection: Remote Qdrant Cloud connection mocking (success & failure).
   - POST /api/config/storage-mode: Mode switching (local vs. cloud) and validation guards.
5. Zero Live Database Pollution:
   - Confirms that all tests run against isolated temporary databases and preserve live data/metadata.db.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.server import app
from src.auth.security import (
    create_access_token,
    generate_password_reset_token,
    verify_password_reset_token,
    hash_password,
    verify_password,
    seed_demo_accounts,
)
from src.config import settings
from src.database.models import Organization, User
import src.database.relational as rel
import src.api.server as srv
import src.auth.security as sec
import src.jobs.ingestion_worker as ing
from src.database.relational import get_db


@pytest.fixture(scope="module", autouse=True)
def setup_phase29_test_environment(tmp_path_factory):
    """Ensure database and demo accounts run inside an isolated temporary SQLite database."""
    test_db_dir = tmp_path_factory.mktemp("identity_storage_test_db")
    test_db_path = test_db_dir / "test_metadata.db"
    test_db_url = f"sqlite:///{test_db_path}"

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
    sec.SessionLocal = test_session_maker
    ing.SessionLocal = test_session_maker

    rel.init_db()
    sec.seed_demo_accounts()

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
    }

    srv.app.dependency_overrides.pop(get_db, None)
    settings.database_url = orig_url
    settings.pdf_upload_dir = orig_upload_dir
    rel.engine = orig_engine
    rel.SessionLocal = orig_session
    srv.SessionLocal = orig_session
    sec.SessionLocal = orig_session
    ing.SessionLocal = orig_session


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def login_as(client: TestClient, email: str, password: str) -> dict[str, Any]:
    """Helper to authenticate and return token payload."""
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"Login failed for {email}: {resp.text}"
    return resp.json()


# ==============================================================================
# 1. Dual-Mode Password Reset Crypto Primitives
# ==============================================================================


def test_password_reset_token_issuance_and_crypto():
    """Verify password reset token generation, verification, and tamper rejection."""
    user_id = str(uuid.uuid4())
    email = "engineer@enterprise.internal"

    # 1. Valid token issuance
    token = generate_password_reset_token(user_id=user_id, email=email, expires_minutes=15)
    assert isinstance(token, str) and len(token) > 20

    # 2. Token verification returns claims
    claims = verify_password_reset_token(token)
    assert claims["sub"] == user_id
    assert claims["email"] == email
    assert claims["type"] == "reset"

    # 3. Master Admin Reset Token bypass
    master_claims = verify_password_reset_token(settings.admin_master_reset_token)
    assert master_claims["type"] == "master_reset"
    assert master_claims["sub"] == "master_admin"

    # 4. Expired token rejection
    expired_token = generate_password_reset_token(user_id=user_id, email=email, expires_minutes=-10)
    with pytest.raises(ValueError, match="expired"):
        verify_password_reset_token(expired_token)

    # 5. Tampered token rejection
    tampered_token = token[:-5] + "abcde"
    with pytest.raises(ValueError, match="Invalid password reset token"):
        verify_password_reset_token(tampered_token)

    # 6. Wrong type token rejection (access token passed instead of reset token)
    access_token = create_access_token(data={"sub": user_id, "email": email, "role": "ADMIN", "org_id": "default_org"})
    with pytest.raises(ValueError, match="Invalid token type"):
        verify_password_reset_token(access_token)


# ==============================================================================
# 2. Forgot Password and Reset Password API Lifecycle
# ==============================================================================


def test_forgot_password_flow(client: TestClient):
    """Verify POST /api/auth/forgot-password generates reset token for valid user and 404s for unknown."""
    # 1. Valid existing user
    resp = client.post("/api/auth/forgot-password", json={"email": "viewer@default.org"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Password reset token generated"
    assert data["email"] == "viewer@default.org"
    assert "reset_token" in data
    assert "master_token_hint" in data
    assert len(data["reset_token"]) > 20

    # 2. Unknown user 404
    resp_unknown = client.post("/api/auth/forgot-password", json={"email": "nonexistent@nowhere.com"})
    assert resp_unknown.status_code == 404
    assert "not found" in resp_unknown.json()["detail"].lower()


def test_reset_password_flow_and_relogin(client: TestClient):
    """Verify full end-to-end forgot-password -> reset-password -> re-login lifecycle."""
    email = "viewer@default.org"
    old_pw = "viewerpassword123"
    new_pw = "NewSecureViewerPassword2026!"

    # 1. Request reset token
    forgot_resp = client.post("/api/auth/forgot-password", json={"email": email})
    assert forgot_resp.status_code == 200
    reset_token = forgot_resp.json()["reset_token"]

    # 2. Attempt reset with mismatched email (token issued for viewer, submitted for admin)
    mismatch_resp = client.post(
        "/api/auth/reset-password",
        json={"email": "admin@default.org", "reset_token": reset_token, "new_password": new_pw},
    )
    assert mismatch_resp.status_code == 400
    assert "does not match" in mismatch_resp.json()["detail"].lower()

    # 3. Attempt reset with invalid token
    invalid_resp = client.post(
        "/api/auth/reset-password",
        json={"email": email, "reset_token": "bogus-invalid-token-12345", "new_password": new_pw},
    )
    assert invalid_resp.status_code == 400

    # 4. Successful password reset
    reset_resp = client.post(
        "/api/auth/reset-password",
        json={"email": email, "reset_token": reset_token, "new_password": new_pw},
    )
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "success"

    # 5. Old password no longer works
    old_login_resp = client.post("/api/auth/login", json={"email": email, "password": old_pw})
    assert old_login_resp.status_code == 401

    # 6. New password works successfully
    new_login_resp = client.post("/api/auth/login", json={"email": email, "password": new_pw})
    assert new_login_resp.status_code == 200
    assert "access_token" in new_login_resp.json()

    # 7. Restore original password for downstream tests
    restore_resp = client.post(
        "/api/auth/reset-password",
        json={
            "email": email,
            "reset_token": settings.admin_master_reset_token,
            "new_password": old_pw,
        },
    )
    assert restore_resp.status_code == 200


def test_admin_master_reset_token_override(client: TestClient):
    """Verify that Master Admin Reset Token can reset any account without an email-bound JWT."""
    email = "analyst@default.org"
    override_pw = "MasterOverriddenAnalyst2026!"

    resp = client.post(
        "/api/auth/reset-password",
        json={
            "email": email,
            "reset_token": settings.admin_master_reset_token,
            "new_password": override_pw,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # Verify re-login with overridden password
    login_resp = client.post("/api/auth/login", json={"email": email, "password": override_pw})
    assert login_resp.status_code == 200

    # Restore default analyst password
    client.post(
        "/api/auth/reset-password",
        json={
            "email": email,
            "reset_token": settings.admin_master_reset_token,
            "new_password": "analystpassword123",
        },
    )


# ==============================================================================
# 3. Team Member Management & RBAC Permissions
# ==============================================================================


def test_team_management_rbac_and_tenant_isolation(client: TestClient):
    """Verify GET /api/auth/users and PATCH /api/auth/users/{id} permission contracts."""
    admin_auth = login_as(client, "admin@default.org", "adminpassword123")
    manager_auth = login_as(client, "manager@default.org", "managerpassword123")
    analyst_auth = login_as(client, "analyst@default.org", "analystpassword123")
    viewer_auth = login_as(client, "viewer@default.org", "viewerpassword123")
    tenant_b_auth = login_as(client, "admin@tenant_b.org", "tenantbadmin123")

    admin_headers = {"Authorization": f"Bearer {admin_auth['access_token']}"}
    manager_headers = {"Authorization": f"Bearer {manager_auth['access_token']}"}
    analyst_headers = {"Authorization": f"Bearer {analyst_auth['access_token']}"}
    viewer_headers = {"Authorization": f"Bearer {viewer_auth['access_token']}"}

    # 1. Admin can list users
    users_resp = client.get("/api/auth/users", headers=admin_headers)
    assert users_resp.status_code == 200
    default_users = users_resp.json()
    assert len(default_users) >= 4
    # Zero tenant leakage
    for u in default_users:
        assert u["org_id"] == "default_org"
        assert "@tenant_b.org" not in u["email"]

    # 2. Manager can list users
    mgr_users_resp = client.get("/api/auth/users", headers=manager_headers)
    assert mgr_users_resp.status_code == 200

    # 3. Analyst and Viewer denied listing users (403 Forbidden)
    assert client.get("/api/auth/users", headers=analyst_headers).status_code == 403
    assert client.get("/api/auth/users", headers=viewer_headers).status_code == 403

    # Locate viewer user to patch
    viewer_user = next((u for u in default_users if u["email"] == "viewer@default.org"), None)
    assert viewer_user is not None
    viewer_id = viewer_user["id"]

    # 4. Admin updates user role to ANALYST
    patch_resp = client.patch(
        f"/api/auth/users/{viewer_id}",
        headers=admin_headers,
        json={"role": "ANALYST"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["role"] == "ANALYST"

    # 5. Non-admin (Manager) forbidden from updating roles
    mgr_patch_resp = client.patch(
        f"/api/auth/users/{viewer_id}",
        headers=manager_headers,
        json={"role": "MANAGER"},
    )
    assert mgr_patch_resp.status_code == 403

    # 6. Admin forbidden from assigning invalid role
    invalid_role_resp = client.patch(
        f"/api/auth/users/{viewer_id}",
        headers=admin_headers,
        json={"role": "SUPER_ADMIN_INVALID"},
    )
    assert invalid_role_resp.status_code == 400
    assert "invalid role" in invalid_role_resp.json()["detail"].lower()

    # 7. Cross-tenant user modification prevented (Admin default_org cannot patch Tenant B user)
    tenant_b_users = client.get(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {tenant_b_auth['access_token']}"},
    ).json()
    tenant_b_user_id = tenant_b_users[0]["id"]

    cross_patch_resp = client.patch(
        f"/api/auth/users/{tenant_b_user_id}",
        headers=admin_headers,
        json={"role": "VIEWER"},
    )
    assert cross_patch_resp.status_code == 404

    # Restore viewer role back to VIEWER
    client.patch(
        f"/api/auth/users/{viewer_id}",
        headers=admin_headers,
        json={"role": "VIEWER"},
    )


# ==============================================================================
# 4. Cloud Database Storage Configuration & Verification Endpoints
# ==============================================================================


def test_storage_status_endpoint(client: TestClient):
    """Verify GET /api/config/storage-status returns local defaults and masked structure."""
    resp = client.get("/api/config/storage-status")
    assert resp.status_code == 200
    data = resp.json()

    assert "storage_mode" in data
    assert data["storage_mode"] in {"local", "cloud"}
    assert "local" in data
    assert data["local"]["status"] == "ready"
    assert "cloud" in data
    assert "qdrant_url_configured" in data["cloud"]
    assert "database_url_configured" in data["cloud"]


def test_storage_connection_testing(client: TestClient):
    """Verify POST /api/config/test-storage-connection with empty, valid mock, and invalid targets."""
    # 1. No parameters provided
    empty_resp = client.post("/api/config/test-storage-connection", json={})
    assert empty_resp.status_code == 200
    assert empty_resp.json()["status"] == "error"
    assert "no cloud storage parameters" in empty_resp.json()["message"].lower()

    # 2. Valid Mock Qdrant Connection
    mock_collections = MagicMock()
    mock_collections.collections = [MagicMock(name="rag_documents_v2")]

    mock_qdrant_client_cls = MagicMock()
    mock_qdrant_instance = MagicMock()
    mock_qdrant_instance.get_collections.return_value = mock_collections
    mock_qdrant_client_cls.return_value = mock_qdrant_instance

    with patch("qdrant_client.QdrantClient", mock_qdrant_client_cls):
        valid_resp = client.post(
            "/api/config/test-storage-connection",
            json={
                "cloud_qdrant_url": "https://mock-cluster.qdrant.io:6333",
                "cloud_qdrant_api_key": "mock-qdrant-secret-key-123",
            },
        )
        assert valid_resp.status_code == 200
        data = valid_resp.json()
        assert data["status"] == "success"
        assert "qdrant" in data["results"]
        assert data["results"]["qdrant"]["status"] == "success"

    # 3. Failing Qdrant Connection
    failing_qdrant_cls = MagicMock()
    failing_instance = MagicMock()
    failing_instance.get_collections.side_effect = ConnectionError("Could not resolve host mock-fail.qdrant.io")
    failing_qdrant_cls.return_value = failing_instance

    with patch("qdrant_client.QdrantClient", failing_qdrant_cls):
        fail_resp = client.post(
            "/api/config/test-storage-connection",
            json={
                "cloud_qdrant_url": "https://mock-fail.qdrant.io:6333",
                "cloud_qdrant_api_key": "invalid-key",
            },
        )
        assert fail_resp.status_code == 200
        fail_data = fail_resp.json()
        assert fail_data["status"] == "error"
        assert fail_data["results"]["qdrant"]["status"] == "error"


def test_storage_mode_switching(client: TestClient):
    """Verify switching between local and cloud storage modes."""
    orig_mode = settings.storage_mode

    try:
        # 1. Switch to cloud mode with valid parameters
        cloud_resp = client.post(
            "/api/config/storage-mode",
            json={
                "storage_mode": "cloud",
                "cloud_qdrant_url": "https://test-qdrant.io:6333",
                "cloud_qdrant_api_key": "test-key-abc",
                "cloud_database_url": "postgresql://user:pass@test-db.io:5432/rag",
            },
        )
        assert cloud_resp.status_code == 200
        assert cloud_resp.json()["storage_mode"] == "cloud"
        assert settings.storage_mode == "cloud"
        assert settings.cloud_qdrant_url == "https://test-qdrant.io:6333"

        # 2. Switch back to local mode
        local_resp = client.post(
            "/api/config/storage-mode",
            json={"storage_mode": "local"},
        )
        assert local_resp.status_code == 200
        assert local_resp.json()["storage_mode"] == "local"
        assert settings.storage_mode == "local"

        # 3. Invalid mode rejected
        invalid_resp = client.post(
            "/api/config/storage-mode",
            json={"storage_mode": "quantum_unsupported"},
        )
        assert invalid_resp.status_code == 400
        assert "invalid storage mode" in invalid_resp.json()["detail"].lower()

    finally:
        settings.storage_mode = orig_mode


# ==============================================================================
# 5. Live Database Cleanliness Verification
# ==============================================================================


def test_live_metadata_db_remains_unpolluted():
    """Confirm live data/metadata.db contains zero temporary unit test accounts or modified passwords."""
    live_db_path = Path("./data/metadata.db")
    if not live_db_path.exists():
        return

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    live_engine = create_engine(
        f"sqlite:///{live_db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    LiveSession = sessionmaker(bind=live_engine)

    with LiveSession() as session:
        # Live accounts must not have been modified by test passwords
        viewer = session.query(User).filter(User.email == "viewer@default.org").first()
        if viewer:
            assert viewer.role == "VIEWER", "Live viewer account role was unexpectedly mutated"
            assert verify_password("viewerpassword123", viewer.password_hash), (
                "Live viewer password was unexpectedly mutated by test suite"
            )

        # No bogus users created
        ghost_user = session.query(User).filter(User.email == "engineer@enterprise.internal").first()
        assert ghost_user is None, "Test user leaked into live database!"
