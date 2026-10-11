"""Phase 2.4 Automated Pytest Suite — Multi-Tenant JWT Auth & Pre-Retrieval RBAC (Sec 3).

Validates:
1. Cryptographic primitives: PBKDF2-HMAC password hashing, timing-safe verification & PyJWT lifecycle.
2. Token security: Expiration enforcement, secret key forgery rejection & payload tampering guards.
3. Auth REST API: User registration, duplicate conflict (409), login token generation & /api/auth/me profile resolution.
4. Role hierarchy & Route ACLs: ADMIN vs. MANAGER vs. ANALYST vs. VIEWER permissions on upload & delete routes.
5. Pre-retrieval Qdrant RBAC filters: Verification of tenant isolation and role visibility boolean conditions.
6. Zero cross-tenant leakage: Proves Tenant A can NEVER retrieve Tenant B chunks under identical queries.
7. Intra-tenant ACL gating: Proves RESTRICTED/ORG document chunks are inaccessible to unauthorized roles.
8. Malicious filter spoofing guard: Proves caller-supplied org_id filters cannot bypass authenticated tenant boundary.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient
from qdrant_client import models

from src.api.server import app
from src.auth.dependencies import build_qdrant_rbac_filter
from src.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    seed_demo_accounts,
    verify_password,
)
from src.config import settings
from src.database.models import DocumentRecord, Organization, User
from src.database.relational import SessionLocal, init_db
from src.database.vector_store import VectorStore, combine_filters
from src.embeddings.embedder import LocalEmbedder
from src.ingestion.chunker import TextChunk
from src.ingestion.classifier import PageType
from src.ingestion.pipeline import IngestionResult
from src.retrieval.retriever import Retriever


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Ensure relational database and demo accounts are initialized."""
    init_db()
    seed_demo_accounts()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    return TestClient(app)


# ==============================================================================
# 1. Cryptographic & JWT Security Primitives
# ==============================================================================


def test_password_hashing_and_jwt_crypto():
    """Verify PBKDF2 salt uniqueness, timing-safe verification, and JWT token crypto."""
    pwd = "EnterpriseSecret2026!"

    # 1. Salt uniqueness: Two hashes of the same password produce different hashes
    hash1 = hash_password(pwd)
    hash2 = hash_password(pwd)
    assert hash1 != hash2, "PBKDF2 hashes must use random unique salts"
    assert hash1.startswith("pbkdf2:sha256:100000$")

    # 2. Timing-safe verification
    assert verify_password(pwd, hash1) is True
    assert verify_password(pwd, hash2) is True
    assert verify_password("WrongPassword!", hash1) is False
    assert verify_password("", hash1) is False

    # 3. Malformed hash crash tolerance
    assert verify_password(pwd, "") is False
    assert verify_password(pwd, "invalid_format") is False
    assert verify_password(pwd, "pbkdf2:sha256:100000$nothex$1234") is False
    assert verify_password(pwd, "corrupted$hash$value$extra") is False

    # 4. JWT generation and decoding
    claims = {"sub": "usr_99", "email": "test@enterprise.com", "org_id": "org_alpha", "role": "ANALYST"}
    token = create_access_token(claims)
    decoded = decode_access_token(token)

    assert decoded["sub"] == "usr_99"
    assert decoded["org_id"] == "org_alpha"
    assert decoded["role"] == "ANALYST"
    assert "exp" in decoded and "iat" in decoded

    # 5. Token expiration handling
    expired_token = create_access_token(claims, expires_delta=timedelta(seconds=-10))
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired_token)

    # 6. Forgery / wrong secret rejection
    forged_token = jwt.encode(
        claims, "wrong-secret-key-32-bytes-minimum-length!", algorithm=settings.jwt_algorithm
    )
    with pytest.raises(jwt.InvalidSignatureError):
        decode_access_token(forged_token)


# ==============================================================================
# 2. Authentication REST Endpoints Lifecycle
# ==============================================================================


def test_auth_endpoints_lifecycle(client: TestClient):
    """Verify user registration, duplicate email rejection, login, and /api/auth/me profile resolution."""
    unique_id = uuid.uuid4().hex[:8]
    test_email = f"qa_user_{unique_id}@customorg.com"
    test_password = "SecurePassword123!"
    org_name = f"Custom Tenant {unique_id}"

    # 1. Register new user with custom tenant org
    reg_resp = client.post(
        "/api/auth/register",
        json={
            "email": test_email,
            "password": test_password,
            "org_name": org_name,
            "role": "MANAGER",
        },
    )
    assert reg_resp.status_code == 201
    user_data = reg_resp.json()
    assert user_data["email"] == test_email
    assert user_data["role"] == "MANAGER"
    assert user_data["org_id"] is not None

    # 2. Duplicate registration rejection (409 Conflict)
    dup_resp = client.post(
        "/api/auth/register",
        json={
            "email": test_email,
            "password": "AnotherPassword!",
            "org_id": user_data["org_id"],
        },
    )
    assert dup_resp.status_code == 409
    assert "already exists" in dup_resp.json()["detail"]

    # 3. Successful login returns JWT access token
    login_resp = client.post(
        "/api/auth/login",
        json={"email": test_email, "password": test_password},
    )
    assert login_resp.status_code == 200
    token_payload = login_resp.json()
    assert "access_token" in token_payload
    assert token_payload["token_type"].lower() == "bearer"
    token = token_payload["access_token"]

    # 4. Invalid password rejection (401)
    bad_login = client.post(
        "/api/auth/login",
        json={"email": test_email, "password": "WrongPassword!"},
    )
    assert bad_login.status_code == 401

    # 5. Access /api/auth/me with Bearer token
    me_resp = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["email"] == test_email
    assert me_data["org_id"] == user_data["org_id"]
    assert me_data["role"] == "MANAGER"

    # 6. Access /api/auth/me with missing / invalid token (401)
    unauth_resp = client.get("/api/auth/me")
    assert unauth_resp.status_code == 401

    bad_token_resp = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer malformed.token.here"},
    )
    assert bad_token_resp.status_code == 401


# ==============================================================================
# 3. Role-Based Permissions & Route Authorization Gating
# ==============================================================================


def test_role_hierarchy_and_route_permissions(client: TestClient):
    """Verify ADMIN and MANAGER can upload while ANALYST and VIEWER are rejected (403),
    and only ADMIN can delete documents."""
    # Obtain tokens for each seeded role in default_org
    def get_token(email: str, pwd: str) -> str:
        resp = client.post("/api/auth/login", json={"email": email, "password": pwd})
        assert resp.status_code == 200, f"Login failed for {email}: {resp.text}"
        return resp.json()["access_token"]

    admin_token = get_token("admin@default.org", "adminpassword123")
    manager_token = get_token("manager@default.org", "managerpassword123")
    analyst_token = get_token("analyst@default.org", "analystpassword123")
    viewer_token = get_token("viewer@default.org", "viewerpassword123")

    import fitz
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()
    page.insert_text((50, 50), "Valid RBAC test document content.")
    valid_pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    # 1. Upload permissions: ADMIN (permitted), MANAGER (permitted)
    for role_name, token in [("ADMIN", admin_token), ("MANAGER", manager_token)]:
        resp = client.post(
            "/api/upload",
            files={"file": (f"test_{role_name.lower()}.pdf", valid_pdf_bytes, "application/pdf")},
            data={"visibility": "ORG"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code in {202, 400}, f"{role_name} should be authorized for upload"

    # 2. Upload permissions: ANALYST (403 Forbidden), VIEWER (403 Forbidden)
    for role_name, token in [("ANALYST", analyst_token), ("VIEWER", viewer_token)]:
        resp = client.post(
            "/api/upload",
            files={"file": (f"test_{role_name.lower()}.pdf", valid_pdf_bytes, "application/pdf")},
            data={"visibility": "ORG"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403, f"{role_name} must receive 403 Forbidden on upload"
        assert "Forbidden" in resp.json()["detail"]

    # 3. Deletion permissions: Only ADMIN can delete
    for role_name, token in [("MANAGER", manager_token), ("ANALYST", analyst_token), ("VIEWER", viewer_token)]:
        resp = client.delete(
            "/api/documents/non_existent_hash",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403, f"{role_name} must receive 403 Forbidden on delete"

    # 4. ADMIN deletion on non-existent document returns 404
    admin_del = client.delete(
        "/api/documents/non_existent_hash",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_del.status_code == 404

    # Drain any background tasks spawned by upload tests
    import concurrent.futures
    import src.api.server as server_module
    if server_module.job_manager and hasattr(server_module.job_manager, "executor"):
        server_module.job_manager.executor.shutdown(wait=True)
        server_module.job_manager.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="rag_ingest_worker"
        )


# ==============================================================================
# 4. Pre-Retrieval Qdrant RBAC Filter Mathematical Construction
# ==============================================================================


def test_pre_retrieval_rbac_filter_construction():
    """Verify build_qdrant_rbac_filter produces strict multi-tenant and role-scoped conditions."""
    # ADMIN user filter
    admin_user = User(id="u_admin", email="admin@corp.com", org_id="tenant_x", role="ADMIN")
    admin_filter = build_qdrant_rbac_filter(admin_user)
    assert admin_filter.must is not None
    assert len(admin_filter.must) == 1
    assert admin_filter.must[0].key == "org_id"
    assert admin_filter.must[0].match.value == "tenant_x"
    assert admin_filter.should is None, "ADMIN should have no role visibility restrictions"

    # VIEWER user filter
    viewer_user = User(id="u_viewer", email="viewer@corp.com", org_id="tenant_x", role="VIEWER")
    viewer_filter = build_qdrant_rbac_filter(viewer_user)
    assert viewer_filter.must is not None
    assert len(viewer_filter.must) == 1
    assert viewer_filter.must[0].key == "org_id"
    assert viewer_filter.must[0].match.value == "tenant_x"
    assert viewer_filter.should is not None
    assert len(viewer_filter.should) == 2

    # Should condition 1: visibility == PUBLIC
    vis_cond = next(c for c in viewer_filter.should if c.key == "visibility")
    assert vis_cond.match.value == "PUBLIC"

    # Should condition 2: allowed_roles contains VIEWER
    role_cond = next(c for c in viewer_filter.should if c.key == "allowed_roles")
    assert "VIEWER" in role_cond.match.any


# ==============================================================================
# 5. Cross-Tenant Zero-Leakage Empirical Guarantee
# ==============================================================================


def test_cross_tenant_zero_leakage_isolation():
    """Verify that under identical queries, Tenant A never retrieves Tenant B documents and vice versa."""
    import src.api.server as server_module
    server_module.ensure_services_initialized()
    vs = server_module.vector_store
    retriever = server_module.retriever

    tenant_a_id = "tenant_alpha_corp"
    tenant_b_id = "tenant_beta_corp"

    # Create synthetic documents with distinct confidential identifiers
    doc_a_hash = "hash_alpha_9999"
    doc_b_hash = "hash_beta_8888"

    text_a = "Confidential Financial Directive: Alpha Corp project budget is exactly 42 Million USD in 2026."
    text_b = "Confidential Financial Directive: Beta Corp project budget is exactly 89 Million USD in 2026."

    res_a = IngestionResult(
        source_pdf="alpha_financials.pdf",
        file_hash=doc_a_hash,
        total_pages=1,
        classification_summary={PageType.TEXT: 1, PageType.IMAGE: 0, PageType.MIXED: 0},
        total_chunks=1,
        chunks=[
            TextChunk(
                text=text_a,
                char_count=len(text_a),
                metadata={
                    "page_number": 1,
                    "org_id": tenant_a_id,
                    "visibility": "ORG",
                    "allowed_roles": ["ADMIN", "MANAGER", "ANALYST", "VIEWER"],
                },
            )
        ],
    )
    setattr(res_a, "org_id", tenant_a_id)

    res_b = IngestionResult(
        source_pdf="beta_financials.pdf",
        file_hash=doc_b_hash,
        total_pages=1,
        classification_summary={PageType.TEXT: 1, PageType.IMAGE: 0, PageType.MIXED: 0},
        total_chunks=1,
        chunks=[
            TextChunk(
                text=text_b,
                char_count=len(text_b),
                metadata={
                    "page_number": 1,
                    "org_id": tenant_b_id,
                    "visibility": "ORG",
                    "allowed_roles": ["ADMIN", "MANAGER", "ANALYST", "VIEWER"],
                },
            )
        ],
    )
    setattr(res_b, "org_id", tenant_b_id)

    # Index into vector database
    vs.add_document(res_a)
    vs.add_document(res_b)

    # Create mock authenticated users for both tenants
    user_a = User(id="user_a", email="a@alpha.com", org_id=tenant_a_id, role="ANALYST")
    user_b = User(id="user_b", email="b@beta.com", org_id=tenant_b_id, role="ANALYST")

    query = "What is the project budget in 2026?"

    # 1. Query as Tenant A User
    outcome_a = retriever.retrieve(query, user=user_a, mode="hybrid_rerank")
    assert outcome_a["abstained"] is False
    assert len(outcome_a["results"]) > 0
    # Every chunk must belong to tenant_a_id and never contain Tenant B content
    for r in outcome_a["results"]:
        assert r["metadata"].get("org_id") == tenant_a_id
        assert "89 Million" not in r["text"]
        assert "Beta Corp" not in r["text"]
    assert "42 Million" in outcome_a["context"]

    # 2. Query as Tenant B User
    outcome_b = retriever.retrieve(query, user=user_b, mode="hybrid_rerank")
    assert outcome_b["abstained"] is False
    assert len(outcome_b["results"]) > 0
    # Every chunk must belong to tenant_b_id and never contain Tenant A content
    for r in outcome_b["results"]:
        assert r["metadata"].get("org_id") == tenant_b_id
        assert "42 Million" not in r["text"]
        assert "Alpha Corp" not in r["text"]
    assert "89 Million" in outcome_b["context"]

    # Cleanup test documents
    vs.delete_document(doc_a_hash, org_id=tenant_a_id)
    vs.delete_document(doc_b_hash, org_id=tenant_b_id)


# ==============================================================================
# 6. Intra-Tenant Role Visibility ACL Enforcement
# ==============================================================================


def test_role_visibility_acl_filtering():
    """Verify that within the same tenant organization, RESTRICTED documents
    are only accessible to authorized roles and blocked for lower-tier roles."""
    import src.api.server as server_module
    server_module.ensure_services_initialized()
    vs = server_module.vector_store
    retriever = server_module.retriever

    org_id = "tenant_acl_test_org"
    restricted_hash = "hash_restricted_1111"
    public_hash = "hash_public_2222"

    # 1. RESTRICTED document: allowed only to ADMIN and MANAGER
    res_restricted = IngestionResult(
        source_pdf="board_compensation.pdf",
        file_hash=restricted_hash,
        total_pages=1,
        classification_summary={PageType.TEXT: 1, PageType.IMAGE: 0, PageType.MIXED: 0},
        total_chunks=1,
        chunks=[
            TextChunk(
                text="Executive Bonus Directive: Executive CEO package is confidential and approved by Board.",
                char_count=len("Executive Bonus Directive: Executive CEO package is confidential and approved by Board."),
                metadata={
                    "page_number": 1,
                    "org_id": org_id,
                    "visibility": "RESTRICTED",
                    "allowed_roles": ["ADMIN", "MANAGER"],
                },
            )
        ],
    )
    setattr(res_restricted, "org_id", org_id)

    # 2. PUBLIC document: visible to all roles
    res_public = IngestionResult(
        source_pdf="holiday_calendar.pdf",
        file_hash=public_hash,
        total_pages=1,
        classification_summary={PageType.TEXT: 1, PageType.IMAGE: 0, PageType.MIXED: 0},
        total_chunks=1,
        chunks=[
            TextChunk(
                text="Corporate Calendar: Annual office holiday is on December 25th for all staff.",
                char_count=len("Corporate Calendar: Annual office holiday is on December 25th for all staff."),
                metadata={
                    "page_number": 1,
                    "org_id": org_id,
                    "visibility": "PUBLIC",
                    "allowed_roles": ["ADMIN", "MANAGER", "ANALYST", "VIEWER"],
                },
            )
        ],
    )
    setattr(res_public, "org_id", org_id)

    vs.add_document(res_restricted)
    vs.add_document(res_public)

    manager_user = User(id="u_mgr", email="m@corp.com", org_id=org_id, role="MANAGER")
    viewer_user = User(id="u_vwr", email="v@corp.com", org_id=org_id, role="VIEWER")

    # 1. Query for confidential executive package
    exec_query = "Executive CEO package confidential board"

    # MANAGER query -> authorized, returns result
    res_mgr = retriever.retrieve(exec_query, user=manager_user, mode="hybrid_rerank")
    assert res_mgr["abstained"] is False
    assert any("Executive CEO package" in r["text"] for r in res_mgr["results"])

    # VIEWER query -> unauthorized, must abstain with zero chunk leakage
    res_vwr = retriever.retrieve(exec_query, user=viewer_user, mode="hybrid_rerank")
    assert res_vwr["abstained"] is True
    assert res_vwr["results"] == []
    assert "Executive CEO package" not in res_vwr["context"]

    # 2. Query for public calendar
    cal_query = "Annual office holiday December 25th"
    cal_vwr = retriever.retrieve(cal_query, user=viewer_user, mode="hybrid_rerank")
    assert cal_vwr["abstained"] is False
    assert any("December 25th" in r["text"] for r in cal_vwr["results"])

    # Cleanup
    vs.delete_document(restricted_hash, org_id=org_id)
    vs.delete_document(public_hash, org_id=org_id)


# ==============================================================================
# 7. Malicious Pre-Retrieval Filter Spoofing Guard
# ==============================================================================


def test_tampered_org_spoofing_guard():
    """Verify that an authenticated user cannot spoof or bypass their tenant org
    by supplying a malicious caller filter targeting another tenant."""
    import src.api.server as server_module
    server_module.ensure_services_initialized()
    vs = server_module.vector_store
    retriever = server_module.retriever

    # Ingest victim tenant chunk
    victim_org = "victim_tenant_target"
    victim_hash = "victim_hash_4444"
    res_victim = IngestionResult(
        source_pdf="victim_secrets.pdf",
        file_hash=victim_hash,
        total_pages=1,
        classification_summary={PageType.TEXT: 1, PageType.IMAGE: 0, PageType.MIXED: 0},
        total_chunks=1,
        chunks=[
            TextChunk(
                text="Victim Confidential Secret: Bank routing PIN code is 987654.",
                char_count=len("Victim Confidential Secret: Bank routing PIN code is 987654."),
                metadata={
                    "page_number": 1,
                    "org_id": victim_org,
                    "visibility": "PUBLIC",
                    "allowed_roles": ["ADMIN", "MANAGER", "ANALYST", "VIEWER"],
                },
            )
        ],
    )
    setattr(res_victim, "org_id", victim_org)
    vs.add_document(res_victim)

    # Attacker authenticated in 'attacker_tenant'
    attacker_user = User(
        id="attacker_1",
        email="hacker@attacker.org",
        org_id="attacker_tenant",
        role="ADMIN",
    )

    # Attacker attempts to forge query_filter targeting 'victim_tenant_target'
    malicious_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="org_id",
                match=models.MatchValue(value=victim_org),
            )
        ]
    )

    # Retriever combines the user's authentic RBAC filter (attacker_tenant)
    # with the malicious filter into an intersection (must=[attacker_tenant, victim_tenant])
    spoofed_outcome = retriever.retrieve(
        "Bank routing PIN code 987654",
        query_filter=malicious_filter,
        user=attacker_user,
        mode="hybrid_rerank",
    )

    # Must abstain and return 0 results because attacker_tenant != victim_tenant
    assert spoofed_outcome["abstained"] is True
    assert spoofed_outcome["results"] == []
    assert "987654" not in spoofed_outcome["context"]

    # Cleanup
    vs.delete_document(victim_hash, org_id=victim_org)
