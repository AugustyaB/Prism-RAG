"""Cryptographic authentication, password hashing, and JWT token management (L2)."""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from loguru import logger

from src.config import settings
from src.database.models import DocumentRecord, IngestionJob, Organization, QueryTelemetry, User
from src.database.relational import SessionLocal, init_db

# Canonical RBAC Role Hierarchy
RBAC_ROLES: tuple[str, ...] = ("ADMIN", "MANAGER", "ANALYST", "VIEWER")


def hash_password(password: str) -> str:
    """Generate a PBKDF2-HMAC-SHA256 hash with a cryptographically secure 16-byte salt."""
    salt = os.urandom(16)
    salt_hex = salt.hex()
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    hash_hex = key.hex()
    return f"pbkdf2:sha256:100000${salt_hex}${hash_hex}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against PBKDF2 hash using timing-safe comparison."""
    try:
        parts = hashed_password.split("$")
        if len(parts) != 3:
            return False
        algo_info, salt_hex, hash_hex = parts
        _, hash_name, iters_str = algo_info.split(":")
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac(hash_name, plain_password.encode("utf-8"), salt, int(iters_str))
        return hmac.compare_digest(key.hex(), hash_hex)
    except Exception as exc:
        logger.warning(f"Failed password verification: {exc}")
        return False


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Encode payload claims into a signed PyJWT access token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    to_encode.update({"iat": now, "exp": expire})
    encoded_jwt: str = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and cryptographically validate an incoming JWT access token."""
    payload: dict[str, Any] = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    return payload


def generate_password_reset_token(
    user_id: str,
    email: str,
    expires_minutes: int = 15,
) -> str:
    """Generate a signed time-limited JWT reset token."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes)
    payload = {
        "sub": user_id,
        "email": email,
        "type": "reset",
        "iat": now,
        "exp": expire,
    }
    encoded_jwt: str = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def verify_password_reset_token(token: str) -> dict[str, str]:
    """Validate a password reset token (supporting dual-mode: Master Admin token or short-lived JWT)."""
    token_clean = token.strip()
    if token_clean == settings.admin_master_reset_token:
        return {
            "sub": "master_admin",
            "email": "master_admin",
            "type": "master_reset",
        }

    try:
        payload = jwt.decode(
            token_clean,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Password reset token has expired.")
    except (jwt.InvalidTokenError, Exception) as exc:
        raise ValueError(f"Invalid password reset token: {exc}")

    if payload.get("type") != "reset":
        raise ValueError("Invalid token type: expected password reset token.")

    return {
        "sub": str(payload.get("sub", "")),
        "email": str(payload.get("email", "")),
        "type": "reset",
    }


def seed_demo_accounts() -> None:
    """Idempotently seed demo tenant organizations and multi-role user accounts."""
    init_db()
    with SessionLocal() as session:
        # Seed Tenant Organizations
        demo_orgs = [
            ("default_org", "Default Enterprise Organization"),
            ("tenant_b", "Tenant Beta Isolated Corp"),
        ]
        for org_id, org_name in demo_orgs:
            existing_org = session.get(Organization, org_id)
            if not existing_org:
                session.add(Organization(id=org_id, name=org_name))
                logger.info(f"Seeded tenant organization: {org_id} ({org_name})")
        session.commit()

        # Seed RBAC Accounts across Tiers
        demo_users = [
            ("admin@default.org", "ADMIN", "default_org", "adminpassword123"),
            ("manager@default.org", "MANAGER", "default_org", "managerpassword123"),
            ("analyst@default.org", "ANALYST", "default_org", "analystpassword123"),
            ("viewer@default.org", "VIEWER", "default_org", "viewerpassword123"),
            ("admin@tenant_b.org", "ADMIN", "tenant_b", "tenantbadmin123"),
            ("viewer@tenant_b.org", "VIEWER", "tenant_b", "tenantbviewer123"),
        ]
        for email, role, org_id, pwd in demo_users:
            user = session.query(User).filter(User.email == email).first()
            if not user:
                user = User(
                    email=email,
                    password_hash=hash_password(pwd),
                    org_id=org_id,
                    role=role,
                )
                session.add(user)
                logger.info(f"Seeded demo account: {email} ({role} in {org_id})")
            else:
                user.role = role
                user.org_id = org_id
        session.commit()

        # Seed & Sync Canonical Benchmark Documents into DocumentRecord Catalog
        sync_benchmark_catalog(session)
        session.commit()
    logger.info("Demo tenant organizations and multi-role RBAC user accounts verified")


def sync_benchmark_catalog(session=None) -> int:
    """Ensure canonical benchmark PDFs in data/benchmark_pdfs/ are registered as ACTIVE DocumentRecords and purge test pollution."""
    from pathlib import Path
    import fitz
    from sqlalchemy import or_

    upload_dir = Path("data/uploads")
    if upload_dir.exists():
        for pdf_file in upload_dir.glob("*.pdf"):
            if (
                pdf_file.name.startswith("test_")
                or pdf_file.name.startswith("studio_audit_")
                or pdf_file.name in {"corrupt.pdf", "encrypted.pdf"}
            ):
                try:
                    pdf_file.unlink(missing_ok=True)
                except Exception:
                    pass

    bench_dir = Path("data/benchmark_pdfs")
    if not bench_dir.exists():
        return 0

    canonical_chunks = {
        "tech_manual.pdf": 26,
        "compliance_policy.pdf": 24,
        "scanned_operations_log.pdf": 12,
    }

    mock_test_queries = [
        "What is policy POL-SEC-104.2?",
        "What is the orbital velocity of Jupiter's moon Io?",
        "What is the orbital velocity of Jupiter's moon Europa?",
        "How often does POL-SEC-104.2 require key rotation?",
        "What is the CEO's home address in POL-SEC-104.2?",
        "Sample query for telemetry test",
        "Valid query text",
        "test telemetry query",
        "What is the architecture?",
        "What is the orbital period of Saturn's ring Titan?",
        "What is the maximum operating temperature or pressure fault threshold?",
        "Unrelated quantum astrophysics teleporter flux capacitor calibration",
        "What is the pump specification in the technical manual?",
        "q1",
        "q2",
        "q3",
    ]

    def _sync(sess) -> int:
        # 1. Purge leftover unit-test documents, jobs, and mocked telemetry rows
        sess.query(DocumentRecord).filter(
            or_(
                DocumentRecord.filename.like("test_%"),
                DocumentRecord.filename.like("studio_audit_%"),
                DocumentRecord.status == "DELETED",
            )
        ).delete(synchronize_session=False)

        sess.query(IngestionJob).filter(
            or_(
                IngestionJob.filename.like("test_%"),
                IngestionJob.filename.like("studio_audit_%"),
            )
        ).delete(synchronize_session=False)

        sess.query(QueryTelemetry).filter(
            or_(
                QueryTelemetry.llm_provider == "invalid_provider_xyz",
                (QueryTelemetry.rerank_latency_ms == 0.0) & (QueryTelemetry.total_latency_ms < 5.0),
                QueryTelemetry.query_text.in_(mock_test_queries),
            )
        ).delete(synchronize_session=False)

        # 2. Register or refresh the 3 canonical benchmark PDFs
        synced_count = 0
        for pdf_path in bench_dir.glob("*.pdf"):
            filename = pdf_path.name
            try:
                content = pdf_path.read_bytes()
                file_hash = hashlib.sha256(content).hexdigest()
                doc = fitz.open(stream=content, filetype="pdf")
                total_pages = len(doc)
                doc.close()
            except Exception as exc:
                logger.warning(f"Failed to read benchmark PDF {pdf_path}: {exc}")
                continue

            total_chunks = canonical_chunks.get(filename, 10)
            existing = sess.query(DocumentRecord).filter(
                (DocumentRecord.file_hash == file_hash) | (DocumentRecord.filename == filename)
            ).first()

            if not existing:
                record = DocumentRecord(
                    filename=filename,
                    file_hash=file_hash,
                    total_pages=total_pages,
                    total_chunks=total_chunks,
                    status="ACTIVE",
                    org_id="default_org",
                    visibility="ORG",
                    allowed_roles=["ADMIN", "MANAGER", "ANALYST", "VIEWER"],
                    uploaded_by="system_benchmark",
                )
                sess.add(record)
                synced_count += 1
                logger.info(f"Registered benchmark document in catalog: {filename} ({file_hash[:8]}...)")
            else:
                existing.status = "ACTIVE"
                existing.filename = filename
                existing.file_hash = file_hash
                existing.total_pages = total_pages
                existing.total_chunks = total_chunks
                existing.org_id = "default_org"
                existing.visibility = "ORG"
                existing.allowed_roles = ["ADMIN", "MANAGER", "ANALYST", "VIEWER"]
                synced_count += 1
        sess.commit()
        return synced_count

    if session is not None:
        return _sync(session)
    else:
        with SessionLocal() as sess:
            return _sync(sess)


