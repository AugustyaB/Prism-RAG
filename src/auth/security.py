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
from src.database.models import Organization, User
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
    logger.info("Demo tenant organizations and multi-role RBAC user accounts verified")
