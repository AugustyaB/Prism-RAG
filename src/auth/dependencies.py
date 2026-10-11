"""FastAPI authentication dependencies and pre-retrieval Qdrant RBAC security filters (L2)."""

from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from loguru import logger
from qdrant_client import models
from sqlalchemy.orm import Session

from src.auth.security import decode_access_token
from src.config import settings
from src.database.models import User
from src.database.relational import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency resolving the authenticated User from Bearer token claims."""
    # When auth is globally disabled and no token is supplied, return default admin user
    if not settings.auth_enabled and token is None:
        admin_user = db.query(User).filter(User.email == "admin@default.org").first()
        if admin_user:
            return admin_user
        return User(
            id="default_admin",
            email="admin@default.org",
            password_hash="",
            org_id="default_org",
            role="ADMIN",
        )

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Signature has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.InvalidTokenError, Exception) as exc:
        logger.warning(f"Invalid access token rejected: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str | None = payload.get("sub")
    org_id: str | None = payload.get("org_id")
    role: str | None = payload.get("role")

    if not user_id or not org_id or not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        # Check by email in case subject was issued as email
        user = db.query(User).filter(User.email == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_role(required_roles: list[str]) -> Callable[..., User]:
    """FastAPI dependency factory enforcing that current user possesses an authorized role."""

    def role_verifier(current_user: User = Depends(get_current_user)) -> User:
        req_normalized = {r.upper() for r in required_roles}
        if current_user.role.upper() not in req_normalized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: insufficient role permissions",
            )
        return current_user

    return role_verifier


def build_qdrant_rbac_filter(user: User) -> models.Filter:
    """Construct pre-retrieval Qdrant security filter enforcing multi-tenant isolation and RBAC.

    - Strict multi-tenant isolation: payload 'org_id' == user.org_id.
    - If user.role == 'ADMIN': tenant match alone is sufficient.
    - If user.role in {'MANAGER', 'ANALYST', 'VIEWER'}:
      Payload must also satisfy: visibility == 'PUBLIC' OR allowed_roles CONTAINS user.role.
    """
    must_conditions = [
        models.FieldCondition(
            key="org_id",
            match=models.MatchValue(value=user.org_id),
        )
    ]

    if user.role.upper() == "ADMIN":
        return models.Filter(must=must_conditions)

    should_conditions = [
        models.FieldCondition(
            key="visibility",
            match=models.MatchValue(value="PUBLIC"),
        ),
        models.FieldCondition(
            key="allowed_roles",
            match=models.MatchAny(any=[user.role.upper()]),
        ),
    ]

    return models.Filter(must=must_conditions, should=should_conditions)
