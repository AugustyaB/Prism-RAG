"""Multi-Tenant Authentication and RBAC Security Module (L2)."""

from src.auth.dependencies import (
    build_qdrant_rbac_filter,
    get_current_user,
    oauth2_scheme,
    require_role,
)
from src.auth.security import (
    RBAC_ROLES,
    create_access_token,
    decode_access_token,
    hash_password,
    seed_demo_accounts,
    verify_password,
)

__all__ = [
    "RBAC_ROLES",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "seed_demo_accounts",
    "oauth2_scheme",
    "get_current_user",
    "require_role",
    "build_qdrant_rbac_filter",
]
