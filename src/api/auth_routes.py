"""Authentication and Multi-Tenant RBAC REST API routes (L4)."""

from __future__ import annotations

from typing import Any

from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_user, require_role
from src.auth.security import (
    RBAC_ROLES,
    create_access_token,
    generate_password_reset_token,
    hash_password,
    seed_demo_accounts,
    verify_password,
    verify_password_reset_token,
)
from src.config import settings
from src.database.models import Organization, User
from src.database.relational import get_db

router = APIRouter(prefix="/api/auth", tags=["Authentication & RBAC"])


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., description="Account email address to request password reset for")


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., description="Account email address")
    reset_token: str = Field(..., description="Time-limited reset token or admin master token")
    new_password: str = Field(..., min_length=6, description="New plaintext password")


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., description="Assigned role: ADMIN, MANAGER, ANALYST, VIEWER")


class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Plaintext password")
    org_name: str | None = Field(default=None, description="Organization name (created if absent)")
    org_id: str | None = Field(default=None, description="Existing organization ID to attach to")
    role: str = Field(default="VIEWER", description="Assigned role: ADMIN, MANAGER, ANALYST, VIEWER")


class LoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="Plaintext password")


class UserResponse(BaseModel):
    id: str
    email: str
    org_id: str
    role: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(req: RegisterRequest, db: Session = Depends(get_db)) -> UserResponse:
    """Register a new user, optionally creating their tenant organization."""
    email_clean = req.email.strip().lower()
    existing_user = db.query(User).filter(User.email == email_clean).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with email '{email_clean}' already exists.",
        )

    target_org_id = req.org_id or "default_org"
    if req.org_name and req.org_name.strip():
        org_name_clean = req.org_name.strip()
        existing_org = db.query(Organization).filter(Organization.name == org_name_clean).first()
        if existing_org:
            target_org_id = existing_org.id
        else:
            new_org = Organization(name=org_name_clean)
            db.add(new_org)
            db.flush()
            target_org_id = new_org.id
    else:
        org = db.get(Organization, target_org_id)
        if not org:
            org = Organization(id=target_org_id, name=target_org_id)
            db.add(org)
            db.flush()

    role_upper = req.role.strip().upper()
    if role_upper not in {"ADMIN", "MANAGER", "ANALYST", "VIEWER"}:
        role_upper = "VIEWER"

    new_user = User(
        email=email_clean,
        password_hash=hash_password(req.password),
        org_id=target_org_id,
        role=role_upper,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        org_id=new_user.org_id,
        role=new_user.role,
        created_at=new_user.created_at.isoformat() if new_user.created_at else "",
    )


@router.post("/login", response_model=TokenResponse)
def login_user(req: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Authenticate with credentials and issue a signed multi-tenant JWT access token."""
    email_clean = req.email.strip().lower()
    user = db.query(User).filter(User.email == email_clean).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_claims = {
        "sub": user.id,
        "email": user.email,
        "org_id": user.org_id,
        "role": user.role,
    }
    access_token = create_access_token(token_claims)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=user.to_dict(),
    )


@router.get("/me")
def get_current_user_profile(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    """Return the profile, tenant organization, and RBAC tier of the authenticated user."""
    return current_user.to_dict()


@router.post("/seed")
def seed_demo_accounts_endpoint(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Seed demo tenant organizations and multi-tier RBAC benchmark accounts."""
    seed_demo_accounts()
    users = db.query(User).all()
    return {
        "message": "Demo tenant accounts successfully initialized.",
        "accounts": [u.to_dict() for u in users],
    }


@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Generate a password reset token for an existing user account."""
    email_clean = req.email.strip().lower()
    user = db.query(User).filter(User.email == email_clean).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email_clean}' not found.",
        )

    token = generate_password_reset_token(user.id, user.email)
    logger.info("Generated password reset token for {}: {}", user.email, token)

    return {
        "message": "Password reset token generated",
        "email": user.email,
        "reset_token": token,
        "master_token_hint": settings.admin_master_reset_token,
    }


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Reset user password using a verified reset token or admin master token."""
    email_clean = req.email.strip().lower()
    user = db.query(User).filter(User.email == email_clean).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email_clean}' not found.",
        )

    try:
        claims = verify_password_reset_token(req.reset_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    if claims.get("type") == "reset":
        if claims.get("email") and claims.get("email").lower() != email_clean:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token does not match the provided email address.",
            )

    user.password_hash = hash_password(req.new_password)
    db.commit()
    logger.info("Password successfully reset for {}", user.email)

    return {
        "status": "success",
        "message": "Password reset successfully. You may now log in with your new password.",
        "email": user.email,
    }


@router.get("/users")
def list_organization_users(
    current_user: User = Depends(require_role(["ADMIN", "MANAGER"])),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List team members scoped to current user's tenant organization (ADMIN or MANAGER)."""
    users = (
        db.query(User)
        .filter(User.org_id == current_user.org_id)
        .order_by(User.created_at.asc())
        .all()
    )
    return [u.to_dict() for u in users]


@router.patch("/users/{user_id}")
def update_user_role(
    user_id: str,
    req: UpdateRoleRequest,
    current_user: User = Depends(require_role(["ADMIN"])),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Update a team member's role within current tenant organization (ADMIN only)."""
    target_role = req.role.strip().upper()
    if target_role not in RBAC_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{req.role}'. Allowed roles: {list(RBAC_ROLES)}",
        )

    user = (
        db.query(User)
        .filter(User.id == user_id, User.org_id == current_user.org_id)
        .first()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in your organization.",
        )

    user.role = target_role
    db.commit()
    db.refresh(user)
    logger.info("User {} role updated to {} by Admin {}", user.email, target_role, current_user.email)
    return user.to_dict()
