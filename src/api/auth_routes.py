"""Authentication and Multi-Tenant RBAC REST API routes (L4)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_user
from src.auth.security import (
    create_access_token,
    hash_password,
    seed_demo_accounts,
    verify_password,
)
from src.database.models import Organization, User
from src.database.relational import get_db

router = APIRouter(prefix="/api/auth", tags=["Authentication & RBAC"])


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
