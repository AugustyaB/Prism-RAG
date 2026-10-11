"""SQLAlchemy 2.0 ORM models for multi-tenant metadata, ingestion jobs, and telemetry (L1)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUIDv4 string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base class for all relational models."""
    pass


class Organization(Base):
    """Tenant organization entity for multi-tenant isolation."""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(Base):
    """User account scoped to an Organization with RBAC role."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    org_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id"), nullable=False, default="default_org", index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="ADMIN")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "org_id": self.org_id,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentRecord(Base):
    """Canonical relational record for an ingested PDF document and its RBAC ACLs."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    org_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id"), nullable=False, default="default_org", index=True
    )
    uploaded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    visibility: Mapped[str] = mapped_column(String(32), nullable=False, default="ORG")
    allowed_roles: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=lambda: ["ADMIN", "MANAGER", "ANALYST", "VIEWER"]
    )
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_hash": self.file_hash,
            "filename": self.filename,
            "org_id": self.org_id,
            "uploaded_by": self.uploaded_by,
            "visibility": self.visibility,
            "allowed_roles": self.allowed_roles,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class IngestionJob(Base):
    """Asynchronous PDF ingestion job state machine and progress tracker."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    document_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("documents.id"), nullable=True, index=True
    )
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    org_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id"), nullable=False, default="default_org", index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING", index=True)
    current_stage: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED")
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "id": self.id,
            "document_id": self.document_id,
            "file_hash": self.file_hash,
            "filename": self.filename,
            "org_id": self.org_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "chunks_created": self.chunks_created,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class QueryTelemetry(Base):
    """Per-query latency breakdown, token consumption, cost estimation, and abstention audit log."""

    __tablename__ = "query_telemetry"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    org_id: Mapped[str] = mapped_column(String(64), nullable=False, default="default_org", index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="hybrid_rerank")
    llm_provider: Mapped[str] = mapped_column(String(64), nullable=False, default="lmstudio")
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False, default="gpt-oss-20b")
    embed_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    search_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rerank_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    llm_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    retrieved_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    top_rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    abstention_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cited_doc_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "query_text": self.query_text,
            "retrieval_mode": self.retrieval_mode,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embed_latency_ms": self.embed_latency_ms,
            "search_latency_ms": self.search_latency_ms,
            "rerank_latency_ms": self.rerank_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "retrieved_count": self.retrieved_count,
            "top_rerank_score": self.top_rerank_score,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cited_doc_ids": self.cited_doc_ids,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
