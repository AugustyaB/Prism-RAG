"""SQLAlchemy relational database engine, session factory, and initialization (L1)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.database.models import Base, Organization


def _build_engine():
    """Construct the SQLAlchemy engine with SQLite multi-thread or remote PostgreSQL compatibility."""
    db_url = (
        settings.cloud_database_url
        if (settings.storage_mode == "cloud" and settings.cloud_database_url)
        else settings.database_url
    )
    if db_url.startswith("sqlite:///"):
        rel_path = db_url.replace("sqlite:///", "", 1)
        db_file = Path(rel_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            future=True,
        )
    return create_engine(db_url, pool_pre_ping=True, future=True)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    """Create all relational tables and seed the default organization if absent."""
    settings.ensure_dirs()
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try:
            from sqlalchemy import text
            columns = [row[1] for row in conn.execute(text("PRAGMA table_info(ingestion_jobs)")).fetchall()]
            if columns and "queue_position" not in columns:
                conn.execute(text("ALTER TABLE ingestion_jobs ADD COLUMN queue_position INTEGER"))
                conn.commit()
        except Exception:
            pass
    with SessionLocal() as session:
        default_org = session.get(Organization, "default_org")
        if default_org is None:
            session.add(
                Organization(
                    id="default_org",
                    name="Default Enterprise Organization",
                )
            )
            session.commit()
            logger.info("Seeded default organization ('default_org') in relational DB")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy Session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
