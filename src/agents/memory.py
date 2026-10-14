"""Conversational session memory and sliding-window context manager (L2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import ChatMessage, ChatSession
from src.database.relational import SessionLocal


def _now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class ConversationMemoryManager:
    """Manages multi-turn conversation sessions, message persistence, and sliding-window context."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        generator: Any = None,
    ) -> None:
        self.session_factory = session_factory or SessionLocal
        self.generator = generator

    def create_session(
        self,
        user_id: str,
        org_id: str = "default_org",
        title: str = "New Conversation",
    ) -> dict[str, Any]:
        """Create a new conversational session for a user within an organization."""
        with self.session_factory() as db:
            session = ChatSession(
                user_id=user_id,
                org_id=org_id,
                title=title,
                summary=None,
                created_at=_now(),
                updated_at=_now(),
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            logger.info(f"Created chat session '{session.id}' for user '{user_id}' (org: '{org_id}')")
            return session.to_dict()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a chat session record by primary key."""
        with self.session_factory() as db:
            session = db.get(ChatSession, session_id)
            return session.to_dict() if session else None

    def list_sessions(
        self,
        user_id: str,
        org_id: str = "default_org",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List active chat sessions for a user, ordered from newest to oldest."""
        with self.session_factory() as db:
            stmt = (
                select(ChatSession)
                .where(ChatSession.user_id == user_id, ChatSession.org_id == org_id)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
            )
            sessions = db.scalars(stmt).all()
            return [s.to_dict() for s in sessions]

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its associated messages."""
        with self.session_factory() as db:
            session = db.get(ChatSession, session_id)
            if not session:
                return False

            # Delete messages first
            db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
            db.delete(session)
            db.commit()
            logger.info(f"Deleted chat session '{session_id}' and all associated messages")
            return True

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Return all messages for a session ordered chronologically."""
        with self.session_factory() as db:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            if limit:
                stmt = stmt.limit(limit)
            messages = db.scalars(stmt).all()
            return [m.to_dict() for m in messages]

    def get_context_window(
        self,
        session_id: str,
        k: int = 3,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Load the sliding window of recent conversational turns.

        Args:
            session_id: Target session identifier.
            k: Number of recent turns to preserve verbatim (1 turn = 2 messages: user + assistant).

        Returns:
            Tuple of (formatted_context_string, list_of_message_dicts_in_window).
        """
        with self.session_factory() as db:
            session = db.get(ChatSession, session_id)
            if not session:
                return "", []

            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            all_messages = list(db.scalars(stmt).all())

            # Sliding window: last 2k messages (or all if fewer)
            window_size = max(1, 2 * k)
            window_messages = all_messages[-window_size:] if len(all_messages) > window_size else all_messages

            context_blocks: list[str] = []
            if session.summary and session.summary.strip():
                context_blocks.append(f"[Previous Conversation Summary: {session.summary.strip()}]")

            for msg in window_messages:
                role_label = msg.role.capitalize()
                context_blocks.append(f"{role_label}: {msg.content}")

            formatted_context = "\n\n".join(context_blocks).strip()
            return formatted_context, [m.to_dict() for m in window_messages]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[Any] | None = None,
        tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Append an individual message to a session."""
        with self.session_factory() as db:
            session = db.get(ChatSession, session_id)
            if not session:
                raise ValueError(f"ChatSession '{session_id}' not found.")

            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                citations=citations or [],
                token_count=tokens,
                latency_ms=latency_ms,
                created_at=_now(),
            )
            db.add(msg)
            session.updated_at = _now()
            db.commit()
            db.refresh(msg)
            return msg.to_dict()

    def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        citations: list[Any] | None = None,
        tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Atomically persist a complete conversational turn (user query + assistant response)."""
        with self.session_factory() as db:
            session = db.get(ChatSession, session_id)
            if not session:
                raise ValueError(f"ChatSession '{session_id}' not found.")

            user_msg = ChatMessage(
                session_id=session_id,
                role="user",
                content=user_text.strip(),
                citations=[],
                token_count=0,
                latency_ms=0.0,
                created_at=_now(),
            )
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=assistant_text.strip(),
                citations=citations or [],
                token_count=tokens,
                latency_ms=latency_ms,
                created_at=_now(),
            )
            db.add(user_msg)
            db.add(assistant_msg)

            # Auto-title if session title is still default
            if session.title in ("New Conversation", "", None):
                first_line = user_text.strip().split("\n")[0]
                session.title = first_line[:50] + ("..." if len(first_line) > 50 else "")

            session.updated_at = _now()
            db.commit()

        # Condense older turns in background if threshold exceeded
        try:
            self.maybe_summarize_background(session_id)
        except Exception as exc:
            logger.warning(f"Background summarization skipped for session '{session_id}': {exc}")

    def maybe_summarize_background(
        self,
        session_id: str,
        threshold: int = 6,
    ) -> str | None:
        """Condense turns older than the active sliding window into a rolling summary.

        Args:
            session_id: Target session identifier.
            threshold: Total message count trigger (default 6 messages = 3 turns).

        Returns:
            Updated summary string or current summary if below threshold.
        """
        with self.session_factory() as db:
            session = db.get(ChatSession, session_id)
            if not session:
                return None

            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            all_messages = list(db.scalars(stmt).all())

            # Only summarize when message count strictly exceeds threshold
            if len(all_messages) <= threshold:
                return session.summary

            # Turns older than the active 3-turn window (last 6 messages)
            older_messages = all_messages[:-6]
            if not older_messages:
                return session.summary

            dialogue_lines: list[str] = []
            for msg in older_messages:
                dialogue_lines.append(f"{msg.role.capitalize()}: {msg.content}")
            dialogue_text = "\n".join(dialogue_lines)

            # Attempt LLM condensation if available; fallback to extractive summary
            condensed = self._condense_dialogue(
                dialogue_text=dialogue_text,
                older_messages=older_messages,
                existing_summary=session.summary,
            )

            session.summary = condensed
            session.updated_at = _now()
            db.commit()
            logger.info(f"Updated rolling summary for session '{session_id}' ({len(older_messages)} turns condensed)")
            return session.summary

    def _condense_dialogue(
        self,
        dialogue_text: str,
        older_messages: list[ChatMessage],
        existing_summary: str | None,
    ) -> str:
        """Condense dialogue using LLM generator with extractive fallback."""
        try:
            generator = self.generator
            if generator is None:
                from src.generation.generator import get_generator
                generator = get_generator()

            prompt = (
                "You are an expert conversation summarizer. Condense the following prior conversation "
                "into 2-3 concise, highly factual sentences. Preserve key topics, specific entities, "
                "user questions, and conclusions. Do NOT include preambles or conversational filler.\n\n"
            )
            if existing_summary and existing_summary.strip():
                prompt += f"Existing Background Summary:\n{existing_summary.strip()}\n\n"
            prompt += f"Dialogue to condense:\n{dialogue_text}\n\nSummary:"

            answer = generator.generate(query="Summarize conversation context", context=prompt)
            cleaned = answer.strip()
            # Verify response is valid text and not an error/unreachable message
            error_keywords = ("error", "cannot connect", "not running", "failed to connect", "i cannot answer")
            if cleaned and not any(kw in cleaned.lower()[:60] for kw in error_keywords):
                return cleaned
        except Exception as exc:
            logger.debug(f"LLM summarizer unavailable ({exc}); falling back to extractive summary")

        return self._extractive_summary(older_messages, existing_summary)

    def _extractive_summary(
        self,
        older_messages: list[ChatMessage],
        existing_summary: str | None,
    ) -> str:
        """Deterministic, offline extractive summary of older conversation turns."""
        user_queries: list[str] = []
        for msg in older_messages:
            if msg.role == "user":
                clean_q = msg.content.strip().split("\n")[0]
                if clean_q:
                    snippet = clean_q[:60] + ("..." if len(clean_q) > 60 else "")
                    user_queries.append(snippet)

        topics_str = "; ".join(f"'{q}'" for q in user_queries[:4])
        if not topics_str:
            topics_str = "various technical inquiries"

        new_segment = f"Earlier discussion covered inquiries regarding: {topics_str}."
        if existing_summary and existing_summary.strip():
            return f"{existing_summary.strip()} {new_segment}"
        return new_segment
