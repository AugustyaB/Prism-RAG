"""Semantic query cache with cosine similarity matching and LRU eviction (L1).

Stores (query_text, query_emb, response_payload, org_id, role, mode) in a thread-safe
in-memory cache. Computes cosine similarity to return cached responses in <10ms
when cos(q, q_cached) >= threshold (default 0.96) within tenant org, role clearance, and mode.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, List, Optional, Tuple

from loguru import logger

from src.config import settings

ROLE_LEVELS: dict[str, int] = {
    "ADMIN": 4,
    "MANAGER": 3,
    "ANALYST": 2,
    "VIEWER": 1,
}


def compute_cosine_similarity(v1: list[float] | Any, v2: list[float] | Any) -> float:
    """Compute cosine similarity between two float vectors with zero-division guard."""
    if not v1 or not v2:
        return 0.0
    try:
        import numpy as np

        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        dot = sum(x * y for x, y in zip(v1, v2))
        norm_a = math.sqrt(sum(x * x for x in v1))
        norm_b = math.sqrt(sum(y * y for y in v2))
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            return 0.0
        return float(dot / (norm_a * norm_b))


@dataclass
class CacheEntry:
    """A cached query-response item with tenant isolation and clearance metadata."""

    query_text: str
    query_emb: list[float]
    response_payload: dict[str, Any]
    org_id: str
    role: str
    mode: str = "hybrid_rerank"
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    associated_docs: list[str] = field(default_factory=list)


class SemanticQueryCache:
    """Thread-safe LRU semantic cache for query responses."""

    def __init__(
        self,
        max_entries: int = 1000,
        similarity_threshold: float = 0.96,
    ) -> None:
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(
        self,
        query_text: str,
        query_emb: list[float],
        org_id: str = "default_org",
        role: str = "VIEWER",
        mode: str | None = None,
        threshold: float | None = None,
    ) -> Optional[Tuple[dict[str, Any], float]]:
        """Look up response by cosine similarity within tenant org and role clearance.

        Returns (response_payload, cosine_similarity) if a matching entry with
        cos >= threshold is found, otherwise None. Execution takes < 10ms.
        """
        effective_threshold = (
            threshold if threshold is not None else self.similarity_threshold
        )
        user_role_level = ROLE_LEVELS.get(role.upper(), 1)
        t_start = time.perf_counter()

        with self._lock:
            # 1. Exact string match fast-path
            for key, entry in list(self._cache.items()):
                if entry.org_id != org_id:
                    continue
                if mode and entry.mode != mode:
                    continue
                entry_role_level = ROLE_LEVELS.get(entry.role.upper(), 1)
                if user_role_level < entry_role_level:
                    continue

                if entry.query_text.strip().lower() == query_text.strip().lower():
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    self._cache.move_to_end(key)
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    logger.debug(
                        "Semantic cache exact hit for '{}' ({:.2f}ms)",
                        query_text[:40],
                        elapsed_ms,
                    )
                    payload = entry.response_payload.copy()
                    payload["cache_lookup_ms"] = round(elapsed_ms, 2)
                    return payload, 1.0

            # 2. Vector cosine similarity scan
            best_entry: Optional[CacheEntry] = None
            best_key: Optional[str] = None
            best_score = -1.0

            for key, entry in list(self._cache.items()):
                if entry.org_id != org_id:
                    continue
                if mode and entry.mode != mode:
                    continue
                entry_role_level = ROLE_LEVELS.get(entry.role.upper(), 1)
                # Clearance check: user clearance must be >= entry role clearance
                if user_role_level < entry_role_level:
                    continue

                sim = compute_cosine_similarity(query_emb, entry.query_emb)
                if sim > best_score:
                    best_score = sim
                    best_entry = entry
                    best_key = key

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            if best_entry is not None and best_score >= effective_threshold and best_key:
                best_entry.last_accessed = time.time()
                best_entry.access_count += 1
                self._cache.move_to_end(best_key)
                logger.debug(
                    "Semantic cache hit: cos={:.4f} >= {:.2f} for '{}' ({:.2f}ms)",
                    best_score,
                    effective_threshold,
                    query_text[:40],
                    elapsed_ms,
                )
                payload = best_entry.response_payload.copy()
                payload["cache_lookup_ms"] = round(elapsed_ms, 2)
                return payload, best_score

        return None

    def set(
        self,
        query_text: str,
        query_emb: list[float],
        response_payload: dict[str, Any],
        org_id: str = "default_org",
        role: str = "VIEWER",
        mode: str = "hybrid_rerank",
        associated_docs: Optional[List[str]] = None,
    ) -> None:
        """Store a query response in the LRU cache."""
        with self._lock:
            # Deduplicate by key if identical text, org, role, mode exists
            key = f"{org_id}:{role.upper()}:{mode}:{query_text.strip().lower()}"

            if key in self._cache:
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.query_emb = query_emb
                entry.response_payload = response_payload
                entry.last_accessed = time.time()
                entry.associated_docs = associated_docs or []
                return

            # LRU eviction if at capacity
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                query_text=query_text,
                query_emb=query_emb,
                response_payload=response_payload,
                org_id=org_id,
                role=role.upper(),
                mode=mode,
                associated_docs=associated_docs or [],
            )

    def invalidate_org(self, org_id: str) -> int:
        """Invalidate all cached responses for a given organization."""
        with self._lock:
            keys_to_remove = [k for k, v in self._cache.items() if v.org_id == org_id]
            for k in keys_to_remove:
                del self._cache[k]
            logger.info("Invalidated {} cache entries for org '{}'", len(keys_to_remove), org_id)
            return len(keys_to_remove)

    def invalidate_document(self, file_hash: str) -> int:
        """Invalidate any cached responses that cite or derive from a specific document."""
        with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items() if file_hash in v.associated_docs
            ]
            for k in keys_to_remove:
                del self._cache[k]
            logger.info(
                "Invalidated {} cache entries associated with doc '{}'",
                len(keys_to_remove),
                file_hash,
            )
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all entries in the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# Global singleton cache instance
_GLOBAL_SEMANTIC_CACHE: Optional[SemanticQueryCache] = None
_INIT_LOCK = threading.Lock()


def get_semantic_cache() -> SemanticQueryCache:
    """Retrieve or initialize the global thread-safe SemanticQueryCache."""
    global _GLOBAL_SEMANTIC_CACHE
    if _GLOBAL_SEMANTIC_CACHE is None:
        with _INIT_LOCK:
            if _GLOBAL_SEMANTIC_CACHE is None:
                _GLOBAL_SEMANTIC_CACHE = SemanticQueryCache(
                    max_entries=1000,
                    similarity_threshold=0.96,
                )
    return _GLOBAL_SEMANTIC_CACHE


def reset_semantic_cache() -> None:
    """Reset the global cache (useful for test isolation)."""
    global _GLOBAL_SEMANTIC_CACHE
    with _INIT_LOCK:
        if _GLOBAL_SEMANTIC_CACHE is not None:
            _GLOBAL_SEMANTIC_CACHE.clear()
        _GLOBAL_SEMANTIC_CACHE = None
