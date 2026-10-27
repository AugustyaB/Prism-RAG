"""Deterministic BM25 Sparse Vector Encoder for Qdrant Hybrid Search (L3)."""

from __future__ import annotations

from collections import Counter
import re
import zlib

# Standard lightweight English stopwords excluded for plain natural-language words
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
        "their", "there", "these", "they", "this", "to", "was", "were", "will",
        "with", "what", "when", "where", "which", "who", "how", "can", "could",
        "should", "would", "do", "does", "did",
    }
)

# Matches hyphenated/dotted codes (ERR-CV-4092, POL-SEC-104.2), snake_case identifiers
# (_render_page_to_image), or standard alphanumeric tokens.
_TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)+)|(?:_[A-Za-z0-9_]+)|(?:[A-Za-z0-9]+)"
)


class SparseBM25Encoder:
    """Encodes text into deterministic sparse term-frequency vectors with BM25+ lower-bound saturation (L3).

    Preserves exact technical identifiers (`ERR-CV-4092`, `POL-SEC-104.2`, `_render_page_to_image`)
    alongside constituent sub-tokens with a delta lower-bound (BM25+) to prevent length penalty starvation.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        avg_dl: float = 65.0,
        delta: float = 0.5,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.avg_dl = avg_dl
        self.delta = delta

    @staticmethod
    def _token_to_index(token: str) -> int:
        """Map a normalized token string to a deterministic 31-bit positive integer index."""
        return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF

    def tokenize(self, text: str) -> list[str]:
        """Extract technical codes, snake_case symbols, and filtered natural language tokens."""
        if not text or not text.strip():
            return []

        tokens: list[str] = []
        raw_fallback: list[str] = []
        for match in _TOKEN_PATTERN.finditer(text):
            raw = match.group(0)
            norm = raw.lower()
            is_compound = any(sep in norm for sep in ("-", "_", "."))

            if is_compound:
                # Preserve exact compound identifier (e.g., 'err-cv-4092', '_render_page_to_image')
                tokens.append(norm)
                raw_fallback.append(norm)
                # Also emit constituent alphanumeric sub-parts for partial identifier recall
                for sub in re.split(r"[-_.]", norm):
                    if sub and (len(sub) > 1 or sub.isdigit()):
                        raw_fallback.append(sub)
                        if sub not in _STOPWORDS:
                            tokens.append(sub)
            else:
                if len(norm) > 1 or norm.isdigit():
                    raw_fallback.append(norm)
                    if norm not in _STOPWORDS:
                        tokens.append(norm)

        return tokens if tokens else raw_fallback

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """Compute saturated BM25 term-frequency sparse vector `(indices, values)` for a document chunk."""
        tokens = self.tokenize(text)
        if not tokens:
            return ([], [])

        doc_len = float(len(tokens))
        idx_counts: Counter[int] = Counter(self._token_to_index(tok) for tok in tokens)

        indices: list[int] = []
        values: list[float] = []
        norm_factor = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_dl))

        for idx in sorted(idx_counts.keys()):
            tf = float(idx_counts[idx])
            weight = (((self.k1 + 1.0) * tf) / (norm_factor + tf)) + self.delta
            indices.append(idx)
            values.append(round(float(weight), 6))

        return (indices, values)

    def encode_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Encode a batch of document texts into sparse `(indices, values)` tuples."""
        return [self.encode_document(t) for t in texts]

    def encode_query(self, query: str) -> tuple[list[int], list[float]]:
        """Encode a search query into sparse `(indices, values)` tuples for Qdrant IDF scoring."""
        tokens = self.tokenize(query)
        if not tokens:
            return ([], [])

        idx_counts: Counter[int] = Counter(self._token_to_index(tok) for tok in tokens)
        indices: list[int] = []
        values: list[float] = []

        for idx in sorted(idx_counts.keys()):
            indices.append(idx)
            values.append(float(idx_counts[idx]))

        return (indices, values)
