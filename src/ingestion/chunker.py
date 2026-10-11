"""Semantic chunker — Cross-Page Sliding Window & Parent-Child hierarchical chunking (L3)."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Tuple

from loguru import logger

from src.config import settings


@dataclass
class TextChunk:
    """A semantic child chunk of text with parent context and page-range metadata."""

    text: str
    char_count: int
    metadata: dict = field(default_factory=dict)
    # metadata includes: source_pdf, page_number, page_end, chunk_index,
    # parent_id, parent_text, section_header


class SemanticChunker:
    """Splits text across page boundaries into ~1200-char Parent blocks and ~300-char Child chunks.

    - Stitches multi-page text into a continuous stream with a character-offset page span map
      so sentences and paragraphs crossing page breaks are never severed in half.
    - Builds Parent blocks (~1200 chars, ~150 overlap) for rich LLM grounding context.
    - Subdivides each Parent block into Child chunks (~300 chars, ~50 overlap) for high-precision
      dense + BM25 retrieval.
    """

    def __init__(
        self,
        max_size: int | None = None,
        min_size: int | None = None,
        overlap: int | None = None,
        parent_size: int | None = None,
        parent_overlap: int | None = None,
        child_size: int | None = None,
        child_overlap: int | None = None,
    ):
        """Initialize Parent-Child chunker with configurable limits."""
        self.parent_size = parent_size or max_size or settings.parent_chunk_size
        self.parent_overlap = (
            parent_overlap
            if parent_overlap is not None
            else (overlap if overlap is not None else settings.parent_overlap_size)
        )
        self.child_size = child_size or settings.child_chunk_size
        self.child_overlap = (
            child_overlap if child_overlap is not None else settings.child_overlap_size
        )
        self.max_size = max_size or self.parent_size
        self.min_size = min_size or settings.min_chunk_size
        self.overlap = overlap if overlap is not None else self.parent_overlap

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text by double newlines."""
        text = text.replace("\r\n", "\n")
        paras = re.split(r"\n{2,}", text)
        return [p.strip() for p in paras if p.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Fallback split by sentence boundaries (. ! ? followed by whitespace)."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _attempt_header_detection(self, paragraph: str) -> str | None:
        """Simple heuristic to detect if a short leading line is a section header."""
        lines = paragraph.split("\n")
        if not lines:
            return None

        first_line = lines[0].strip()
        if 0 < len(first_line) < 60 and not first_line.endswith((".", ":", ",")):
            if first_line.isupper() or re.match(r"^\d+(\.\d+)*\s+[A-Z]", first_line, re.I):
                return first_line

        return None

    def _slice_with_overlap(
        self,
        text: str,
        target_size: int,
        overlap_size: int,
    ) -> list[tuple[int, int, str]]:
        """Split `text` into segments of at most `target_size` chars with `overlap_size` overlap,
        returning `[(start_offset, end_offset, segment_text)]` relative to `text`.
        """
        n = len(text)
        if n == 0:
            return []
        if n <= target_size:
            stripped = text.strip()
            return [(0, n, stripped)] if stripped else []

        # Collect atomic candidate break positions (paragraph breaks, sentence breaks, spaces)
        segments: list[tuple[int, int, str]] = []
        start = 0

        while start < n:
            # Skip leading whitespace
            while start < n and text[start].isspace():
                start += 1
            if start >= n:
                break

            ideal_end = min(start + target_size, n)
            if ideal_end == n:
                seg_text = text[start:n].strip()
                if seg_text:
                    segments.append((start, n, seg_text))
                break

            window = text[start:ideal_end]
            cut_offset = -1

            # 1. Prefer double newline (paragraph break) in the second half of the window
            para_pos = window.rfind("\n\n")
            if para_pos >= target_size // 3:
                cut_offset = start + para_pos

            # 2. Fall back to sentence boundary (. ! ? followed by space or newline)
            if cut_offset == -1:
                sent_matches = list(re.finditer(r"[.!?](?=\s)", window))
                if sent_matches and sent_matches[-1].end() >= target_size // 4:
                    cut_offset = start + sent_matches[-1].end()

            # 3. Fall back to line break or word boundary (whitespace)
            if cut_offset == -1:
                space_pos = max(window.rfind("\n"), window.rfind(" "))
                if space_pos > 0:
                    cut_offset = start + space_pos
                else:
                    cut_offset = ideal_end

            cut_offset = max(start + 1, cut_offset)
            seg_text = text[start:cut_offset].strip()
            if seg_text:
                segments.append((start, cut_offset, seg_text))

            if cut_offset >= n:
                break

            # Advance `start` keeping `overlap_size` characters of context
            if overlap_size > 0:
                next_start = max(start + 1, cut_offset - overlap_size)
                # Snap forward to word/line boundary so we never cut a token in half
                if next_start > start and next_start < cut_offset:
                    candidates = [
                        idx
                        for idx in (
                            text.find(" ", next_start, cut_offset),
                            text.find("\n", next_start, cut_offset),
                        )
                        if idx != -1
                    ]
                    if candidates:
                        next_start = min(candidates) + 1
                start = next_start
            else:
                start = cut_offset

        return segments

    @staticmethod
    def _resolve_page_range(
        start_char: int,
        end_char: int,
        page_spans: list[tuple[int, int, int]],
    ) -> tuple[int, int]:
        """Resolve `(page_number, page_end)` for a character interval `[start_char, end_char)`."""
        if not page_spans:
            return (1, 1)

        pages_hit = [
            page_num
            for span_start, span_end, page_num in page_spans
            if span_end > start_char and span_start < end_char
        ]
        if not pages_hit:
            return (page_spans[0][2], page_spans[0][2])
        return (min(pages_hit), max(pages_hit))

    def _detect_latest_header(self, text_up_to_block: str, block_text: str) -> str:
        """Detect the active section header for a parent block."""
        for para in self._split_into_paragraphs(block_text):
            hdr = self._attempt_header_detection(para)
            if hdr:
                return hdr
        for para in reversed(self._split_into_paragraphs(text_up_to_block)):
            hdr = self._attempt_header_detection(para)
            if hdr:
                return hdr
        return ""

    def chunk_text(self, text: str, metadata_base: dict) -> List[TextChunk]:
        """Chunk a single text string into Parent-Child `TextChunk` objects."""
        cleaned = text.replace("\r\n", "\n").strip()
        if not cleaned or len(cleaned) < 10:
            return []

        source_pdf = metadata_base.get("source_pdf", "unknown.pdf")
        page_num = int(metadata_base.get("page_number", 1))
        page_end = int(metadata_base.get("page_end", page_num))
        start_chunk_idx = int(metadata_base.get("chunk_index", 0))

        page_spans = [(0, len(cleaned), page_num)]
        chunks = self._build_parent_child_chunks(
            full_text=cleaned,
            page_spans=page_spans,
            source_pdf=source_pdf,
            base_metadata=metadata_base,
            start_chunk_idx=start_chunk_idx,
            override_page_end=page_end if page_end > page_num else None,
        )
        return chunks

    def _build_parent_child_chunks(
        self,
        full_text: str,
        page_spans: list[tuple[int, int, int]],
        source_pdf: str,
        base_metadata: dict | None = None,
        start_chunk_idx: int = 0,
        override_page_end: int | None = None,
    ) -> List[TextChunk]:
        """Build ~1200-char Parent blocks from `full_text` and subdivide into ~300-char Child chunks."""
        parent_segments = self._slice_with_overlap(
            full_text,
            target_size=self.parent_size,
            overlap_size=self.parent_overlap,
        )

        all_children: List[TextChunk] = []
        chunk_idx = start_chunk_idx

        for parent_idx, (p_start, p_end, parent_text) in enumerate(parent_segments):
            if len(parent_text.strip()) < 10:
                continue

            page_start, resolved_page_end = self._resolve_page_range(
                p_start, p_end, page_spans
            )
            if override_page_end is not None and override_page_end > resolved_page_end:
                resolved_page_end = override_page_end

            parent_id = f"{source_pdf}::parent_{parent_idx}"
            section_header = self._detect_latest_header(full_text[:p_start], parent_text)

            child_segments = self._slice_with_overlap(
                parent_text,
                target_size=self.child_size,
                overlap_size=self.child_overlap,
            )

            for _, _, child_text in child_segments:
                cleaned_child = child_text.strip()
                if len(cleaned_child) < 10:
                    continue

                meta = (base_metadata or {}).copy()
                meta.update(
                    {
                        "source_pdf": source_pdf,
                        "page_number": page_start,
                        "page_end": resolved_page_end,
                        "chunk_index": chunk_idx,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                    }
                )
                if section_header:
                    meta["section_header"] = section_header

                all_children.append(
                    TextChunk(
                        text=cleaned_child,
                        char_count=len(cleaned_child),
                        metadata=meta,
                    )
                )
                chunk_idx += 1

        return all_children

    def chunk_pages(
        self, pages: List[Tuple[int, str, str]]
    ) -> List[TextChunk]:
        """Stitch multiple pages into a continuous cross-page stream, build Parent blocks,
        and subdivide into Child chunks tagged with `parent_id`, `parent_text`, `page_number`, and `page_end`.
        """
        if not pages:
            return []

        source_pdf = pages[0][2] if pages[0][2] else "unknown.pdf"
        stream_parts: list[str] = []
        page_spans: list[tuple[int, int, int]] = []
        cursor = 0

        for page_num, raw_text, pdf_name in pages:
            if pdf_name:
                source_pdf = pdf_name
            cleaned_page = (raw_text or "").replace("\r\n", "\n").strip()
            if not cleaned_page:
                continue

            if stream_parts:
                # Join across page boundary with a space/newline so sentences crossing pages stay unified
                sep = " " if not stream_parts[-1].endswith(("\n", ".", "!", "?")) else "\n"
                stream_parts.append(sep)
                cursor += len(sep)

            start_char = cursor
            stream_parts.append(cleaned_page)
            cursor += len(cleaned_page)
            page_spans.append((start_char, cursor, int(page_num)))

        if not stream_parts:
            return []

        full_text = "".join(stream_parts)
        all_chunks = self._build_parent_child_chunks(
            full_text=full_text,
            page_spans=page_spans,
            source_pdf=source_pdf,
            start_chunk_idx=0,
        )

        logger.info(
            "Chunked {} pages into {} child chunks (avg child size: {:.0f} chars)",
            len(pages),
            len(all_chunks),
            sum(c.char_count for c in all_chunks) / len(all_chunks) if all_chunks else 0,
        )
        return all_chunks
