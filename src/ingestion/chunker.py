"""Semantic chunker — splits text into coherent, metadata-tagged chunks."""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from loguru import logger

from src.config import settings


@dataclass
class TextChunk:
    """A semantic chunk of text with metadata."""
    
    text: str
    char_count: int
    metadata: dict = field(default_factory=dict)
    # metadata includes: source_pdf, page_number, chunk_index, section_header


class SemanticChunker:
    """Splits text into chunks prioritizing structural boundaries.
    
    Uses double newlines (paragraphs) as primary boundaries. Falls back
    to sentences for oversized paragraphs. Merges small paragraphs.
    Maintains a rolling overlap for context continuity.
    """

    def __init__(
        self,
        max_size: int | None = None,
        min_size: int | None = None,
        overlap: int | None = None,
    ):
        """Initialize chunker with configurable limits.
        
        Args:
            max_size: Max characters per chunk. Defaults to settings.
            min_size: Min characters per chunk. Defaults to settings.
            overlap: Character overlap between consecutive chunks.
        """
        self.max_size = max_size or settings.max_chunk_size
        self.min_size = min_size or settings.min_chunk_size
        self.overlap = overlap or settings.overlap_size

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text by double newlines."""
        # Normalize line endings
        text = text.replace("\r\n", "\n")
        # Split by 2 or more newlines
        paras = re.split(r'\n{2,}', text)
        return [p.strip() for p in paras if p.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Fallback split by basic sentence boundaries."""
        # Basic regex for sentence split (. ! ?) followed by space and Capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _attempt_header_detection(self, paragraph: str) -> str | None:
        """Simple heuristic to detect if a short line is a header."""
        lines = paragraph.split('\n')
        if not lines:
            return None
        
        first_line = lines[0].strip()
        # If the first line is short, doesn't end with punctuation, and is all caps
        # or starts with a number (e.g. "1. Introduction"), it might be a header.
        if (
            len(first_line) < 60 
            and not first_line.endswith(('.', ':', ','))
        ):
            if first_line.isupper() or re.match(r'^\d+(\.\d+)*\s+[A-Z]', first_line, re.I):
                return first_line
        
        return None

    def chunk_text(self, text: str, metadata_base: dict) -> List[TextChunk]:
        """Chunk a single block of text (typically a page).
        
        Args:
            text: Raw text to chunk.
            metadata_base: Base metadata (e.g., source_pdf, page_number).
            
        Returns:
            List of TextChunk objects.
        """
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk_text = ""
        current_header = ""

        # Start chunk index counter (can be refined across pages later)
        chunk_idx = metadata_base.get("chunk_index", 0)

        for para in paragraphs:
            # Check if this paragraph starts with a header
            potential_header = self._attempt_header_detection(para)
            if potential_header:
                current_header = potential_header

            # Strategy: if adding paragraph to current chunk exceeds max size,
            # finalize current chunk and start a new one.
            potential_len = len(current_chunk_text) + (1 if current_chunk_text else 0) + len(para)
            
            if potential_len > self.max_size and current_chunk_text:
                # Need to push the current chunk before adding the large paragraph
                # But check if current_chunk is at least min_size
                if len(current_chunk_text) >= self.min_size:
                    meta = metadata_base.copy()
                    meta["chunk_index"] = chunk_idx
                    if current_header:
                        meta["section_header"] = current_header
                    
                    chunks.append(
                        TextChunk(
                            text=current_chunk_text,
                            char_count=len(current_chunk_text),
                            metadata=meta
                        )
                    )
                    chunk_idx += 1
                    
                    # Start new chunk with overlap from the END of the current chunk
                    overlap_text = current_chunk_text[-self.overlap:] if self.overlap > 0 else ""
                    # Ensure we don't snap words in half if possible (find previous space)
                    if overlap_text:
                        space_idx = overlap_text.find(' ')
                        if space_idx != -1 and space_idx < len(overlap_text) // 2:
                            overlap_text = overlap_text[space_idx+1:]
                    
                    current_chunk_text = overlap_text + ("\n" if overlap_text else "") + para
                else:
                    # Current chunk is very small, force append and accept breaking max_size slightly, 
                    # OR we split the paragraph. Let's just append for context continuity.
                    current_chunk_text += ("\n" if current_chunk_text else "") + para
            
            elif potential_len > self.max_size and not current_chunk_text:
                # The single paragraph ITSELF is larger than max_size.
                # Fall back to sentence splitting.
                sentences = self._split_into_sentences(para)
                temp_chunk = ""
                for sentence in sentences:
                    temp_len = len(temp_chunk) + (1 if temp_chunk else 0) + len(sentence)
                    if temp_len > self.max_size and temp_chunk:
                        meta = metadata_base.copy()
                        meta["chunk_index"] = chunk_idx
                        if current_header:
                            meta["section_header"] = current_header
                        
                        chunks.append(
                            TextChunk(text=temp_chunk, char_count=len(temp_chunk), metadata=meta)
                        )
                        chunk_idx += 1
                        
                        overlap_text = temp_chunk[-self.overlap:] if self.overlap > 0 else ""
                        temp_chunk = overlap_text + (" " if overlap_text else "") + sentence
                    else:
                        temp_chunk += (" " if temp_chunk else "") + sentence
                
                # Assign what's left to current_chunk_text
                current_chunk_text = temp_chunk
            else:
                # Normal accumulation
                current_chunk_text += ("\n" if current_chunk_text else "") + para

        # Flush the final chunk
        if current_chunk_text.strip() and len(current_chunk_text.strip()) >= 10: # Noise guard
            meta = metadata_base.copy()
            meta["chunk_index"] = chunk_idx
            if current_header:
                meta["section_header"] = current_header
            
            cleaned_text = current_chunk_text.strip()
            chunks.append(
                TextChunk(text=cleaned_text, char_count=len(cleaned_text), metadata=meta)
            )

        return chunks

    def chunk_pages(
        self, pages: List[Tuple[int, str, str]]
    ) -> List[TextChunk]:
        """Process multiple pages into cross-page continuous chunks.
        
        Args:
            pages: List of (page_num, text, source_pdf).
            
        Returns:
            Flat list of all TextChunks generated.
        """
        all_chunks = []
        global_chunk_idx = 0
        
        # We concatenate text first and maintain a mapping, or simply
        # chunk per page and allow minor breaks. For simplicity in V1,
        # chunking per page with metadata is safer for precise citations.
        
        for page_num, text, source_pdf in pages:
            if not text.strip():
                continue
                
            metadata_base = {
                "source_pdf": source_pdf,
                "page_number": page_num,
                "chunk_index": global_chunk_idx
            }
            
            page_chunks = self.chunk_text(text, metadata_base)
            
            if page_chunks:
                all_chunks.extend(page_chunks)
                global_chunk_idx += len(page_chunks)
                
        logger.info(
            "Chunked {} pages into {} chunks (avg size: {:.0f} chars)",
            len(pages),
            len(all_chunks),
            sum(c.char_count for c in all_chunks) / len(all_chunks) if all_chunks else 0
        )
        return all_chunks
