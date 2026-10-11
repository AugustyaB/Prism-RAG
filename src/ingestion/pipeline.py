"""Main ingestion pipeline wiring parser, classifier, OCR, and chunker."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Dict

import fitz
import psutil
from loguru import logger

from src.config import settings
from src.ingestion.pdf_parser import PdfParser
from src.ingestion.classifier import PageClassifier, PageType
from src.ingestion.ocr import OcrEngine
from src.ingestion.chunker import SemanticChunker, TextChunk


@dataclass
class IngestionResult:
    """Consolidated result of a single PDF ingestion."""
    
    source_pdf: str
    file_hash: str
    total_pages: int
    classification_summary: Dict[PageType, int]
    total_chunks: int
    chunks: List[TextChunk]
    errors: List[str] = field(default_factory=list)


class IngestionPipeline:
    """Wires together the end-to-end PDF ingestion flow."""

    def __init__(self):
        """Initialize the pipeline components."""
        self.parser = PdfParser()
        self.classifier = PageClassifier()
        self.chunker = SemanticChunker()
        self.ocr_engine = None  # Lazy load

    def _check_memory(self):
        """Memory guard failsafe."""
        mem = psutil.virtual_memory()
        percent = mem.percent
        if percent > settings.memory_warn_percent:
            logger.warning(
                "Memory pressure high: {:.1f}% used. Available: {:.1f}GB", 
                percent, 
                mem.available / (1024**3)
            )

    def get_ocr_engine(self) -> OcrEngine:
        """Lazily initialize OCR only if an image/mixed page is detected."""
        if self.ocr_engine is None:
            self.ocr_engine = OcrEngine()
        return self.ocr_engine

    def process_pdf(
        self,
        pdf_path: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> IngestionResult:
        """Process a single PDF from path to chunks using a single fitz.open handle."""
        pdf_path = Path(pdf_path)
        
        # Failsafe: basic file check
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        logger.info("Starting ingestion for {}", pdf_path.name)
        
        # 1. Compute SHA-256 hash once
        file_hash = self.parser.compute_file_hash(pdf_path)
        source_pdf = pdf_path.name
        
        extracted_pages: List[tuple[int, str, str]] = []
        classifications: Dict[PageType, int] = {
            PageType.TEXT: 0,
            PageType.IMAGE: 0,
            PageType.MIXED: 0,
        }
        errors: List[str] = []
        total_pages = 0

        # 2. Open PDF once and iterate pages directly
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {pdf_path} — {e}") from e

        try:
            if doc.is_encrypted:
                logger.warning("PDF is password-protected, skipping: {}", source_pdf)
                errors.append(f"PDF is password-protected: {source_pdf}")
            else:
                total_pages = len(doc)
                if progress_callback:
                    progress_callback(0, total_pages, "PARSING")

                for idx in range(total_pages):
                    self._check_memory()
                    page_num = idx + 1
                    try:
                        page_result = self.parser.extract_page(doc, idx, source_pdf)
                        page_type = self.classifier.classify(page_result)
                        classifications[page_type] += 1

                        stage = "PARSING" if page_type == PageType.TEXT else "OCR"
                        if progress_callback:
                            progress_callback(idx, total_pages, stage)

                        final_text = ""
                        if page_type == PageType.TEXT:
                            final_text = page_result.raw_text
                        elif page_type == PageType.IMAGE:
                            ocr = self.get_ocr_engine()
                            final_text = ocr.extract_text(doc[idx], page_num)
                        elif page_type == PageType.MIXED:
                            ocr = self.get_ocr_engine()
                            final_text = ocr.extract_from_regions(
                                doc[idx],
                                page_num,
                                page_result.raw_text,
                            )

                        extracted_pages.append((page_num, final_text, source_pdf))
                        if progress_callback:
                            progress_callback(page_num, total_pages, stage)

                    except Exception as e:
                        err_msg = f"Failed to route/extract page {page_num}: {e}"
                        logger.error(err_msg)
                        errors.append(err_msg)
        finally:
            doc.close()

        # 3. Chunk collected text
        if progress_callback:
            progress_callback(total_pages, total_pages, "CHUNKING")
        chunks = self.chunker.chunk_pages(extracted_pages)
        
        result = IngestionResult(
            source_pdf=source_pdf,
            file_hash=file_hash,
            total_pages=total_pages,
            classification_summary=classifications,
            total_chunks=len(chunks),
            chunks=chunks,
            errors=errors
        )
        
        logger.info(
            "Completed ingestion for {} — Chunks: {}, Errors: {}",
            result.source_pdf,
            result.total_chunks,
            len(result.errors)
        )
        return result

    def process_directory(self, dir_path: Path) -> List[IngestionResult]:
        """Process all PDFs in a directory."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Is not a directory: {dir_path}")
            
        results = []
        pdf_files = list(dir_path.glob("*.pdf"))
        
        logger.info("Found {} PDF files in {}", len(pdf_files), dir_path)
        
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf(pdf_file)
                results.append(result)
            except Exception as e:
                logger.error("Failed to process PDF {}: {}", pdf_file.name, e)
                
        return results
