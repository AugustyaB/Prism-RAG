"""PDF parser — loads PDFs and extracts text/image info per page."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from loguru import logger


@dataclass
class PdfDocument:
    """Metadata for a loaded PDF document."""

    file_path: Path
    file_hash: str
    total_pages: int
    metadata: dict = field(default_factory=dict)


@dataclass
class PageResult:
    """Extraction result for a single PDF page."""

    page_number: int
    raw_text: str
    image_count: int
    text_length: int
    source_pdf: str


class PdfParser:
    """Parses PDFs and extracts text/image info page by page.

    Uses lazy page-by-page iteration to avoid loading
    entire documents into memory.
    """

    @staticmethod
    def compute_file_hash(pdf_path: Path) -> str:
        """Compute SHA-256 hash of the PDF file for deduplication."""
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def load(pdf_path: Path) -> PdfDocument:
        """Load PDF metadata without extracting page content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            PdfDocument with file metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            RuntimeError: If the PDF cannot be opened.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        file_hash = PdfParser.compute_file_hash(pdf_path)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {pdf_path} — {e}") from e

        pdf_doc = PdfDocument(
            file_path=pdf_path,
            file_hash=file_hash,
            total_pages=len(doc),
            metadata=dict(doc.metadata) if doc.metadata else {},
        )

        doc.close()
        logger.info(
            "Loaded PDF: {} ({} pages, hash: {}...)",
            pdf_path.name,
            pdf_doc.total_pages,
            file_hash[:12],
        )
        return pdf_doc

    @staticmethod
    def extract_page(doc: fitz.Document, page_num: int, source_pdf: str) -> PageResult:
        """Extract text and image info from a single page.

        Args:
            doc: An open PyMuPDF document.
            page_num: Zero-indexed page number.
            source_pdf: Name of the source PDF file.

        Returns:
            PageResult with extracted text and image count.
        """
        page = doc[page_num]
        raw_text = page.get_text("text") or ""
        images = page.get_images(full=True)

        return PageResult(
            page_number=page_num + 1,  # 1-indexed for user display
            raw_text=raw_text,
            image_count=len(images),
            text_length=len(raw_text.strip()),
            source_pdf=source_pdf,
        )

    @staticmethod
    def iter_pages(
        pdf_path: Path, doc: fitz.Document | None = None
    ) -> Generator[PageResult, None, None]:
        """Lazily iterate over all pages in a PDF, yielding PageResult per page.

        Wraps each page extraction in try/except so a single
        corrupted page doesn't crash the entire pipeline.

        Args:
            pdf_path: Path to the PDF file.
            doc: Optional already-opened PyMuPDF Document handle to avoid re-opening.

        Yields:
            PageResult for each successfully processed page.
        """
        pdf_path = Path(pdf_path)
        source_pdf = pdf_path.name
        owns_doc = doc is None

        if owns_doc:
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as e:
                logger.error("Cannot open PDF {}: {}", pdf_path.name, e)
                return

        try:
            if doc.is_encrypted:
                logger.warning(
                    "PDF is password-protected, skipping: {}", pdf_path.name
                )
                return

            total = len(doc)
            logger.info("Iterating {} pages of {}", total, pdf_path.name)

            for page_num in range(total):
                try:
                    result = PdfParser.extract_page(doc, page_num, source_pdf)
                    if (page_num + 1) % 100 == 0 or page_num == 0:
                        logger.debug(
                            "Page {}/{} — {} chars, {} images",
                            page_num + 1,
                            total,
                            result.text_length,
                            result.image_count,
                        )
                    yield result
                except Exception as e:
                    logger.error(
                        "Error on page {}/{} of {}: {}",
                        page_num + 1,
                        total,
                        pdf_path.name,
                        e,
                    )
                    continue

            logger.info("Finished iterating {}", pdf_path.name)
        finally:
            if owns_doc and doc is not None:
                doc.close()
