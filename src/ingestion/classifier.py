"""Page classifier — routes PDF pages to the correct extraction path."""

from enum import Enum
from typing import Iterable

from loguru import logger

from src.config import settings
from src.ingestion.pdf_parser import PageResult


class PageType(Enum):
    """Classification of a PDF page by content type."""

    TEXT = "text"    # Rich extractable text, no significant images
    IMAGE = "image"  # Scanned/image page, needs OCR
    MIXED = "mixed"  # Some text + significant images


class PageClassifier:
    """Classifies PDF pages based on text length and image presence.

    Uses a simple heuristic: if extractable text exceeds a threshold,
    the page is text-rich. Otherwise it needs OCR. Pages with both
    text and images are classified as mixed.
    """

    def __init__(self, text_threshold: int | None = None):
        """Initialize classifier with configurable threshold.

        Args:
            text_threshold: Minimum chars to consider a page text-rich.
                            Defaults to settings.text_threshold.
        """
        self.text_threshold = text_threshold or settings.text_threshold

    def classify(self, page_result: PageResult) -> PageType:
        """Classify a single page.

        Args:
            page_result: Extraction result from PdfParser.

        Returns:
            PageType indicating the content type.
        """
        has_text = page_result.text_length >= self.text_threshold
        has_images = page_result.image_count > 0

        if has_text and not has_images:
            return PageType.TEXT
        elif not has_text:
            return PageType.IMAGE
        else:
            return PageType.MIXED

    def classify_document(
        self, pages: Iterable[PageResult]
    ) -> dict[PageType, int]:
        """Classify all pages and return a summary distribution.

        Args:
            pages: Iterable of PageResult objects.

        Returns:
            Dict mapping PageType to count of pages with that type.
        """
        summary: dict[PageType, int] = {
            PageType.TEXT: 0,
            PageType.IMAGE: 0,
            PageType.MIXED: 0,
        }

        for page in pages:
            page_type = self.classify(page)
            summary[page_type] += 1

        total = sum(summary.values())
        logger.info(
            "Classification: {} total — TEXT: {}, IMAGE: {}, MIXED: {}",
            total,
            summary[PageType.TEXT],
            summary[PageType.IMAGE],
            summary[PageType.MIXED],
        )

        return summary
