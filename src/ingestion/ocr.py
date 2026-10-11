"""OCR engine — GPU-accelerated text extraction from image-based PDF pages."""

import signal
import numpy as np
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from loguru import logger

from src.config import settings


class OcrTimeoutError(Exception):
    """Raised when OCR exceeds the configured timeout."""
    pass


class OcrEngine:
    """GPU-accelerated OCR using EasyOCR.

    Lazily initializes the EasyOCR reader on first use to avoid
    loading the model if only text-rich pages are processed.
    """

    def __init__(
        self,
        gpu: bool | None = None,
        languages: list[str] | None = None,
    ):
        """Initialize OCR engine configuration.

        Args:
            gpu: Use GPU for OCR. Defaults to settings.ocr_gpu.
            languages: OCR languages. Defaults to settings.ocr_languages.
        """
        import torch
        requested_gpu = gpu if gpu is not None else settings.ocr_gpu
        self.gpu = requested_gpu and torch.cuda.is_available()
        if requested_gpu and not self.gpu:
            logger.warning("GPU requested, but Torch is not compiled with CUDA. Falling back to CPU.")
            
        self.languages = languages or settings.ocr_languages
        self._reader = None  # Lazy init

    @property
    def reader(self):
        """Lazily initialize EasyOCR reader on first access."""
        if self._reader is None:
            logger.info(
                "Initializing EasyOCR (GPU={}, languages={})",
                self.gpu,
                self.languages,
            )
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                )
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize EasyOCR: {}", e)
                raise
        return self._reader

    def _render_page_to_image(
        self, page: fitz.Page, dpi: int | None = None
    ) -> np.ndarray:
        """Render a PDF page to a numpy array image.

        Args:
            page: PyMuPDF page object.
            dpi: Resolution for rendering. Defaults to settings.ocr_dpi.

        Returns:
            Numpy array of the rendered page image.
        """
        dpi = dpi or settings.ocr_dpi
        zoom = dpi / 72  # 72 is the default PDF DPI
        matrix = fitz.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(matrix=matrix)

        # Memory guard: check pixmap size
        pixmap_mb = len(pixmap.samples) / (1024 * 1024)
        if pixmap_mb > settings.ocr_max_pixmap_mb:
            logger.warning(
                "Pixmap too large ({:.1f}MB), reducing DPI from {} to {}",
                pixmap_mb,
                dpi,
                dpi // 2,
            )
            pixmap = None  # Free memory
            return self._render_page_to_image(page, dpi=dpi // 2)

        # Convert pixmap to numpy array
        img = Image.open(BytesIO(pixmap.tobytes("png")))
        return np.array(img)

    def _readtext_with_timeout(self, img_array: np.ndarray) -> list[str]:
        """Run EasyOCR readtext bounded by settings.ocr_timeout_seconds."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        timeout_s = max(1, int(settings.ocr_timeout_seconds))
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.reader.readtext, img_array, detail=0)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise OcrTimeoutError(
                f"OCR exceeded timeout of {timeout_s}s"
            ) from exc
        finally:
            executor.shutdown(wait=False)

    def extract_text(self, page: fitz.Page, page_num: int) -> str:
        """Extract text from a full page image using OCR.

        Args:
            page: PyMuPDF page object.
            page_num: Page number for logging (1-indexed).

        Returns:
            Extracted text or empty string on failure.
        """
        try:
            img_array = self._render_page_to_image(page)

            results = self._readtext_with_timeout(img_array)

            text = " ".join(results)
            logger.debug(
                "OCR page {}: extracted {} chars",
                page_num,
                len(text),
            )
            return text

        except OcrTimeoutError:
            logger.warning("OCR timeout on page {}, skipping", page_num)
            return ""
        except Exception as e:
            logger.error("OCR failed on page {}: {}", page_num, e)
            return ""

    def extract_from_regions(
        self,
        page: fitz.Page,
        page_num: int,
        existing_text: str,
    ) -> str:
        """Extract text from image regions only (for MIXED pages).

        Identifies image bounding boxes on the page and runs OCR
        only on those regions, combining with existing extracted text.

        Args:
            page: PyMuPDF page object.
            page_num: Page number for logging (1-indexed).
            existing_text: Text already extracted via PyMuPDF.

        Returns:
            Combined text (existing + OCR from image regions).
        """
        try:
            images = page.get_images(full=True)
            if not images:
                return existing_text

            ocr_texts = []
            for img_index, img_info in enumerate(images):
                try:
                    xref = img_info[0]
                    base_image = page.parent.extract_image(xref)
                    if base_image is None:
                        continue

                    image_bytes = base_image["image"]
                    img = Image.open(BytesIO(image_bytes))
                    img_array = np.array(img)

                    # Skip tiny images (icons, decorations)
                    if img_array.shape[0] < 50 or img_array.shape[1] < 50:
                        continue

                    results = self._readtext_with_timeout(img_array)
                    if results:
                        ocr_texts.append(" ".join(results))

                except Exception as e:
                    logger.debug(
                        "Failed to OCR image {} on page {}: {}",
                        img_index,
                        page_num,
                        e,
                    )
                    continue

            if ocr_texts:
                combined = existing_text + "\n\n[OCR from images]\n" + "\n".join(ocr_texts)
                logger.debug(
                    "MIXED page {}: {} images OCR'd, {} extra chars",
                    page_num,
                    len(ocr_texts),
                    sum(len(t) for t in ocr_texts),
                )
                return combined

            return existing_text

        except Exception as e:
            logger.error(
                "Region OCR failed on page {}: {}", page_num, e
            )
            return existing_text
