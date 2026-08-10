"""Deterministic text extraction from PDF and DOCX (build prompt §9).

No model is involved. pdfplumber first; if a page yields too little text to be
a real text layer, that page is rasterised and passed to Tesseract. The density
threshold is the whole point of the fallback: a scanned RFP produces a valid PDF
with almost no extractable text, and silently returning an empty string would
look like an RFP with no questions rather than like a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from src.contracts.enums import DocumentFormat

#: Characters per page below which we stop believing the text layer.
#: A genuine text page of an RFP carries hundreds; a scan carries a handful of
#: stray marks.
TEXT_DENSITY_THRESHOLD = 120

#: Resolution used when rasterising a page for OCR.
OCR_DPI = 300


@dataclass(frozen=True)
class ExtractedText:
    """Raw text plus how it was obtained.

    `ocr_pages` is carried forward rather than discarded because OCR text is
    materially less reliable, and a downstream extraction failure on an OCR'd
    document deserves a different diagnosis than one on a text-layer document.
    """

    text: str
    document_format: DocumentFormat
    page_count: int
    ocr_pages: tuple[int, ...] = ()

    @property
    def used_ocr(self) -> bool:
        return bool(self.ocr_pages)


class ExtractionError(RuntimeError):
    """The document could not be read at all."""


def _ocr_page(page: object) -> str:
    """OCR one pdfplumber page. Imported lazily so Tesseract is optional.

    Tesseract is a system binary, not a Python dependency. A repository that
    only ever ingests text-layer PDFs should not fail to import because it is
    missing.
    """
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ExtractionError(
            "page has no usable text layer and pytesseract is not installed"
        ) from exc

    image = page.to_image(resolution=OCR_DPI).original  # type: ignore[attr-defined]
    return str(pytesseract.image_to_string(image))


def extract_pdf(path: Path, *, ocr_enabled: bool = True) -> ExtractedText:
    """Text from a PDF, falling back to OCR per page where the layer is thin."""
    pages: list[str] = []
    ocr_pages: list[int] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for index, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < TEXT_DENSITY_THRESHOLD and ocr_enabled:
                    try:
                        ocr_text = _ocr_page(page)
                    except ExtractionError:
                        ocr_text = ""
                    # Only prefer OCR if it actually produced more than the layer.
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        ocr_pages.append(index)
                pages.append(text)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"could not read PDF {path}: {exc}") from exc

    return ExtractedText(
        text="\n".join(pages),
        document_format=DocumentFormat.PDF,
        page_count=len(pages),
        ocr_pages=tuple(ocr_pages),
    )


def extract_docx(path: Path) -> ExtractedText:
    """Text from a DOCX, paragraph per line."""
    try:
        from docx import Document

        document = Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"could not read DOCX {path}: {exc}") from exc

    lines = [paragraph.text for paragraph in document.paragraphs]
    return ExtractedText(
        text="\n".join(lines),
        document_format=DocumentFormat.DOCX,
        page_count=1,
    )


def extract(path: Path, *, ocr_enabled: bool = True) -> ExtractedText:
    """Dispatch on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, ocr_enabled=ocr_enabled)
    if suffix == ".docx":
        return extract_docx(path)
    raise ExtractionError(f"unsupported document format: {suffix or '(no extension)'}")
