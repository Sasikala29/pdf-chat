"""PDF validation and page-aware text extraction services."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from src.pdf_chatbot.core.config import get_settings


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be validated or processed."""


class ExtractedPage(BaseModel):
    """Text extracted from one PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(
        ge=1,
        description="One-based PDF page number",
    )
    text: str = Field(
        description="Extracted text with meaningful line breaks preserved",
    )


class PDFDocument(BaseModel):
    """Metadata and extracted pages from a PDF document."""

    model_config = ConfigDict(frozen=True)

    filename: str
    page_count: int = Field(ge=1)
    pages: tuple[ExtractedPage, ...]


@dataclass(frozen=True)
class _PDFInput:
    """Normalized PDF input held privately by the service."""

    filename: str
    content: bytes


def _validate_pdf_input(
    filename: str,
    content: bytes,
    max_size_bytes: int,
) -> _PDFInput:
    """Validate basic file properties and return normalized input."""
    if not filename or not filename.strip():
        raise PDFProcessingError("A PDF filename is required.")

    if Path(filename).suffix.lower() != ".pdf":
        raise PDFProcessingError("Only PDF files are supported.")

    if not isinstance(content, bytes) or not content:
        raise PDFProcessingError("The uploaded PDF is empty.")

    if len(content) > max_size_bytes:
        max_size_mb = max_size_bytes // (1024 * 1024)
        raise PDFProcessingError(
            f"The PDF exceeds the maximum size of {max_size_mb} MB."
        )

    return _PDFInput(
        filename=Path(filename).name,
        content=content,
    )


def _normalize_page_text(text: str) -> str:
    """Normalize whitespace while preserving meaningful line boundaries."""
    normalized_lines = [
        " ".join(line.split()) for line in text.splitlines() if line.strip()
    ]

    return "\n".join(normalized_lines)


def extract_pdf_text(filename: str, content: bytes) -> PDFDocument:
    """Validate a PDF and extract its text page by page.

    Args:
        filename: Original name of the uploaded file.
        content: Complete PDF file contents as bytes.

    Returns:
        A validated PDFDocument with one-based page numbers.

    Raises:
        PDFProcessingError: If the file is invalid, too large, encrypted,
            empty, or contains no extractable text.
    """
    settings = get_settings()
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    pdf_input = _validate_pdf_input(
        filename=filename,
        content=content,
        max_size_bytes=max_size_bytes,
    )

    try:
        with pymupdf.open(
            stream=BytesIO(pdf_input.content),
            filetype="pdf",
        ) as pdf:
            if pdf.is_encrypted:
                raise PDFProcessingError("Password-protected PDFs are not supported.")

            if pdf.page_count == 0:
                raise PDFProcessingError("The PDF contains no pages.")

            extracted_pages: list[ExtractedPage] = []

            for page_index, page in enumerate(pdf):
                text = _normalize_page_text(page.get_text("text"))

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

            if not any(page.text for page in extracted_pages):
                raise PDFProcessingError(
                    "No extractable text was found. "
                    "The PDF may contain scanned images; OCR is not "
                    "implemented in this version."
                )

            return PDFDocument(
                filename=pdf_input.filename,
                page_count=pdf.page_count,
                pages=tuple(extracted_pages),
            )

    except PDFProcessingError:
        raise
    except (pymupdf.FileDataError, ValueError, RuntimeError) as exc:
        raise PDFProcessingError(
            "The PDF is invalid, corrupted, or could not be read."
        ) from exc
