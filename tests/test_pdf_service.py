"""Automated tests for PDF validation and text extraction."""

from io import BytesIO

import fitz
import pytest

from src.pdf_chatbot.services.pdf_service import (
    PDFProcessingError,
    extract_pdf_text,
)


def create_test_pdf(*page_texts: str) -> bytes:
    """Generate a small PDF in memory for automated tests."""
    output = BytesIO()

    with fitz.open() as document:
        for page_text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), page_text)

        document.save(output)

    return output.getvalue()


def test_extracts_text_and_preserves_page_numbers() -> None:
    """Text from multiple pages should retain its source page."""
    content = create_test_pdf(
        "First page contains account information.",
        "Second page contains contact information.",
    )

    result = extract_pdf_text("sample.pdf", content)

    assert result.filename == "sample.pdf"
    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert "account information" in result.pages[0].text
    assert result.pages[1].page_number == 2
    assert "contact information" in result.pages[1].text


def test_rejects_non_pdf_filename() -> None:
    """Files with unsupported extensions should be rejected."""
    with pytest.raises(PDFProcessingError, match="Only PDF"):
        extract_pdf_text("notes.txt", b"not a PDF")


def test_rejects_empty_content() -> None:
    """Empty uploaded content should be rejected."""
    with pytest.raises(PDFProcessingError, match="empty"):
        extract_pdf_text("empty.pdf", b"")


def test_rejects_corrupted_pdf() -> None:
    """Non-PDF bytes must not be accepted as a valid document."""
    with pytest.raises(PDFProcessingError, match="invalid, corrupted"):
        extract_pdf_text("broken.pdf", b"%PDF-1.7 invalid content")


def test_rejects_pdf_without_extractable_text() -> None:
    """A PDF containing only blank pages should be rejected."""
    content = create_test_pdf("   ")

    with pytest.raises(PDFProcessingError, match="No extractable text"):
        extract_pdf_text("blank.pdf", content)


def test_rejects_pdf_exceeding_size_limit(monkeypatch) -> None:
    """The configured file-size limit must be enforced."""

    class TestSettings:
        max_upload_size_mb = 1

    monkeypatch.setattr(
        "src.pdf_chatbot.services.pdf_service.get_settings",
        lambda: TestSettings(),
    )

    with pytest.raises(PDFProcessingError, match="maximum size"):
        extract_pdf_text("large.pdf", b"x" * (1024 * 1024 + 1))
