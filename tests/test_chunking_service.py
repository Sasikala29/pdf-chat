"""Tests for page-aware PDF text chunking."""

import pytest

from src.pdf_chatbot.models.document import DocumentChunk
from src.pdf_chatbot.services.chunking_service import chunk_pages
from src.pdf_chatbot.services.pdf_service import ExtractedPage


def make_page(page_number: int, text: str) -> ExtractedPage:
    """Create a validated page record for testing."""
    return ExtractedPage(page_number=page_number, text=text)


def test_empty_input_returns_empty_list() -> None:
    """No pages should produce no chunks."""
    assert chunk_pages([]) == []


def test_short_page_produces_one_chunk() -> None:
    """Short page text should remain together."""
    result = chunk_pages([make_page(1, "This is a short PDF page.")])

    assert len(result) == 1
    assert isinstance(result[0], DocumentChunk)
    assert result[0].page_number == 1
    assert result[0].chunk_index == 1
    assert result[0].chunk_id == "page-1-chunk-1"
    assert result[0].text == "This is a short PDF page."


def test_blank_pages_are_skipped() -> None:
    """Blank pages should not create searchable chunks."""
    pages = [
        make_page(1, "   "),
        make_page(2, "Useful content appears here."),
    ]

    result = chunk_pages(pages)

    assert len(result) == 1
    assert result[0].page_number == 2


def test_chunks_preserve_source_page_numbers() -> None:
    """Chunks must retain the page from which their text was extracted."""
    pages = [
        make_page(1, "Account information is stored here."),
        make_page(2, "Contact information is stored here."),
    ]

    result = chunk_pages(pages)

    assert [chunk.page_number for chunk in result] == [1, 2]
    assert result[0].chunk_id == "page-1-chunk-1"
    assert result[1].chunk_id == "page-2-chunk-1"


def test_long_page_is_split_into_multiple_chunks(monkeypatch) -> None:
    """Long text should be split according to the configured chunk size."""

    class TestSettings:
        chunk_size = 100
        chunk_overlap = 20

    monkeypatch.setattr(
        "src.pdf_chatbot.services.chunking_service.get_settings",
        lambda: TestSettings(),
    )

    text = " ".join(f"word{i}" for i in range(100))
    result = chunk_pages([make_page(1, text)])

    assert len(result) > 1
    assert all(len(chunk.text) <= 100 for chunk in result)
    assert all(chunk.page_number == 1 for chunk in result)


def test_duplicate_page_numbers_are_rejected() -> None:
    """Duplicate page numbers make source metadata ambiguous."""
    pages = [
        make_page(1, "First page."),
        make_page(1, "Duplicate page number."),
    ]

    with pytest.raises(ValueError, match="duplicate page numbers"):
        chunk_pages(pages)


def test_out_of_order_pages_are_rejected() -> None:
    """Pages must arrive in their original document order."""
    pages = [
        make_page(2, "Second page."),
        make_page(1, "First page."),
    ]

    with pytest.raises(ValueError, match="ascending page order"):
        chunk_pages(pages)
