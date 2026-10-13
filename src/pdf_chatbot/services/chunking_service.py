"""Split extracted PDF pages into searchable, page-aware text chunks."""

from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pdf_chatbot.core.config import get_settings
from src.pdf_chatbot.models.document import DocumentChunk
from src.pdf_chatbot.services.pdf_service import ExtractedPage


def _create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Build a text splitter using the validated application settings."""
    settings = get_settings()

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        keep_separator=True,
        strip_whitespace=True,
    )


def chunk_pages(pages: Sequence[ExtractedPage]) -> list[DocumentChunk]:
    """Split extracted pages while retaining their original page numbers.

    Args:
        pages: Page-aware text records produced by the PDF service.

    Returns:
        An ordered list of chunks. Pages with only whitespace are skipped.

    Raises:
        ValueError: If page numbers are duplicated or out of order.
    """
    if not pages:
        return []

    page_numbers = [page.page_number for page in pages]

    if len(page_numbers) != len(set(page_numbers)):
        raise ValueError("Input contains duplicate page numbers.")

    if page_numbers != sorted(page_numbers):
        raise ValueError("Input pages must be in ascending page order.")

    splitter = _create_text_splitter()
    chunks: list[DocumentChunk] = []

    for page in pages:
        if not page.text.strip():
            continue

        for chunk_index, text in enumerate(
            splitter.split_text(page.text),
            start=1,
        ):
            normalized_text = text.strip()

            if not normalized_text:
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=f"page-{page.page_number}-chunk-{chunk_index}",
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=normalized_text,
                )
            )

    return chunks
