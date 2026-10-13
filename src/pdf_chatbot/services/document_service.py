
"""Coordinate PDF upload, processing, embedding, and SQLite persistence."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from src.pdf_chatbot.core.config import get_settings
from src.pdf_chatbot.repositories.sqlite_repository import (
    save_chunks_and_embeddings,
    save_document,
)
from src.pdf_chatbot.services.chunking_service import chunk_pages
from src.pdf_chatbot.services.embedding_service import embed_chunks
from src.pdf_chatbot.services.pdf_service import extract_pdf_text


# Store original PDFs separately from the SQLite database.
UPLOAD_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "data" / "uploads"
)


@dataclass(frozen=True)
class ProcessedDocument:
    """Summary of a PDF successfully processed and saved."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int


def process_uploaded_pdf(
    filename: str,
    content: bytes,
) -> ProcessedDocument:
    """Process an uploaded PDF and persist its searchable data.

    Args:
        filename: Original uploaded filename.
        content: Complete PDF file contents.

    Returns:
        Document identifier and processing summary.

    Raises:
        PDFProcessingError: If PDF validation or extraction fails.
        RuntimeError: If embedding generation fails.
    """
    settings = get_settings()

    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise ValueError("The uploaded PDF exceeds the configured size limit.")

    # Validate and extract text before saving the original file.
    document = extract_pdf_text(filename, content)

    chunks = chunk_pages(document.pages)

    if not chunks:
        raise ValueError("No searchable text chunks were produced.")

    embedded_chunks = embed_chunks(chunks)

    if not embedded_chunks:
        raise ValueError("No embeddings were generated for the PDF.")

    # Generate a server-side ID instead of trusting the uploaded filename.
    document_id = uuid4().hex
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Use the generated ID as the stored filename to prevent path traversal.
    pdf_path = UPLOAD_DIRECTORY / f"{document_id}.pdf"

    try:
        # Persist the original PDF, then its metadata and search index.
        pdf_path.write_bytes(content)
        save_document(document_id, document.filename)
        save_chunks_and_embeddings(document_id, embedded_chunks)

    except Exception:
        # Avoid leaving an orphaned original PDF if persistence fails.
        pdf_path.unlink(missing_ok=True)
        raise

    return ProcessedDocument(
        document_id=document_id,
        filename=document.filename,
        page_count=document.page_count,
        chunk_count=len(embedded_chunks),
    )
