"""Safely delete a PDF and its associated conversations."""

from pathlib import Path

from src.pdf_chatbot.repositories.sqlite_repository import (
    delete_conversation,
    delete_document,
    list_conversation_documents,
    list_conversations,
    list_documents,
)

UPLOAD_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "data" / "uploads"
).resolve()


def delete_pdf_and_chats(document_id: str) -> dict[str, int]:
    """Delete a saved PDF, its indexed data, and associated conversations.

    Conversations associated with multiple PDFs are also deleted in full.
    The stored PDF path is validated before any deletion is attempted.

    Args:
        document_id: ID of the document to delete.

    Returns:
        A dictionary containing the document and conversation deletion counts.

    Raises:
        ValueError: If the document does not exist or its file path is unsafe.
        OSError: If deleting the PDF file fails.
    """
    # Find the requested document.
    document = next(
        (
            item
            for item in list_documents()
            if item["document_id"] == document_id
        ),
        None,
    )

    if document is None:
        raise ValueError("This PDF no longer exists. Refresh the page.")

    # Resolve and validate the stored path before modifying any data.
    stored_path = Path(document["file_path"]).resolve()

    if stored_path.parent != UPLOAD_DIRECTORY:
        raise ValueError(
            "The stored PDF path is outside the upload directory. "
            "No data was deleted."
        )

    # Identify every conversation associated with this PDF.
    associated_conversation_ids: list[str] = []

    for conversation in list_conversations():
        conversation_id = conversation["conversation_id"]
        associations = list_conversation_documents(conversation_id)

        if any(
            item["document_id"] == document_id
            for item in associations
        ):
            associated_conversation_ids.append(conversation_id)

    # Remove the physical PDF first. If this fails, preserve database records.
    stored_path.unlink(missing_ok=True)

    # Delete associated chat histories and their conversation records.
    for conversation_id in associated_conversation_ids:
        delete_conversation(conversation_id)

    # Delete the document and its dependent indexed data.
    delete_document(document_id)

    return {
        "documents_deleted": 1,
        "conversations_deleted": len(associated_conversation_ids),
    }
