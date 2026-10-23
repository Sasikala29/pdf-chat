"""Application-level coordination for the DocuMind Streamlit interface."""

from src.pdf_chatbot.database.connection import initialize_database as _initialize_database
from src.pdf_chatbot.repositories.sqlite_repository import (
    create_conversation as _create_conversation,
    delete_conversation,
    get_conversation as _get_conversation,
    list_conversation_documents,
    list_conversations,
    list_documents,
    set_conversation_documents,
)
from src.pdf_chatbot.services.chat_service import (
    answer_question,
    get_document_chat_history,
    create_chat_for_documents as _create_chat_for_documents,
)
from src.pdf_chatbot.services.deletion_service import delete_pdf_and_chats
from src.pdf_chatbot.services.document_service import process_uploaded_pdf
from src.pdf_chatbot.utils.pdf_identity import unique_documents_by_content


def initialize_application() -> None:
    """Initialize the database schema."""
    _initialize_database()


def load_application_data() -> tuple[list[dict], list[dict]]:
    """Load saved PDFs and conversations, hiding duplicate PDF content in the UI."""
    return unique_documents_by_content(list_documents()), list_conversations()


def get_active_conversation(conversation_id: str | None) -> dict | None:
    """Return the selected conversation if it still exists."""
    if not conversation_id:
        return None
    return _get_conversation(conversation_id)


def create_chat_for_documents(document_ids: list[str]) -> str:
    """Create a conversation for valid, existing PDFs."""
    unique_document_ids = list(dict.fromkeys(
        document_id for document_id in document_ids if document_id
    ))
    if not unique_document_ids:
        raise ValueError("Select at least one PDF before starting a chat.")

    saved_document_ids = {
        document["document_id"] for document in list_documents()
    }
    invalid_ids = [
        document_id for document_id in unique_document_ids
        if document_id not in saved_document_ids
    ]
    if invalid_ids:
        raise ValueError(
            "The selected PDF list contains invalid document IDs. "
            "Refresh the page and select the PDFs again."
        )

    return _create_chat_for_documents(unique_document_ids)


def get_conversation_documents(conversation_id: str) -> list[dict]:
    """Return the PDFs associated with a conversation."""
    return list_conversation_documents(conversation_id)


def save_conversation_documents(
    conversation_id: str,
    document_ids: list[str],
) -> None:
    """Save the selected PDF associations for a conversation."""
    set_conversation_documents(conversation_id, document_ids)


def delete_conversation_by_id(conversation_id: str) -> None:
    """Delete a conversation and its associated chat history."""
    delete_conversation(conversation_id)
