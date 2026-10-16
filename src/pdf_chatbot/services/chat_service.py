"""Answer PDF questions using retrieval-augmented generation."""

from src.pdf_chatbot.core.constants import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_RETRIEVAL_TOP_K,
    NO_ANSWER,
)
from src.pdf_chatbot.repositories.sqlite_repository import (
    get_conversation,
    list_conversation_documents,
    load_chat_history,
    load_embedded_chunks,
    save_chat_message,
)
from src.pdf_chatbot.services.embedding_service import embed_query
from src.pdf_chatbot.services.llm_service import generate_answer
from src.pdf_chatbot.services.retrieval_service import retrieve_relevant_chunks


def _save_message(
    document_id: str,
    role: str,
    content: str,
    source_pages: tuple[int, ...] | list[int] = (),
    conversation_id: str | None = None,
) -> None:
    """Save a message while preserving legacy single-PDF compatibility."""
    if conversation_id is None:
        # Keep the original function call compatible with existing tests
        # and older single-PDF chat flows.
        save_chat_message(document_id, role, content, source_pages)
        return

    save_chat_message(
        document_id,
        role,
        content,
        source_pages=source_pages,
        conversation_id=conversation_id,
    )


def answer_question(
    document_id: str,
    question: str,
    conversation_id: str | None = None,
) -> str:
    """Answer using one PDF or all PDFs linked to a conversation."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Enter a question before submitting.")

    if conversation_id is None:
        document_ids = [document_id]
        message_document_id = document_id
        filenames: dict[str, str] = {}
    else:
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found.")

        # Retain the conversation's original document ID for message storage.
        message_document_id = conversation["document_id"]

        associated_documents = list_conversation_documents(conversation_id)
        document_ids = [item["document_id"] for item in associated_documents]
        filenames = {
            item["document_id"]: item["filename"] for item in associated_documents
        }

        if not document_ids:
            raise ValueError("No PDFs are associated with this conversation.")

    # Combine chunks from every PDF associated with this conversation.
    stored_chunks = []
    for linked_document_id in document_ids:
        stored_chunks.extend(load_embedded_chunks(linked_document_id))

    if not stored_chunks:
        raise ValueError("No indexed text was found for the selected PDFs.")

    _save_message(
        message_document_id,
        "user",
        cleaned_question,
        conversation_id=conversation_id,
    )

    query_embedding = embed_query(cleaned_question)

    retrieved_chunks = retrieve_relevant_chunks(
        query_embedding,
        stored_chunks,
        top_k=DEFAULT_RETRIEVAL_TOP_K,
        min_similarity=DEFAULT_MIN_SIMILARITY,
    )

    if not retrieved_chunks:
        answer = NO_ANSWER
        source_pages: list[int] = []
    else:
        context_chunks: list[tuple[int, str]] = []
        source_references: list[str] = []

        for retrieved in retrieved_chunks:
            chunk = retrieved.item.chunk

            # Persisted chunk IDs are prefixed with their document ID.
            chunk_document_id = chunk.chunk_id.split(":", 1)[0]
            filename = filenames.get(chunk_document_id, "Selected PDF")

            context_text = (
                f"[Source PDF: {filename}; Page: {chunk.page_number}]\n{chunk.text}"
            )
            context_chunks.append((chunk.page_number, context_text))

            source_reference = f"{filename}, page {chunk.page_number}"
            if source_reference not in source_references:
                source_references.append(source_reference)

        generated_answer = generate_answer(
            cleaned_question,
            context_chunks,
        )
        answer = generated_answer.strip() or NO_ANSWER

        if answer == NO_ANSWER:
            source_pages = []
        else:
            source_pages = list(
                dict.fromkeys(
                    retrieved.item.chunk.page_number for retrieved in retrieved_chunks
                )
            )

            if conversation_id is None:
                # Preserve the existing single-PDF response format.
                answer += f"\n\nSource pages: {', '.join(map(str, source_pages))}"
            else:
                answer += "\n\nSources: " + "; ".join(source_references)

    _save_message(
        message_document_id,
        "assistant",
        answer,
        source_pages=source_pages,
        conversation_id=conversation_id,
    )

    return answer


def get_document_chat_history(
    document_id: str,
    conversation_id: str | None = None,
) -> list[dict[str, object]]:
    """Load chat history for a conversation or a legacy PDF chat."""
    if conversation_id is None:
        return load_chat_history(document_id)

    return load_chat_history(document_id, conversation_id)
