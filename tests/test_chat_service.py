"""Test retrieval-based question answering and chat-history persistence."""

import numpy as np
import pytest

from src.pdf_chatbot.core.constants import NO_ANSWER
from src.pdf_chatbot.models.document import DocumentChunk
from src.pdf_chatbot.services import chat_service
from src.pdf_chatbot.services.embedding_service import EmbeddedChunk
from src.pdf_chatbot.services.retrieval_service import RetrievedChunk


@pytest.fixture
def embedded_chunk() -> EmbeddedChunk:
    """Provide one document chunk with a predictable embedding."""
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        page_number=2,
        chunk_index=1,
        text="A tokenizer splits text into tokens and converts them into IDs.",
    )
    return EmbeddedChunk(
        chunk=chunk,
        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_answer_question_uses_retrieval_and_saves_source_pages(
    monkeypatch: pytest.MonkeyPatch,
    embedded_chunk: EmbeddedChunk,
) -> None:
    """Verify retrieval, answer generation, citations, and persistence."""
    saved_messages: list[tuple[str, str, list[int]]] = []

    monkeypatch.setattr(
        chat_service,
        "load_embedded_chunks",
        lambda document_id: [embedded_chunk],
    )
    monkeypatch.setattr(
        chat_service,
        "save_chat_message",
        lambda document_id, role, content, source_pages=(): saved_messages.append(
            (role, content, list(source_pages))
        ),
    )
    monkeypatch.setattr(
        chat_service,
        "embed_query",
        lambda question: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        chat_service,
        "retrieve_relevant_chunks",
        lambda *args, **kwargs: [RetrievedChunk(item=embedded_chunk, similarity=1.0)],
    )
    monkeypatch.setattr(
        chat_service,
        "generate_answer",
        lambda question, context_chunks: (
            "A tokenizer splits text into tokens and converts them into IDs."
        ),
    )

    answer = chat_service.answer_question(
        "doc-1",
        "What does a tokenizer do?",
    )

    assert answer.endswith("Source pages: 2")
    assert "A tokenizer splits text into tokens" in answer
    assert saved_messages[0] == (
        "user",
        "What does a tokenizer do?",
        [],
    )
    assert saved_messages[1] == ("assistant", answer, [2])


def test_answer_question_returns_fallback_when_nothing_is_retrieved(
    monkeypatch: pytest.MonkeyPatch,
    embedded_chunk: EmbeddedChunk,
) -> None:
    """Verify that empty retrieval results do not trigger answer generation."""
    saved_messages: list[tuple[str, str, list[int]]] = []

    monkeypatch.setattr(
        chat_service,
        "load_embedded_chunks",
        lambda document_id: [embedded_chunk],
    )
    monkeypatch.setattr(
        chat_service,
        "save_chat_message",
        lambda document_id, role, content, source_pages=(): saved_messages.append(
            (role, content, list(source_pages))
        ),
    )
    monkeypatch.setattr(
        chat_service,
        "embed_query",
        lambda question: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        chat_service,
        "retrieve_relevant_chunks",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        chat_service,
        "generate_answer",
        lambda *args, **kwargs: pytest.fail(
            "Answer generation must not run when retrieval returns no chunks."
        ),
    )

    answer = chat_service.answer_question(
        "doc-1",
        "What is quantum computing?",
    )

    assert answer == NO_ANSWER
    assert saved_messages[-1] == ("assistant", NO_ANSWER, [])


def test_answer_question_rejects_empty_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject blank questions before accessing the repository."""
    monkeypatch.setattr(
        chat_service,
        "load_embedded_chunks",
        lambda document_id: pytest.fail("Repository should not be called."),
    )

    with pytest.raises(ValueError, match="Enter a question"):
        chat_service.answer_question("doc-1", "   ")


def test_answer_question_rejects_document_without_indexed_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report when a document has no searchable chunks."""
    monkeypatch.setattr(
        chat_service,
        "load_embedded_chunks",
        lambda document_id: [],
    )

    with pytest.raises(ValueError, match="No indexed text"):
        chat_service.answer_question(
            "doc-1",
            "What is this document about?",
        )


def test_get_document_chat_history_returns_saved_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the repository's saved messages unchanged."""
    expected_history = [
        {
            "role": "user",
            "content": "What does a tokenizer do?",
            "source_pages": [],
            "created_at": "2026-10-10T10:00:00",
        }
    ]

    monkeypatch.setattr(
        chat_service,
        "load_chat_history",
        lambda document_id: expected_history,
    )

    assert chat_service.get_document_chat_history("doc-1") == expected_history
