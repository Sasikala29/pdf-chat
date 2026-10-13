"""Focused tests for local document embedding generation."""

import numpy as np
import pytest

from src.pdf_chatbot.models.document import DocumentChunk
from src.pdf_chatbot.services import embedding_service
from src.pdf_chatbot.services.embedding_service import (
    EmbeddedChunk,
    EmbeddingProcessingError,
    embed_chunks,
)


class FakeEmbeddingModel:
    """Provide deterministic vectors without loading a real model."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.received_texts: list[str] = []

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        """Record input text and return predefined vectors."""
        self.received_texts = sentences
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return self.vectors


def make_chunk(
    chunk_id: str = "page-1-chunk-1",
    page_number: int = 1,
    text: str = "Example PDF text.",
) -> DocumentChunk:
    """Create a valid document chunk for testing."""
    return DocumentChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=1,
        text=text,
    )


def test_empty_input_returns_empty_list() -> None:
    """Avoid loading the model when there are no chunks."""
    assert embed_chunks([]) == []


def test_embeddings_preserve_chunks_and_page_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the source chunk metadata paired with its embedding."""
    chunk_one = make_chunk()
    chunk_two = make_chunk(
        chunk_id="page-2-chunk-1",
        page_number=2,
        text="Second page text.",
    )
    model = FakeEmbeddingModel([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(
        embedding_service,
        "_load_embedding_model",
        lambda model_name: model,
    )

    result = embed_chunks([chunk_one, chunk_two])

    assert len(result) == 2
    assert isinstance(result[0], EmbeddedChunk)
    assert result[0].chunk == chunk_one
    assert result[1].chunk.page_number == 2
    assert model.received_texts == [
        "Example PDF text.",
        "Second page text.",
    ]
    assert np.allclose(result[0].embedding, [1.0, 0.0])
    assert np.allclose(result[1].embedding, [0.0, 1.0])


def test_model_failure_raises_processing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert model-loading and encoding failures to a clear service error."""

    def fail_to_load(model_name: str) -> FakeEmbeddingModel:
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(
        embedding_service,
        "_load_embedding_model",
        fail_to_load,
    )

    with pytest.raises(EmbeddingProcessingError):
        embed_chunks([make_chunk()])


@pytest.mark.parametrize(
    ("vectors", "chunk_count"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], 1),
        ([[1.0, 0.0]], 2),
        ([[float("nan"), 0.0]], 1),
        ([[0.0, 0.0]], 1),
        ([[]], 1),
    ],
)
def test_invalid_embedding_vectors_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    vectors: list[list[float]],
    chunk_count: int,
) -> None:
    """Reject vector-count mismatches and invalid numerical vectors."""
    model = FakeEmbeddingModel(vectors)
    monkeypatch.setattr(
        embedding_service,
        "_load_embedding_model",
        lambda model_name: model,
    )

    chunks = [
        make_chunk(
            chunk_id=f"page-1-chunk-{index}",
            text=f"Sample chunk {index}.",
        )
        for index in range(1, chunk_count + 1)
    ]

    with pytest.raises(ValueError):
        embed_chunks(chunks)
