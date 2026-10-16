"""Tests for ranking and validating retrieved PDF chunks."""

import numpy as np
import pytest

from src.pdf_chatbot.models.document import DocumentChunk
from src.pdf_chatbot.services.embedding_service import EmbeddedChunk
from src.pdf_chatbot.services.retrieval_service import (
    RetrievedChunk,
    retrieve_relevant_chunks,
)

# Input: Chunk ID, page number, and embedding vector.
# Return: An EmbeddedChunk containing document metadata and its embedding.
# Purpose: Creates a test chunk for retrieval testing.
def make_embedded_chunk(
    chunk_id: str,
    page_number: int,
    vector: list[float],
) -> EmbeddedChunk:
    """Create an embedded chunk with source-page metadata."""
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=1,
        text=f"Content from page {page_number}.",
    )
    return EmbeddedChunk(
        chunk=chunk,
        embedding=np.asarray(vector, dtype=np.float32),
    )

# Input: Three embedded chunks and a query vector.
# Return: None.
# Purpose: Verifies that retrieved chunks are ranked by relevance and limited by top_k.
def test_returns_chunks_in_relevance_order() -> None:
    """Rank the most similar chunk first."""
    chunks = [
        make_embedded_chunk("chunk-a", 1, [0.0, 1.0]),
        make_embedded_chunk("chunk-b", 2, [1.0, 0.0]),
        make_embedded_chunk("chunk-c", 3, [0.8, 0.6]),
    ]

    result = retrieve_relevant_chunks(
        np.asarray([1.0, 0.0], dtype=np.float32),
        chunks,
        top_k=2,
    )

    assert len(result) == 2
    assert all(isinstance(item, RetrievedChunk) for item in result)
    assert result[0].item.chunk.chunk_id == "chunk-b"
    assert result[1].item.chunk.chunk_id == "chunk-c"
    assert result[0].item.chunk.page_number == 2
    assert result[0].similarity > result[1].similarity

# Input: A query vector and an empty chunk list.
# Return: None.
# Purpose: Verifies that retrieval returns an empty list when no chunks are available.
def test_empty_chunk_list_returns_empty_list() -> None:
    """Return no results when the document has no embedded chunks."""
    result = retrieve_relevant_chunks(
        np.asarray([1.0, 0.0], dtype=np.float32),
        [],
    )

    assert result == []

# Input: A query vector and relevant and irrelevant chunks.
# Return: None.
# Purpose: Verifies that chunks below the minimum similarity threshold are excluded.
def test_minimum_similarity_filters_irrelevant_chunks() -> None:
    """Exclude chunks that do not meet the relevance threshold."""
    chunks = [
        make_embedded_chunk("relevant", 1, [1.0, 0.0]),
        make_embedded_chunk("irrelevant", 2, [0.0, 1.0]),
    ]

    result = retrieve_relevant_chunks(
        np.asarray([1.0, 0.0], dtype=np.float32),
        chunks,
        min_similarity=0.5,
    )

    assert [item.item.chunk.chunk_id for item in result] == ["relevant"]

# Input: A top_k value of zero or a negative number.
# Return: None.
# Purpose: Verifies that invalid result limits raise a ValueError.
@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    """Require a positive result limit."""
    with pytest.raises(ValueError, match="top_k"):
        retrieve_relevant_chunks(
            np.asarray([1.0, 0.0], dtype=np.float32),
            [],
            top_k=top_k,
        )

# Input: A zero-valued query vector.
# Return: None.
# Purpose: Verifies that a zero query vector raises a ValueError.
def test_zero_query_vector_is_rejected() -> None:
    """Prevent undefined cosine similarity calculations."""
    with pytest.raises(ValueError, match="zero vector"):
        retrieve_relevant_chunks(
            np.asarray([0.0, 0.0], dtype=np.float32),
            [],
        )

# Input: A query vector and a chunk with a different embedding dimension.
# Return: None.
# Purpose: Verifies that mismatched vector dimensions raise a ValueError.
def test_mismatched_vector_dimensions_are_rejected() -> None:
    """Require query and document vectors to have matching dimensions."""
    chunks = [make_embedded_chunk("chunk-a", 1, [1.0, 0.0, 0.0])]

    with pytest.raises(ValueError, match="matching dimensions"):
        retrieve_relevant_chunks(
            np.asarray([1.0, 0.0], dtype=np.float32),
            chunks,
        )
