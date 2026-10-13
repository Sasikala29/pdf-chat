"""Retrieve relevant PDF chunks using cosine similarity."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.pdf_chatbot.services.embedding_service import EmbeddedChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """Represent a retrieved chunk and its relevance score."""

    item: EmbeddedChunk
    similarity: float


def retrieve_relevant_chunks(
    query_embedding: NDArray[np.floating],
    embedded_chunks: list[EmbeddedChunk],
    *,
    top_k: int = 4,
    min_similarity: float = 0.0,
) -> list[RetrievedChunk]:
    """Rank document chunks by cosine similarity to a query embedding.

    Args:
        query_embedding: Vector representing the user's question.
        embedded_chunks: Document chunks paired with their embeddings.
        top_k: Maximum number of chunks to return.
        min_similarity: Minimum cosine similarity score to include.

    Returns:
        Relevant chunks sorted from highest to lowest similarity.

    Raises:
        ValueError: If parameters or embedding vectors are invalid.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1.")

    query = np.asarray(query_embedding, dtype=np.float32)

    if query.ndim != 1 or query.size == 0:
        raise ValueError("The query embedding must be a non-empty vector.")

    if not np.isfinite(query).all():
        raise ValueError("The query embedding contains invalid values.")

    query_norm = float(np.linalg.norm(query))

    if query_norm == 0.0:
        raise ValueError("The query embedding cannot be a zero vector.")

    if not embedded_chunks:
        return []

    query = query / query_norm
    candidates: list[RetrievedChunk] = []

    for item in embedded_chunks:
        vector = np.asarray(item.embedding, dtype=np.float32)

        if vector.ndim != 1 or vector.shape != query.shape:
            raise ValueError(
                "Query and document embeddings must have matching dimensions."
            )

        if not np.isfinite(vector).all():
            raise ValueError("A document embedding contains invalid values.")

        vector_norm = float(np.linalg.norm(vector))

        if vector_norm == 0.0:
            raise ValueError("A document embedding cannot be a zero vector.")

        similarity = float(np.dot(query, vector / vector_norm))

        if similarity >= min_similarity:
            candidates.append(RetrievedChunk(item=item, similarity=similarity))

    candidates.sort(key=lambda result: result.similarity, reverse=True)
    return candidates[:top_k]
