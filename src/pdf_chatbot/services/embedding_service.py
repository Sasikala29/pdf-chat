"""Generate local vector embeddings for PDF document chunks and user queries."""

from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.pdf_chatbot.core.config import get_settings
from src.pdf_chatbot.models.document import DocumentChunk

# Standard vector type shared by embedding and retrieval services.
EmbeddingVector = NDArray[np.float32]


class EmbeddingModel(Protocol):
    """Define the encoding interface required by this service."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> NDArray[np.floating]:
        """Convert text strings into numerical vectors."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """Pair a source document chunk with its numerical embedding."""

    chunk: DocumentChunk
    embedding: EmbeddingVector


class EmbeddingProcessingError(RuntimeError):
    """Represent a failure while generating document embeddings."""


# Prevent simultaneous first-time model loads from competing.
_MODEL_LOCK = Lock()


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str) -> EmbeddingModel:
    """Load and cache the configured embedding model on first use."""
    from sentence_transformers import SentenceTransformer

    with _MODEL_LOCK:
        return SentenceTransformer(model_name)


def _generate_embeddings(
    texts: list[str],
    *,
    error_message: str,
) -> NDArray[np.float32]:
    """Generate and validate normalized vectors for non-empty text input.

    Args:
        texts: Text strings to convert into vectors.
        error_message: Safe message used when model loading or encoding fails.

    Returns:
        A two-dimensional float32 array containing one vector per input.

    Raises:
        ValueError: If the input is empty or the returned vectors are invalid.
        EmbeddingProcessingError: If model loading or encoding fails.
    """
    if not texts:
        raise ValueError("At least one text string is required.")

    settings = get_settings()

    try:
        model = _load_embedding_model(settings.embedding_model)
        vectors = np.asarray(
            model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise EmbeddingProcessingError(error_message) from exc

    # Validate the output before it reaches persistence or retrieval.
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(
            "The embedding model returned an unexpected number of vectors."
        )

    if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError(
            "The embedding model returned empty or non-finite vector values."
        )

    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("The embedding model returned a zero-length vector.")

    return vectors


def embed_chunks(chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
    """Embed PDF chunks while preserving their page and chunk metadata.

    Returns:
        An empty list for empty input; otherwise, embedded chunks.
    """
    if not chunks:
        return []

    vectors = _generate_embeddings(
        [chunk.text for chunk in chunks],
        error_message="Could not generate document embeddings.",
    )

    return [
        EmbeddedChunk(chunk=chunk, embedding=vector.copy())
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def embed_query(query: str) -> EmbeddingVector:
    """Generate a normalized embedding for a user's question.

    The same model and encoding settings are used for document chunks and
    questions so that retrieval can compare their vectors consistently.

    Args:
        query: The user's question.

    Returns:
        A one-dimensional float32 embedding vector.

    Raises:
        ValueError: If the question is blank or the generated vector is invalid.
        EmbeddingProcessingError: If the model cannot be loaded or used.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    vectors = _generate_embeddings(
        [query.strip()],
        error_message="Could not generate the query embedding.",
    )

    return vectors[0].copy()
