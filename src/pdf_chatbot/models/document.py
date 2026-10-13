"""Data models used to represent document chunks."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """One searchable section of text extracted from a source document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        min_length=1,
        description="Unique identifier for this chunk within the document.",
    )
    page_number: int = Field(
        ge=1,
        description="One-based page number from the original PDF.",
    )
    chunk_index: int = Field(
        ge=1,
        description="One-based chunk number within its source page.",
    )
    text: str = Field(
        min_length=1,
        description="Text content used for embedding and retrieval.",
    )
