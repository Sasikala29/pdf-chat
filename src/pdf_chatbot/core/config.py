"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root independently of the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Validate and centralize configuration for the PDF chatbot."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SecretStr helps prevent accidental disclosure when the secret is displayed.
    groq_api_key: SecretStr = Field(
        ...,
        description="API key used to authenticate with Groq.",
    )

    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        min_length=1,
        description="Groq model used to generate chatbot responses.",
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
        description="Model used to generate text embeddings locally.",
    )

    max_upload_size_mb: int = Field(
        default=20,
        gt=0,
        le=100,
        description="Maximum allowed PDF upload size in megabytes.",
    )

    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Target number of characters in each text chunk.",
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Number of overlapping characters between chunks.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one validated settings instance for the application."""
    settings = Settings()

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    return settings
