"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
# Input: Current file location.
# Return: Absolute path to the project root directory.
# Purpose: Resolves the project root independently of the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Input: Environment variables and values from the .env file.
    # Return: A validated Settings instance.
    # Purpose: Centralizes and validates the PDF chatbot's configuration.
    """Validate and centralize configuration for the PDF chatbot."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # Input: Groq API key from the environment or .env file.
    # Return: A SecretStr containing the API key.
    # Purpose: Stores the API key while reducing accidental secret exposure.
    groq_api_key: SecretStr = Field(
        ...,
        description="API key used to authenticate with Groq.",
    )
    # Input: GROQ_MODEL environment variable or the default model name.
    # Return: A non-empty string containing the model name.
    # Purpose: Configures the Groq model used to generate chatbot responses.
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        min_length=1,
        description="Groq model used to generate chatbot responses.",
    )
    # Input: EMBEDDING_MODEL environment variable or the default model name.
    # Return: A non-empty string containing the embedding model name.
    # Purpose: Configures the local model used to generate text embeddings.
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
        description="Model used to generate text embeddings locally.",
    )
    # Input: MAX_UPLOAD_SIZE_MB environment variable or the default size.
    # Return: An integer between 1 and 100.
    # Purpose: Sets the maximum permitted PDF upload size in megabytes.
    max_upload_size_mb: int = Field(
        default=20,
        gt=0,
        le=100,
        description="Maximum allowed PDF upload size in megabytes.",
    )
    # Input: CHUNK_SIZE environment variable or the default chunk size.
    # Return: A positive integer.
    # Purpose: Sets the target number of characters in each text chunk.
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Target number of characters in each text chunk.",
    )
    # Input: CHUNK_OVERLAP environment variable or the default overlap.
    # Return: A non-negative integer.
    # Purpose: Sets the number of overlapping characters between consecutive chunks.
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Number of overlapping characters between chunks.",
    )

# Input: No arguments.
# Return: A validated Settings instance.
# Purpose: Creates and caches one settings instance and checks that the chunk overlap is smaller than the chunk size.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one validated settings instance for the application."""
    settings = Settings()

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    return settings
