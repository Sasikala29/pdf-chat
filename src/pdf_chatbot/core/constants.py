"""Shared application constants and default values."""

# Standard response when the PDF does not contain enough information.
NO_ANSWER = (
    "The uploaded PDF does not contain enough information to answer this question."
)

# Maximum number of PDF chunks passed to the language model.
DEFAULT_RETRIEVAL_TOP_K = 4

# Minimum cosine similarity accepted during retrieval.
DEFAULT_MIN_SIMILARITY = 0.0

# Groq request timeout and retry configuration.
LLM_REQUEST_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 2

# Maximum completion length requested from the language model.
LLM_MAX_COMPLETION_TOKENS = 1024

# Number of bytes in one mebibyte, used for upload-size validation.
BYTES_PER_MEGABYTE = 1024 * 1024
