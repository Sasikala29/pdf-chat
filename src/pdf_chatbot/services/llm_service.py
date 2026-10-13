"""Generate source-faithful answers from retrieved PDF passages."""

from functools import lru_cache

from groq import Groq

from src.pdf_chatbot.core.config import get_settings
from src.pdf_chatbot.core.constants import (
    LLM_MAX_COMPLETION_TOKENS,
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT_SECONDS,
    NO_ANSWER,
)
from src.pdf_chatbot.core.prompts import SYSTEM_PROMPT, build_user_prompt


@lru_cache(maxsize=1)
def _get_groq_client() -> Groq:
    """Create and reuse the configured Groq client."""
    settings = get_settings()

    return Groq(
        api_key=settings.groq_api_key.get_secret_value(),
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )


def generate_answer(
    question: str,
    context_chunks: list[tuple[int, str]],
) -> str:
    """Generate an answer grounded in retrieved PDF passages.

    Args:
        question: The user's question.
        context_chunks: Retrieved passages as (page_number, text) pairs.

    Returns:
        The generated answer, or NO_ANSWER when no passages are supplied.

    Raises:
        ValueError: If the question or a supplied passage is invalid.
        RuntimeError: If Groq fails or returns an empty response.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context_chunks:
        return NO_ANSWER

    context_parts: list[str] = []

    for page_number, text in context_chunks:
        if page_number < 1 or not text.strip():
            raise ValueError("Each context passage must have a valid page and text.")

        context_parts.append(f"[Source page {page_number}]\n{text.strip()}")

    context = "\n\n".join(context_parts)
    user_prompt = build_user_prompt(question, context)
    settings = get_settings()

    try:
        response = _get_groq_client().chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
            include_reasoning=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Groq could not generate an answer. Check the API configuration, "
            "model access, network connection, and API limits."
        ) from exc

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty answer.")

    return answer.strip()
