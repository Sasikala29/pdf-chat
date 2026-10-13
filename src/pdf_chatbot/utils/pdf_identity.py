"""Identify identical PDF files and remove duplicates from display lists."""

import hashlib
from pathlib import Path
from collections.abc import Iterable


def sha256_bytes(content: bytes) -> str:
    """Return a stable SHA-256 fingerprint for file contents."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file fingerprint, or None when the file is unavailable."""
    try:
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()
    except OSError:
        return None


def unique_documents_by_content(
    documents: Iterable[dict],
) -> list[dict]:
    """Keep only the newest visible record for each identical PDF.

    Documents are expected to be ordered newest first by the repository.
    Missing files fall back to document ID so unrelated records are retained.
    """
    unique_documents: list[dict] = []
    seen_hashes: set[str] = set()

    for document in documents:
        file_path = Path(str(document.get("file_path") or ""))
        fingerprint = sha256_file(file_path) if file_path.is_file() else None

        if fingerprint is not None:
            if fingerprint in seen_hashes:
                continue

            seen_hashes.add(fingerprint)

        unique_documents.append(document)

    return unique_documents
