"""Persist documents, embeddings, conversations, and chat messages in SQLite."""

import json
from collections.abc import Sequence
from uuid import uuid4

import numpy as np

from src.pdf_chatbot.database.connection import (
    get_connection,
    initialize_database,
)
from src.pdf_chatbot.models.document import DocumentChunk
from src.pdf_chatbot.services.embedding_service import EmbeddedChunk


def save_document(
    document_id: str,
    filename: str,
    file_path: str,
) -> None:
    """Register a PDF and the path of its saved file."""
    initialize_database()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (document_id, filename, file_path)
            VALUES (?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename = excluded.filename,
                file_path = excluded.file_path
            """,
            (document_id, filename, file_path),
        )


def list_documents() -> list[dict[str, str]]:
    """Return all saved PDFs, newest first."""
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT document_id, filename, file_path, uploaded_at
            FROM documents
            ORDER BY uploaded_at DESC, rowid DESC
            """
        ).fetchall()

    return [
        {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "file_path": row["file_path"] or "",
            "uploaded_at": row["uploaded_at"],
        }
        for row in rows
    ]


def delete_document(document_id: str) -> None:
    """Delete a PDF and its dependent records according to database rules."""
    initialize_database()

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM documents WHERE document_id = ?",
            (document_id,),
        )


def save_chunks_and_embeddings(
    document_id: str,
    items: Sequence[EmbeddedChunk],
) -> None:
    """Persist a PDF's text chunks and corresponding embedding vectors."""
    initialize_database()

    with get_connection() as connection:
        for item in items:
            chunk = item.chunk
            persisted_chunk_id = f"{document_id}:{chunk.chunk_id}"
            vector = np.asarray(item.embedding, dtype=np.float32)

            connection.execute(
                """
                INSERT INTO document_chunks
                    (chunk_id, document_id, page_number, chunk_index, text)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    page_number = excluded.page_number,
                    chunk_index = excluded.chunk_index,
                    text = excluded.text
                """,
                (
                    persisted_chunk_id,
                    document_id,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.text,
                ),
            )

            connection.execute(
                """
                INSERT INTO embeddings (chunk_id, vector)
                VALUES (?, ?)
                ON CONFLICT(chunk_id)
                DO UPDATE SET vector = excluded.vector
                """,
                (persisted_chunk_id, vector.tobytes()),
            )


def load_embedded_chunks(document_id: str) -> list[EmbeddedChunk]:
    """Load only the chunks and embeddings belonging to one PDF."""
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.chunk_id,
                c.page_number,
                c.chunk_index,
                c.text,
                e.vector
            FROM document_chunks AS c
            JOIN embeddings AS e ON e.chunk_id = c.chunk_id
            WHERE c.document_id = ?
            ORDER BY c.page_number, c.chunk_index
            """,
            (document_id,),
        ).fetchall()

    return [
        EmbeddedChunk(
            chunk=DocumentChunk(
                chunk_id=row["chunk_id"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                text=row["text"],
            ),
            embedding=np.frombuffer(
                row["vector"],
                dtype=np.float32,
            ).copy(),
        )
        for row in rows
    ]


def create_conversation(
    document_id: str,
    title: str = "New chat",
) -> dict[str, str]:
    """Create a conversation linked initially to one existing PDF."""
    cleaned_title = title.strip() or "New chat"
    conversation_id = uuid4().hex

    initialize_database()

    with get_connection() as connection:
        document = connection.execute(
            "SELECT document_id FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()

        if document is None:
            raise ValueError("The selected PDF does not exist.")

        connection.execute(
            """
            INSERT INTO conversations
                (conversation_id, document_id, title)
            VALUES (?, ?, ?)
            """,
            (conversation_id, document_id, cleaned_title),
        )

        connection.execute(
            """
            INSERT INTO conversation_documents
                (conversation_id, document_id)
            VALUES (?, ?)
            """,
            (conversation_id, document_id),
        )

        row = connection.execute(
            """
            SELECT
                conversation_id,
                document_id,
                title,
                created_at,
                updated_at
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

    return dict(row)


def set_conversation_documents(
    conversation_id: str,
    document_ids: Sequence[str],
) -> None:
    """Replace a conversation's PDF associations with the selected PDFs."""
    requested_ids = list(dict.fromkeys(document_ids))

    if not requested_ids:
        raise ValueError("Select at least one PDF for this conversation.")

    initialize_database()

    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT conversation_id
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            raise ValueError("Conversation not found.")

        placeholders = ", ".join("?" for _ in requested_ids)
        existing_rows = connection.execute(
            f"""
            SELECT document_id
            FROM documents
            WHERE document_id IN ({placeholders})
            """,
            tuple(requested_ids),
        ).fetchall()

        existing_ids = {row["document_id"] for row in existing_rows}
        missing_ids = set(requested_ids) - existing_ids

        if missing_ids:
            raise ValueError("One or more selected PDFs no longer exist.")

        # Keep the legacy anchor aligned with the first selected PDF.
        # It must not force a previously linked PDF to remain associated.
        connection.execute(
            """
            UPDATE conversations
            SET document_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (requested_ids[0], conversation_id),
        )

        # Replace only this conversation's links. PDF files, chunks,
        # embeddings, and messages are not deleted by this operation.
        connection.execute(
            """
            DELETE FROM conversation_documents
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )

        connection.executemany(
            """
            INSERT INTO conversation_documents
                (conversation_id, document_id)
            VALUES (?, ?)
            """,
            [(conversation_id, selected_id) for selected_id in requested_ids],
        )


def list_conversation_documents(
    conversation_id: str,
) -> list[dict[str, str]]:
    """Return only the PDFs explicitly linked to this conversation."""
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                d.document_id,
                d.filename,
                d.file_path,
                d.uploaded_at
            FROM conversation_documents AS cd
            JOIN documents AS d ON d.document_id = cd.document_id
            WHERE cd.conversation_id = ?
            ORDER BY cd.added_at, d.filename
            """,
            (conversation_id,),
        ).fetchall()

    return [
        {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "file_path": row["file_path"] or "",
            "uploaded_at": row["uploaded_at"],
        }
        for row in rows
    ]


def list_conversations(
    document_id: str | None = None,
) -> list[dict[str, str]]:
    """List conversations by recent activity, optionally filtered by PDF."""
    initialize_database()

    query = """
        SELECT
            c.conversation_id,
            c.document_id,
            c.title,
            c.created_at,
            c.updated_at,
            d.filename,
            COALESCE(
                (
                    SELECT group_concat(associated.filename, ', ')
                    FROM conversation_documents AS cd
                    JOIN documents AS associated
                        ON associated.document_id = cd.document_id
                    WHERE cd.conversation_id = c.conversation_id
                ),
                d.filename
            ) AS filenames
        FROM conversations AS c
        JOIN documents AS d ON d.document_id = c.document_id
    """
    parameters: tuple[str, ...] = ()

    if document_id is not None:
        query += """
            WHERE EXISTS (
                SELECT 1
                FROM conversation_documents AS cd
                WHERE cd.conversation_id = c.conversation_id
                  AND cd.document_id = ?
            )
        """
        parameters = (document_id,)

    query += " ORDER BY c.updated_at DESC, c.created_at DESC"

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]


def get_conversation(
    conversation_id: str,
) -> dict[str, str] | None:
    """Return conversation details and associated PDF filenames."""
    initialize_database()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                c.conversation_id,
                c.document_id,
                c.title,
                c.created_at,
                c.updated_at,
                d.filename,
                COALESCE(
                    (
                        SELECT group_concat(associated.filename, ', ')
                        FROM conversation_documents AS cd
                        JOIN documents AS associated
                            ON associated.document_id = cd.document_id
                        WHERE cd.conversation_id = c.conversation_id
                    ),
                    d.filename
                ) AS filenames
            FROM conversations AS c
            JOIN documents AS d ON d.document_id = c.document_id
            WHERE c.conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

    return dict(row) if row else None


def rename_conversation(
    conversation_id: str,
    title: str,
) -> None:
    """Rename a conversation and update its activity timestamp."""
    cleaned_title = title.strip()

    if not cleaned_title:
        raise ValueError("Conversation title cannot be empty.")

    initialize_database()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (cleaned_title[:120], conversation_id),
        )

        if cursor.rowcount == 0:
            raise ValueError("Conversation not found.")


def delete_conversation(conversation_id: str) -> None:
    """Delete one conversation and its messages, not its PDF files."""
    initialize_database()

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM conversation_documents WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )


def _get_or_create_default_conversation(document_id: str) -> str:
    """Find or create a default conversation for legacy callers."""
    initialize_database()

    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT conversation_id
            FROM conversations
            WHERE document_id = ?
            ORDER BY created_at, conversation_id
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()

        if existing:
            return str(existing["conversation_id"])

    return create_conversation(document_id)["conversation_id"]


def save_chat_message(
    document_id: str,
    role: str,
    content: str,
    source_pages: Sequence[int] = (),
    conversation_id: str | None = None,
) -> None:
    """Save a message if its PDF belongs to the target conversation."""
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'.")

    cleaned_content = content.strip()
    if not cleaned_content:
        raise ValueError("Chat message cannot be empty.")

    initialize_database()

    target_conversation_id = conversation_id or _get_or_create_default_conversation(
        document_id
    )

    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT conversation_id
            FROM conversations
            WHERE conversation_id = ?
            """,
            (target_conversation_id,),
        ).fetchone()

        if conversation is None:
            raise ValueError("Conversation not found.")

        # In a multi-PDF chat, a message may reference any PDF linked
        # to that conversation, not only its legacy anchor PDF.
        association = connection.execute(
            """
            SELECT 1
            FROM conversation_documents
            WHERE conversation_id = ?
              AND document_id = ?
            """,
            (target_conversation_id, document_id),
        ).fetchone()

        if association is None:
            raise ValueError("The selected PDF is not linked to this conversation.")

        connection.execute(
            """
            INSERT INTO chat_messages
                (document_id, conversation_id, role, content, source_pages)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                target_conversation_id,
                role,
                cleaned_content,
                json.dumps(list(source_pages)),
            ),
        )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (target_conversation_id,),
        )

        current_title = connection.execute(
            """
            SELECT title
            FROM conversations
            WHERE conversation_id = ?
            """,
            (target_conversation_id,),
        ).fetchone()["title"]

        if role == "user" and current_title == "New chat":
            connection.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE conversation_id = ?
                """,
                (cleaned_content[:60], target_conversation_id),
            )


def load_chat_history(
    document_id: str,
    conversation_id: str | None = None,
) -> list[dict[str, object]]:
    """Load one conversation's history or legacy history for one PDF."""
    initialize_database()

    if conversation_id is not None:
        query = """
            SELECT role, content, source_pages, created_at
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY message_id
        """
        parameters: tuple[str, ...] = (conversation_id,)
    else:
        query = """
            SELECT role, content, source_pages, created_at
            FROM chat_messages
            WHERE document_id = ?
            ORDER BY message_id
        """
        parameters = (document_id,)

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [
        {
            "role": row["role"],
            "content": row["content"],
            "source_pages": json.loads(row["source_pages"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
