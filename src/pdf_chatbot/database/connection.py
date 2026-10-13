
"""SQLite connection management and backward-compatible database migrations."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Resolve the database location from the project root.
DATABASE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "pdf_chatbot.db"
)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Provide a configured SQLite connection and close it after use."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    """Create required tables and migrate existing data without deleting it."""
    with get_connection() as connection:
        # Create the original application tables if they do not exist.
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT,
                uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL CHECK (page_number >= 1),
                chunk_index INTEGER NOT NULL CHECK (chunk_index >= 1),
                text TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (chunk_id)
                    REFERENCES document_chunks(chunk_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                source_pages TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE
            );
            """
        )

        # Inspect existing columns before applying migrations.
        document_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(documents)"
            ).fetchall()
        }

        if "file_path" not in document_columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN file_path TEXT"
            )

        conversation_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(conversations)"
            ).fetchall()
        }

        if "title" not in conversation_columns:
            connection.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat'
                """
            )

        if "created_at" not in conversation_columns:
            connection.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN created_at TEXT
                """
            )
            connection.execute(
                """
                UPDATE conversations
                SET created_at = CURRENT_TIMESTAMP
                WHERE created_at IS NULL
                """
            )

        if "updated_at" not in conversation_columns:
            connection.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN updated_at TEXT
                """
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = COALESCE(created_at, CURRENT_TIMESTAMP)
                WHERE updated_at IS NULL
                """
            )

        message_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(chat_messages)"
            ).fetchall()
        }

        if "source_pages" not in message_columns:
            connection.execute(
                """
                ALTER TABLE chat_messages
                ADD COLUMN source_pages TEXT NOT NULL DEFAULT '[]'
                """
            )

        if "conversation_id" not in message_columns:
            connection.execute(
                """
                ALTER TABLE chat_messages
                ADD COLUMN conversation_id TEXT
                """
            )

        # Ensure every existing PDF has a default conversation.
        documents = connection.execute(
            "SELECT document_id FROM documents"
        ).fetchall()

        for document in documents:
            document_id = document["document_id"]

            existing_conversation = connection.execute(
                """
                SELECT conversation_id
                FROM conversations
                WHERE document_id = ?
                ORDER BY created_at, conversation_id
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()

            if existing_conversation is None:
                # UUID-based identifiers are generated for new conversations.
                import uuid

                conversation_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO conversations (
                        conversation_id, document_id, title
                    )
                    VALUES (?, ?, 'New chat')
                    """,
                    (conversation_id, document_id),
                )

        # Link older messages to a conversation for their original PDF.
        connection.execute(
            """
            UPDATE chat_messages
            SET conversation_id = (
                SELECT c.conversation_id
                FROM conversations AS c
                WHERE c.document_id = chat_messages.document_id
                ORDER BY c.created_at, c.conversation_id
                LIMIT 1
            )
            WHERE conversation_id IS NULL
            """
        )

        # Create the missing many-to-many association table.
        # A conversation can now be linked to multiple existing PDFs.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_documents (
                conversation_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (conversation_id, document_id),
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE
            )
            """
        )

        # Backfill existing conversations with their original PDF.
        # INSERT OR IGNORE makes this migration safe to run repeatedly.
        connection.execute(
            """
            INSERT OR IGNORE INTO conversation_documents (
                conversation_id, document_id
            )
            SELECT conversation_id, document_id
            FROM conversations
            """
        )

        # Add indexes for common retrieval and history queries.
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_chunks_document
            ON document_chunks(document_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_document
            ON chat_messages(document_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
            ON chat_messages(conversation_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_documents_document
            ON conversation_documents(document_id)
            """
        )
