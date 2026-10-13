"""Verify that the conversation-to-PDF association table exists."""

from src.pdf_chatbot.database.connection import (
    get_connection,
    initialize_database,
)


def main() -> None:
    """Initialize the schema and print the table verification result."""
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = ? AND name = ?
            """,
            ("table", "conversation_documents"),
        ).fetchall()

        print([row["name"] for row in rows])


if __name__ == "__main__":
    main()