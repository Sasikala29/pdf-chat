
"""Streamlit interface for DocuMind, a multi-PDF conversational assistant."""

import logging

import streamlit as st
from pydantic import ValidationError

from src.pdf_chatbot.core.config import get_settings
from src.pdf_chatbot.database.connection import initialize_database
from src.pdf_chatbot.repositories.sqlite_repository import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversation_documents,
    list_conversations,
    list_documents,
    set_conversation_documents,
)
from src.pdf_chatbot.services.chat_service import (
    answer_question,
    get_document_chat_history,
)
from src.pdf_chatbot.services.document_service import process_uploaded_pdf
from src.pdf_chatbot.services.deletion_service import delete_pdf_and_chats
from src.pdf_chatbot.services.pdf_service import PDFProcessingError
from src.pdf_chatbot.utils.pdf_identity import unique_documents_by_content

logger = logging.getLogger(__name__)

# Configure the browser tab and overall page layout.
st.set_page_config(
    page_title="DocuMind | Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep styling lightweight and compatible with Streamlit's theme.
st.markdown(
    """
    <style>
    .stApp {
        background: var(--background-color);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.25);
    }

    .app-brand {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        opacity: 0.72;
        margin-bottom: 1.25rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.85rem 1rem;
    }

    [data-testid="stChatInput"] {
        padding-top: 0.75rem;
    }

    .section-caption {
        opacity: 0.7;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def prepare_application() -> bool:
    """Initialize the existing database once per Streamlit process."""
    initialize_database()
    return True


try:
    prepare_application()
except Exception:
    logger.exception("Database initialization failed.")
    st.error(
        "DocuMind could not initialize its database. "
        "Check the terminal for details."
    )
    st.stop()


def load_application_data() -> tuple[list[dict], list[dict]]:
    """Load saved PDFs and conversations from the existing database."""
    # Hide legacy duplicate uploads by comparing actual file contents.
    # This changes only the display list; it does not alter existing records.
    return unique_documents_by_content(list_documents()), list_conversations()


def get_active_conversation(
    conversation_id: str | None,
) -> dict | None:
    """Return the selected conversation if it still exists."""
    if not conversation_id:
        return None

    return get_conversation(conversation_id)


def create_chat_for_documents(document_ids: list[str]) -> str:
    """Create a conversation and associate it with existing PDFs."""
    unique_document_ids = list(
        dict.fromkeys(
            document_id
            for document_id in document_ids
            if document_id
        )
    )

    if not unique_document_ids:
        raise ValueError("Select at least one PDF before starting a chat.")

    # Validate selected IDs so UI labels cannot be mistaken for document IDs.
    saved_document_ids = {
        document["document_id"] for document in list_documents()
    }

    invalid_ids = [
        document_id
        for document_id in unique_document_ids
        if document_id not in saved_document_ids
    ]

    if invalid_ids:
        raise ValueError(
            "The selected PDF list contains invalid document IDs. "
            "Refresh the page and select the PDFs again."
        )

    conversation = create_conversation(
        unique_document_ids[0],
        title="New chat",
    )
    conversation_id = conversation["conversation_id"]

    set_conversation_documents(
        conversation_id,
        unique_document_ids,
    )

    return conversation_id


# Initialize session state without resetting an existing selection.
if "active_conversation_id" not in st.session_state:
    st.session_state["active_conversation_id"] = None


try:
    documents, conversations = load_application_data()
except Exception:
    logger.exception("Could not load saved PDFs and conversations.")
    st.error(
        "Saved PDFs and conversations could not be loaded. "
        "Check the terminal for details."
    )
    st.stop()


# Restore the selected conversation, if it still exists.
active_conversation_id = st.session_state["active_conversation_id"]

try:
    active_conversation = get_active_conversation(active_conversation_id)
except Exception:
    logger.exception("Could not load the selected conversation.")
    active_conversation = None

if active_conversation is None:
    st.session_state["active_conversation_id"] = None
    active_conversation_id = None


# Build ID-to-document and display-label-to-ID mappings.
document_by_id = {
    document["document_id"]: document
    for document in documents
}

document_options = {
    (
        f"{document['filename']} "
        f"· {document['document_id'][:8]}"
    ): document["document_id"]
    for document in documents
}

document_id_by_label = {
    label: document_id
    for label, document_id in document_options.items()
}


# Load the PDFs currently associated with the selected conversation.
current_associations: list[dict] = []
if active_conversation_id:
    try:
        current_associations = list_conversation_documents(
            active_conversation_id
        )
        current_document_ids = list(
            dict.fromkeys(
                item["document_id"]
                for item in current_associations
                if item["document_id"] in document_by_id
            )
        )
    except Exception:
        logger.exception("Could not load conversation PDF associations.")
        current_document_ids = []
else:
    current_document_ids = []


# ---------------------------------------------------------------------------
# Sidebar: one uploader, saved-PDF selection, explicit action menus, history.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 DocuMind")
    st.caption("Your intelligent document assistant.")

    # Session-state keys keep each menu and confirmation independent.
    st.session_state.setdefault("open_pdf_menu_id", None)
    st.session_state.setdefault("confirm_delete_pdf_id", None)
    st.session_state.setdefault("open_chat_menu_id", None)
    st.session_state.setdefault("confirm_delete_chat_id", None)
    st.session_state.setdefault("new_chat_selection_version", 0)

    st.divider()
    st.markdown("### Upload PDFs")
    st.caption("Upload new PDFs here. Saved PDFs can be reused below.")

    try:
        max_size_mb = get_settings().max_upload_size_mb
    except (ValidationError, ValueError):
        logger.exception("Application configuration is invalid.")
        st.error("Check your application configuration and .env file.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum file size per PDF: {max_size_mb} MB.",
        key="pdf_multi_uploader",
    )

    if st.button(
        "Process uploaded PDFs",
        use_container_width=True,
        disabled=not uploaded_files,
        key="process_uploaded_pdfs_button",
    ):
        processed_ids: list[str] = []
        failures: list[str] = []

        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    processed = process_uploaded_pdf(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )
                processed_ids.append(processed.document_id)
                if getattr(processed, "reused_existing", False):
                    st.success(
                        f"{uploaded_file.name}: identical PDF already saved; "
                        "reused the existing indexed document."
                    )
                else:
                    st.success(
                        f"{processed.filename}: {processed.page_count} pages indexed."
                    )
            except (PDFProcessingError, ValueError) as exc:
                failures.append(f"{uploaded_file.name}: {exc}")
            except Exception:
                logger.exception("Failed to process uploaded PDF %s.", uploaded_file.name)
                failures.append(f"{uploaded_file.name}: unexpected processing error.")

        if processed_ids:
            try:
                existing_id = st.session_state.get("active_conversation_id")
                existing_conversation = get_active_conversation(existing_id)
                if existing_conversation is not None:
                    associations = list_conversation_documents(existing_id)
                    combined_ids = [item["document_id"] for item in associations]
                    combined_ids.extend(processed_ids)
                    set_conversation_documents(existing_id, list(dict.fromkeys(combined_ids)))
                else:
                    new_id = create_chat_for_documents(processed_ids)
                    st.session_state["active_conversation_id"] = new_id
                st.rerun()
            except Exception:
                logger.exception("Processed PDFs, but could not attach them to a conversation.")
                st.error(
                    "The PDFs were processed, but could not be attached to the chat. "
                    "Their saved records were not intentionally deleted."
                )

        for failure in failures:
            st.error(failure)

    st.divider()
    st.markdown("### Saved PDFs")
    st.caption("Tick saved PDFs to use them in a new chat. Use ⋮ to manage a PDF.")

    if not documents:
        st.info("No PDFs saved yet. Upload a PDF above to get started.")
    else:
        selection_version = st.session_state["new_chat_selection_version"]
        selected_ids_key = f"new_chat_selected_ids_{selection_version}"
        if selected_ids_key not in st.session_state:
            st.session_state[selected_ids_key] = []

        valid_ids = {doc["document_id"] for doc in documents}
        st.session_state[selected_ids_key] = [
            doc_id for doc_id in st.session_state[selected_ids_key]
            if doc_id in valid_ids
        ]

        for document in documents:
            document_id = document["document_id"]
            filename = document.get("filename") or "Untitled PDF"
            left, right = st.columns([0.82, 0.18], vertical_alignment="center")
            with left:
                selected = st.checkbox(
                    filename,
                    value=document_id in st.session_state[selected_ids_key],
                    key=f"new_chat_pdf_checkbox_{selection_version}_{document_id}",
                    help="Select this saved PDF for the next new chat.",
                )
                selected_ids = list(st.session_state[selected_ids_key])
                if selected and document_id not in selected_ids:
                    selected_ids.append(document_id)
                    st.session_state[selected_ids_key] = selected_ids
                elif not selected and document_id in selected_ids:
                    st.session_state[selected_ids_key] = [
                        item for item in selected_ids if item != document_id
                    ]
            with right:
                if st.button(
                    "⋮",
                    key=f"pdf_menu_button_{document_id}",
                    help=f"Actions for {filename}",
                    use_container_width=True,
                ):
                    st.session_state["open_pdf_menu_id"] = (
                        None if st.session_state["open_pdf_menu_id"] == document_id
                        else document_id
                    )
                    st.session_state["confirm_delete_pdf_id"] = None
                    st.rerun()

            if st.session_state["open_pdf_menu_id"] == document_id:
                with st.container(border=True):
                    st.caption(filename)
                    if st.button(
                        "🗑️ Delete PDF",
                        key=f"pdf_delete_action_{document_id}",
                        use_container_width=True,
                    ):
                        st.session_state["confirm_delete_pdf_id"] = document_id
                        st.rerun()
                    if st.button(
                        "Close menu",
                        key=f"pdf_close_menu_{document_id}",
                        use_container_width=True,
                    ):
                        st.session_state["open_pdf_menu_id"] = None
                        st.rerun()

            if st.session_state["confirm_delete_pdf_id"] == document_id:
                with st.container(border=True):
                    st.warning(
                        f"Delete **{filename}** and its associated conversations? "
                        "This cannot be undone."
                    )
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button(
                            "Confirm delete",
                            key=f"pdf_confirm_delete_{document_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                delete_pdf_and_chats(document_id)
                                if st.session_state.get("active_conversation_id"):
                                    current_id = st.session_state["active_conversation_id"]
                                    if get_active_conversation(current_id) is None:
                                        st.session_state["active_conversation_id"] = None
                                st.session_state["open_pdf_menu_id"] = None
                                st.session_state["confirm_delete_pdf_id"] = None
                                st.session_state["new_chat_selection_version"] += 1
                                st.success(f"Deleted {filename} and its associated chats.")
                                st.rerun()
                            except Exception:
                                logger.exception("Could not delete PDF %s.", document_id)
                                st.error("PDF deletion failed. Check the terminal for details.")
                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"pdf_cancel_delete_{document_id}",
                            use_container_width=True,
                        ):
                            st.session_state["confirm_delete_pdf_id"] = None
                            st.session_state["open_pdf_menu_id"] = None
                            st.rerun()

        if st.button(
            "＋ New Chat",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state[selected_ids_key],
            key=f"create_new_chat_{selection_version}",
        ):
            try:
                new_id = create_chat_for_documents(st.session_state[selected_ids_key])
                st.session_state["active_conversation_id"] = new_id
                st.session_state["new_chat_selection_version"] += 1
                st.rerun()
            except Exception:
                logger.exception("Could not create a new conversation.")
                st.error("Could not create a new chat. Check the terminal for details.")

    st.divider()
    st.markdown("### PDFs in this chat")
    if active_conversation_id and documents:
        selection_key = f"pdf_selection_{active_conversation_id}"
        valid_current_ids = [
            item["document_id"] for item in current_associations
            if item.get("document_id") in document_by_id
        ] if active_conversation_id else []

        if selection_key not in st.session_state:
            st.session_state[selection_key] = [
                label for label, doc_id in document_options.items()
                if doc_id in valid_current_ids
            ]

        selected_pdf_labels = st.multiselect(
            "Attach saved PDFs to this conversation",
            options=list(document_options.keys()),
            key=selection_key,
            help="This changes which saved PDFs the current conversation searches.",
        )
        selected_pdf_ids = [
            document_options[label] for label in selected_pdf_labels
            if label in document_options
        ]
        if st.button("Save PDF selection", use_container_width=True, key=f"save_pdf_selection_{active_conversation_id}"):
            try:
                if not selected_pdf_ids:
                    st.warning("Select at least one PDF for this conversation.")
                else:
                    set_conversation_documents(active_conversation_id, selected_pdf_ids)
                    st.success("PDF selection saved.")
                    st.rerun()
            except Exception:
                logger.exception("Could not update conversation PDF associations.")
                st.error("Could not save the PDF selection. Check the terminal for details.")
    elif not active_conversation_id:
        st.caption("Create or open a chat to manage its PDF selection.")
    else:
        st.caption("Upload or save a PDF before attaching it to a chat.")

    st.divider()
    st.markdown("### Chat history")
    if not conversations:
        st.caption("Your saved conversations will appear here.")
    else:
        for conversation in conversations:
            conversation_id = conversation["conversation_id"]
            title = conversation.get("title") or "New chat"
            filenames = conversation.get("filenames") or conversation.get("filename", "")
            if isinstance(filenames, list):
                pdf_summary = ", ".join(dict.fromkeys(name for name in filenames if name))
            else:
                pdf_summary = str(filenames or "")

            left, right = st.columns([0.82, 0.18], vertical_alignment="center")
            with left:
                button_label = f"💬 {title}"
                if pdf_summary:
                    button_label += f"\n{pdf_summary}"
                if st.button(
                    button_label,
                    key=f"open_conversation_{conversation_id}",
                    use_container_width=True,
                    type="primary" if conversation_id == active_conversation_id else "secondary",
                ):
                    st.session_state["active_conversation_id"] = conversation_id
                    st.rerun()
            with right:
                if st.button(
                    "⋮",
                    key=f"chat_menu_button_{conversation_id}",
                    help=f"Actions for {title}",
                    use_container_width=True,
                ):
                    st.session_state["open_chat_menu_id"] = (
                        None if st.session_state["open_chat_menu_id"] == conversation_id
                        else conversation_id
                    )
                    st.session_state["confirm_delete_chat_id"] = None
                    st.rerun()

            if st.session_state["open_chat_menu_id"] == conversation_id:
                with st.container(border=True):
                    st.caption(title)
                    if st.button(
                        "🗑️ Delete chat",
                        key=f"chat_delete_action_{conversation_id}",
                        use_container_width=True,
                    ):
                        st.session_state["confirm_delete_chat_id"] = conversation_id
                        st.rerun()
                    if st.button(
                        "Close menu",
                        key=f"chat_close_menu_{conversation_id}",
                        use_container_width=True,
                    ):
                        st.session_state["open_chat_menu_id"] = None
                        st.rerun()

            if st.session_state["confirm_delete_chat_id"] == conversation_id:
                with st.container(border=True):
                    st.warning(
                        f"Delete chat **{title}** and its messages? "
                        "The saved PDFs will remain available."
                    )
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button(
                            "Confirm delete",
                            key=f"chat_confirm_delete_{conversation_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                delete_conversation(conversation_id)
                                if st.session_state.get("active_conversation_id") == conversation_id:
                                    st.session_state["active_conversation_id"] = None
                                st.session_state["open_chat_menu_id"] = None
                                st.session_state["confirm_delete_chat_id"] = None
                                st.rerun()
                            except Exception:
                                logger.exception("Could not delete conversation %s.", conversation_id)
                                st.error("Chat deletion failed. Check the terminal for details.")
                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"chat_cancel_delete_{conversation_id}",
                            use_container_width=True,
                        ):
                            st.session_state["confirm_delete_chat_id"] = None
                            st.session_state["open_chat_menu_id"] = None
                            st.rerun()


# ---------------------------------------------------------------------------
# Main area: DocuMind branding, active conversation, and chat messages.
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-brand">📚 DocuMind</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">'
    "Your intelligent document assistant. "
    "Ask questions and discover insights across your PDFs."
    "</div>",
    unsafe_allow_html=True,
)

active_conversation_id = st.session_state.get(
    "active_conversation_id"
)

try:
    active_conversation = get_active_conversation(
        active_conversation_id
    )
except Exception:
    logger.exception("Could not load the active conversation.")
    active_conversation = None

if active_conversation is None:
    if documents:
        st.info(
            "Select a saved conversation or click **New Chat** "
            "in the sidebar to begin."
        )
    else:
        st.info(
            "Upload and process one or more PDFs in the sidebar "
            "to begin your first conversation."
        )
    st.stop()


active_document_id = active_conversation["document_id"]

try:
    active_associations = list_conversation_documents(
        active_conversation_id
    )
except Exception:
    logger.exception("Could not load PDFs for the active conversation.")
    active_associations = []

# Deduplicate filenames for display without changing database associations.
active_pdf_names = list(
    dict.fromkeys(
        item["filename"]
        for item in active_associations
        if item.get("filename")
    )
)

st.subheader(active_conversation.get("title") or "New chat")

if active_pdf_names:
    st.caption(
        "PDFs in this conversation: " + ", ".join(active_pdf_names)
    )
else:
    st.warning(
        "No PDFs are associated with this conversation. "
        "Select PDFs in the sidebar and save the selection."
    )

try:
    history = get_document_chat_history(
        active_document_id,
        active_conversation_id,
    )
except Exception:
    logger.exception(
        "Could not load history for conversation %s.",
        active_conversation_id,
    )
    history = []
    st.warning(
        "Previous messages could not be loaded. "
        "Check the terminal for details."
    )

for message in history:
    role = str(message.get("role", "assistant"))
    if role not in {"user", "assistant"}:
        continue

    with st.chat_message(role):
        st.markdown(str(message.get("content", "")))


question = st.chat_input(
    "Ask a question about your selected PDFs...",
    key=f"question_{active_conversation_id}",
    disabled=not active_pdf_names,
)

if question:
    try:
        with st.spinner(
            "Searching your PDFs and preparing an answer..."
        ):
            answer_question(
                active_document_id,
                question,
                conversation_id=active_conversation_id,
            )
        st.rerun()
    except ValueError as exc:
        st.warning(str(exc))
    except Exception:
        logger.exception(
            "Could not answer a question for conversation %s.",
            active_conversation_id,
        )
        st.error(
            "The question could not be answered. "
            "Check the terminal for details."
        )
