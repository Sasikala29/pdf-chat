# PDF Chatbot

A Python-based PDF question-answering application built with Streamlit. Users can upload PDF documents, ask questions about their content, and receive answers with source-page references.

## Features

- Upload PDF documents through a Streamlit interface.
- Extract text while preserving original page numbers.
- Split extracted text into searchable chunks.
- Generate local text embeddings using Sentence Transformers.
- Store document metadata, chunks, embeddings, and chat history in SQLite.
- Retrieve answers from the uploaded document's original text.
- Display source-page references with supported answers.
- Return a clear fallback message when the document does not contain enough information.
- Keep chat history associated with individual documents.

## Technology Stack

- **Python** — application logic
- **Streamlit** — user interface
- **PyMuPDF** — PDF text extraction
- **Sentence Transformers** — local text embeddings
- **SQLite** — local persistence
- **Groq API** — configured language-model integration
- **Pydantic Settings** — configuration validation
- **Pytest** — automated testing

## Project Structure

```text
pdf-chatbot/
├── app.py
├── requirements.in
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── uploads/
│   └── pdf_chatbot.db
├── src/
│   └── pdf_chatbot/
│       ├── core/
│       │   └── config.py
│       ├── database/
│       │   └── connection.py
│       ├── models/
│       ├── repositories/
│       ├── services/
│       └── utils/
└── tests/
```

The database and uploaded files are generated locally and should not be committed to version control.

## Prerequisites

- Python 3.14, if supported by all installed dependencies
- A Groq API key
- Internet access for Groq API calls and initial model downloads

## Installation

Open PowerShell in the project root.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Create a local environment file from the example:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure your API key and model:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_supported_groq_model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_UPLOAD_SIZE_MB=20
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

Replace the API key and model value with valid values for your Groq account. Never commit `.env` or share its contents.

## Run the Application

From the project root, with the virtual environment activated:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, upload a PDF, and ask a question about its contents.

## Run Tests

Run the complete automated test suite:

```powershell
python -m pytest -q
```

Run the chat-answering tests only:

```powershell
python -m pytest tests/test_chat_service.py -q
```

## Answering Behavior

The application is designed to return text supported by the uploaded PDF rather than inventing an answer. Supported answers include a source-page reference. If the document does not provide enough information, the application returns:

> The uploaded PDF does not contain enough information to answer this question.

## Security Notes

- Keep API keys in `.env`, never in source code.
- Keep `.env`, `.venv`, local databases, uploaded PDFs, and generated caches out of Git.
- Validate uploaded file types and file sizes.
- Treat uploaded documents as untrusted input.
- Do not expose internal errors, secrets, or sensitive document contents in logs.
- Use appropriate access controls before deploying the application for multiple users.

## Limitations

- Answers depend on the text successfully extracted from the PDF.
- Scanned PDFs may require OCR support.
- Answer matching may not handle every paraphrase or complex question.
- Groq API access requires a valid key, network connectivity, and model availability.
- SQLite is intended for local persistence and modest workloads; production multi-user deployments may require additional access controls and a different database strategy.

## License

Add the appropriate license before distributing this project publicly.