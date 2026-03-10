# RAG Production Streamlit

A small local RAG app built with Streamlit, FastAPI, Inngest, Qdrant, and Ollama.

You can:

- upload documents for ingestion
- chunk and embed it with Ollama
- store vectors in Qdrant
- ask questions in Streamlit
- generate answers from retrieved context with Ollama

## Stack

- `Streamlit` for the UI
- `FastAPI` to serve Inngest functions
- `Inngest` for ingestion and query workflows
- `Qdrant` as the vector store
- `Ollama` for embeddings and chat
- `uv` for Python dependency management

## Prerequisites

Install these locally before running the app:

- `uv`
- `Docker`
- `Ollama`
- `inngest` CLI

The helper install script currently does:

```bash
./scripts/install.sh
```

That script runs:

- `uv sync`
- `brew install inngest/tap/inngest`

## Environment Setup

Copy the example environment file and adjust models if needed:

```bash
cp .env.example .env
```

Current environment variables:

- `OLLAMA_BASE_URL`: Ollama server URL
- `OLLAMA_CHAT_MODEL`: model used for answer generation
- `OLLAMA_EMBED_MODEL`: model used for embeddings
- `INNGEST_API_BASE`: optional override for polling Inngest dev API
- `QDRANT_URL`: optional override for Qdrant
- `QDRANT_COLLECTION`: optional override for the collection name

Example:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=phi3:3.8b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Pull Ollama Models

Make sure the required models exist locally:

```bash
ollama pull phi3:3.8b
ollama pull nomic-embed-text
```

You can confirm with:

```bash
ollama list
```

## Run The App

Start everything with:

```bash
./scripts/run.sh
```

This starts:

- Qdrant on `http://127.0.0.1:6333`
- FastAPI on `http://127.0.0.1:8000`
- Inngest dev server
- Streamlit app

Then open the Streamlit URL shown in your terminal.

## How It Works

### Ingest flow

1. Upload a supported document in the Streamlit app.
2. The UI saves the file into `uploads/`.
3. Streamlit sends a `rag/ingest_file` event to Inngest.
4. The backend loads and chunks the document.
5. Chunks are embedded with Ollama.
6. Vectors and source metadata are stored in Qdrant.

### Query flow

1. Ask a question in Streamlit.
2. Streamlit sends a `rag/query_pdf_ai` event.
3. The backend embeds the question with Ollama.
4. Qdrant returns the most relevant chunks.
5. Ollama generates an answer using only the retrieved context.

## Project Structure

Key files:

- `streamlit_app.py`: UI for upload and querying
- `main.py`: Inngest-backed ingestion and query functions
- `data_loader.py`: document loading, chunking, and embedding bridge
- `ollama_client.py`: direct Ollama chat and embedding calls
- `vector_db.py`: Qdrant collection and search logic
- `custom_types.py`: Pydantic models for workflow outputs
- `scripts/run.sh`: local dev startup script
- `scripts/install.sh`: dependency/bootstrap helper

## Notes

- Supported upload types are `pdf`, `txt`, `md`, and `docx`.
- Uploaded files are stored locally in `uploads/`.
- `.env` is ignored by git; use `.env.example` as the shared template.
