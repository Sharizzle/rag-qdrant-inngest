import logging
import os
from pathlib import Path

from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import datetime
from data_loader import SUPPORTED_FILE_SUFFIXES, load_and_chunk_file, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
from ollama_client import chat as ollama_chat

load_dotenv()
logger = logging.getLogger("uvicorn")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


def _is_truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _base_files_dir() -> Path:
    return Path(os.getenv("BASE_FILES_DIR", "base_files"))


def _load_file_chunks(file_path: str, source_id: str | None = None) -> RAGChunkAndSrc:
    resolved_source_id = source_id or file_path
    chunks = load_and_chunk_file(file_path)
    return RAGChunkAndSrc(chunks=chunks, source_id=resolved_source_id)


def _upsert_chunks(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
    chunks = chunks_and_src.chunks
    source_id = chunks_and_src.source_id
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    QdrantStorage().upsert(ids, vecs, payloads)
    return RAGUpsertResult(ingested=len(chunks))


def _iter_default_files() -> list[Path]:
    base_dir = _base_files_dir()
    if not base_dir.exists():
        logger.info("Default ingest enabled but '%s' does not exist; skipping.", base_dir)
        return []

    return sorted(
        path for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_SUFFIXES
    )

@inngest_client.create_function(
    fn_id="RAG: Ingest File",
    trigger=inngest.TriggerEvent(event="rag/ingest_file"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_file(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        file_path = ctx.event.data["file_path"]
        source_id = ctx.event.data.get("source_id", file_path)
        return _load_file_chunks(file_path, source_id)

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert_chunks(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query Documents",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    def _answer(question: str, found: RAGSearchResult) -> RAQQueryResult:
        context_block = "\n\n".join(f"- {c}" for c in found.contexts)
        user_content = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context above."
        )

        answer = ollama_chat(
            [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            num_predict=1024,
        )
        return RAQQueryResult(answer=answer, sources=found.sources, num_contexts=len(found.contexts))

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)
    result = await ctx.step.run("answer-with-ollama", lambda: _answer(question, found), output_type=RAQQueryResult)
    return result.model_dump()

app = FastAPI()


@app.on_event("startup")
async def ingest_default_files_on_startup():
    if not _is_truthy_env("INGEST_DEFAULT_FILES"):
        return

    default_files = _iter_default_files()
    if not default_files:
        logger.info("No supported files found in '%s'; skipping default ingest.", _base_files_dir())
        return

    logger.info("Default ingest enabled; ingesting %s file(s) from '%s'.", len(default_files), _base_files_dir())
    for file_path in default_files:
        source_id = str(file_path.relative_to(_base_files_dir()))
        try:
            chunks_and_src = _load_file_chunks(str(file_path.resolve()), source_id=source_id)
            ingested = _upsert_chunks(chunks_and_src)
            logger.info("Default-ingested '%s' with %s chunk(s).", source_id, ingested.ingested)
        except Exception:
            logger.exception("Failed to default-ingest '%s'.", file_path)


inngest.fast_api.serve(app, inngest_client, [rag_ingest_file, rag_query_pdf_ai])