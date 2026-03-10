from pathlib import Path

from docx import Document
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from ollama_client import embed_texts as ollama_embed_texts

load_dotenv()

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
SUPPORTED_FILE_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}


def _load_pdf_text(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)
    return [d.text for d in docs if getattr(d, "text", None)]


def _load_text_file(path: str) -> list[str]:
    file_path = Path(path)
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return [file_path.read_text(encoding=encoding)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode text file: {path}")


def _load_docx_text(path: str) -> list[str]:
    document = Document(path)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    return [text] if text else []


def load_and_chunk_file(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        texts = _load_pdf_text(path)
    elif suffix in {".txt", ".md"}:
        texts = _load_text_file(path)
    elif suffix == ".docx":
        texts = _load_docx_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    return ollama_embed_texts(texts)