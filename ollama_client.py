import os

import requests
from dotenv import load_dotenv


load_dotenv()


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def default_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2")


def chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", default_model())


def embed_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_MODEL", "nomic-embed-text"))


def _post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    response = requests.post(
        f"{ollama_base_url()}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _should_try_legacy_embeddings(exc: requests.HTTPError) -> bool:
    response = exc.response
    if response is None or response.status_code != 404:
        return False

    try:
        error_message = response.json().get("error", "").lower()
    except ValueError:
        error_message = response.text.lower()

    # Ollama uses 404 for missing models too. Only fall back when the
    # route itself looks unavailable, not when the model is missing.
    endpoint_markers = (
        "not found",
        "404 page not found",
        "endpoint",
        "route",
    )
    model_markers = (
        "model",
        "pull",
    )

    if any(marker in error_message for marker in model_markers):
        return False

    return any(marker in error_message for marker in endpoint_markers) or not error_message


def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        data = _post(
            "/api/embed",
            {
                "model": embed_model(),
                "input": texts,
            },
        )
        embeddings = data.get("embeddings")
        if embeddings:
            return embeddings
    except requests.HTTPError as exc:
        # Older Ollama builds expose embeddings one prompt at a time.
        if not _should_try_legacy_embeddings(exc):
            raise

    embeddings = []
    for text in texts:
        data = _post(
            "/api/embeddings",
            {
                "model": embed_model(),
                "prompt": text,
            },
        )
        embeddings.append(data["embedding"])
    return embeddings


def embedding_dimension() -> int:
    return len(embed_texts(["dimension probe"])[0])


def chat(messages: list[dict], temperature: float = 0.2, num_predict: int = 1024) -> str:
    data = _post(
        "/api/chat",
        {
            "model": chat_model(),
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
    )
    return data["message"]["content"].strip()
