import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from ollama_client import embed_model, embedding_dimension


class QdrantStorage:
    def __init__(self, url=None, collection=None, dim=None):
        self.client = QdrantClient(url=url or os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=30)
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "docs")
        self.dim = dim or embedding_dimension()
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            return

        collection_info = self.client.get_collection(self.collection)
        vectors_config = collection_info.config.params.vectors
        current_dim = getattr(vectors_config, "size", None)
        if current_dim is None and isinstance(vectors_config, dict):
            first_vector = next(iter(vectors_config.values()))
            current_dim = getattr(first_vector, "size", None)

        if current_dim != self.dim:
            raise ValueError(
                f"Qdrant collection '{self.collection}' expects vectors of size {current_dim}, "
                f"but Ollama model '{embed_model()}' "
                f"returns size {self.dim}. Recreate the collection or use a matching embedding model."
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}