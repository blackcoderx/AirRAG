import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)


class QdrantStore:
    def __init__(self, path: str, embed_dim: int):
        self._client = QdrantClient(path=path)
        self._embed_dim = embed_dim

    def _ensure_collection(self, collection_name: str) -> None:
        # Creates the Qdrant collection on first use; cosine distance matches the Gemini embedding space
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self._embed_dim, distance=Distance.COSINE),
            )

    def upsert(
        self,
        collection_name: str,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._ensure_collection(collection_name)
        points = [
            PointStruct(
                # uuid5 derives a deterministic UUID from the string chunk_id, making re-ingestion idempotent
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                vector=embedding,
                # Store the raw text/label alongside caller metadata so search results are self-contained
                payload={"chunk_id": chunk_id, "document": doc, **meta},
            )
            for chunk_id, embedding, doc, meta in zip(chunk_ids, embeddings, documents, metadatas)
        ]
        self._client.upsert(collection_name=collection_name, points=points, wait=True)

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        if not self._client.collection_exists(collection_name):
            return []
        count = self._client.count(collection_name).count
        if count == 0:
            return []
        # min(top_k, count) prevents requesting more results than exist, which errors in some qdrant versions
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=min(top_k, count),
            with_payload=True,
        )
        return [
            {
                "chunk_id": r.payload.get("chunk_id"),
                "document": r.payload.get("document"),
                # Strip internal keys so callers only see the original metadata fields
                "metadata": {k: v for k, v in r.payload.items() if k not in ("chunk_id", "document")},
                "score": r.score,
            }
            for r in response.points
        ]

    def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        # Silently returns if the collection was already deleted (e.g. cascade from collection delete)
        if not self._client.collection_exists(collection_name):
            return
        self._client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            ),
            wait=True,
        )

    def delete_collection(self, collection_name: str) -> None:
        # Guard prevents an error if the collection was already removed
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
