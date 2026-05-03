import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)


class QdrantStore:
    """Wraps Qdrant vector database for storing and searching embeddings.

    All embeddings (text, images, PDF, audio, video) are stored in the same
    3072-dim vector space, enabling cross-modal retrieval.

    Used by: ingestor.py (store embeddings), query.py (search embeddings).
    Related: embedder.py (generates the vectors stored here).
    """

    def __init__(self, url: str, embed_dim: int):
        "Initialize Qdrant client (supports :memory: for testing)."
        if url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=url)
        self._embed_dim = embed_dim

    def _ensure_collection(self, collection_name: str) -> None:
        """Create Qdrant collection on first use (cosine distance = Gemini embedding space).

        Cosine similarity is the standard for Gemini embeddings.
        """
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._embed_dim, distance=Distance.COSINE
                ),
            )

    def upsert(
        self,
        collection_name: str,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Store embeddings with metadata in Qdrant."""
        self._ensure_collection(collection_name)
        points = [
            PointStruct(
                # uuid5 derives deterministic UUID from string, making re-ingestion idempotent
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                vector=embedding,
                # Store raw text alongside caller metadata so search results are self-contained
                payload={"chunk_id": chunk_id, "document": doc, **meta},
            )
            for chunk_id, embedding, doc, meta in zip(
                chunk_ids, embeddings, documents, metadatas
            )
        ]
        self._client.upsert(collection_name=collection_name, points=points, wait=True)

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Search for similar vectors in Qdrant.

        Returns: List of dicts with chunk_id, document text, metadata, and score.
        Used by: query.py (after embedding user's query).
        """
        if not self._client.collection_exists(collection_name):
            return []
        count = self._client.count(collection_name).count
        if count == 0:
            return []
        # min(top_k, count) prevents requesting more results than exist
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=min(top_k, count),
            with_payload=True,
        )
        return [
            {
                "chunk_id": r.payload.get("chunk_id")
                if r.payload is not None
                else None,
                "document": r.payload.get("document")
                if r.payload is not None
                else None,
                # Strip internal keys so callers only see original metadata fields
                "metadata": {
                    k: v
                    for k, v in (r.payload or {}).items()
                    if k not in ("chunk_id", "document")
                },
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
                    must=[
                        FieldCondition(
                            key="document_id", match=MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_collection(self, collection_name: str) -> None:
        # Guard prevents an error if the collection was already removed
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
