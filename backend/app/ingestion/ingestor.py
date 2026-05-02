from app.ingestion.parser import DocumentParser
from app.ingestion.embedder import GeminiEmbedder
from app.retrieval.vector_store import QdrantStore

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
SUPPORTED_DOC_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/markdown",
}


class Ingestor:
    def __init__(self, embedder: GeminiEmbedder, vector_store: QdrantStore):
        self._embedder = embedder
        self._store = vector_store
        self._parser = DocumentParser()

    def ingest(
        self,
        collection_id: str,
        document_id: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> int:
        if content_type in SUPPORTED_IMAGE_TYPES:
            chunks = self._parser.parse_image(content, content_type, source_id=document_id)
            chunk_ids = [f"{document_id}-img-0"]
            embeddings = [self._embedder.embed_image(chunks[0].image_bytes, chunks[0].mime_type)]
            documents = [f"[IMAGE: {filename}]"]
            metadatas = [{"document_id": document_id, "filename": filename, "content_type": "image", "mime_type": content_type}]
            self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
            return 1

        if content_type in SUPPORTED_DOC_TYPES:
            chunks = self._parser.parse_document_bytes(content, filename, source_id=document_id)
            if not chunks:
                return 0
            chunk_ids = [f"{document_id}-chunk-{i}" for i in range(len(chunks))]
            embeddings = self._embedder.embed_texts([c.text for c in chunks])
            documents = [c.text for c in chunks]
            metadatas = [{"document_id": document_id, "filename": filename, "content_type": "text", "chunk_index": str(c.chunk_index)} for c in chunks]
            self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
            return len(chunks)

        raise ValueError(f"Unsupported content type: {content_type}")

    def delete_document(self, collection_id: str, document_id: str) -> None:
        self._store.delete_by_document_id(collection_id, document_id)
