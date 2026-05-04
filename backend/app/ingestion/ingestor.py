from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.converter import convert_to_pdf
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.media_chunker import MediaChunker
from app.ingestion.parser import DocumentParser
from app.ingestion.pdf_chunker import PDFChunker
from app.ingestion.vision_enricher import VisionEnricher
from app.retrieval.vector_store import QdrantStore
from app.storage.minio_client import MinIOClient

def _fmt_sec(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
SUPPORTED_PDF_TYPES = {"application/pdf"}
SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
}
SUPPORTED_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
}
SUPPORTED_TEXT_TYPES = {"text/plain", "text/markdown"}
CONVERTIBLE_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class Ingestor:
    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: QdrantStore,
        minio_client: MinIOClient,
        vision_enricher: VisionEnricher,
        audio_chunker: MediaChunker,
        video_chunker: MediaChunker,
        pdf_chunker: PDFChunker,
    ):
        self._embedder = embedder
        self._store = vector_store
        self._minio = minio_client
        self._vision = vision_enricher
        self._audio_chunker = audio_chunker
        self._video_chunker = video_chunker
        self._pdf_chunker = pdf_chunker
        self._parser = DocumentParser()

    def ingest(
        self,
        collection_id: str,
        document_id: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> int:
        ingested_at = datetime.now(timezone.utc).isoformat()
        object_name = MinIOClient.object_name(document_id, filename)
        blob_url = self._minio.upload(object_name, content, content_type)

        suffix = Path(filename).suffix.lower()
        if suffix in CONVERTIBLE_TYPES:
            pdf_bytes = convert_to_pdf(content, filename)
            return self._ingest_pdf(
                collection_id, document_id, filename, pdf_bytes, blob_url, ingested_at
            )
        if content_type in SUPPORTED_IMAGE_TYPES:
            return self._ingest_image(
                collection_id,
                document_id,
                filename,
                content_type,
                content,
                blob_url,
                ingested_at,
            )
        if content_type in SUPPORTED_PDF_TYPES:
            return self._ingest_pdf(
                collection_id, document_id, filename, content, blob_url, ingested_at
            )
        if content_type in SUPPORTED_AUDIO_TYPES:
            return self._ingest_audio(
                collection_id,
                document_id,
                filename,
                content_type,
                content,
                blob_url,
                ingested_at,
            )
        if content_type in SUPPORTED_VIDEO_TYPES:
            return self._ingest_video(
                collection_id,
                document_id,
                filename,
                content_type,
                content,
                blob_url,
                ingested_at,
            )
        if content_type in SUPPORTED_TEXT_TYPES:
            return self._ingest_text(
                collection_id,
                document_id,
                filename,
                content_type,
                content,
                blob_url,
                ingested_at,
            )
        raise ValueError(f"Unsupported content type: {content_type}")

    def _ingest_image(
        self,
        collection_id,
        document_id,
        filename,
        mime_type,
        content,
        blob_url,
        ingested_at,
    ) -> int:
        vision_description = self._vision.describe(content, mime_type)
        embedding = self._embedder.embed_image(content, mime_type)
        doc_text = vision_description or f"[IMAGE: {filename}]"
        self._store.upsert(
            collection_id,
            [f"{document_id}-img-0"],
            [embedding],
            [doc_text],
            [
                {
                    "document_id": document_id,
                    "filename": filename,
                    "media_type": "image",
                    "mime_type": mime_type,
                    "chunk_index": 0,
                    "blob_url": blob_url,
                    "vision_description": vision_description,
                    "ingested_at": ingested_at,
                }
            ],
        )
        return 1

    def _ingest_pdf(
        self, collection_id, document_id, filename, content, blob_url, ingested_at
    ) -> int:
        chunks = self._pdf_chunker.chunk(content)
        if not chunks:
            return 0
        chunk_ids, embeddings, documents, metadatas = [], [], [], []
        for i, (chunk_bytes, page_start, page_end) in enumerate(chunks):
            embedding = self._embedder.embed_pdf_chunk(chunk_bytes)
            text_chunks = self._parser.parse_document_bytes(
                chunk_bytes, filename, source_id=f"{document_id}-{i}"
            )
            doc_text = (
                " ".join(c.text for c in text_chunks)
                if text_chunks
                else f"[PDF pages {page_start}-{page_end}]"
            )
            chunk_ids.append(f"{document_id}-pdf-{i}")
            embeddings.append(embedding)
            documents.append(doc_text)
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "media_type": "pdf",
                    "mime_type": "application/pdf",
                    "chunk_index": i,
                    "blob_url": blob_url,
                    "page_start": page_start,
                    "page_end": page_end,
                    "ingested_at": ingested_at,
                }
            )
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(chunks)

    def _ingest_audio(
        self,
        collection_id,
        document_id,
        filename,
        mime_type,
        content,
        blob_url,
        ingested_at,
    ) -> int:
        media_chunks = self._audio_chunker.process(content, mime_type)
        if not media_chunks:
            return 0
        chunk_ids = [f"{document_id}-audio-{i}" for i in range(len(media_chunks))]
        with ThreadPoolExecutor(max_workers=6) as pool:
            embeddings = list(
                pool.map(lambda c: self._embedder.embed_audio(c.data, c.mime_type), media_chunks)
            )
        documents = [
            f"[AUDIO: {filename} {_fmt_sec(c.start_sec)}–{_fmt_sec(c.end_sec)}]"
            for c in media_chunks
        ]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "media_type": "audio",
                "mime_type": mime_type,
                "chunk_index": i,
                "blob_url": blob_url,
                "chunk_start_sec": c.start_sec,
                "chunk_end_sec": c.end_sec,
                "ingested_at": ingested_at,
            }
            for i, c in enumerate(media_chunks)
        ]
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(media_chunks)

    def _ingest_video(
        self,
        collection_id,
        document_id,
        filename,
        mime_type,
        content,
        blob_url,
        ingested_at,
    ) -> int:
        media_chunks = self._video_chunker.process(content, mime_type)
        if not media_chunks:
            return 0

        n = len(media_chunks)
        embeddings: list = [None] * n
        descriptions: list[str] = [""] * n

        def _embed(i: int, mc) -> tuple:
            return "embed", i, self._embedder.embed_video(mc.data, mc.mime_type)

        def _describe(i: int, mc) -> tuple:
            return "describe", i, self._vision.describe(mc.data, mc.mime_type) or ""

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i, mc in enumerate(media_chunks):
                futures.append(pool.submit(_embed, i, mc))
                futures.append(pool.submit(_describe, i, mc))

            for fut in futures:
                kind, idx, result = fut.result()
                if kind == "embed":
                    embeddings[idx] = result
                else:
                    descriptions[idx] = result

        chunk_ids, emb_list, documents, metadatas = [], [], [], []
        for i, mc in enumerate(media_chunks):
            vision_description = descriptions[i]
            doc_text = (
                vision_description
                or f"[VIDEO: {filename} {_fmt_sec(mc.start_sec)}–{_fmt_sec(mc.end_sec)}]"
            )
            chunk_ids.append(f"{document_id}-video-{i}")
            emb_list.append(embeddings[i])
            documents.append(doc_text)
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "media_type": "video",
                    "mime_type": mime_type,
                    "chunk_index": i,
                    "blob_url": blob_url,
                    "chunk_start_sec": mc.start_sec,
                    "chunk_end_sec": mc.end_sec,
                    "vision_description": vision_description,
                    "ingested_at": ingested_at,
                }
            )

        self._store.upsert(collection_id, chunk_ids, emb_list, documents, metadatas)
        return len(media_chunks)

    def _ingest_text(
        self,
        collection_id,
        document_id,
        filename,
        mime_type,
        content,
        blob_url,
        ingested_at,
    ) -> int:
        chunks = self._parser.parse_document_bytes(
            content, filename, source_id=document_id
        )
        if not chunks:
            return 0
        chunk_ids = [f"{document_id}-chunk-{i}" for i in range(len(chunks))]
        embeddings = self._embedder.embed_texts([c.text for c in chunks])
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "media_type": "text",
                "mime_type": mime_type,
                "chunk_index": c.chunk_index,
                "blob_url": blob_url,
                "ingested_at": ingested_at,
            }
            for c in chunks
        ]
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(chunks)

    def delete_document(
        self, collection_id: str, document_id: str, filename: str = ""
    ) -> None:
        self._store.delete_by_document_id(collection_id, document_id)
        if filename:
            try:
                self._minio.delete(MinIOClient.object_name(document_id, filename))
            except Exception:
                pass
