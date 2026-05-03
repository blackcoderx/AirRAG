from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.audio_processor import AudioProcessor
from app.ingestion.converter import convert_to_pdf
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.parser import DocumentParser
from app.ingestion.pdf_chunker import PDFChunker
from app.ingestion.video_processor import VideoProcessor
from app.ingestion.vision_enricher import VisionEnricher
from app.retrieval.vector_store import QdrantStore
from app.storage.minio_client import MinIOClient

def _fmt_sec(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
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
    "video/x-msvideo",
    "video/webm",
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
        audio_processor: AudioProcessor,
        video_processor: VideoProcessor,
        pdf_chunker: PDFChunker,
    ):
        self._embedder = embedder
        self._store = vector_store
        self._minio = minio_client
        self._vision = vision_enricher
        self._audio = audio_processor
        self._video = video_processor
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
        audio_chunks = self._audio.process(content, mime_type)
        if not audio_chunks:
            return 0
        chunk_ids = [f"{document_id}-audio-{i}" for i in range(len(audio_chunks))]
        # Parallel embedding — Gemini API calls are the bottleneck; 6 workers ~4x speedup
        with ThreadPoolExecutor(max_workers=6) as pool:
            embeddings = list(
                pool.map(lambda c: self._embedder.embed_audio(c.data, c.mime_type), audio_chunks)
            )
        documents = [
            f"[AUDIO: {filename} {_fmt_sec(c.start_sec)}–{_fmt_sec(c.end_sec)}]"
            for c in audio_chunks
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
            for i, c in enumerate(audio_chunks)
        ]
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(audio_chunks)

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
        video_chunks = self._video.process(content, mime_type)
        if not video_chunks:
            return 0
        chunk_ids, embeddings, documents, metadatas = [], [], [], []
        for i, vc in enumerate(video_chunks):
            vision_description = self._vision.describe(vc.data, vc.mime_type)
            try:
                audio_bytes = self._video.extract_audio(vc.data)
                transcript = self._audio.transcribe(audio_bytes, ".mp3")
            except Exception:
                transcript = ""
            embedding = self._embedder.embed_video(vc.data, vc.mime_type)
            parts = [
                p
                for p in [
                    vision_description,
                    f"Transcript: {transcript}" if transcript else "",
                ]
                if p
            ]
            doc_text = (
                "\n\n".join(parts)
                or f"[VIDEO: {filename} {vc.start_sec}-{vc.end_sec}s]"
            )
            chunk_ids.append(f"{document_id}-video-{i}")
            embeddings.append(embedding)
            documents.append(doc_text)
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "media_type": "video",
                    "mime_type": mime_type,
                    "chunk_index": i,
                    "blob_url": blob_url,
                    "chunk_start_sec": vc.start_sec,
                    "chunk_end_sec": vc.end_sec,
                    "vision_description": vision_description,
                    "audio_transcript": transcript,
                    "ingested_at": ingested_at,
                }
            )
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(video_chunks)

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
