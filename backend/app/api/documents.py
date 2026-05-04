from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.ingestor import Ingestor
from app.ingestion.media_chunker import MediaChunker
from app.ingestion.pdf_chunker import PDFChunker
from app.ingestion.vision_enricher import VisionEnricher
from app.models.db_models import Collection, Document
from app.models.schemas import DocumentResponse
from app.retrieval.vector_store import QdrantStore
from app.storage.minio_client import MinIOClient

router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])


def _make_ingestor() -> Ingestor:
    """Factory function that creates a fresh Ingestor per request (stateless architecture).
    
    Each ingestion gets its own:
    - GeminiEmbedder (for creating embeddings)
    - QdrantStore (for vector storage)
    - MinIOClient (for raw file storage)
    - VisionEnricher (for image/video descriptions)
    - MediaChunker for audio and video (configurable chunk sizes)
    - PDFChunker (for splitting PDFs into 6-page segments)
    
    This per-request construction keeps the API layer stateless.
    """
    return Ingestor(
        embedder=GeminiEmbedder(api_key=settings.gemini_api_key, model=settings.gemini_embed_model),
        vector_store=QdrantStore(url=settings.qdrant_url, embed_dim=settings.gemini_embed_dim),
        minio_client=MinIOClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
            public_url=settings.minio_public_url,
        ),
        vision_enricher=VisionEnricher(api_key=settings.gemini_api_key, model=settings.gemini_gen_model),
        audio_chunker=MediaChunker(
            chunk_duration=settings.audio_chunk_duration,
            overlap=settings.audio_overlap,
            hard_limit=180,
            media_type="audio",
        ),
        video_chunker=MediaChunker(
            chunk_duration=settings.video_chunk_duration,
            overlap=settings.video_overlap,
            hard_limit=120,
            media_type="video",
        ),
        pdf_chunker=PDFChunker(),
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload and ingest a document into a collection.

    Flow:
    1. Validate collection exists
    2. Create Document record with "processing" status
    3. Read file bytes and content type
    4. Run ingestion pipeline (parse → embed → store in Qdrant + MinIO)
    5. Update status to "ready" with chunk_count, or "error" on failure
    """
    coll = db.get(Collection, collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")

    content = await file.read()
    content_type = file.content_type or "application/octet-stream"
    doc = Document(
        collection_id=collection_id,
        filename=file.filename,
        content_type=content_type,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        chunk_count = _make_ingestor().ingest(
            collection_id=collection_id,
            document_id=doc.id,
            filename=file.filename or "",
            content_type=content_type,
            content=content,
        )
        doc.status = "ready"
        doc.chunk_count = chunk_count
    except ValueError as e:
        doc.status = "error"
        db.commit()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        doc.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    db.commit()
    db.refresh(doc)
    return doc


@router.get("", response_model=list[DocumentResponse])
def list_documents(collection_id: str, db: Session = Depends(get_db)):
    """List all documents in a collection."""
    if not db.get(Collection, collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    return db.query(Document).filter(Document.collection_id == collection_id).all()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    collection_id: str, document_id: str, db: Session = Depends(get_db)
):
    """Delete a document (removes from DB, Qdrant vectors, and MinIO storage)."""
    doc = db.get(Document, document_id)
    if not doc or doc.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Document not found")
    _make_ingestor().delete_document(collection_id, document_id, doc.filename or "")
    db.delete(doc)
    db.commit()


@router.get("/{document_id}/media")
def stream_document_media(
    collection_id: str,
    document_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Stream the original media file from MinIO with Range request support.

    Supports HTTP Range headers for seeking in audio/video players.
    Returns 206 Partial Content for range requests, 200 OK for full file.
    """
    doc = db.get(Document, document_id)
    if not doc or doc.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Document not found")

    minio = MinIOClient(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
        public_url=settings.minio_public_url,
    )
    object_name = MinIOClient.object_name(document_id, doc.filename or "")
    total_size, content_type = minio.stat(object_name)
    range_header = request.headers.get("range")

    if range_header:
        try:
            range_str = range_header.replace("bytes=", "").strip()
            range_start_str, range_end_str = range_str.split("-", 1)
            range_start = int(range_start_str)
            range_end = int(range_end_str) if range_end_str else total_size - 1
            range_end = min(range_end, total_size - 1)
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="Invalid Range header")

        chunk_size = range_end - range_start + 1
        data = minio.download_range(object_name, range_start, range_end + 1)

        headers = {
            "Content-Range": f"bytes {range_start}-{range_end}/{total_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": content_type,
        }
        return StreamingResponse(
            iter([data]),
            status_code=206,
            headers=headers,
        )

    data = minio.download(object_name)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total_size),
        "Content-Type": content_type,
    }
    return StreamingResponse(
        iter([data]),
        headers=headers,
    )
