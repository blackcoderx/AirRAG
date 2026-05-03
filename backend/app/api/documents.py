from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.ingestion.audio_processor import AudioProcessor
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.ingestor import Ingestor
from app.ingestion.pdf_chunker import PDFChunker
from app.ingestion.video_processor import VideoProcessor
from app.ingestion.vision_enricher import VisionEnricher
from app.models.db_models import Collection, Document
from app.models.schemas import DocumentResponse
from app.retrieval.vector_store import QdrantStore
from app.storage.minio_client import MinIOClient

router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])


def _make_ingestor() -> Ingestor:
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
        audio_processor=AudioProcessor(whisper_url=settings.whisper_server_url),
        video_processor=VideoProcessor(),
        pdf_chunker=PDFChunker(),
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
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
    if not db.get(Collection, collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    return db.query(Document).filter(Document.collection_id == collection_id).all()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    collection_id: str, document_id: str, db: Session = Depends(get_db)
):
    doc = db.get(Document, document_id)
    if not doc or doc.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Document not found")
    _make_ingestor().delete_document(collection_id, document_id, doc.filename or "")
    db.delete(doc)
    db.commit()
