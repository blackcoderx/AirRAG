from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.ingestor import Ingestor
from app.models.db_models import Collection, Document
from app.models.schemas import DocumentResponse
from app.retrieval.vector_store import QdrantStore

router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])


def _make_ingestor() -> Ingestor:
    # Builds a fresh Ingestor per request; Qdrant local client is cheap to construct
    embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key, model=settings.gemini_embed_model
    )
    store = QdrantStore(
        path=str(settings.qdrant_dir), embed_dim=settings.gemini_embed_dim
    )
    return Ingestor(embedder=embedder, vector_store=store)


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
    _make_ingestor().delete_document(collection_id, document_id)
    db.delete(doc)
    db.commit()
