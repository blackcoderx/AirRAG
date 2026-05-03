from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.db_models import Collection, Document
from app.models.schemas import CollectionCreate, CollectionResponse

router = APIRouter(prefix="/collections", tags=["collections"])


def _to_response(coll: Collection, db: Session) -> CollectionResponse:
    """Convert ORM Collection to response schema with live document count.

    Computes document_count dynamically so it stays accurate after deletions.
    """
    doc_count = db.query(Document).filter(Document.collection_id == coll.id).count()
    return CollectionResponse(
        id=coll.id,
        name=coll.name,
        description=coll.description,
        created_at=coll.created_at,
        document_count=doc_count,
    )


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
def create_collection(payload: CollectionCreate, db: Session = Depends(get_db)):
    """Create a new knowledge collection (the container for documents)."""
    coll = Collection(name=payload.name, description=payload.description)
    db.add(coll)
    db.commit()
    db.refresh(coll)
    return _to_response(coll, db)


@router.get("", response_model=list[CollectionResponse])
def list_collections(db: Session = Depends(get_db)):
    """List all collections with their document counts."""
    return [_to_response(c, db) for c in db.query(Collection).all()]


@router.get("/{collection_id}", response_model=CollectionResponse)
def get_collection(collection_id: str, db: Session = Depends(get_db)):
    """Get a single collection by ID."""
    coll = db.get(Collection, collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    return _to_response(coll, db)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: str, db: Session = Depends(get_db)):
    """Delete a collection (cascades to documents in DB, must also clean Qdrant/MinIO)."""
    coll = db.get(Collection, collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    db.delete(coll)
    db.commit()
