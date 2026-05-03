from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.ingestion.embedder import GeminiEmbedder
from app.models.db_models import Collection
from app.models.schemas import ChunkResult, QueryResponse
from app.retrieval.generator import GeminiGenerator
from app.retrieval.vector_store import QdrantStore

router = APIRouter(prefix="/collections/{collection_id}/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_collection(
    collection_id: str,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(5),
    db: Session = Depends(get_db),
):
    if not db.get(Collection, collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    if not text and not image:
        raise HTTPException(status_code=400, detail="Provide text or image for query")

    embedder = GeminiEmbedder(api_key=settings.gemini_api_key, model=settings.gemini_embed_model)
    store = QdrantStore(url=settings.qdrant_url, embed_dim=settings.gemini_embed_dim)
    generator = GeminiGenerator(api_key=settings.gemini_api_key, model=settings.gemini_gen_model)

    if image:
        image_bytes = await image.read()
        query_embedding = embedder.embed_image_query(image_bytes, image.content_type or "image/jpeg")
        query_text = text or "[IMAGE QUERY]"
    else:
        query_embedding = embedder.embed_query(text or "")
        query_text = text

    results = store.search(collection_name=collection_id, query_embedding=query_embedding, top_k=top_k)
    sources = [
        ChunkResult(
            document_id=r["metadata"].get("document_id", ""),
            filename=r["metadata"].get("filename", ""),
            content=r["document"] or "",
            content_type=r["metadata"].get("media_type", "text"),
            media_type=r["metadata"].get("media_type", "text"),
            score=r["score"],
            blob_url=r["metadata"].get("blob_url"),
            chunk_start_sec=r["metadata"].get("chunk_start_sec"),
            chunk_end_sec=r["metadata"].get("chunk_end_sec"),
            vision_description=r["metadata"].get("vision_description"),
            page_start=r["metadata"].get("page_start"),
            page_end=r["metadata"].get("page_end"),
        )
        for r in results
    ]
    answer = generator.generate(query=query_text or "", context_chunks=[r["document"] for r in results])
    return QueryResponse(answer=answer, sources=sources)
