from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CollectionCreate(BaseModel):
    name: str
    description: str = ""


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    document_count: int = 0

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    id: str
    collection_id: str
    filename: str
    content_type: str
    status: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    text: Optional[str] = None
    top_k: int = 5


class ChunkResult(BaseModel):
    document_id: str
    filename: str
    content: str
    content_type: str
    media_type: str
    score: float
    blob_url: Optional[str] = None
    chunk_start_sec: Optional[float] = None
    chunk_end_sec: Optional[float] = None
    vision_description: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]
