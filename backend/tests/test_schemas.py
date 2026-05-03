import pytest

from app.models.schemas import ChunkResult, CollectionCreate, QueryRequest


def test_collection_create_requires_name():
    with pytest.raises(Exception):
        CollectionCreate()


def test_collection_create_valid():
    c = CollectionCreate(name="my-kb")
    assert c.name == "my-kb"
    assert c.description == ""


def test_query_request_defaults():
    q = QueryRequest(text="what is in this document?")
    assert q.top_k == 5
    assert q.text == "what is in this document?"


def test_chunk_result_fields():
    r = ChunkResult(
        document_id="doc1",
        filename="file.pdf",
        content="some text",
        content_type="text",
        media_type="text",
        score=0.95,
    )
    assert r.score == 0.95


def test_chunk_result_includes_blob_url():
    chunk = ChunkResult(
        document_id="d1",
        filename="video.mp4",
        content="description",
        content_type="video",
        media_type="video",
        score=0.9,
        blob_url="http://localhost:9000/airrag/d1/video.mp4",
        chunk_start_sec=60.0,
        chunk_end_sec=120.0,
    )
    assert chunk.blob_url == "http://localhost:9000/airrag/d1/video.mp4"
    assert chunk.chunk_start_sec == 60.0


def test_chunk_result_optional_fields_default_to_none():
    chunk = ChunkResult(
        document_id="d1",
        filename="doc.txt",
        content="hello",
        content_type="text",
        media_type="text",
        score=0.8,
    )
    assert chunk.blob_url is None
    assert chunk.chunk_start_sec is None
    assert chunk.vision_description is None
    assert chunk.page_start is None
