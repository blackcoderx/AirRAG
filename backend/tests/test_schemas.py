import pytest
from app.models.schemas import CollectionCreate, QueryRequest, ChunkResult


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
        score=0.95,
    )
    assert r.score == 0.95
