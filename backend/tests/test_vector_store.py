import pytest
from app.retrieval.vector_store import QdrantStore


@pytest.fixture
def store():
    return QdrantStore(url="http://localhost:6333", embed_dim=4)


def test_upsert_and_search(store):
    store.upsert(
        collection_name="test-coll",
        chunk_ids=["chunk-1"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["hello world"],
        metadatas=[{"document_id": "doc1", "filename": "a.txt", "content_type": "text"}],
    )
    results = store.search(
        collection_name="test-coll",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=1,
    )
    assert len(results) == 1
    assert results[0]["document"] == "hello world"
    assert results[0]["metadata"]["document_id"] == "doc1"


def test_delete_by_document_id(store):
    store.upsert(
        collection_name="test-coll",
        chunk_ids=["chunk-1", "chunk-2"],
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        documents=["text 1", "text 2"],
        metadatas=[
            {"document_id": "doc1", "filename": "a.txt", "content_type": "text"},
            {"document_id": "doc1", "filename": "a.txt", "content_type": "text"},
        ],
    )
    store.delete_by_document_id("test-coll", "doc1")
    results = store.search(
        collection_name="test-coll",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=10,
    )
    assert len(results) == 0


def test_delete_collection(store):
    store.upsert(
        collection_name="to-delete",
        chunk_ids=["c1"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["text"],
        metadatas=[{"document_id": "d1", "filename": "f.txt", "content_type": "text"}],
    )
    store.delete_collection("to-delete")
    results = store.search("to-delete", [0.1, 0.2, 0.3, 0.4], top_k=5)
    assert results == []
