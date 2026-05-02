import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.core.database import Base, get_db
from main import app

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_create_collection():
    resp = client.post("/collections", json={"name": "my-kb", "description": "test"})
    assert resp.status_code == 201
    assert resp.json()["name"] == "my-kb"
    assert "id" in resp.json()


def test_list_collections():
    resp = client.get("/collections")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_collection_not_found():
    resp = client.get("/collections/nonexistent-id")
    assert resp.status_code == 404


def test_get_collection():
    create_resp = client.post("/collections", json={"name": "kb2"})
    coll_id = create_resp.json()["id"]
    resp = client.get(f"/collections/{coll_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == coll_id


def test_delete_collection():
    create_resp = client.post("/collections", json={"name": "kb-del"})
    coll_id = create_resp.json()["id"]
    resp = client.delete(f"/collections/{coll_id}")
    assert resp.status_code == 204
    assert client.get(f"/collections/{coll_id}").status_code == 404
