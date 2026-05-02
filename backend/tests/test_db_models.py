import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.database import Base
from app.models.db_models import Collection, Document


@pytest.fixture
def db():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    Session = sessionmaker(bind=eng)
    session = Session()
    yield session
    session.close()


def test_create_collection(db):
    coll = Collection(name="kb-1", description="test")
    db.add(coll)
    db.commit()
    db.refresh(coll)
    assert coll.id is not None
    assert coll.name == "kb-1"


def test_create_document(db):
    coll = Collection(name="kb-1")
    db.add(coll)
    db.commit()
    doc = Document(
        collection_id=coll.id,
        filename="report.pdf",
        content_type="application/pdf",
        status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    assert doc.id is not None
    assert doc.collection_id == coll.id
    assert doc.chunk_count == 0
