from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings


# SQLite engine with check_same_thread=False for FastAPI's async nature
# Used by: db_models.py (Base parent class), get_db() dependency
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)
# Session factory for creating database sessions (used in FastAPI dependencies)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models (Collection, Document).
    Tables are created on app startup via lifespan() in main.py.
    """
    pass


def get_db():
    """FastAPI dependency that yields a database session per request.
    Used by: collections.py, documents.py, query.py via Depends(get_db).
    Always closes the session after the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
