from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

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
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
