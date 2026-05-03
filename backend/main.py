from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.core.database import engine, Base
from app.api.collections import router as collections_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - runs on startup/shutdown.
    
    Startup: Create all database tables (collections, documents) if they don't exist.
    Shutdown: (currently no cleanup needed, connections are managed per-request).
    """
    Base.metadata.create_all(bind=engine)
    yield


# FastAPI application instance
# Includes three routers: collections (CRUD), documents (upload/delete), query (search+generate)
app = FastAPI(title="AirRAG", version="0.1.0", lifespan=lifespan)
app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(query_router)
