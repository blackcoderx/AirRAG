from pathlib import Path

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Loads all configuration from environment variables and .env file.

    Used by: database.py (DB URL), embedder.py (Gemini API key + model),
    generator.py (Gemini model), vector_store.py (Qdrant URL),
    MinIOClient (storage credentials), audio_processor.py (Whisper URL).
    """

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Gemini API key for both embedding and generation models
    gemini_api_key: str = ""
    # Directory for temporary file storage (currently reserved for future use)
    storage_dir: Path = Path("./storage")
    # Qdrant vector database connection URL (Docker container)
    qdrant_url: str = "http://localhost:6333"
    # SQLite database URL for collection/document metadata
    database_url: str = "sqlite:///./airrag.db"
    # Gemini embedding model (gemini-embedding-2 supports text, images, PDF, audio, video)
    gemini_embed_model: str = "gemini-embedding-2"
    # Output dimension for embeddings (3072 = full, also supports 768, 1536 via Matryoshka)
    gemini_embed_dim: int = 3072
    # Gemini generation model for RAG answer synthesis
    gemini_gen_model: str = "gemini-2.0-flash"

    # MinIO object storage for raw file storage (images, PDFs, audio, video)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "airrag"
    minio_secure: bool = False
    # Public URL for accessing stored files (returned in query results as blob_url)
    minio_public_url: str = "http://localhost:9000"

    # Local Whisper server URL for audio transcription (used by audio_processor.py)
    whisper_server_url: str = "http://localhost:9010"


# Singleton settings instance imported by all modules needing configuration
settings = Settings()
