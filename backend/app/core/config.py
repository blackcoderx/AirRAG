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

    gemini_api_key: str = ""
    storage_dir: Path = Path("./storage")
    qdrant_url: str = "http://localhost:6333"
    database_url: str = "sqlite:///./airrag.db"
    gemini_embed_model: str = "gemini-embedding-2"
    gemini_embed_dim: int = 3072
    gemini_gen_model: str = "gemini-2.0-flash"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "airrag"
    minio_secure: bool = False
    minio_public_url: str = "http://localhost:9000"
    whisper_server_url: str = "http://localhost:9010"


# Singleton settings instance imported by all modules needing configuration
settings = Settings()
