from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str = ""
    storage_dir: Path = Path("./storage")
    qdrant_dir: Path = Path("./qdrant_data")
    database_url: str = "sqlite:///./airrag.db"
    gemini_embed_model: str = "gemini-embedding-exp-03-07"
    gemini_embed_dim: int = 3072
    gemini_gen_model: str = "gemini-2.0-flash"


settings = Settings()
