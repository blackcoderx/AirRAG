from app.core.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    settings = Settings()
    assert settings.gemini_api_key == "test-key"


def test_settings_defaults(monkeypatch):
    settings = Settings()
    assert settings.gemini_embed_model == "gemini-embedding-2"
    assert settings.gemini_embed_dim == 3072
    assert settings.gemini_gen_model == "gemini-2.0-flash"
    assert settings.qdrant_url == "http://localhost:6333"


def test_minio_defaults():
    s = Settings(_env_file=None)
    assert s.minio_endpoint == "localhost:9000"
    assert s.minio_access_key == "minioadmin"
    assert s.minio_secret_key == "minioadmin"
    assert s.minio_bucket == "airrag"
    assert s.minio_secure is False
    assert s.minio_public_url == "http://localhost:9000"


def test_chunking_defaults():
    s = Settings(_env_file=None)
    assert s.audio_chunk_duration == 150
    assert s.video_chunk_duration == 60
    assert s.audio_overlap == 15
    assert s.video_overlap == 15
