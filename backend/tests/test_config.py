from app.core.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    settings = Settings()
    assert settings.gemini_api_key == "test-key"


def test_settings_defaults(monkeypatch):
    settings = Settings()
    assert settings.gemini_embed_model == "gemini-embedding-exp-03-07"
    assert settings.gemini_embed_dim == 3072
    assert settings.gemini_gen_model == "gemini-2.0-flash"
    assert settings.qdrant_url == "http://localhost:6333"
