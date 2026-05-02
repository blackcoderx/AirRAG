from unittest.mock import MagicMock, patch


def test_embed_text_returns_floats():
    with patch("app.ingestion.embedder.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1, 0.2, 0.3])]
        )
        from app.ingestion.embedder import GeminiEmbedder
        embedder = GeminiEmbedder(api_key="fake", model="gemini-embedding-exp-03-07")
        result = embedder.embed_text("hello world")
        assert result == [0.1, 0.2, 0.3]


def test_embed_texts_calls_once_per_text():
    with patch("app.ingestion.embedder.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1, 0.2])]
        )
        from app.ingestion.embedder import GeminiEmbedder
        embedder = GeminiEmbedder(api_key="fake", model="gemini-embedding-exp-03-07")
        results = embedder.embed_texts(["a", "b"])
        assert len(results) == 2
        assert mock_client.models.embed_content.call_count == 2


def test_embed_image_calls_api():
    with patch("app.ingestion.embedder.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.5, 0.6])]
        )
        from app.ingestion.embedder import GeminiEmbedder
        embedder = GeminiEmbedder(api_key="fake", model="gemini-embedding-exp-03-07")
        result = embedder.embed_image(b"fake_bytes", "image/jpeg")
        assert result == [0.5, 0.6]
        mock_client.models.embed_content.assert_called_once()
