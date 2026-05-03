from unittest.mock import MagicMock, patch

from app.ingestion.vision_enricher import VisionEnricher


def test_describe_returns_model_text():
    with patch("app.ingestion.vision_enricher.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.text = "A golden retriever playing guitar in a sunlit room."
        MockClient.return_value.models.generate_content.return_value = mock_response

        enricher = VisionEnricher(api_key="fake-key")
        result = enricher.describe(b"fakeimagebytes", "image/jpeg")

        assert result == "A golden retriever playing guitar in a sunlit room."


def test_describe_returns_empty_string_on_none_text():
    with patch("app.ingestion.vision_enricher.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.text = None
        MockClient.return_value.models.generate_content.return_value = mock_response

        enricher = VisionEnricher(api_key="fake-key")
        result = enricher.describe(b"fakevideobytes", "video/mp4")

        assert result == ""


def test_describe_passes_blob_to_model():
    with patch("app.ingestion.vision_enricher.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.text = "description"
        MockClient.return_value.models.generate_content.return_value = mock_response

        enricher = VisionEnricher(api_key="fake-key", model="gemini-2.0-flash")
        enricher.describe(b"bytes", "image/png")

        MockClient.return_value.models.generate_content.assert_called_once()
        call_kwargs = MockClient.return_value.models.generate_content.call_args
        assert call_kwargs[1]["model"] == "gemini-2.0-flash"
