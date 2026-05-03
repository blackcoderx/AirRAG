from google import genai
from google.genai import types


class GeminiEmbedder:
    """Wraps Google Gemini API for multimodal embeddings (text, images, PDF, audio, video).

    All modalities are mapped to the same 3072-dim vector space (configurable via Matryoshka).
    """

    def __init__(self, api_key: str, model: str):
        """Initialize Gemini client with API key and model name."""
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_text(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[float]:
        """Embed plain text string (up to 8192 tokens).

        task_type: RETRIEVAL_DOCUMENT (storage) vs RETRIEVAL_QUERY (search).
        Note: gemini-embedding-2 ignores task_type; use prompt prefixes instead.
        """
        response = self._client.models.embed_content(
            model=self._model,
            contents=[text],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return list(
            response.embeddings[0].values
            if response.embeddings and response.embeddings[0].values
            else {}
        )

    def embed_texts(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Embed multiple texts (calls API once per text - no batch endpoint).

        Used by: ingestor.py (_ingest_text for multiple chunks).
        """
        return [self.embed_text(t, task_type) for t in texts]

    def embed_image(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> list[float]:
        """Embed image bytes into the same vector space as text (cross-modal retrieval).

        Supports: PNG, JPEG (up to 6 images per call).
        Used by: ingestor.py (_ingest_image).
        """
        response = self._client.models.embed_content(
            model=self._model,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type, data=image_bytes
                            )
                        )
                    ]
                )
            ],
        )
        return list(
            response.embeddings[0].values
            if response.embeddings and response.embeddings[0].values
            else {}
        )

    def embed_query(self, text: str) -> list[float]:
        "Embed query text optimized for similarity search (vs document storage)."
        return self.embed_text(text, task_type="RETRIEVAL_QUERY")

    def embed_image_query(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> list[float]:
        "Embed image query for similarity search (e.g., find documents similar to image)."
        response = self._client.models.embed_content(
            model=self._model,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type, data=image_bytes
                            )
                        )
                    ]
                )
            ],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return (
            list(response.embeddings[0].values)
            if response.embeddings and response.embeddings[0].values
            else []
        )

    def embed_bytes(self, data: bytes, mime_type: str) -> list[float]:
        "Generic method to embed raw bytes (PDF, audio, video)."
        response = self._client.models.embed_content(
            model=self._model,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(mime_type=mime_type, data=data)
                        )
                    ]
                )
            ],
        )
        return list(
            response.embeddings[0].values
            if response.embeddings and response.embeddings[0].values
            else []
        )

    def embed_audio(
        self, audio_bytes: bytes, mime_type: str = "audio/mpeg"
    ) -> list[float]:
        "Embed audio bytes (up to 180 seconds, MP3/WAV)."
        return self.embed_bytes(audio_bytes, mime_type)

    def embed_video(
        self, video_bytes: bytes, mime_type: str = "video/mp4"
    ) -> list[float]:
        "Embed video bytes (up to 120 seconds, MP4/MOV)."
        return self.embed_bytes(video_bytes, mime_type)

    def embed_pdf_chunk(self, pdf_bytes: bytes) -> list[float]:
        "Embed PDF bytes (up to 6 pages per call)."
        return self.embed_bytes(pdf_bytes, "application/pdf")
