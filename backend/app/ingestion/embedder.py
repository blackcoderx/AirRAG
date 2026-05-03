from google import genai
from google.genai import types


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_text(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[float]:
        # task_type tells Gemini whether this is a document being stored or a query being searched
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
        # Embeds each text individually; Gemini has no batch endpoint so this calls the API once per chunk
        return [self.embed_text(t, task_type) for t in texts]

    def embed_image(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> list[float]:
        # Images are embedded into the same 3072-dim vector space as text — enabling cross-modal retrieval
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
        # RETRIEVAL_QUERY optimises the vector for similarity search rather than storage
        return self.embed_text(text, task_type="RETRIEVAL_QUERY")

    def embed_image_query(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> list[float]:
        # Same as embed_image but tagged as a query so Gemini adjusts the embedding direction
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

    def embed_audio(self, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> list[float]:
        return self.embed_bytes(audio_bytes, mime_type)

    def embed_video(self, video_bytes: bytes, mime_type: str = "video/mp4") -> list[float]:
        return self.embed_bytes(video_bytes, mime_type)

    def embed_pdf_chunk(self, pdf_bytes: bytes) -> list[float]:
        return self.embed_bytes(pdf_bytes, "application/pdf")
