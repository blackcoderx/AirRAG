from google import genai
from google.genai import types


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=[text],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return list(response.embeddings[0].values)

    def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        return [self.embed_text(t, task_type) for t in texts]

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=[
                types.Content(
                    parts=[types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes))]
                )
            ],
        )
        return list(response.embeddings[0].values)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text, task_type="RETRIEVAL_QUERY")

    def embed_image_query(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=[
                types.Content(
                    parts=[types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes))]
                )
            ],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return list(response.embeddings[0].values)
