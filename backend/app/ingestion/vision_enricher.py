from google import genai
from google.genai import types

_VISION_PROMPT = (
    "Describe this media in rich detail. Include:\n"
    "- What is happening / what is shown\n"
    "- Key objects, people, animals, actions\n"
    "- Setting, environment, mood\n"
    "- Any text visible\n"
    "- Any audio or speech if applicable\n"
    "Be specific and descriptive. This description will be used for semantic search retrieval."
)


class VisionEnricher:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def describe(self, media_bytes: bytes, mime_type: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(mime_type=mime_type, data=media_bytes)
                        ),
                        types.Part(text=_VISION_PROMPT),
                    ]
                )
            ],
        )
        return response.text or ""
