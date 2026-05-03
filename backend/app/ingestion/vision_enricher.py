from google import genai
from google.genai import types

# Prompt sent to Gemini to generate rich descriptions of images/video frames
# These descriptions are embedded (not the raw image) for text-based search
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
    """Generates text descriptions of images/video frames using Gemini Vision.

    Purpose: Convert visual content to text descriptions that can be:
    1. Embedded for text-based search (images)
    2. Stored as metadata for search result display
    3. Combined with audio transcripts for video chunks

    Used by: ingestor.py (_ingest_image, _ingest_video).
    Related: embedder.py (embeds the resulting descriptions).
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """Initialize Gemini client for vision tasks (uses gemini-2.0-flash, not embedding model)."""
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def describe(self, media_bytes: bytes, mime_type: str) -> str:
        """Generate rich text description of image or video frame.

        Used for: Image search results show vision_description; video chunks combine this with transcript.
        """
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type, data=media_bytes
                            )
                        ),
                        types.Part(text=_VISION_PROMPT),
                    ]
                )
            ],
        )
        return response.text or ""
