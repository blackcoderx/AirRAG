import io
from dataclasses import dataclass
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter


@dataclass
class TextChunk:
    """Represents a chunk of text ready for embedding.

    Used by: ingestor.py (_ingest_text, _ingest_pdf) to pass to embedder.
    """

    text: str
    source_id: str
    chunk_index: int


@dataclass
class ImageChunk:
    """Represents a validated image ready for embedding.

    Used by: ingestor.py (_ingest_image) to pass to embedder.
    """

    image_bytes: bytes
    mime_type: str
    source_id: str


class DocumentParser:
    """Handles text chunking (TXT/MD) and image validation.

    Related files:
    - Note: DOCX/PPTX now converted to PDF via converter.py
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """Initialize with LlamaIndex SentenceSplitter for semantic text chunking.

        Args:
            chunk_size: Target tokens per chunk (512 ≈ 400-500 words)
            chunk_overlap: Overlapping tokens between chunks (preserves context)
        """
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def parse_text(self, text: str, source_id: str) -> list[TextChunk]:
        """Split text into semantic chunks using LlamaIndex SentenceSplitter.

        Used for: TXT and Markdown files (read as string, then chunked).
        Note: Gemini can embed plain text directly, chunking is for handling >8192 tokens.
        """
        from llama_index.core.schema import Document as LIDoc

        nodes = self._splitter.get_nodes_from_documents([LIDoc(text=text)])
        return [
            TextChunk(text=getattr(n, "text", ""), source_id=source_id, chunk_index=i)
            for i, n in enumerate(nodes)
        ]

    def parse_document_bytes(
        self, content: bytes, filename: str, source_id: str
    ) -> list[TextChunk]:
        """Parse text files (TXT/MD) by decoding bytes to string, then chunking.

        Note: DOCX/PPTX are no longer handled here - they go through converter.py first.
        PDF chunks are handled separately in ingestor.py using embed_pdf_chunk().
        """
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            return self.parse_text(content.decode("utf-8", errors="replace"), source_id)
        raise ValueError(f"Unsupported text format: {suffix}")

    def parse_image(
        self, image_bytes: bytes, mime_type: str, source_id: str
    ) -> list[ImageChunk]:
        """Validate image integrity using Pillow (not parsing - Gemini handles actual embedding).

        Used by: ingestor.py (_ingest_image) before sending to embedder.embed_image().
        """
        from PIL import Image

        Image.open(io.BytesIO(image_bytes)).verify()
        return [
            ImageChunk(
                image_bytes=image_bytes, mime_type=mime_type, source_id=source_id
            )
        ]
