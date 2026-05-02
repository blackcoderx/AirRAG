import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from llama_index.core.node_parser import SentenceSplitter
from markitdown import MarkItDown


@dataclass
class TextChunk:
    text: str
    source_id: str
    chunk_index: int


@dataclass
class ImageChunk:
    image_bytes: bytes
    mime_type: str
    source_id: str


class DocumentParser:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._converter = MarkItDown()

    def parse_text(self, text: str, source_id: str) -> list[TextChunk]:
        from llama_index.core.schema import Document as LIDoc
        nodes = self._splitter.get_nodes_from_documents([LIDoc(text=text)])
        return [TextChunk(text=n.text, source_id=source_id, chunk_index=i) for i, n in enumerate(nodes)]

    def parse_document_bytes(self, content: bytes, filename: str, source_id: str) -> list[TextChunk]:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            result = self._converter.convert(tmp_path)
            return self.parse_text(result.text_content or "", source_id)
        finally:
            os.unlink(tmp_path)

    def parse_image(self, image_bytes: bytes, mime_type: str, source_id: str) -> list[ImageChunk]:
        Image.open(io.BytesIO(image_bytes)).verify()
        return [ImageChunk(image_bytes=image_bytes, mime_type=mime_type, source_id=source_id)]
