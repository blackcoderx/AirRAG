import io
import pytest
from PIL import Image
from app.ingestion.parser import DocumentParser, TextChunk, ImageChunk


@pytest.fixture
def parser():
    return DocumentParser(chunk_size=100, chunk_overlap=10)


def test_parse_text_produces_chunks(parser):
    long_text = "This is a sentence. " * 30
    chunks = parser.parse_text(long_text, source_id="doc1")
    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.source_id == "doc1" for c in chunks)


def test_parse_image_validates_and_returns_chunk(parser):
    img = Image.new("RGB", (64, 64), color=(100, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    chunks = parser.parse_image(image_bytes, "image/jpeg", source_id="img1")
    assert len(chunks) == 1
    assert isinstance(chunks[0], ImageChunk)
    assert chunks[0].source_id == "img1"
    assert chunks[0].mime_type == "image/jpeg"


def test_parse_image_rejects_invalid_bytes(parser):
    with pytest.raises(Exception):
        parser.parse_image(b"not an image", "image/jpeg", source_id="bad")
