import io

from pypdf import PdfWriter

from app.ingestion.pdf_chunker import PDFChunker


def _make_pdf(num_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_three_pages_returns_one_chunk():
    chunks = PDFChunker(pages_per_chunk=6).chunk(_make_pdf(3))
    assert len(chunks) == 1
    assert chunks[0][1] == 1
    assert chunks[0][2] == 3


def test_seven_pages_returns_two_chunks():
    chunks = PDFChunker(pages_per_chunk=6).chunk(_make_pdf(7))
    assert len(chunks) == 2
    assert chunks[0][1] == 1
    assert chunks[0][2] == 6
    assert chunks[1][1] == 7
    assert chunks[1][2] == 7


def test_twelve_pages_returns_two_full_chunks():
    chunks = PDFChunker(pages_per_chunk=6).chunk(_make_pdf(12))
    assert len(chunks) == 2
    assert chunks[0][2] == 6
    assert chunks[1][1] == 7
    assert chunks[1][2] == 12


def test_chunk_bytes_are_valid_pdf():
    from pypdf import PdfReader
    chunks = PDFChunker(pages_per_chunk=6).chunk(_make_pdf(4))
    reader = PdfReader(io.BytesIO(chunks[0][0]))
    assert len(reader.pages) == 4
