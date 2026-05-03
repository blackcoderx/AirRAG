import io

from pypdf import PdfReader, PdfWriter


class PDFChunker:
    def __init__(self, pages_per_chunk: int = 6):
        self._pages_per_chunk = pages_per_chunk

    def chunk(self, pdf_bytes: bytes) -> list[tuple[bytes, int, int]]:
        """Return list of (chunk_pdf_bytes, page_start_1indexed, page_end_1indexed)."""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = len(reader.pages)
        result = []
        for start_idx in range(0, total, self._pages_per_chunk):
            end_idx = min(start_idx + self._pages_per_chunk, total)
            writer = PdfWriter()
            for i in range(start_idx, end_idx):
                writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)
            result.append((buf.getvalue(), start_idx + 1, end_idx))
        return result
