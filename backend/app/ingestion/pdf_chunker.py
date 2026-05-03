import io

from pypdf import PdfReader, PdfWriter


class PDFChunker:
    """Splits PDFs into chunks of N pages for Gemini's embedding limits.

    Gemini's gemini-embedding-2 supports up to 6 pages per embedding call.
    This class chunks large PDFs into 6-page segments, each embedded separately.

    """

    def __init__(self, pages_per_chunk: int = 6):
        """Initialize with pages per chunk (6 = Gemini's limit)."""
        self._pages_per_chunk = pages_per_chunk

    def chunk(self, pdf_bytes: bytes) -> list[tuple[bytes, int, int]]:
        """Return list of (chunk_pdf_bytes, page_start_1indexed, page_end_1indexed).

        Each chunk is a valid PDF containing a subset of pages.
        Gemini embeds each chunk as a single unit (visual + text content).
        """
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
