from pathlib import Path

from pdf_oxide import OfficeConverter


def convert_to_pdf(content: bytes, filename: str) -> bytes:
    """Convert DOCX or PPTX bytes to PDF using pdf_oxide (pure Python, no system deps).

    This enables native Gemini PDF embedding for DOCX/PPTX files.
    Gemini's gemini-embedding-2 can embed PDFs directly (up to 6 pages per call).

    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".docx":
        return OfficeConverter.from_docx_bytes(content).to_bytes()
    if suffix == ".pptx":
        return OfficeConverter.from_pptx_bytes(content).to_bytes()
    raise ValueError(f"Unsupported conversion from {suffix} to PDF")
