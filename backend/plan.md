# AirRAG Multimodal Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand AirRAG from text+image-only into a full multimodal RAG pipeline supporting PDF (6-page chunks embedded as bytes), audio (Whisper transcription + chunking), and video (ffmpeg chunking + vision enrichment + audio extraction), with MinIO blob storage and rich Qdrant payloads that surface source assets back to the user.

**Architecture:** All media types embed into the same gemini-embedding-2 vector space via binary blob embedding. Raw files live in MinIO; Qdrant stores the embedding plus a JSON payload with blob_url, timestamps, and vision/transcript descriptions. At query time, retrieved payloads include blob_url and timestamps so the frontend can seek directly to the relevant moment.

**Tech Stack:** gemini-embedding-2, Gemini Flash (vision enrichment), MinIO (minio Python SDK), Whisper server (hwdsl2/whisper-server via httpx), pypdf (PDF page splitting), ffmpeg (subprocess, binary installed in Dockerfile), FastAPI, Qdrant, SQLite.

---

## File Structure

### New Files
- `backend/app/storage/__init__.py` — package marker
- `backend/app/storage/minio_client.py` — MinIO upload/delete wrapper
- `backend/app/ingestion/vision_enricher.py` — Gemini Flash description generator
- `backend/app/ingestion/pdf_chunker.py` — 6-page PDF window splitter using pypdf
- `backend/app/ingestion/audio_processor.py` — ffmpeg audio chunking + Whisper transcription
- `backend/app/ingestion/video_processor.py` — ffmpeg video chunking + audio extraction
- `backend/tests/test_minio_client.py`
- `backend/tests/test_vision_enricher.py`
- `backend/tests/test_pdf_chunker.py`
- `backend/tests/test_audio_processor.py`
- `backend/tests/test_video_processor.py`
- `backend/tests/test_ingestor.py`

### Modified Files
- `docker-compose.yaml` — add MinIO + Whisper services
- `backend/pyproject.toml` — add minio, pypdf, httpx dependencies
- `backend/Dockerfile` — add ffmpeg binary
- `backend/.env.example` — add MinIO + Whisper env vars
- `backend/app/core/config.py` — add MinIO + Whisper settings
- `backend/app/ingestion/embedder.py` — add embed_bytes, embed_audio, embed_video, embed_pdf_chunk
- `backend/app/ingestion/ingestor.py` — full multimodal orchestration rewrite
- `backend/app/models/schemas.py` — extend ChunkResult with blob_url, timestamps, vision_description
- `backend/app/api/documents.py` — update _make_ingestor to wire all new dependencies
- `backend/app/api/query.py` — map new payload fields into ChunkResult
- `backend/tests/test_config.py` — add MinIO + Whisper defaults tests
- `backend/tests/test_embedder.py` — add audio/video/pdf embed tests
- `backend/tests/test_schemas.py` — add new ChunkResult fields tests

---

## Task 1: Infrastructure — pyproject.toml, Dockerfile, docker-compose.yaml

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/Dockerfile`
- Modify: `docker-compose.yaml` (repo root)

- [x] **Step 1: Write failing import test for new deps**

Create `backend/tests/test_deps.py`:
```python
def test_minio_importable():
    import minio  # noqa: F401

def test_pypdf_importable():
    import pypdf  # noqa: F401
```

- [x] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_deps.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'minio'`

- [x] **Step 3: Add dependencies to pyproject.toml**

In `backend/pyproject.toml`, update the `dependencies` list:
```toml
dependencies = [
    "fastapi>=0.136.1",
    "google-genai>=1.74.0",
    "llama-index-core>=0.14.21",
    "markitdown[docx,pdf,pptx]>=0.1.5",
    "uvicorn[standard]>=0.34.0",
    "qdrant-client>=1.9.0",
    "sqlalchemy>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-multipart>=0.0.20",
    "pillow>=11.0.0",
    "aiofiles>=24.0.0",
    "python-dotenv>=1.0.0",
    "minio>=7.2.0",
    "pypdf>=4.0.0",
    "httpx>=0.28.0",
]
```

- [x] **Step 4: Install new deps**

```
cd backend && uv sync
```
Expected: resolves and installs minio, pypdf, httpx

- [x] **Step 5: Run import tests to confirm pass**

```
cd backend && uv run pytest tests/test_deps.py -v
```
Expected: PASS

- [x] **Step 6: Update Dockerfile to install ffmpeg**

Replace the `FROM` block in `backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

RUN mkdir -p storage qdrant_data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [x] **Step 7: Update docker-compose.yaml to add MinIO and Whisper services**

Replace `docker-compose.yaml` at repo root with:
```yaml
services:
  airrag-qdrant:
    image: qdrant/qdrant:latest
    container_name: airrag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  airrag-minio:
    image: minio/minio:latest
    container_name: airrag-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"

  airrag-whisper:
    image: hwdsl2/whisper-server:latest
    container_name: airrag-whisper
    ports:
      - "9010:9000"
    volumes:
      - whisper_models:/root/.cache/whisper

volumes:
  qdrant_data:
  minio_data:
  whisper_models:
```

- [x] **Step 8: Commit**

```bash
git add backend/pyproject.toml backend/Dockerfile docker-compose.yaml backend/tests/test_deps.py
git commit -m "feat: add minio/pypdf/httpx deps, ffmpeg to Dockerfile, MinIO+Whisper to docker-compose"
```

---

## Task 2: Config — New MinIO and Whisper Settings

**Files:**
- Modify: `backend/app/core/config.py`
- Modify: `backend/.env.example`
- Modify: `backend/tests/test_config.py`

- [x] **Step 1: Write failing config tests**

Open `backend/tests/test_config.py` and add at the end:
```python
def test_minio_defaults():
    s = Settings(_env_file=None)
    assert s.minio_endpoint == "localhost:9000"
    assert s.minio_access_key == "minioadmin"
    assert s.minio_secret_key == "minioadmin"
    assert s.minio_bucket == "airrag"
    assert s.minio_secure is False
    assert s.minio_public_url == "http://localhost:9000"

def test_whisper_defaults():
    s = Settings(_env_file=None)
    assert s.whisper_server_url == "http://localhost:9010"
```

- [x] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_config.py -v
```
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'minio_endpoint'`

- [x] **Step 3: Update config.py**

Replace `backend/app/core/config.py` with:
```python
from pathlib import Path

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str = ""
    storage_dir: Path = Path("./storage")
    qdrant_url: str = "http://localhost:6333"
    database_url: str = "sqlite:///./airrag.db"
    gemini_embed_model: str = "gemini-embedding-2"
    gemini_embed_dim: int = 3072
    gemini_gen_model: str = "gemini-2.0-flash"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "airrag"
    minio_secure: bool = False
    minio_public_url: str = "http://localhost:9000"

    whisper_server_url: str = "http://localhost:9010"


settings = Settings()
```

- [x] **Step 4: Update .env.example**

Replace `backend/.env.example` with:
```
GEMINI_API_KEY=your_gemini_api_key_here
STORAGE_DIR=./storage
QDRANT_URL=http://localhost:6333
DATABASE_URL=sqlite:///./airrag.db
GEMINI_EMBED_MODEL=gemini-embedding-2
GEMINI_EMBED_DIM=3072
GEMINI_GEN_MODEL=gemini-2.0-flash

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=airrag
MINIO_SECURE=false
MINIO_PUBLIC_URL=http://localhost:9000

WHISPER_SERVER_URL=http://localhost:9010
```

- [x] **Step 5: Run config tests**

```
cd backend && uv run pytest tests/test_config.py -v
```
Expected: PASS (all 4 tests)

- [x] **Step 6: Commit**

```bash
git add backend/app/core/config.py backend/.env.example backend/tests/test_config.py
git commit -m "feat: add MinIO and Whisper config settings with defaults"
```

---

## Task 3: MinIO Client

**Files:**
- Create: `backend/app/storage/__init__.py`
- Create: `backend/app/storage/minio_client.py`
- Create: `backend/tests/test_minio_client.py`

- [x] **Step 1: Write failing tests**

Create `backend/tests/test_minio_client.py`:
```python
from unittest.mock import MagicMock, patch

from app.storage.minio_client import MinIOClient


def _make_client(mock_minio_cls):
    mock_instance = MagicMock()
    mock_instance.bucket_exists.return_value = True
    mock_minio_cls.return_value = mock_instance
    return MinIOClient(
        endpoint="localhost:9000",
        access_key="key",
        secret_key="secret",
        bucket="test-bucket",
        public_url="http://localhost:9000",
    ), mock_instance


def test_upload_returns_url():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        url = client.upload("doc123/file.mp4", b"data", "video/mp4")
        assert url == "http://localhost:9000/test-bucket/doc123/file.mp4"
        mock_instance.put_object.assert_called_once()


def test_upload_calls_put_object_with_correct_args():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        client.upload("myobj", b"hello", "text/plain")
        call_kwargs = mock_instance.put_object.call_args
        assert call_kwargs[0][0] == "test-bucket"
        assert call_kwargs[0][1] == "myobj"
        assert call_kwargs[0][3] == 5  # length of b"hello"


def test_delete_calls_remove_object():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        client.delete("doc123/file.mp4")
        mock_instance.remove_object.assert_called_once_with("test-bucket", "doc123/file.mp4")


def test_object_name_combines_document_id_and_filename():
    assert MinIOClient.object_name("abc123", "video.mp4") == "abc123/video.mp4"


def test_ensure_bucket_creates_if_missing():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        mock_instance = MagicMock()
        mock_instance.bucket_exists.return_value = False
        MockMinio.return_value = mock_instance
        MinIOClient("localhost:9000", "k", "s", "new-bucket")
        mock_instance.make_bucket.assert_called_once_with("new-bucket")
```

- [x] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_minio_client.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.storage'`

- [x] **Step 3: Create package and implementation**

Create `backend/app/storage/__init__.py` (empty file).

Create `backend/app/storage/minio_client.py`:
```python
import io

from minio import Minio


class MinIOClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        public_url: str = "",
    ):
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._public_url = public_url.rstrip("/")
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def upload(self, object_name: str, data: bytes, content_type: str) -> str:
        self._client.put_object(
            self._bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )
        return f"{self._public_url}/{self._bucket}/{object_name}"

    def delete(self, object_name: str) -> None:
        self._client.remove_object(self._bucket, object_name)

    @staticmethod
    def object_name(document_id: str, filename: str) -> str:
        return f"{document_id}/{filename}"
```

- [x] **Step 4: Run tests**

```
cd backend && uv run pytest tests/test_minio_client.py -v
```
Expected: PASS (5 tests)

- [x] **Step 5: Commit**

```bash
git add backend/app/storage/ backend/tests/test_minio_client.py
git commit -m "feat: add MinIO blob storage client with upload/delete"
```

---

## Task 4: Vision Enricher

**Files:**
- Create: `backend/app/ingestion/vision_enricher.py`
- Create: `backend/tests/test_vision_enricher.py`

- [x] **Step 1: Write failing tests**

Create `backend/tests/test_vision_enricher.py`:
```python
from unittest.mock import MagicMock, patch

from app.ingestion.vision_enricher import VisionEnricher


def test_describe_returns_model_text():
    with patch("app.ingestion.vision_enricher.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.text = "A golden retriever playing guitar in a sunlit room."
        MockClient.return_value.models.generate_content.return_value = mock_response

        enricher = VisionEnricher(api_key="fake-key")
        result = enricher.describe(b"fakeimagebytes", "image/jpeg")

        assert result == "A golden retriever playing guitar in a sunlit room."


def test_describe_returns_empty_string_on_none_text():
    with patch("app.ingestion.vision_enricher.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.text = None
        MockClient.return_value.models.generate_content.return_value = mock_response

        enricher = VisionEnricher(api_key="fake-key")
        result = enricher.describe(b"fakevideobytes", "video/mp4")

        assert result == ""


def test_describe_passes_blob_to_model():
    with patch("app.ingestion.vision_enricher.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.text = "description"
        MockClient.return_value.models.generate_content.return_value = mock_response

        enricher = VisionEnricher(api_key="fake-key", model="gemini-2.0-flash")
        enricher.describe(b"bytes", "image/png")

        MockClient.return_value.models.generate_content.assert_called_once()
        call_kwargs = MockClient.return_value.models.generate_content.call_args
        assert call_kwargs[1]["model"] == "gemini-2.0-flash"
```

- [x] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_vision_enricher.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ingestion.vision_enricher'`

- [x] **Step 3: Create vision_enricher.py**

Create `backend/app/ingestion/vision_enricher.py`:
```python
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
```

- [x] **Step 4: Run tests**

```
cd backend && uv run pytest tests/test_vision_enricher.py -v
```
Expected: PASS (3 tests)

- [x] **Step 5: Commit**

```bash
git add backend/app/ingestion/vision_enricher.py backend/tests/test_vision_enricher.py
git commit -m "feat: add VisionEnricher using Gemini Flash for image/video descriptions"
```

---

## Task 5: PDF Chunker

**Files:**
- Create: `backend/app/ingestion/pdf_chunker.py`
- Create: `backend/tests/test_pdf_chunker.py`

- [x] **Step 1: Write failing tests**

Create `backend/tests/test_pdf_chunker.py`:
```python
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
```

- [x] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_pdf_chunker.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [x] **Step 3: Create pdf_chunker.py**

Create `backend/app/ingestion/pdf_chunker.py`:
```python
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
```

- [x] **Step 4: Run tests**

```
cd backend && uv run pytest tests/test_pdf_chunker.py -v
```
Expected: PASS (4 tests)

- [x] **Step 5: Commit**

```bash
git add backend/app/ingestion/pdf_chunker.py backend/tests/test_pdf_chunker.py
git commit -m "feat: add PDFChunker splitting PDFs into 6-page windows for byte embedding"
```

---

## Task 6: Audio Processor

**Files:**
- Create: `backend/app/ingestion/audio_processor.py`
- Create: `backend/tests/test_audio_processor.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_audio_processor.py`:
```python
from unittest.mock import MagicMock, patch

from app.ingestion.audio_processor import AudioChunk, AudioProcessor


def test_process_short_audio_returns_one_chunk():
    processor = AudioProcessor(whisper_url="http://localhost:9010")
    with (
        patch.object(processor, "_get_duration", return_value=30.0),
        patch.object(processor, "_extract_segment", return_value=b"segmentbytes"),
        patch.object(processor, "transcribe", return_value="hello world"),
    ):
        chunks = processor.process(b"fakeaudio", "audio/mpeg")

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 30
    assert chunks[0].transcript == "hello world"
    assert chunks[0].data == b"segmentbytes"


def test_process_long_audio_returns_multiple_chunks():
    processor = AudioProcessor(whisper_url="http://localhost:9010")
    with (
        patch.object(processor, "_get_duration", return_value=400.0),
        patch.object(processor, "_extract_segment", return_value=b"seg"),
        patch.object(processor, "transcribe", return_value="text"),
    ):
        chunks = processor.process(b"fakeaudio", "audio/mpeg")

    assert len(chunks) >= 3


def test_audio_chunk_has_correct_fields():
    chunk = AudioChunk(data=b"d", mime_type="audio/mpeg", start_sec=0, end_sec=30, transcript="hi")
    assert chunk.transcript == "hi"
    assert chunk.start_sec == 0


def test_transcribe_posts_to_whisper():
    processor = AudioProcessor(whisper_url="http://localhost:9010")
    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "transcribed text"}
    mock_response.raise_for_status = MagicMock()

    with patch("app.ingestion.audio_processor.httpx.post", return_value=mock_response):
        result = processor.transcribe(b"audiodata", ".mp3")

    assert result == "transcribed text"
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_audio_processor.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create audio_processor.py**

Create `backend/app/ingestion/audio_processor.py`:
```python
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

import httpx


@dataclass
class AudioChunk:
    data: bytes
    mime_type: str
    start_sec: int
    end_sec: int
    transcript: str


class AudioProcessor:
    _CHUNK_DURATION = 170
    _OVERLAP = 10

    def __init__(self, whisper_url: str):
        self._whisper_url = whisper_url.rstrip("/")

    def _get_duration(self, input_path: str) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", input_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])

    def _extract_segment(self, input_path: str, start: int, end: int, suffix: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as out:
            out_path = out.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-ss", str(start), "-to", str(end), "-c", "copy", out_path],
                capture_output=True,
                check=True,
            )
            with open(out_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(out_path)

    def transcribe(self, audio_bytes: bytes, suffix: str = ".mp3") -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            with open(tmp_path, "rb") as f:
                response = httpx.post(
                    f"{self._whisper_url}/v1/audio/transcriptions",
                    files={"file": (f"audio{suffix}", f, "audio/mpeg")},
                    data={"model": "whisper-1", "response_format": "json"},
                    timeout=120.0,
                )
            response.raise_for_status()
            return response.json().get("text", "")
        finally:
            os.unlink(tmp_path)

    def process(self, audio_bytes: bytes, mime_type: str) -> list[AudioChunk]:
        suffix = ".mp3" if "mp3" in mime_type else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name
        try:
            duration = self._get_duration(input_path)
            chunks: list[AudioChunk] = []
            start = 0
            while start < int(duration):
                end = min(start + self._CHUNK_DURATION, int(duration))
                segment_bytes = self._extract_segment(input_path, start, end, suffix)
                transcript = self.transcribe(segment_bytes, suffix)
                chunks.append(
                    AudioChunk(
                        data=segment_bytes,
                        mime_type=mime_type,
                        start_sec=start,
                        end_sec=end,
                        transcript=transcript,
                    )
                )
                if end >= int(duration):
                    break
                start = end - self._OVERLAP
            return chunks
        finally:
            os.unlink(input_path)
```

- [ ] **Step 4: Run tests**

```
cd backend && uv run pytest tests/test_audio_processor.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/audio_processor.py backend/tests/test_audio_processor.py
git commit -m "feat: add AudioProcessor with ffmpeg chunking and Whisper transcription"
```

---

## Task 7: Video Processor

**Files:**
- Create: `backend/app/ingestion/video_processor.py`
- Create: `backend/tests/test_video_processor.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_video_processor.py`:
```python
from unittest.mock import patch

from app.ingestion.video_processor import VideoChunk, VideoProcessor


def test_process_short_video_returns_one_chunk():
    processor = VideoProcessor()
    with (
        patch.object(processor, "_get_duration", return_value=50.0),
        patch.object(processor, "_extract_video_segment", return_value=b"fakevideo"),
    ):
        chunks = processor.process(b"rawvideo", "video/mp4")

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 50
    assert chunks[0].data == b"fakevideo"
    assert chunks[0].mime_type == "video/mp4"


def test_process_long_video_returns_multiple_chunks():
    processor = VideoProcessor()
    with (
        patch.object(processor, "_get_duration", return_value=300.0),
        patch.object(processor, "_extract_video_segment", return_value=b"seg"),
    ):
        chunks = processor.process(b"rawvideo", "video/mp4")

    assert len(chunks) >= 3


def test_video_chunk_has_correct_fields():
    chunk = VideoChunk(data=b"d", mime_type="video/mp4", start_sec=0, end_sec=115)
    assert chunk.start_sec == 0
    assert chunk.end_sec == 115
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_video_processor.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create video_processor.py**

Create `backend/app/ingestion/video_processor.py`:
```python
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class VideoChunk:
    data: bytes
    mime_type: str
    start_sec: int
    end_sec: int


class VideoProcessor:
    _CHUNK_DURATION = 115
    _OVERLAP = 5

    def _get_duration(self, input_path: str) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", input_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])

    def _extract_video_segment(self, input_path: str, start: int, end: int) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out:
            out_path = out.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-ss", str(start), "-to", str(end), "-c", "copy", out_path],
                capture_output=True,
                check=True,
            )
            with open(out_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(out_path)

    def extract_audio(self, video_bytes: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as inp:
            inp.write(video_bytes)
            input_path = inp.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as out:
            out_path = out.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "libmp3lame", out_path],
                capture_output=True,
                check=True,
            )
            with open(out_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(input_path)
            os.unlink(out_path)

    def process(self, video_bytes: bytes, mime_type: str = "video/mp4") -> list[VideoChunk]:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            input_path = f.name
        try:
            duration = self._get_duration(input_path)
            chunks: list[VideoChunk] = []
            start = 0
            while start < int(duration):
                end = min(start + self._CHUNK_DURATION, int(duration))
                segment_bytes = self._extract_video_segment(input_path, start, end)
                chunks.append(VideoChunk(data=segment_bytes, mime_type=mime_type, start_sec=start, end_sec=end))
                if end >= int(duration):
                    break
                start = end - self._OVERLAP
            return chunks
        finally:
            os.unlink(input_path)
```

- [ ] **Step 4: Run tests**

```
cd backend && uv run pytest tests/test_video_processor.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/video_processor.py backend/tests/test_video_processor.py
git commit -m "feat: add VideoProcessor with ffmpeg chunking and audio extraction"
```

---

## Task 8: Embedder Extension

**Files:**
- Modify: `backend/app/ingestion/embedder.py`
- Modify: `backend/tests/test_embedder.py`

- [ ] **Step 1: Write failing tests**

Append to `backend/tests/test_embedder.py`:
```python
def test_embed_audio_returns_vector():
    with patch("app.ingestion.embedder.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
        MockClient.return_value.models.embed_content.return_value = mock_response

        embedder = GeminiEmbedder(api_key="fake", model="gemini-embedding-2")
        result = embedder.embed_audio(b"fakeaudio", "audio/mpeg")

        assert result == [0.1, 0.2, 0.3]


def test_embed_video_returns_vector():
    with patch("app.ingestion.embedder.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.4, 0.5, 0.6])]
        MockClient.return_value.models.embed_content.return_value = mock_response

        embedder = GeminiEmbedder(api_key="fake", model="gemini-embedding-2")
        result = embedder.embed_video(b"fakevideo", "video/mp4")

        assert result == [0.4, 0.5, 0.6]


def test_embed_pdf_chunk_returns_vector():
    with patch("app.ingestion.embedder.genai.Client") as MockClient:
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.7, 0.8, 0.9])]
        MockClient.return_value.models.embed_content.return_value = mock_response

        embedder = GeminiEmbedder(api_key="fake", model="gemini-embedding-2")
        result = embedder.embed_pdf_chunk(b"fakepdf")

        assert result == [0.7, 0.8, 0.9]
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_embedder.py -v
```
Expected: 3 new tests FAIL — `AttributeError: 'GeminiEmbedder' has no attribute 'embed_audio'`

- [ ] **Step 3: Add methods to embedder.py**

Append to the `GeminiEmbedder` class in `backend/app/ingestion/embedder.py`:
```python
    def embed_bytes(self, data: bytes, mime_type: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(mime_type=mime_type, data=data)
                        )
                    ]
                )
            ],
        )
        return list(
            response.embeddings[0].values
            if response.embeddings and response.embeddings[0].values
            else []
        )

    def embed_audio(self, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> list[float]:
        return self.embed_bytes(audio_bytes, mime_type)

    def embed_video(self, video_bytes: bytes, mime_type: str = "video/mp4") -> list[float]:
        return self.embed_bytes(video_bytes, mime_type)

    def embed_pdf_chunk(self, pdf_bytes: bytes) -> list[float]:
        return self.embed_bytes(pdf_bytes, "application/pdf")
```

- [ ] **Step 4: Run all embedder tests**

```
cd backend && uv run pytest tests/test_embedder.py -v
```
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/embedder.py backend/tests/test_embedder.py
git commit -m "feat: add embed_bytes/audio/video/pdf_chunk to GeminiEmbedder"
```

---

## Task 9: Schema Update

**Files:**
- Modify: `backend/app/models/schemas.py`
- Modify: `backend/tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

Append to `backend/tests/test_schemas.py`:
```python
def test_chunk_result_includes_blob_url():
    chunk = ChunkResult(
        document_id="d1",
        filename="video.mp4",
        content="description",
        content_type="video",
        media_type="video",
        score=0.9,
        blob_url="http://localhost:9000/airrag/d1/video.mp4",
        chunk_start_sec=60.0,
        chunk_end_sec=120.0,
    )
    assert chunk.blob_url == "http://localhost:9000/airrag/d1/video.mp4"
    assert chunk.chunk_start_sec == 60.0


def test_chunk_result_optional_fields_default_to_none():
    chunk = ChunkResult(
        document_id="d1",
        filename="doc.txt",
        content="hello",
        content_type="text",
        media_type="text",
        score=0.8,
    )
    assert chunk.blob_url is None
    assert chunk.chunk_start_sec is None
    assert chunk.vision_description is None
    assert chunk.page_start is None
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_schemas.py -v
```
Expected: FAIL — `ChunkResult` missing fields

- [ ] **Step 3: Update schemas.py**

Replace the `ChunkResult` class in `backend/app/models/schemas.py`:
```python
class ChunkResult(BaseModel):
    document_id: str
    filename: str
    content: str
    content_type: str
    media_type: str
    score: float
    blob_url: Optional[str] = None
    chunk_start_sec: Optional[float] = None
    chunk_end_sec: Optional[float] = None
    vision_description: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
```

- [ ] **Step 4: Run all schema tests**

```
cd backend && uv run pytest tests/test_schemas.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/schemas.py backend/tests/test_schemas.py
git commit -m "feat: extend ChunkResult schema with blob_url, timestamps, vision_description"
```

---

## Task 10: Ingestor Rewrite

**Files:**
- Modify: `backend/app/ingestion/ingestor.py`
- Create: `backend/tests/test_ingestor.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_ingestor.py`:
```python
from unittest.mock import MagicMock, patch

from app.ingestion.ingestor import Ingestor


def _make_ingestor():
    embedder = MagicMock()
    embedder.embed_image.return_value = [0.1] * 3072
    embedder.embed_texts.return_value = [[0.2] * 3072]
    embedder.embed_audio.return_value = [0.3] * 3072
    embedder.embed_video.return_value = [0.4] * 3072
    embedder.embed_pdf_chunk.return_value = [0.5] * 3072

    store = MagicMock()
    minio = MagicMock()
    minio.upload.return_value = "http://localhost:9000/airrag/doc1/file.jpg"
    vision = MagicMock()
    vision.describe.return_value = "A red running shoe."
    audio = MagicMock()

    from app.ingestion.audio_processor import AudioChunk
    audio.process.return_value = [
        AudioChunk(data=b"seg", mime_type="audio/mpeg", start_sec=0, end_sec=30, transcript="hello")
    ]
    video = MagicMock()

    from app.ingestion.video_processor import VideoChunk
    video.process.return_value = [
        VideoChunk(data=b"vseg", mime_type="video/mp4", start_sec=0, end_sec=50)
    ]
    video.extract_audio.return_value = b"audiotrack"
    audio.transcribe.return_value = "spoken words"

    pdf_chunker = MagicMock()
    pdf_chunker.chunk.return_value = [(b"pdfchunk", 1, 6)]

    return Ingestor(
        embedder=embedder,
        vector_store=store,
        minio_client=minio,
        vision_enricher=vision,
        audio_processor=audio,
        video_processor=video,
        pdf_chunker=pdf_chunker,
    ), store, minio


def test_ingest_image_uploads_to_minio_and_stores_vector():
    ingestor, store, minio = _make_ingestor()
    count = ingestor.ingest("col1", "doc1", "photo.jpg", "image/jpeg", b"imgdata")
    assert count == 1
    minio.upload.assert_called_once()
    store.upsert.assert_called_once()
    payload = store.upsert.call_args[0][4][0]
    assert payload["media_type"] == "image"
    assert payload["blob_url"] == "http://localhost:9000/airrag/doc1/file.jpg"
    assert payload["vision_description"] == "A red running shoe."


def test_ingest_audio_creates_chunks():
    ingestor, store, _ = _make_ingestor()
    count = ingestor.ingest("col1", "doc1", "talk.mp3", "audio/mpeg", b"audiodata")
    assert count == 1
    payload = store.upsert.call_args[0][4][0]
    assert payload["media_type"] == "audio"
    assert payload["transcript"] == "hello"
    assert payload["chunk_start_sec"] == 0


def test_ingest_video_creates_chunks_with_vision():
    ingestor, store, _ = _make_ingestor()
    count = ingestor.ingest("col1", "doc1", "clip.mp4", "video/mp4", b"videodata")
    assert count == 1
    payload = store.upsert.call_args[0][4][0]
    assert payload["media_type"] == "video"
    assert "vision_description" in payload


def test_ingest_pdf_chunks_by_pages():
    ingestor, store, _ = _make_ingestor()
    with patch.object(ingestor._parser, "parse_document_bytes", return_value=[]):
        count = ingestor.ingest("col1", "doc1", "paper.pdf", "application/pdf", b"pdfdata")
    assert count == 1
    payload = store.upsert.call_args[0][4][0]
    assert payload["media_type"] == "pdf"
    assert payload["page_start"] == 1
    assert payload["page_end"] == 6


def test_ingest_text_embeds_chunks():
    ingestor, store, _ = _make_ingestor()
    count = ingestor.ingest("col1", "doc1", "readme.txt", "text/plain", b"hello world")
    assert count >= 1
    payload = store.upsert.call_args[0][4][0]
    assert payload["media_type"] == "text"


def test_unsupported_type_raises_value_error():
    ingestor, _, _ = _make_ingestor()
    try:
        ingestor.ingest("col1", "doc1", "file.xyz", "application/xyz", b"data")
        assert False, "should have raised"
    except ValueError:
        pass


def test_delete_document_removes_from_store_and_minio():
    ingestor, store, minio = _make_ingestor()
    ingestor.delete_document("col1", "doc1", "photo.jpg")
    store.delete_by_document_id.assert_called_once_with("col1", "doc1")
    minio.delete.assert_called_once_with("doc1/photo.jpg")
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && uv run pytest tests/test_ingestor.py -v
```
Expected: FAIL — import errors and wrong Ingestor signature

- [ ] **Step 3: Rewrite ingestor.py**

Replace `backend/app/ingestion/ingestor.py` with:
```python
from datetime import datetime, timezone

from app.ingestion.audio_processor import AudioProcessor
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.parser import DocumentParser
from app.ingestion.pdf_chunker import PDFChunker
from app.ingestion.video_processor import VideoProcessor
from app.ingestion.vision_enricher import VisionEnricher
from app.retrieval.vector_store import QdrantStore
from app.storage.minio_client import MinIOClient

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
SUPPORTED_PDF_TYPES = {"application/pdf"}
SUPPORTED_AUDIO_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav"}
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
SUPPORTED_TEXT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/markdown",
}


class Ingestor:
    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: QdrantStore,
        minio_client: MinIOClient,
        vision_enricher: VisionEnricher,
        audio_processor: AudioProcessor,
        video_processor: VideoProcessor,
        pdf_chunker: PDFChunker,
    ):
        self._embedder = embedder
        self._store = vector_store
        self._minio = minio_client
        self._vision = vision_enricher
        self._audio = audio_processor
        self._video = video_processor
        self._pdf_chunker = pdf_chunker
        self._parser = DocumentParser()

    def ingest(
        self,
        collection_id: str,
        document_id: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> int:
        ingested_at = datetime.now(timezone.utc).isoformat()
        object_name = MinIOClient.object_name(document_id, filename)
        blob_url = self._minio.upload(object_name, content, content_type)

        if content_type in SUPPORTED_IMAGE_TYPES:
            return self._ingest_image(collection_id, document_id, filename, content_type, content, blob_url, ingested_at)
        if content_type in SUPPORTED_PDF_TYPES:
            return self._ingest_pdf(collection_id, document_id, filename, content, blob_url, ingested_at)
        if content_type in SUPPORTED_AUDIO_TYPES:
            return self._ingest_audio(collection_id, document_id, filename, content_type, content, blob_url, ingested_at)
        if content_type in SUPPORTED_VIDEO_TYPES:
            return self._ingest_video(collection_id, document_id, filename, content_type, content, blob_url, ingested_at)
        if content_type in SUPPORTED_TEXT_TYPES:
            return self._ingest_text(collection_id, document_id, filename, content_type, content, blob_url, ingested_at)
        raise ValueError(f"Unsupported content type: {content_type}")

    def _ingest_image(self, collection_id, document_id, filename, mime_type, content, blob_url, ingested_at) -> int:
        vision_description = self._vision.describe(content, mime_type)
        embedding = self._embedder.embed_image(content, mime_type)
        doc_text = vision_description or f"[IMAGE: {filename}]"
        self._store.upsert(
            collection_id,
            [f"{document_id}-img-0"],
            [embedding],
            [doc_text],
            [{
                "document_id": document_id, "filename": filename,
                "media_type": "image", "mime_type": mime_type,
                "chunk_index": 0, "blob_url": blob_url,
                "vision_description": vision_description, "ingested_at": ingested_at,
            }],
        )
        return 1

    def _ingest_pdf(self, collection_id, document_id, filename, content, blob_url, ingested_at) -> int:
        chunks = self._pdf_chunker.chunk(content)
        if not chunks:
            return 0
        chunk_ids, embeddings, documents, metadatas = [], [], [], []
        for i, (chunk_bytes, page_start, page_end) in enumerate(chunks):
            embedding = self._embedder.embed_pdf_chunk(chunk_bytes)
            text_chunks = self._parser.parse_document_bytes(chunk_bytes, filename, source_id=f"{document_id}-{i}")
            doc_text = " ".join(c.text for c in text_chunks) if text_chunks else f"[PDF pages {page_start}-{page_end}]"
            chunk_ids.append(f"{document_id}-pdf-{i}")
            embeddings.append(embedding)
            documents.append(doc_text)
            metadatas.append({
                "document_id": document_id, "filename": filename,
                "media_type": "pdf", "mime_type": "application/pdf",
                "chunk_index": i, "blob_url": blob_url,
                "page_start": page_start, "page_end": page_end, "ingested_at": ingested_at,
            })
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(chunks)

    def _ingest_audio(self, collection_id, document_id, filename, mime_type, content, blob_url, ingested_at) -> int:
        audio_chunks = self._audio.process(content, mime_type)
        if not audio_chunks:
            return 0
        chunk_ids = [f"{document_id}-audio-{i}" for i in range(len(audio_chunks))]
        embeddings = [self._embedder.embed_audio(c.data, c.mime_type) for c in audio_chunks]
        documents = [c.transcript or f"[AUDIO: {filename} {c.start_sec}-{c.end_sec}s]" for c in audio_chunks]
        metadatas = [
            {
                "document_id": document_id, "filename": filename,
                "media_type": "audio", "mime_type": mime_type,
                "chunk_index": i, "blob_url": blob_url,
                "chunk_start_sec": c.start_sec, "chunk_end_sec": c.end_sec,
                "transcript": c.transcript, "ingested_at": ingested_at,
            }
            for i, c in enumerate(audio_chunks)
        ]
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(audio_chunks)

    def _ingest_video(self, collection_id, document_id, filename, mime_type, content, blob_url, ingested_at) -> int:
        video_chunks = self._video.process(content, mime_type)
        if not video_chunks:
            return 0
        chunk_ids, embeddings, documents, metadatas = [], [], [], []
        for i, vc in enumerate(video_chunks):
            vision_description = self._vision.describe(vc.data, vc.mime_type)
            try:
                audio_bytes = self._video.extract_audio(vc.data)
                transcript = self._audio.transcribe(audio_bytes, ".mp3")
            except Exception:
                transcript = ""
            embedding = self._embedder.embed_video(vc.data, vc.mime_type)
            parts = [p for p in [vision_description, f"Transcript: {transcript}" if transcript else ""] if p]
            doc_text = "\n\n".join(parts) or f"[VIDEO: {filename} {vc.start_sec}-{vc.end_sec}s]"
            chunk_ids.append(f"{document_id}-video-{i}")
            embeddings.append(embedding)
            documents.append(doc_text)
            metadatas.append({
                "document_id": document_id, "filename": filename,
                "media_type": "video", "mime_type": mime_type,
                "chunk_index": i, "blob_url": blob_url,
                "chunk_start_sec": vc.start_sec, "chunk_end_sec": vc.end_sec,
                "vision_description": vision_description, "audio_transcript": transcript,
                "ingested_at": ingested_at,
            })
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(video_chunks)

    def _ingest_text(self, collection_id, document_id, filename, mime_type, content, blob_url, ingested_at) -> int:
        chunks = self._parser.parse_document_bytes(content, filename, source_id=document_id)
        if not chunks:
            return 0
        chunk_ids = [f"{document_id}-chunk-{i}" for i in range(len(chunks))]
        embeddings = self._embedder.embed_texts([c.text for c in chunks])
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": document_id, "filename": filename,
                "media_type": "text", "mime_type": mime_type,
                "chunk_index": c.chunk_index, "blob_url": blob_url, "ingested_at": ingested_at,
            }
            for c in chunks
        ]
        self._store.upsert(collection_id, chunk_ids, embeddings, documents, metadatas)
        return len(chunks)

    def delete_document(self, collection_id: str, document_id: str, filename: str = "") -> None:
        self._store.delete_by_document_id(collection_id, document_id)
        if filename:
            try:
                self._minio.delete(MinIOClient.object_name(document_id, filename))
            except Exception:
                pass
```

- [ ] **Step 4: Run ingestor tests**

```
cd backend && uv run pytest tests/test_ingestor.py -v
```
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/ingestor.py backend/tests/test_ingestor.py
git commit -m "feat: rewrite Ingestor with full multimodal pipeline (image/pdf/audio/video/text)"
```

---

## Task 11: API Updates

**Files:**
- Modify: `backend/app/api/documents.py`
- Modify: `backend/app/api/query.py`

- [ ] **Step 1: Update documents.py — wire all new Ingestor dependencies**

Replace `_make_ingestor` and the delete route in `backend/app/api/documents.py`:

```python
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.ingestion.audio_processor import AudioProcessor
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.ingestor import Ingestor
from app.ingestion.pdf_chunker import PDFChunker
from app.ingestion.video_processor import VideoProcessor
from app.ingestion.vision_enricher import VisionEnricher
from app.models.db_models import Collection, Document
from app.models.schemas import DocumentResponse
from app.retrieval.vector_store import QdrantStore
from app.storage.minio_client import MinIOClient

router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])


def _make_ingestor() -> Ingestor:
    return Ingestor(
        embedder=GeminiEmbedder(api_key=settings.gemini_api_key, model=settings.gemini_embed_model),
        vector_store=QdrantStore(url=settings.qdrant_url, embed_dim=settings.gemini_embed_dim),
        minio_client=MinIOClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
            public_url=settings.minio_public_url,
        ),
        vision_enricher=VisionEnricher(api_key=settings.gemini_api_key, model=settings.gemini_gen_model),
        audio_processor=AudioProcessor(whisper_url=settings.whisper_server_url),
        video_processor=VideoProcessor(),
        pdf_chunker=PDFChunker(),
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    coll = db.get(Collection, collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")

    content = await file.read()
    content_type = file.content_type or "application/octet-stream"
    doc = Document(
        collection_id=collection_id,
        filename=file.filename,
        content_type=content_type,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        chunk_count = _make_ingestor().ingest(
            collection_id=collection_id,
            document_id=doc.id,
            filename=file.filename or "",
            content_type=content_type,
            content=content,
        )
        doc.status = "ready"
        doc.chunk_count = chunk_count
    except ValueError as e:
        doc.status = "error"
        db.commit()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        doc.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    db.commit()
    db.refresh(doc)
    return doc


@router.get("", response_model=list[DocumentResponse])
def list_documents(collection_id: str, db: Session = Depends(get_db)):
    if not db.get(Collection, collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    return db.query(Document).filter(Document.collection_id == collection_id).all()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    collection_id: str, document_id: str, db: Session = Depends(get_db)
):
    doc = db.get(Document, document_id)
    if not doc or doc.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Document not found")
    _make_ingestor().delete_document(collection_id, document_id, doc.filename or "")
    db.delete(doc)
    db.commit()
```

- [ ] **Step 2: Update query.py — map new payload fields into ChunkResult**

Replace `backend/app/api/query.py` with:
```python
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.ingestion.embedder import GeminiEmbedder
from app.models.db_models import Collection
from app.models.schemas import ChunkResult, QueryResponse
from app.retrieval.generator import GeminiGenerator
from app.retrieval.vector_store import QdrantStore

router = APIRouter(prefix="/collections/{collection_id}/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_collection(
    collection_id: str,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(5),
    db: Session = Depends(get_db),
):
    if not db.get(Collection, collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    if not text and not image:
        raise HTTPException(status_code=400, detail="Provide text or image for query")

    embedder = GeminiEmbedder(api_key=settings.gemini_api_key, model=settings.gemini_embed_model)
    store = QdrantStore(url=settings.qdrant_url, embed_dim=settings.gemini_embed_dim)
    generator = GeminiGenerator(api_key=settings.gemini_api_key, model=settings.gemini_gen_model)

    if image:
        image_bytes = await image.read()
        query_embedding = embedder.embed_image_query(image_bytes, image.content_type or "image/jpeg")
        query_text = text or "[IMAGE QUERY]"
    else:
        query_embedding = embedder.embed_query(text or "")
        query_text = text

    results = store.search(collection_name=collection_id, query_embedding=query_embedding, top_k=top_k)
    sources = [
        ChunkResult(
            document_id=r["metadata"].get("document_id", ""),
            filename=r["metadata"].get("filename", ""),
            content=r["document"] or "",
            content_type=r["metadata"].get("media_type", "text"),
            media_type=r["metadata"].get("media_type", "text"),
            score=r["score"],
            blob_url=r["metadata"].get("blob_url"),
            chunk_start_sec=r["metadata"].get("chunk_start_sec"),
            chunk_end_sec=r["metadata"].get("chunk_end_sec"),
            vision_description=r["metadata"].get("vision_description"),
            page_start=r["metadata"].get("page_start"),
            page_end=r["metadata"].get("page_end"),
        )
        for r in results
    ]
    answer = generator.generate(query=query_text or "", context_chunks=[r["document"] for r in results])
    return QueryResponse(answer=answer, sources=sources)
```

- [ ] **Step 3: Run full test suite**

```
cd backend && uv run pytest tests/ -v --ignore=tests/test_deps.py
```
Expected: All existing + new tests PASS. (Note: test_api.py tests that call `_make_ingestor` will fail if MinIO is not running — those are integration tests. Skip with `-k "not test_api"` if running without Docker.)

- [ ] **Step 4: Run just the unit tests**

```
cd backend && uv run pytest tests/ -v -k "not test_api and not test_vector_store"
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/documents.py backend/app/api/query.py
git commit -m "feat: update API to wire multimodal Ingestor and return blob_url/timestamps in query"
```

---

## Task 12: End-to-End Verification

- [ ] **Step 1: Start all services**

```bash
docker compose up -d
```
Expected: airrag-qdrant, airrag-minio, airrag-whisper all start. Check with `docker compose ps`.

- [ ] **Step 2: Start the backend**

```bash
cd backend && uv run uvicorn main:app --reload --port 8000
```
Expected: Server starts, MinIO bucket `airrag` is created on first request.

- [ ] **Step 3: Create a collection**

```bash
curl -s -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "multimodal-test", "description": "End-to-end test"}' | python -m json.tool
```
Expected: JSON response with collection `id`. Copy the `id` for next steps.

- [ ] **Step 4: Upload an image**

```bash
curl -s -X POST http://localhost:8000/collections/{COLLECTION_ID}/documents \
  -F "file=@/path/to/test.jpg" | python -m json.tool
```
Expected: `"status": "ready"`, `"chunk_count": 1`

- [ ] **Step 5: Upload a PDF**

```bash
curl -s -X POST http://localhost:8000/collections/{COLLECTION_ID}/documents \
  -F "file=@/path/to/test.pdf" | python -m json.tool
```
Expected: `"status": "ready"`, chunk_count matches ceil(pages/6)

- [ ] **Step 6: Upload audio (requires Whisper running)**

```bash
curl -s -X POST http://localhost:8000/collections/{COLLECTION_ID}/documents \
  -F "file=@/path/to/test.mp3" | python -m json.tool
```
Expected: `"status": "ready"`, chunk_count >= 1

- [ ] **Step 7: Query and verify blob_url in response**

```bash
curl -s -X POST http://localhost:8000/collections/{COLLECTION_ID}/query \
  -F "text=describe the content" \
  -F "top_k=3" | python -m json.tool
```
Expected: Response contains `sources` with `blob_url`, `media_type`, and optional `chunk_start_sec` fields populated.

- [ ] **Step 8: Run the full test suite one final time**

```
cd backend && uv run pytest tests/ -v -k "not test_api and not test_vector_store"
```
Expected: All unit tests PASS.

---

## Verification Summary

| Test | Command | Expected |
|------|---------|----------|
| Unit tests | `uv run pytest tests/ -k "not test_api and not test_vector_store"` | All PASS |
| Config defaults | `uv run pytest tests/test_config.py -v` | 4 tests PASS |
| MinIO client | `uv run pytest tests/test_minio_client.py -v` | 5 tests PASS |
| Vision enricher | `uv run pytest tests/test_vision_enricher.py -v` | 3 tests PASS |
| PDF chunker | `uv run pytest tests/test_pdf_chunker.py -v` | 4 tests PASS |
| Audio processor | `uv run pytest tests/test_audio_processor.py -v` | 4 tests PASS |
| Video processor | `uv run pytest tests/test_video_processor.py -v` | 3 tests PASS |
| Embedder extensions | `uv run pytest tests/test_embedder.py -v` | 6 tests PASS |
| Schema extensions | `uv run pytest tests/test_schemas.py -v` | All PASS |
| Ingestor | `uv run pytest tests/test_ingestor.py -v` | 7 tests PASS |
| Integration (Docker) | `docker compose up -d && curl POST /collections/{id}/documents` | 201 with blob_url |
