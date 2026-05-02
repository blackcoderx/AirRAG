# AirRAG Backend

Lightweight multimodal RAG (Retrieval-Augmented Generation) as a Service. Upload PDFs, Office documents, and images into named collections, then query them with text or an image. Gemini's unified embedding space means a single query can retrieve both text chunks and images in one pass.

---

## How it works

```
Upload file → parse → chunk → embed (Gemini) → store (Qdrant)
Query text/image → embed → vector search → generate answer (Gemini)
```

- **Embeddings:** `gemini-embedding-exp-03-07` — text and images share a single 3072-dim vector space
- **Vector store:** Qdrant running in Docker (`docker compose up -d`)
- **Metadata:** SQLite via SQLAlchemy (`./airrag.db`)
- **Generation:** `gemini-2.0-flash` with grounded prompts (answers only from retrieved context)

---

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A [Gemini API key](https://aistudio.google.com/app/apikey)

---

## Setup

```bash
# 1. Copy the env template and add your key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 2. Install dependencies
uv sync

# 3. Start Qdrant (requires Docker)
docker compose up -d

# 4. Start the server
uv run uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

---

## API

### Collections

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections` | Create a knowledge base |
| `GET` | `/collections` | List all collections |
| `GET` | `/collections/{id}` | Get collection details |
| `DELETE` | `/collections/{id}` | Delete collection and all its documents |

**Create collection:**
```bash
curl -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my-kb", "description": "optional description"}'
```

---

### Documents

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/{id}/documents` | Upload and ingest a file |
| `GET` | `/collections/{id}/documents` | List documents in a collection |
| `DELETE` | `/collections/{id}/documents/{doc_id}` | Remove a document and its vectors |

**Supported file types:** PDF, DOCX, PPTX, TXT, Markdown, JPEG, PNG, WebP, GIF

**Upload a file:**
```bash
curl -X POST http://localhost:8000/collections/{id}/documents \
  -F "file=@report.pdf"
```

---

### Query

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/{id}/query` | Search and generate an answer |

Accepts a multipart form with either `text`, `image`, or both. The query is embedded into the same vector space as stored content, so a text query can retrieve images and vice versa.

**Text query:**
```bash
curl -X POST http://localhost:8000/collections/{id}/query \
  -F "text=what are the main findings?" \
  -F "top_k=5"
```

**Image query (cross-modal):**
```bash
curl -X POST http://localhost:8000/collections/{id}/query \
  -F "image=@photo.jpg"
```

**Response:**
```json
{
  "answer": "Based on the context, the main findings are...",
  "sources": [
    {
      "document_id": "...",
      "filename": "report.pdf",
      "content": "chunk text or [IMAGE: filename]",
      "content_type": "text",
      "score": 0.91
    }
  ]
}
```

---

## Project structure

```
backend/
├── main.py                    # FastAPI app + DB init on startup
├── .env.example               # Environment variable template
├── Dockerfile                 # Production container
├── app/
│   ├── core/
│   │   ├── config.py          # Settings loaded from environment
│   │   └── database.py        # SQLAlchemy engine and session
│   ├── models/
│   │   ├── db_models.py       # Collection + Document ORM tables
│   │   └── schemas.py         # Pydantic request/response shapes
│   ├── ingestion/
│   │   ├── embedder.py        # Gemini multimodal embedding calls
│   │   ├── parser.py          # Document → text chunks / image bytes
│   │   └── ingestor.py        # Orchestrates parse → embed → store
│   ├── retrieval/
│   │   ├── vector_store.py    # Qdrant wrapper (upsert, search, delete)
│   │   └── generator.py       # Grounded answer generation via Gemini
│   └── api/
│       ├── collections.py     # /collections CRUD router
│       ├── documents.py       # /documents upload router
│       └── query.py           # /query search + generate router
└── tests/                     # 22 unit + integration tests
```

---

## Configuration

All settings are read from environment variables (or a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | _(required)_ | Google AI Studio API key |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-exp-03-07` | Embedding model |
| `GEMINI_EMBED_DIM` | `3072` | Embedding vector size |
| `GEMINI_GEN_MODEL` | `gemini-2.0-flash` | Generation model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `DATABASE_URL` | `sqlite:///./airrag.db` | SQLite metadata database |
| `STORAGE_DIR` | `./storage` | Raw file storage (future use) |

---

## Docker

Qdrant runs as a separate container managed by `docker-compose.yml`:

```bash
# Start Qdrant
docker compose up -d

# Build and run the backend (connects to Qdrant on the Docker network)
docker build -t airrag .
docker run -p 8000:8000 \
  --network backend_default \
  -e GEMINI_API_KEY=your_key \
  -e QDRANT_URL=http://airrag-qdrant:6333 \
  -v $(pwd)/airrag.db:/app/airrag.db \
  airrag
```

---

## Tests

```bash
uv run pytest tests/ -v
# 22 passed
```
