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
        AudioChunk(data=b"seg", mime_type="audio/mpeg", start_sec=0, end_sec=30)
    ]
    video = MagicMock()

    from app.ingestion.video_processor import VideoChunk
    video.process.return_value = [
        VideoChunk(data=b"vseg", mime_type="video/mp4", start_sec=0, end_sec=50)
    ]
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
    assert payload["chunk_start_sec"] == 0


def test_ingest_video_creates_chunks_with_vision():
    ingestor, store, _ = _make_ingestor()
    count = ingestor.ingest("col1", "doc1", "clip.mp4", "video/mp4", b"videodata")
    assert count == 1
    payload = store.upsert.call_args[0][4][0]
    assert payload["media_type"] == "video"
    assert payload["vision_description"] == "A red running shoe."
    assert payload["chunk_start_sec"] == 0
    assert payload["chunk_end_sec"] == 50


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
