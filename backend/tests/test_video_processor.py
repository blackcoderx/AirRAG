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
