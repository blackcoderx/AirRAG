from unittest.mock import patch

from app.ingestion.audio_processor import AudioChunk, AudioProcessor


def test_process_short_audio_returns_one_chunk():
    processor = AudioProcessor()
    with (
        patch.object(processor, "_get_duration", return_value=30.0),
        patch.object(processor, "_extract_segment", return_value=b"segmentbytes"),
    ):
        chunks = processor.process(b"fakeaudio", "audio/mpeg")

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 30
    assert chunks[0].data == b"segmentbytes"


def test_process_long_audio_returns_multiple_chunks():
    processor = AudioProcessor()
    with (
        patch.object(processor, "_get_duration", return_value=400.0),
        patch.object(processor, "_extract_segment", return_value=b"seg"),
    ):
        chunks = processor.process(b"fakeaudio", "audio/mpeg")

    assert len(chunks) >= 3


def test_audio_chunk_has_correct_fields():
    chunk = AudioChunk(data=b"d", mime_type="audio/mpeg", start_sec=0, end_sec=30)
    assert chunk.start_sec == 0
    assert chunk.end_sec == 30
    assert chunk.data == b"d"


def test_process_overlap_applies_between_chunks():
    processor = AudioProcessor()
    with (
        patch.object(processor, "_get_duration", return_value=400.0),
        patch.object(processor, "_extract_segment", return_value=b"seg"),
    ):
        chunks = processor.process(b"fakeaudio", "audio/mpeg")

    # With 170s chunks and 15s overlap: chunk 0 ends at 170, chunk 1 starts at 155
    assert chunks[1].start_sec == 170 - AudioProcessor._OVERLAP
