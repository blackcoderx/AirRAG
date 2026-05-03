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
