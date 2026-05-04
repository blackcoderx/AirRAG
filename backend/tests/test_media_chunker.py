from unittest.mock import patch

from app.ingestion.media_chunker import MediaChunk, MediaChunker


def test_short_audio_returns_one_chunk():
    chunker = MediaChunker(chunk_duration=150, overlap=15, hard_limit=180, media_type="audio")
    with (
        patch.object(chunker, "_get_duration", return_value=30.0),
        patch.object(chunker, "_extract_audio_segment", return_value=b"segmentbytes"),
    ):
        chunks = chunker.process(b"fakeaudio", "audio/mpeg")

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 30
    assert chunks[0].data == b"segmentbytes"
    assert chunks[0].mime_type == "audio/mpeg"


def test_short_video_returns_one_chunk():
    chunker = MediaChunker(chunk_duration=60, overlap=15, hard_limit=120, media_type="video")
    with (
        patch.object(chunker, "_get_duration", return_value=50.0),
        patch.object(chunker, "_extract_video_segment", return_value=b"fakevideo"),
    ):
        chunks = chunker.process(b"rawvideo", "video/mp4")

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 50
    assert chunks[0].data == b"fakevideo"
    assert chunks[0].mime_type == "video/mp4"


def test_exact_match_returns_one_chunk():
    chunker = MediaChunker(chunk_duration=60, overlap=15, hard_limit=120, media_type="video")
    with (
        patch.object(chunker, "_get_duration", return_value=60.0),
        patch.object(chunker, "_extract_video_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"rawvideo", "video/mp4")

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 60


def test_just_over_boundary_returns_two_chunks_with_overlap():
    chunker = MediaChunker(chunk_duration=60, overlap=15, hard_limit=120, media_type="video")
    with (
        patch.object(chunker, "_get_duration", return_value=61.0),
        patch.object(chunker, "_extract_video_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"rawvideo", "video/mp4")

    assert len(chunks) == 2
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 60
    assert chunks[1].start_sec == 45
    assert chunks[1].end_sec == 61


def test_long_audio_returns_multiple_chunks():
    chunker = MediaChunker(chunk_duration=150, overlap=15, hard_limit=180, media_type="audio")
    with (
        patch.object(chunker, "_get_duration", return_value=400.0),
        patch.object(chunker, "_extract_audio_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"fakeaudio", "audio/mpeg")

    assert len(chunks) >= 3


def test_long_video_returns_multiple_chunks():
    chunker = MediaChunker(chunk_duration=60, overlap=15, hard_limit=120, media_type="video")
    with (
        patch.object(chunker, "_get_duration", return_value=300.0),
        patch.object(chunker, "_extract_video_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"rawvideo", "video/mp4")

    assert len(chunks) >= 3


def test_overlap_applies_between_audio_chunks():
    chunker = MediaChunker(chunk_duration=150, overlap=15, hard_limit=180, media_type="audio")
    with (
        patch.object(chunker, "_get_duration", return_value=400.0),
        patch.object(chunker, "_extract_audio_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"fakeaudio", "audio/mpeg")

    assert chunks[1].start_sec == chunks[0].end_sec - chunker._overlap


def test_overlap_applies_between_video_chunks():
    chunker = MediaChunker(chunk_duration=60, overlap=15, hard_limit=120, media_type="video")
    with (
        patch.object(chunker, "_get_duration", return_value=300.0),
        patch.object(chunker, "_extract_video_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"rawvideo", "video/mp4")

    assert chunks[1].start_sec == chunks[0].end_sec - chunker._overlap


def test_no_overlap_on_single_chunk_file():
    chunker = MediaChunker(chunk_duration=60, overlap=15, hard_limit=120, media_type="video")
    with (
        patch.object(chunker, "_get_duration", return_value=45.0),
        patch.object(chunker, "_extract_video_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"rawvideo", "video/mp4")

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 45


def test_audio_at_hard_limit():
    chunker = MediaChunker(chunk_duration=200, overlap=15, hard_limit=180, media_type="audio")
    with (
        patch.object(chunker, "_get_duration", return_value=180.0),
        patch.object(chunker, "_extract_audio_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"fakeaudio", "audio/mpeg")

    assert len(chunks) == 1
    assert chunks[0].end_sec == 180
    assert chunker._chunk_duration == 180


def test_video_chunk_clamped_to_hard_limit():
    chunker = MediaChunker(chunk_duration=200, overlap=15, hard_limit=120, media_type="video")
    assert chunker._chunk_duration == 120


def test_overlap_clamped_when_exceeds_chunk():
    chunker = MediaChunker(chunk_duration=10, overlap=20, hard_limit=180, media_type="audio")
    assert chunker._overlap == 9


def test_media_chunk_dataclass_fields():
    chunk = MediaChunk(data=b"d", mime_type="audio/mpeg", start_sec=0, end_sec=30)
    assert chunk.start_sec == 0
    assert chunk.end_sec == 30
    assert chunk.data == b"d"
    assert chunk.mime_type == "audio/mpeg"


def test_180s_audio_chunks_correctly():
    chunker = MediaChunker(chunk_duration=150, overlap=15, hard_limit=180, media_type="audio")
    with (
        patch.object(chunker, "_get_duration", return_value=180.0),
        patch.object(chunker, "_extract_audio_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"fakeaudio", "audio/mpeg")

    assert len(chunks) == 2
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 150
    assert chunks[1].start_sec == 135
    assert chunks[1].end_sec == 180


def test_120s_video_chunks_correctly():
    chunker = MediaChunker(chunk_duration=60, overlap=15, hard_limit=120, media_type="video")
    with (
        patch.object(chunker, "_get_duration", return_value=120.0),
        patch.object(chunker, "_extract_video_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"rawvideo", "video/mp4")

    assert len(chunks) == 3
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 60
    assert chunks[1].start_sec == 45
    assert chunks[1].end_sec == 105
    assert chunks[2].start_sec == 90
    assert chunks[2].end_sec == 120


def test_59s_video_returns_single_chunk():
    chunker = MediaChunker(chunk_duration=60, overlap=15, hard_limit=120, media_type="video")
    with (
        patch.object(chunker, "_get_duration", return_value=59.0),
        patch.object(chunker, "_extract_video_segment", return_value=b"seg"),
    ):
        chunks = chunker.process(b"rawvideo", "video/mp4")

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0
    assert chunks[0].end_sec == 59
