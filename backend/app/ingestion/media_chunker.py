import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class MediaChunk:
    """Represents a segment of audio or video with raw bytes and timing metadata.

    Used by: ingestor.py (_ingest_audio, _ingest_video) to pass to embedder.
    """

    data: bytes
    mime_type: str
    start_sec: int
    end_sec: int


class MediaChunker:
    """Segments audio or video files into overlapping chunks for Gemini embedding.

    Uses ffmpeg for time-based extraction with configurable chunk size,
    overlap, and hard limit to stay within Gemini's per-modality constraints.

    Edge cases handled:
    - Files smaller than chunk_size → single chunk, no overlap
    - Chunk size > hard_limit → clamped to hard_limit
    - Overlap >= chunk_size → clamped to chunk_size - 1 (prevents step ≤ 0)
    """

    def __init__(
        self,
        chunk_duration: int,
        overlap: int,
        hard_limit: int,
        media_type: str,
    ):
        """Initialize with chunking parameters.

        Args:
            chunk_duration: Target seconds per chunk.
            overlap: Seconds of overlap between adjacent chunks.
            hard_limit: Absolute maximum seconds (Gemini's modality limit).
            media_type: "audio" or "video" — determines ffmpeg pipeline.
        """
        self._chunk_duration = min(chunk_duration, hard_limit)
        self._overlap = min(overlap, self._chunk_duration - 1)
        self._hard_limit = hard_limit
        self._media_type = media_type
        self._max_workers = 6 if media_type == "audio" else 4

    def _compute_segments(self, duration: float) -> list[tuple[int, int]]:
        """Compute start/end timestamps for all segments."""
        segments: list[tuple[int, int]] = []
        start = 0
        while start < int(duration):
            end = min(start + self._chunk_duration, int(duration))
            segments.append((start, end))
            if end >= int(duration):
                break
            start = end - self._overlap
        return segments

    def _get_duration(self, input_path: str) -> float:
        """Get media duration in seconds using ffprobe."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                input_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])

    def _extract_audio_segment(
        self, input_path: str, start: int, end: int, fmt: str
    ) -> bytes:
        """Extract audio segment via ffmpeg pipe:1 (no temp file)."""
        result = subprocess.run(
            [
                "ffmpeg",
                "-ss", str(start),
                "-i", input_path,
                "-t", str(end - start),
                "-c", "copy",
                "-f", fmt,
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout

    def _extract_video_segment(
        self, input_path: str, start: int, end: int
    ) -> bytes:
        """Extract video segment via temp file to preserve container integrity."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out:
            out_path = out.name
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss", str(start),
                    "-i", input_path,
                    "-t", str(end - start),
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    out_path,
                ],
                capture_output=True,
                check=True,
            )
            with open(out_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(out_path)

    def process(self, media_bytes: bytes, mime_type: str) -> list[MediaChunk]:
        """Segment media into overlapping chunks; extraction runs in parallel."""
        if self._media_type == "audio":
            fmt = "mp3" if "mp3" in mime_type else "wav"
            suffix = f".{fmt}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(media_bytes)
                input_path = f.name
            try:
                duration = self._get_duration(input_path)
                segments = self._compute_segments(duration)

                def _extract(seg: tuple[int, int]) -> MediaChunk:
                    s, e = seg
                    return MediaChunk(
                        data=self._extract_audio_segment(input_path, s, e, fmt),
                        mime_type=mime_type,
                        start_sec=s,
                        end_sec=e,
                    )

                with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                    return list(pool.map(_extract, segments))
            finally:
                os.unlink(input_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(media_bytes)
                input_path = f.name
            try:
                duration = self._get_duration(input_path)
                segments = self._compute_segments(duration)

                def _extract(seg: tuple[int, int]) -> MediaChunk:
                    s, e = seg
                    return MediaChunk(
                        data=self._extract_video_segment(input_path, s, e),
                        mime_type=mime_type,
                        start_sec=s,
                        end_sec=e,
                    )

                with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                    return list(pool.map(_extract, segments))
            finally:
                os.unlink(input_path)
