import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Represents a segment of audio with timing metadata.

    Used by: ingestor.py (_ingest_audio) to pass to embedder and metadata storage.
    """

    data: bytes
    mime_type: str
    start_sec: int
    end_sec: int


class AudioProcessor:
    "Processes audio files: segments into chunks for native Gemini embedding."

    _CHUNK_DURATION = 170
    _OVERLAP = 15   # 15s overlap prevents context loss at speaker transitions
    _MAX_WORKERS = 6

    def _get_duration(self, input_path: str) -> float:
        """Get audio duration in seconds using ffprobe."""
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

    def _extract_segment(
        self, input_path: str, start: int, duration: int, fmt: str
    ) -> bytes:
        """Extract a time segment from audio using ffmpeg.

        -ss before -i enables fast seek (10-100x faster for large files).
        pipe:1 writes to stdout, avoiding a temp file round-trip per chunk.
        """
        result = subprocess.run(
            [
                "ffmpeg",
                "-ss", str(start),
                "-i", input_path,
                "-t", str(duration),
                "-c", "copy",
                "-f", fmt,
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout

    def process(self, audio_bytes: bytes, mime_type: str) -> list[AudioChunk]:
        "Segment audio into overlapping chunks; extraction runs in parallel."
        fmt = "mp3" if "mp3" in mime_type else "wav"
        suffix = f".{fmt}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name
        try:
            duration = self._get_duration(input_path)
            segments: list[tuple[int, int]] = []
            start = 0
            while start < int(duration):
                end = min(start + self._CHUNK_DURATION, int(duration))
                segments.append((start, end))
                if end >= int(duration):
                    break
                start = end - self._OVERLAP

            def _extract(seg: tuple[int, int]) -> AudioChunk:
                s, e = seg
                return AudioChunk(
                    data=self._extract_segment(input_path, s, e - s, fmt),
                    mime_type=mime_type,
                    start_sec=s,
                    end_sec=e,
                )

            with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as pool:
                return list(pool.map(_extract, segments))
        finally:
            os.unlink(input_path)
