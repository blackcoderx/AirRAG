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
