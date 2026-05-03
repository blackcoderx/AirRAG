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
