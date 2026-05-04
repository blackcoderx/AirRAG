import json
import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class VideoChunk:
    """Represents a segment of video with raw bytes and timing metadata.

    Used by: ingestor.py (_ingest_video) to pass to embedder and vision_enricher.
    """

    data: bytes
    mime_type: str
    start_sec: int
    end_sec: int


class VideoProcessor:
    """Processes video files: segments into chunks for native Gemini embedding.

    Pipeline: Video bytes → segment by time → each chunk embedded + described.
    """

    # 60s chunks: one semantic unit, comfortably under 120s Gemini video embed limit.
    _CHUNK_DURATION = 60
    _OVERLAP = 10

    def _get_duration(self, input_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
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

    def _extract_video_segment(self, input_path: str, start: int, end: int) -> bytes:
        """Extract a time segment from video using ffmpeg.

        -ss before -i = fast seek. -avoid_negative_ts make_zero prevents audio
        sync drift when cutting at non-keyframe boundaries with stream copy.
        """
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

    def process(
        self, video_bytes: bytes, mime_type: str = "video/mp4"
    ) -> list[VideoChunk]:
        """Segment video into overlapping chunks and return VideoChunks.

        Vision description and embedding are applied later in ingestor.py.
        """
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
                chunks.append(
                    VideoChunk(
                        data=segment_bytes,
                        mime_type=mime_type,
                        start_sec=start,
                        end_sec=end,
                    )
                )
                if end >= int(duration):
                    break
                start = end - self._OVERLAP
            return chunks
        finally:
            os.unlink(input_path)
