import io

from minio import Minio


class MinIOClient:
    """Client for MinIO object storage (stores raw uploaded files).

    Purpose: Keep original files accessible for reference/debugging.
    Stored files are referenced via blob_url in search results.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        public_url: str = "",
    ):
        "Initialize MinIO client and ensure bucket exists."
        self._client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )
        self._bucket = bucket
        self._public_url = public_url.rstrip("/")
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create MinIO bucket on first use."""
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def upload(self, object_name: str, data: bytes, content_type: str) -> str:
        "Upload file bytes to MinIO, return public URL for access."
        self._client.put_object(
            self._bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )
        return f"{self._public_url}/{self._bucket}/{object_name}"

    def delete(self, object_name: str) -> None:
        """Delete file from MinIO (called when document is deleted)."""
        self._client.remove_object(self._bucket, object_name)

    @staticmethod
    def object_name(document_id: str, filename: str) -> str:
        """Generate MinIO object name (preserves directory structure).

        Format: "{document_id}/{filename}"
        Used by: ingestor.py (upload), documents.py (delete).
        """
        return f"{document_id}/{filename}"
