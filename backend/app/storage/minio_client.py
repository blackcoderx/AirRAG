import io

from minio import Minio


class MinIOClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        public_url: str = "",
    ):
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._public_url = public_url.rstrip("/")
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def upload(self, object_name: str, data: bytes, content_type: str) -> str:
        self._client.put_object(
            self._bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )
        return f"{self._public_url}/{self._bucket}/{object_name}"

    def delete(self, object_name: str) -> None:
        self._client.remove_object(self._bucket, object_name)

    @staticmethod
    def object_name(document_id: str, filename: str) -> str:
        return f"{document_id}/{filename}"
