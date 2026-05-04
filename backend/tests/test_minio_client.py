from unittest.mock import MagicMock, patch

from app.storage.minio_client import MinIOClient


def _make_client(mock_minio_cls):
    mock_instance = MagicMock()
    mock_instance.bucket_exists.return_value = True
    mock_minio_cls.return_value = mock_instance
    return MinIOClient(
        endpoint="localhost:9000",
        access_key="key",
        secret_key="secret",
        bucket="test-bucket",
        public_url="http://localhost:9000",
    ), mock_instance


def test_upload_returns_url():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        url = client.upload("doc123/file.mp4", b"data", "video/mp4")
        assert url == "http://localhost:9000/test-bucket/doc123/file.mp4"
        mock_instance.put_object.assert_called_once()


def test_upload_calls_put_object_with_correct_args():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        client.upload("myobj", b"hello", "text/plain")
        call_kwargs = mock_instance.put_object.call_args
        assert call_kwargs[0][0] == "test-bucket"
        assert call_kwargs[0][1] == "myobj"
        assert call_kwargs[0][3] == 5


def test_delete_calls_remove_object():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        client.delete("doc123/file.mp4")
        mock_instance.remove_object.assert_called_once_with("test-bucket", "doc123/file.mp4")


def test_object_name_combines_document_id_and_filename():
    assert MinIOClient.object_name("abc123", "video.mp4") == "abc123/video.mp4"


def test_ensure_bucket_creates_if_missing():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        mock_instance = MagicMock()
        mock_instance.bucket_exists.return_value = False
        MockMinio.return_value = mock_instance
        MinIOClient("localhost:9000", "k", "s", "new-bucket")
        mock_instance.make_bucket.assert_called_once_with("new-bucket")


def test_download_returns_bytes():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        mock_response = MagicMock()
        mock_response.read.return_value = b"filecontent"
        mock_instance.get_object.return_value = mock_response
        data = client.download("doc123/audio.mp3")
        assert data == b"filecontent"
        mock_instance.get_object.assert_called_once_with("test-bucket", "doc123/audio.mp3")


def test_download_range_returns_partial_bytes():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        mock_response = MagicMock()
        mock_response.read.return_value = b"partial"
        mock_instance.get_object.return_value = mock_response
        data = client.download_range("doc123/audio.mp3", 0, 100)
        assert data == b"partial"
        mock_instance.get_object.assert_called_once_with(
            "test-bucket", "doc123/audio.mp3", offset=0, length=100
        )


def test_stat_returns_size_and_content_type():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        mock_stat = MagicMock()
        mock_stat.size = 1048576
        mock_stat.content_type = "audio/mpeg"
        mock_instance.stat_object.return_value = mock_stat
        size, content_type = client.stat("doc123/audio.mp3")
        assert size == 1048576
        assert content_type == "audio/mpeg"
        mock_instance.stat_object.assert_called_once_with("test-bucket", "doc123/audio.mp3")


def test_stat_fallback_content_type():
    with patch("app.storage.minio_client.Minio") as MockMinio:
        client, mock_instance = _make_client(MockMinio)
        mock_stat = MagicMock()
        mock_stat.size = 512
        mock_stat.content_type = None
        mock_instance.stat_object.return_value = mock_stat
        size, content_type = client.stat("doc123/unknown")
        assert content_type == "application/octet-stream"
