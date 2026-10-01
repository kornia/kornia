# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import http.server
import os
import threading
from pathlib import Path
from urllib.error import ContentTooShortError, HTTPError

import pytest

from kornia.onnx.download import CachedDownloader

pytestmark = pytest.mark.device_agnostic


@pytest.fixture
def download_server(monkeypatch):
    """Serve bytes with a controllable Content-Length to exercise short transfers."""
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    responses: dict[str, tuple[bytes, int]] = {}
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            if self.path not in responses:
                self.send_error(404)
                return
            body, length = responses[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(length))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield responses, f"http://127.0.0.1:{server.server_port}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class TestCachedDownloader:
    @pytest.mark.parametrize("absolute", [False, True])
    def test_download_preserves_cache_directory(self, download_server, monkeypatch, tmp_path, absolute):
        monkeypatch.chdir(tmp_path)
        responses, base_url, hits = download_server
        responses["/model"] = (b"weights", 7)
        cache_dir = str(tmp_path / "cache") if absolute else "cache"

        path = CachedDownloader.download_to_cache(
            f"{base_url}/model", os.path.join("operators", "model"), cache_dir=cache_dir, suffix=".pth"
        )

        assert path == os.path.join(cache_dir, "operators", "model.pth")
        assert (tmp_path / "cache" / "operators" / "model.pth").read_bytes() == b"weights"
        assert hits == ["/model"]

    def test_short_transfer_is_not_cached(self, download_server, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        responses, base_url, hits = download_server
        payload = bytes(range(256)) * 40
        responses["/model"] = (payload[: len(payload) // 2], len(payload))
        url = f"{base_url}/model"

        with pytest.raises(ContentTooShortError):
            CachedDownloader.download_to_cache(url, "model", cache_dir="cache", suffix=".pth")

        assert list((tmp_path / "cache").iterdir()) == []
        responses["/model"] = (payload, len(payload))
        path = CachedDownloader.download_to_cache(url, "model", cache_dir="cache", suffix=".pth")
        assert Path(path).read_bytes() == payload
        assert CachedDownloader.download_to_cache(url, "model", cache_dir="cache", suffix=".pth") == path
        assert hits == ["/model", "/model"]
        assert list((tmp_path / "cache").iterdir()) == [tmp_path / "cache" / "model.pth"]

    def test_http_error_does_not_leave_cache_entry(self, download_server, tmp_path):
        _, base_url, _ = download_server
        path = tmp_path / "cache" / "missing.pth"
        with pytest.raises(ValueError, match="Error in resolving") as exc:
            CachedDownloader.download(f"{base_url}/missing", str(path))
        assert isinstance(exc.value.__cause__, HTTPError)
        assert list(path.parent.iterdir()) == []

    def test_download_disabled(self, download_server, tmp_path):
        _, base_url, hits = download_server
        with pytest.raises(ValueError, match="not found"):
            CachedDownloader.download(f"{base_url}/model", str(tmp_path / "missing.pth"), False)
        assert hits == []

    @pytest.mark.parametrize("download", [False, True])
    def test_existing_cache_entry_is_preserved(self, download_server, tmp_path, download):
        _, base_url, hits = download_server
        path = tmp_path / "model.pth"
        path.write_bytes(b"cached weights")
        CachedDownloader.download(f"{base_url}/missing", str(path), download)
        assert path.read_bytes() == b"cached weights"
        assert hits == []
