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
import sys
import threading
import time
from pathlib import Path
from unittest import mock
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

    def test_download_expands_user_directory(self, download_server, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        responses, base_url, _ = download_server
        responses["/model"] = (b"weights", 7)

        path = CachedDownloader.download_to_cache(f"{base_url}/model", "model", cache_dir="~/cache", suffix=".onnx")

        assert Path(path) == tmp_path / "cache" / "model.onnx"
        assert Path(path).read_bytes() == b"weights"

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

    def test_stalled_transfer_obeys_timeout_environment_variable(self, monkeypatch, tmp_path):
        for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("no_proxy", "127.0.0.1")
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", "0.2")
        stalled, release = threading.Event(), threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"x" * 10)
                self.wfile.flush()
                stalled.set()
                release.wait(5)

            def log_message(self, *args: object) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        path = tmp_path / "cache" / "model.onnx"
        url = f"http://127.0.0.1:{server.server_port}/model.onnx"
        errors: list[BaseException] = []

        def download() -> None:
            try:
                CachedDownloader.download(url, str(path))
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=download, daemon=True)
        try:
            worker.start()
            assert stalled.wait(5), "the server did not start the transfer"
            worker.join(5)
            assert not worker.is_alive(), "download remained blocked past KORNIA_DOWNLOAD_TIMEOUT"
        finally:
            release.set()
            worker.join(5)
            server.shutdown()
            server.server_close()
            server_thread.join(5)

        assert len(errors) == 1 and isinstance(errors[0], TimeoutError), errors
        assert not path.exists()
        assert not list(path.parent.glob("*.partial"))

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

    def test_interrupted_transfer_leaves_no_file(self, tmp_path):
        # KeyboardInterrupt is a BaseException: the cleanup must not be an `except Exception`
        path = tmp_path / "cache" / "model.pth"

        def interrupted(url, filename, **kwargs):
            raise KeyboardInterrupt

        with mock.patch("kornia.core.download._download_url_to_file", side_effect=interrupted):
            with pytest.raises(KeyboardInterrupt):
                CachedDownloader.download("http://127.0.0.1:9/model.pth", str(path))

        assert list(path.parent.iterdir()) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_cache_file_has_default_permissions(self, tmp_path):
        # a downloaded file gets the permissions of any new file (umask applied), not mkstemp's owner-only 0o600,
        # so a cache filled by one user stays readable by another
        path = tmp_path / "cache" / "model.pth"
        reference = tmp_path / "reference"
        reference.write_bytes(b"")

        def write(url, filename, **kwargs):
            Path(filename).write_bytes(b"weights")

        with mock.patch("kornia.core.download._download_url_to_file", side_effect=write):
            CachedDownloader.download("http://127.0.0.1:9/model.pth", str(path))

        assert path.stat().st_mode == reference.stat().st_mode

    def test_partial_transfer_is_not_visible_at_cache_path(self, monkeypatch, tmp_path):
        # a second caller checking the cache mid-transfer must not find (and return) a half-written file
        for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("no_proxy", "127.0.0.1")
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        payload = bytes(range(256)) * 400
        half_sent, release = threading.Event(), threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload[: len(payload) // 2])
                self.wfile.flush()
                half_sent.set()
                release.wait(10)
                self.wfile.write(payload[len(payload) // 2 :])

            def log_message(self, *args: object) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        path = tmp_path / "cache" / "model.pth"
        url = f"http://127.0.0.1:{server.server_port}/model.pth"
        worker = threading.Thread(target=CachedDownloader.download, args=(url, str(path)))
        try:
            worker.start()
            assert half_sent.wait(10)
            deadline = time.monotonic() + 10
            while not (path.parent.exists() and any(p != path for p in path.parent.iterdir())):
                assert time.monotonic() < deadline, "the transfer never started writing"
                time.sleep(0.01)
            assert not path.exists()
        finally:
            release.set()
            worker.join(10)
            server.shutdown()
            server.server_close()
        assert path.read_bytes() == payload
