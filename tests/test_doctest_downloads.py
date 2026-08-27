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

"""Tests for the doctest download guard (see #4005).

The guard itself is exercised for real by ``pixi run doctest``; these tests pin
the pieces that would otherwise only be checked by running that whole task.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from testing.doctest_downloads import (
    _DOWNLOAD_PRIMITIVES,
    DOWNLOAD_ENV_VAR,
    downloads_allowed,
    install_download_guard,
    skip_reason,
)


class _Blocked(Exception):
    """Raised by the test callback in place of a real download."""


def _record_and_raise(sink: list[str]):
    def _callback(url: str) -> None:
        sink.append(url)
        raise _Blocked(url)

    return _callback


class TestDownloadsAllowed:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " 1 "])
    def test_enabled(self, value: str) -> None:
        assert downloads_allowed({DOWNLOAD_ENV_VAR: value}) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_disabled(self, value: str) -> None:
        assert downloads_allowed({DOWNLOAD_ENV_VAR: value}) is False

    def test_unset_defaults_to_disabled(self) -> None:
        assert downloads_allowed({}) is False


class TestInstallDownloadGuard:
    def test_patches_are_undone_by_monkeypatch(self, monkeypatch) -> None:
        original = torch.hub.download_url_to_file
        with monkeypatch.context() as m:
            install_download_guard(m.setattr, lambda url: pytest.fail("unreachable"))
            assert torch.hub.download_url_to_file is not original
        assert torch.hub.download_url_to_file is original

    def test_callback_receives_url(self, monkeypatch) -> None:
        seen: list[str] = []

        def record(url: str) -> None:
            seen.append(url)
            raise RuntimeError("blocked")

        with monkeypatch.context() as m:
            install_download_guard(m.setattr, record)
            with pytest.raises(RuntimeError, match="blocked"):
                torch.hub.download_url_to_file("http://example.com/w.pth", "/tmp/w.pth")

        assert seen == ["http://example.com/w.pth"]

    def test_unknown_module_is_skipped_not_raised(self, monkeypatch) -> None:
        with monkeypatch.context() as m:
            patched = install_download_guard(
                m.setattr,
                lambda url: pytest.fail("unreachable"),
                [("kornia.this_module_does_not_exist", "download")],
            )
        assert patched == []

    def test_returning_callback_is_an_error(self, monkeypatch) -> None:
        # A callback that returns would let the download silently proceed as a no-op.
        with monkeypatch.context() as m:
            install_download_guard(m.setattr, lambda url: None)
            with pytest.raises(AssertionError, match="returned instead of raising"):
                torch.hub.download_url_to_file("http://example.com/w.pth", "/tmp/w.pth")


class TestPrimitivesStillExist:
    """The guard is only as good as its list of entry points.

    Each pair names a function that runs *only* on a cache miss. If an upstream
    rename makes one disappear, the guard silently stops covering that path and
    doctests quietly start downloading again, so pin them here.
    """

    @pytest.mark.parametrize(("module_name", "attribute"), _DOWNLOAD_PRIMITIVES)
    def test_primitive_is_importable(self, module_name: str, attribute: str) -> None:
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute))

    def test_guard_covers_lightglue_downloader(self, monkeypatch, tmp_path) -> None:
        # lightglue_onnx binds download_url_to_file at import time, so patching
        # torch.hub alone would leave this path downloading.
        from kornia.feature.lightglue_onnx.utils.download import download_onnx_from_url

        seen: list[str] = []
        with monkeypatch.context() as m:
            install_download_guard(m.setattr, _record_and_raise(seen))
            with pytest.raises(_Blocked):
                download_onnx_from_url("http://example.com/model.onnx", model_dir=str(tmp_path))

        assert seen == ["http://example.com/model.onnx"]

    def test_guard_covers_onnx_cached_downloader(self, monkeypatch, tmp_path) -> None:
        from kornia.onnx.download import CachedDownloader

        seen: list[str] = []
        with monkeypatch.context() as m:
            install_download_guard(m.setattr, _record_and_raise(seen))
            with pytest.raises(_Blocked):
                CachedDownloader.download("http://example.com/model.onnx", str(tmp_path / "sub" / "model.onnx"))

        assert seen == ["http://example.com/model.onnx"]

    def test_cached_file_is_not_blocked(self, monkeypatch, tmp_path) -> None:
        # The guard patches primitives that run only on a cache miss, which is what
        # lets an already-cached doctest keep running for real.
        cached = tmp_path / "model.onnx"
        cached.write_bytes(b"already here")

        from kornia.onnx.download import CachedDownloader

        with monkeypatch.context() as m:
            install_download_guard(m.setattr, lambda url: pytest.fail("a cached file must not trigger a download"))
            CachedDownloader.download("http://example.com/model.onnx", str(cached))

        assert cached.read_bytes() == b"already here"


class TestSkipReason:
    def test_names_the_url_and_the_opt_in(self) -> None:
        reason = skip_reason("http://example.com/w.pth")
        assert "http://example.com/w.pth" in reason
        assert DOWNLOAD_ENV_VAR in reason
