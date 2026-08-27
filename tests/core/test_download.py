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

import sys
import threading
import time
import warnings
from unittest.mock import call, patch

import pytest
import torch

from kornia.core.download import hf_url, load_state_dict_from_url


class TestHfUrl:
    def test_format(self) -> None:
        assert hf_url("hardnet", "HardNetPP.pth") == (
            "https://huggingface.co/kornia/hardnet/resolve/main/HardNetPP.pth"
        )

    def test_subdirectory(self) -> None:
        url = hf_url("loftr", "loftr_outdoor.ckpt")
        assert url.startswith("https://huggingface.co/kornia/loftr/resolve/main/")


class TestLoadStateDictFromUrl:
    _SD = {"weight": 1}
    _MOCK_TARGET = "kornia.core.download.torch.hub.load_state_dict_from_url"

    @pytest.fixture(autouse=True)
    def _isolated_cache(self, monkeypatch, tmp_path):
        """Keep the wrapper's prefetch step off the network and out of weights/."""
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(
            torch.hub,
            "download_url_to_file",
            lambda url, dst, *a, **k: torch.save({"weight": torch.zeros(1)}, dst),
        )

    def test_single_url_success(self) -> None:
        with patch(self._MOCK_TARGET, return_value=self._SD) as mock:
            result = load_state_dict_from_url("http://example.com/model.pth")
        assert result == self._SD
        mock.assert_called_once_with("http://example.com/model.pth")

    def test_list_single_url_success(self) -> None:
        # A single-element list behaves like a plain str — no file_name injection
        with patch(self._MOCK_TARGET, return_value=self._SD) as mock:
            result = load_state_dict_from_url(["http://example.com/model.pth"])
        assert result == self._SD
        mock.assert_called_once_with("http://example.com/model.pth")

    def test_fallback_on_failure(self) -> None:
        primary = "http://primary.example.com/model.pth"
        fallback = "http://fallback.example.com/model.pth"

        def side_effect(url: str, **kwargs: object) -> dict:
            if url == primary:
                raise OSError("primary down")
            return self._SD

        with patch(self._MOCK_TARGET, side_effect=side_effect) as mock:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = load_state_dict_from_url([primary, fallback])

        assert result == self._SD
        assert mock.call_count == 2
        assert any("primary down" in str(warning.message) for warning in w)

    def test_all_fail_raises_runtime_error(self) -> None:
        with patch(self._MOCK_TARGET, side_effect=OSError("down")):
            with pytest.raises(RuntimeError, match="Failed to load weights from all 2 source"):
                load_state_dict_from_url(["http://a.com/m.pth", "http://b.com/m.pth"])

    def test_file_name_pinned_to_primary(self) -> None:
        primary = "http://primary.example.com/weights-abc123.pth"
        fallback = "http://fallback.example.com/weights.pth"

        def side_effect(url: str, **kwargs: object) -> dict:
            if url == primary:
                raise OSError("primary down")
            return self._SD

        with patch(self._MOCK_TARGET, side_effect=side_effect) as mock:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                load_state_dict_from_url([primary, fallback])

        # fallback call must carry the primary's filename, not the fallback's
        fallback_call = mock.call_args_list[1]
        assert fallback_call == call(fallback, file_name="weights-abc123.pth")

    def test_explicit_file_name_not_overridden(self) -> None:
        primary = "http://primary.example.com/model.pth"
        fallback = "http://fallback.example.com/model.pth"

        def side_effect(url: str, **kwargs: object) -> dict:
            if url == primary:
                raise OSError("primary down")
            return self._SD

        with patch(self._MOCK_TARGET, side_effect=side_effect) as mock:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                load_state_dict_from_url([primary, fallback], file_name="custom.pth")

        for c in mock.call_args_list:
            assert c.kwargs.get("file_name") == "custom.pth"

    def test_kwargs_forwarded(self) -> None:
        with patch(self._MOCK_TARGET, return_value=self._SD) as mock:
            load_state_dict_from_url("http://example.com/model.pth", map_location="cpu")
        mock.assert_called_once_with("http://example.com/model.pth", map_location="cpu")


class TestProgressGoesToStderr:
    """torch.hub writes its 'Downloading: ...' line to stdout (torch 2.x).

    Status output on stdout corrupts callers that treat stdout as data. The most
    visible victim is ``pytest --doctest-modules kornia/``: the line is captured as
    unexpected example output and fails any example that downloads on a cold cache
    (#4005). The wrapper populates the cache itself so torch never reaches that
    line, rather than redirecting the process-global stdout around the call.
    """

    _URL = "http://example.com/model.pth"

    @staticmethod
    def _cold_cache(monkeypatch, tmp_path, *, on_download=None):
        """Point the hub cache at an empty tmp dir and stub the actual transfer.

        torch's own ``load_state_dict_from_url`` still runs for real, so a
        transfer count of exactly one also proves the path the wrapper
        prefetches to is the path torch looks in -- a disagreement would make
        torch download the file a second time.
        """
        transfers: list[str] = []
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

        def fake_download(url, dst, hash_prefix=None, progress=True):
            transfers.append(url)
            if on_download is not None:
                on_download()
            torch.save({"weight": torch.zeros(1)}, dst)

        monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
        return transfers

    def test_cold_cache_writes_nothing_to_stdout(self, capsys, monkeypatch, tmp_path) -> None:
        transfers = self._cold_cache(monkeypatch, tmp_path)

        result = load_state_dict_from_url(self._URL)

        captured = capsys.readouterr()
        assert "weight" in result
        assert captured.out == ""
        assert f'Downloading: "{self._URL}"' in captured.err
        # Exactly one transfer: torch found the prefetched file where it expected it.
        assert transfers == [self._URL]

    def test_warm_cache_is_silent(self, capsys, monkeypatch, tmp_path) -> None:
        transfers = self._cold_cache(monkeypatch, tmp_path)
        load_state_dict_from_url(self._URL)
        capsys.readouterr()

        load_state_dict_from_url(self._URL)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert len(transfers) == 1

    def test_stdout_object_is_never_replaced(self, monkeypatch, tmp_path) -> None:
        seen: list[object] = []
        self._cold_cache(monkeypatch, tmp_path, on_download=lambda: seen.append(sys.stdout))
        original = sys.stdout

        load_state_dict_from_url(self._URL)

        # Not merely restored afterwards -- untouched *during* the transfer.
        assert seen == [original]
        assert sys.stdout is original


class TestConcurrentLoadsDoNotDisturbStdout:
    """Regression tests for the review finding on #4039.

    An earlier revision wrapped the torch call in ``contextlib.redirect_stdout``.
    That mutates process-global ``sys.stdout`` for the whole transfer, so unrelated
    threads lost their output to stderr, and two overlapping calls restored out of
    order and left ``sys.stdout`` permanently pointing at ``sys.stderr``.
    """

    _URL = "http://example.com/model.pth"

    @staticmethod
    def _stub_transfer(monkeypatch, tmp_path, hook):
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

        def fake_download(url, dst, hash_prefix=None, progress=True):
            hook(url)
            torch.save({"weight": torch.zeros(1)}, dst)

        monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)

    def test_overlapping_calls_leave_stdout_intact(self, monkeypatch, tmp_path) -> None:
        barrier = threading.Barrier(2)

        def hook(url: str) -> None:
            barrier.wait(timeout=5)
            # Force the two calls to finish in the opposite order they started.
            time.sleep(0.05 if url.endswith("a.pth") else 0.15)

        self._stub_transfer(monkeypatch, tmp_path, hook)
        original = sys.stdout

        threads = [
            threading.Thread(target=load_state_dict_from_url, args=(f"http://example.com/{name}.pth",))
            for name in ("a", "b")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert sys.stdout is original
        assert sys.stdout is not sys.stderr

    def test_unrelated_thread_keeps_its_stdout(self, monkeypatch, tmp_path) -> None:
        in_flight = threading.Event()
        release = threading.Event()

        def hook(url: str) -> None:
            in_flight.set()
            release.wait(timeout=5)

        self._stub_transfer(monkeypatch, tmp_path, hook)

        written: list[str] = []

        class _Spy:
            def write(self, text: str) -> int:
                written.append(text)
                return len(text)

            def flush(self) -> None:
                pass

        monkeypatch.setattr(sys, "stdout", _Spy())

        loader = threading.Thread(target=load_state_dict_from_url, args=(self._URL,))
        loader.start()
        assert in_flight.wait(timeout=5), "download stub never ran"

        print("unrelated thread output")  # printed while the transfer is in flight

        release.set()
        loader.join(timeout=10)

        assert "unrelated thread output" in "".join(written)
