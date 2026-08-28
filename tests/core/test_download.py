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

import http.client
import os
import sys
import threading
import time
import warnings
from email.message import Message
from email.utils import formatdate
from unittest.mock import call, patch
from urllib.error import HTTPError, URLError

import pytest
import torch

from kornia.core import download as download_mod
from kornia.core.download import hf_url, load_state_dict_from_url


@pytest.fixture(autouse=True)
def _clear_discard_ledger():
    """The one-discard-per-path bound is process-global; keep tests independent."""
    download_mod._DISCARDED_CACHE_PATHS.clear()
    yield
    download_mod._DISCARDED_CACHE_PATHS.clear()


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


def _http_error(url: str, code: int, headers: dict[str, str] | None = None) -> HTTPError:
    hdrs = None
    if headers is not None:
        hdrs = Message()
        for name, value in headers.items():
            hdrs[name] = value
    return HTTPError(url, code, "boom", hdrs=hdrs, fp=None)


class _FakeTime:
    """Stand-in for the ``time`` module that records sleeps instead of taking them."""

    NOW = 1_700_000_000.0

    def __init__(self) -> None:
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def time(self) -> float:
        return self.NOW


class TestPoisonedCacheEntry:
    """A bad cache entry must not disable the fallback URLs.

    All URLs in a list share one cache path, because ``file_name`` is pinned to
    the primary. ``_prefetch_to_cache`` skips a path that already exists, so
    before the discard step a single bad write was handed straight back to torch
    by every remaining source -- the fallback URL was named in the error without
    ever being fetched -- and the bad file outlived the process, so every later
    run failed the same way until the cache was cleared by hand.
    """

    _PRIMARY = "http://primary.example.com/model.pth"
    _FALLBACK = "http://fallback.example.com/model.pth"
    _GOOD = {"weight": torch.zeros(1)}
    _MOCK_TARGET = "kornia.core.download.torch.hub.load_state_dict_from_url"

    @staticmethod
    def _cache(monkeypatch, tmp_path, bad_urls):
        """Point the hub cache at *tmp_path*; URLs in *bad_urls* download garbage."""
        transfers: list[str] = []
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

        def fake_download(url, dst, hash_prefix=None, progress=True):
            transfers.append(url)
            if url in bad_urls:
                # A rate-limit page served with a 200, or a truncated transfer.
                with open(dst, "wb") as f:
                    f.write(b"<html>429 Too Many Requests</html>")
            else:
                torch.save(TestPoisonedCacheEntry._GOOD, dst)

        monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
        return transfers

    def test_fallback_recovers_from_bad_primary_download(self, monkeypatch, tmp_path) -> None:
        transfers = self._cache(monkeypatch, tmp_path, bad_urls={self._PRIMARY})

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        assert "weight" in result
        # The fallback was really fetched, not just named in an error message.
        assert transfers == [self._PRIMARY, self._FALLBACK]

    def test_fallback_recovers_from_preexisting_bad_cache_file(self, monkeypatch, tmp_path) -> None:
        transfers = self._cache(monkeypatch, tmp_path, bad_urls=set())
        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"<html>429 Too Many Requests</html>")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        assert "weight" in result
        # The primary short-circuits on the poisoned file, so only the fallback transfers.
        assert transfers == [self._FALLBACK]

    def test_bad_entry_never_survives_into_the_next_run(self, monkeypatch, tmp_path) -> None:
        """A run in which every source fails may leave its last write behind.

        The discard is bounded to one per cache path per process, so the file the
        final source wrote is still there afterwards. What must not survive is the
        *poisoning*: the next process spends its own discard on that path, so the
        fallback is reached again rather than being locked out for good.
        """
        bad = {self._PRIMARY, self._FALLBACK}
        transfers = self._cache(monkeypatch, tmp_path, bad_urls=bad)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError):
                load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        cached = tmp_path / "checkpoints" / "model.pth"
        assert cached.read_bytes().startswith(b"<html>")

        # A later run, with the fallback healthy again: the ledger is per-process.
        download_mod._DISCARDED_CACHE_PATHS.clear()
        bad.discard(self._FALLBACK)
        transfers.clear()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        assert "weight" in result
        assert transfers == [self._FALLBACK]

    def test_load_side_failure_refetches_once_not_once_per_call(self, monkeypatch, tmp_path) -> None:
        """An unloadable-but-intact cache entry must not re-download on every call.

        A failure the cache cannot fix -- a bad ``map_location``, a ``weights_only``
        rejection -- used to cost a full transfer of every source on each call,
        because the discard fired unconditionally and the entry never came back.
        Once per model construction across a matrix is the storm this module
        exists to prevent.
        """
        transfers = self._cache(monkeypatch, tmp_path, bad_urls=set())
        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._GOOD, cached)

        def always_fails(url: str, **kwargs: object) -> dict:
            raise RuntimeError("Attempting to deserialize object on a CUDA device")

        with patch(self._MOCK_TARGET, side_effect=always_fails):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                for _ in range(3):
                    with pytest.raises(RuntimeError):
                        load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        # One refetch in total, not one per source per call.
        assert transfers == [self._FALLBACK]

    def test_single_source_recovers_from_a_poisoned_entry_in_the_same_call(self, monkeypatch, tmp_path) -> None:
        """With no fallback to refetch the path, the failing URL must refetch it itself.

        Twenty-four of the library's checkpoints have a single source, so for
        them every URL is the last one. A discard with nothing after it deletes
        the entry and fetches nothing, leaving the call to fail and the recovery
        to the *next* process -- one guaranteed spurious failure per poisoned
        entry, on models where the fallback list cannot help.
        """
        transfers = self._cache(monkeypatch, tmp_path, bad_urls=set())
        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"<html>429 Too Many Requests</html>")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url(self._PRIMARY)

        assert "weight" in result
        assert transfers == [self._PRIMARY]

    def test_single_source_load_failure_leaves_the_cache_entry_in_place(self, monkeypatch, tmp_path) -> None:
        """A failure the cache cannot fix must not cost the caller its checkpoint.

        ``map_location='cuda'`` on a CPU-only build is the everyday trigger, and
        it is what ``ModelBase.load_checkpoint`` passes. An unpaired discard would
        delete an intact file -- up to 2.4 GB for ``sam.vit_h`` -- and refetch
        nothing, which on an offline machine turns a call that used to succeed
        into one that cannot.
        """
        transfers = self._cache(monkeypatch, tmp_path, bad_urls=set())
        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._GOOD, cached)

        def always_fails(url: str, **kwargs: object) -> dict:
            raise RuntimeError("Attempting to deserialize object on a CUDA device")

        with patch(self._MOCK_TARGET, side_effect=always_fails):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError):
                    load_state_dict_from_url(self._PRIMARY)

                # Within the failing call itself: the discard was paid for by a
                # refetch, so the caller ends it with the entry it started with.
                assert transfers == [self._PRIMARY]
                assert cached.exists()

                for _ in range(2):
                    with pytest.raises(RuntimeError):
                        load_state_dict_from_url(self._PRIMARY)

        # And the bound holds: one refetch per path per process, not one per call.
        assert transfers == [self._PRIMARY]
        assert cached.exists()

    def test_multi_source_load_failure_offline_leaves_the_cache_entry_in_place(self, monkeypatch, tmp_path) -> None:
        """A discard nothing could pay for is undone rather than left as a deletion.

        Pairing the discard with a refetch on the *last* URL alone left the
        primary's discard to be paid by the fallback, which offline writes
        nothing: ``DISK.from_pretrained('depth', device='cuda')`` on a CPU-only
        build destroyed an intact checkpoint, and the corrected ``device='cpu'``
        call -- which used to succeed offline -- then failed too.
        """
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        transfers: list[str] = []

        def offline(url, dst, hash_prefix=None, progress=True):
            transfers.append(url)
            raise URLError("network is unreachable")

        monkeypatch.setattr(torch.hub, "download_url_to_file", offline)

        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._GOOD, cached)
        before = cached.read_bytes()

        def always_fails(url: str, **kwargs: object) -> dict:
            raise RuntimeError("Attempting to deserialize object on a CUDA device")

        with patch(self._MOCK_TARGET, side_effect=always_fails):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError):
                    load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        # The fallback really was tried, and each of its transient failures retried.
        assert transfers == [self._FALLBACK] * download_mod._MAX_ATTEMPTS
        assert cached.read_bytes() == before  # and the caller kept its checkpoint
        assert not list(cached.parent.glob("*" + download_mod._QUARANTINE_SUFFIX))

    def test_a_discard_nothing_refetched_is_not_charged_to_the_bound(self, monkeypatch, tmp_path) -> None:
        """The bound counts refetches, so a call that fetched nothing must not spend it.

        Otherwise the first offline call in a process would use up the single
        allowed discard on a path it could not repair, and a later call -- with
        the network back -- could never clear a genuinely poisoned entry.
        """
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        offline = True
        transfers: list[str] = []

        def fake_download(url, dst, hash_prefix=None, progress=True):
            transfers.append(url)
            if offline:
                raise URLError("network is unreachable")
            torch.save(self._GOOD, dst)

        monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)

        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"<html>429 Too Many Requests</html>")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError):
                load_state_dict_from_url(self._PRIMARY)

            assert cached.read_bytes() == b"<html>429 Too Many Requests</html>"

            offline = False
            result = load_state_dict_from_url(self._PRIMARY)

        assert "weight" in result

    def test_unremovable_entry_is_reported(self, monkeypatch, tmp_path) -> None:
        """A cache the process cannot write to locks the fallback out silently."""
        self._cache(monkeypatch, tmp_path, bad_urls={self._PRIMARY})

        # ``download_mod.os`` *is* the os module, so delegate everything that is
        # not this cache entry rather than break renaming process-wide.
        cached = str(tmp_path / "checkpoints" / "model.pth")
        real_replace = os.replace

        def refuse(src: str, dst: str) -> None:
            if os.path.abspath(src) == cached:
                raise PermissionError(13, "Permission denied")
            real_replace(src, dst)

        monkeypatch.setattr(download_mod.os, "replace", refuse)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError):
                load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        assert any("Could not discard the cache entry" in str(w.message) for w in caught)


class TestTransientRetry:
    """Rate limits are the common CI failure; a retry is cheaper than a failed job."""

    _URL = "http://example.com/model.pth"

    @staticmethod
    def _cache(monkeypatch, tmp_path, side_effects):
        """Each call pops one entry off *side_effects*: an exception to raise, or None."""
        attempts: list[str] = []
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

        def fake_download(url, dst, hash_prefix=None, progress=True):
            attempts.append(url)
            outcome = side_effects.pop(0)
            if outcome is not None:
                raise outcome
            torch.save({"weight": torch.zeros(1)}, dst)

        monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
        return attempts

    @pytest.mark.parametrize("code", [408, 425, 429, 500, 502, 503, 504])
    def test_transient_http_status_is_retried(self, monkeypatch, tmp_path, code) -> None:
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, code), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url(self._URL)

        assert "weight" in result
        assert len(attempts) == 2
        assert clock.slept == [1.0]

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 410])
    def test_permanent_http_status_is_not_retried(self, monkeypatch, tmp_path, code) -> None:
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, code)])

        with pytest.raises(RuntimeError):
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 1
        assert clock.slept == []

    def test_connection_error_is_retried(self, monkeypatch, tmp_path) -> None:
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [URLError("dns"), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2

    def test_rate_limited_403_is_retried_and_honours_retry_after(self, monkeypatch, tmp_path) -> None:
        """GitHub answers a rate limit with 403 as well as 429, and says when to return.

        A bare 403 stays permanent -- it is also the "you may not have this file"
        answer -- so the rate-limit headers are what tells the two apart.
        """
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 403, {"Retry-After": "5"}), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url(self._URL)

        assert "weight" in result
        assert len(attempts) == 2
        assert clock.slept == [5.0]  # the server's number, not the 1s guess

    def test_rate_limit_reset_header_is_honoured(self, monkeypatch, tmp_path) -> None:
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(_FakeTime.NOW) + 7)}
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 403, headers), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2
        assert clock.slept == [7.0]

    def test_retry_after_accepts_an_http_date(self, monkeypatch, tmp_path) -> None:
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        when = formatdate(time.time() + 4, usegmt=True)  # the other form RFC 9110 allows
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 429, {"Retry-After": when}), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2
        assert clock.slept[0] == pytest.approx(4.0, abs=1.5)

    def test_server_requested_delay_is_clamped(self, monkeypatch, tmp_path) -> None:
        """A host may name minutes; holding a CI job open that long costs more than a refetch."""
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 429, {"Retry-After": "3600"}), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2
        assert clock.slept == [download_mod._MAX_BACKOFF_SECONDS]

    def test_unparseable_retry_after_still_marks_the_failure_transient(self, monkeypatch, tmp_path) -> None:
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 403, {"Retry-After": "soon"}), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2
        assert clock.slept == [1.0]  # falls back to the exponential guess

    def test_incomplete_read_is_retried(self, monkeypatch, tmp_path) -> None:
        """A truncated chunked response is neither a URLError nor a ConnectionError."""
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [http.client.IncompleteRead(b"partial"), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2

    def test_attempts_are_capped_with_exponential_backoff(self, monkeypatch, tmp_path) -> None:
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 429)] * 3)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError):
                load_state_dict_from_url(self._URL)

        assert len(attempts) == download_mod._MAX_ATTEMPTS == 3
        assert clock.slept == [1.0, 2.0]  # no sleep after the final attempt

    def test_every_url_gets_its_own_retries(self, monkeypatch, tmp_path) -> None:
        primary = "http://primary.example.com/model.pth"
        fallback = "http://fallback.example.com/model.pth"
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(
            monkeypatch, tmp_path, [_http_error(primary, 429)] * 3 + [_http_error(fallback, 429), None]
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url([primary, fallback])

        assert "weight" in result
        assert attempts == [primary] * 3 + [fallback] * 2

    @pytest.mark.parametrize("code", [408, 500, 503])
    def test_retry_after_is_honoured_on_any_transient_status(self, monkeypatch, tmp_path, code) -> None:
        """``Retry-After`` is defined for 503 and 408 too, not only for the rate limits."""
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, code, {"Retry-After": "10"}), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2
        assert clock.slept == [10.0]  # the server's number, not the 1s guess

    def test_rate_limit_reset_is_ignored_on_an_unrelated_failure(self, monkeypatch, tmp_path) -> None:
        """GitHub sends the window's end on every response, limited or not."""
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        headers = {"X-RateLimit-Remaining": "57", "X-RateLimit-Reset": str(int(_FakeTime.NOW) + 3000)}
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 500, headers), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2
        assert clock.slept == [1.0]

    @pytest.mark.parametrize("value", ["nan", "inf", "-5", "0"])
    def test_unusable_retry_after_falls_back_to_the_guess(self, monkeypatch, tmp_path, value) -> None:
        """``time.sleep(nan)`` raises, from inside the handler, so the 429 is never retried."""
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 429, {"Retry-After": value}), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url(self._URL)

        assert "weight" in result
        assert len(attempts) == 2
        assert clock.slept == [1.0]

    def test_a_call_stops_waiting_once_its_sleep_budget_is_gone(self, monkeypatch, tmp_path) -> None:
        """Clamping each wait does not bound the call: four waits at the clamp is four minutes."""
        primary = "http://primary.example.com/model.pth"
        fallback = "http://fallback.example.com/model.pth"
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        limited = _http_error(primary, 429, {"Retry-After": "3600"})
        attempts = self._cache(monkeypatch, tmp_path, [limited] * 6)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError):
                load_state_dict_from_url([primary, fallback])

        assert sum(clock.slept) <= download_mod._MAX_CALL_SLEEP_SECONDS
        # Two attempts on the primary -- the second is what the one wait bought --
        # then one on the fallback, which has nothing left to wait with.
        assert attempts == [primary] * 2 + [fallback]


class TestFailureMessageCarriesCause:
    """A CI failure summary shows only the exception's own message.

    Without the cause in that line, a rate limit, a DNS failure and a dead link
    are indistinguishable -- which is what made the OriNet CI failures read as
    broken URLs that were in fact serving fine.
    """

    def test_message_names_the_underlying_error(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(
            torch.hub,
            "download_url_to_file",
            lambda url, dst, *a, **k: (_ for _ in ()).throw(_http_error(url, 429)),
        )
        monkeypatch.setattr(download_mod, "time", _FakeTime())

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError) as excinfo:
                load_state_dict_from_url(["http://a.example.com/m.pth", "http://b.example.com/m.pth"])

        message = str(excinfo.value)
        assert "Failed to load weights from all 2 source" in message
        assert "HTTPError" in message
        assert "429" in message
        # And the cache path, so a caller stuck behind a corrupt entry knows what to delete.
        assert str(tmp_path) in message and "m.pth" in message
        # The original exception stays chained for a full traceback.
        assert isinstance(excinfo.value.__cause__, HTTPError)
