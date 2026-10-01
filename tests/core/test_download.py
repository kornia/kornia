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

import collections
import copy
import hashlib
import http.client
import http.server
import io
import ntpath
import os
import pickle
import socket
import sys
import threading
import time
import warnings
from email.message import Message
from email.utils import formatdate
from pathlib import Path
from unittest.mock import call, patch
from urllib.error import HTTPError, URLError

import pytest
import torch

from kornia.core import download as download_mod
from kornia.core.download import (
    _hf_cache_file_name,
    download_file_from_url,
    download_hf_file,
    hf_url,
    load_state_dict_from_url,
)

from testing.base import BaseTester
from testing.pickle_payload import CreatesMarkerOnLoad, load_without_running_payload


@pytest.fixture(autouse=True)
def _clear_discard_ledger():
    """The one-discard-per-path bound is process-global; keep tests independent."""
    download_mod._DISCARDED_CACHE_PATHS.clear()
    yield
    download_mod._DISCARDED_CACHE_PATHS.clear()


@pytest.fixture(autouse=True)
def _no_timeout_from_the_environment(monkeypatch):
    """A ``KORNIA_DOWNLOAD_TIMEOUT`` in the developer's shell must not change these tests."""
    monkeypatch.delenv("KORNIA_DOWNLOAD_TIMEOUT", raising=False)


class TestHfUrl:
    def test_format(self) -> None:
        assert hf_url("hardnet", "HardNetPP.pth") == (
            "https://huggingface.co/kornia/hardnet/resolve/main/HardNetPP.pth"
        )

    def test_subdirectory(self) -> None:
        url = hf_url("loftr", "loftr_outdoor.ckpt")
        assert url.startswith("https://huggingface.co/kornia/loftr/resolve/main/")

    def test_a_full_repo_id_keeps_its_owner(self) -> None:
        """A repo name cannot contain a ``/``, so one marks an ``owner/name`` id."""
        assert hf_url("google/siglip2-base-patch16-224", "model.safetensors") == (
            "https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/model.safetensors"
        )


class TestHfCacheFileName:
    """The cache is one flat directory, and every HF repo names its checkpoint alike."""

    def test_repo_id_is_folded_into_the_name(self) -> None:
        assert _hf_cache_file_name("kornia/kimi-vl-a3b-instruct-vision", "model.safetensors") == (
            "kornia--kimi-vl-a3b-instruct-vision--model.safetensors"
        )

    def test_both_spellings_of_a_kornia_repo_agree(self) -> None:
        """They resolve to one URL, so they must resolve to one cache entry."""
        assert _hf_cache_file_name("hardnet", "HardNetPP.pth") == _hf_cache_file_name("kornia/hardnet", "HardNetPP.pth")
        assert _hf_cache_file_name("hardnet", "HardNetPP.pth") == "kornia--hardnet--HardNetPP.pth"

    def test_two_repos_do_not_collide(self) -> None:
        first = _hf_cache_file_name("google/siglip2-base-patch16-224", "model.safetensors")
        second = _hf_cache_file_name("google/siglip2-base-patch16-256", "model.safetensors")
        assert first != second

    def test_the_result_is_one_path_component(self) -> None:
        name = _hf_cache_file_name("google/siglip2-base-patch16-224", "model.safetensors")
        assert os.sep not in name and "/" not in name


class TestLoadStateDictFromUrl:
    _SD = {"weight": 1}
    _MOCK_TARGET = "kornia.core.download.torch.hub.load_state_dict_from_url"

    @pytest.fixture(autouse=True)
    def _isolated_cache(self, monkeypatch, tmp_path):
        """Keep the wrapper's prefetch step off the network and out of weights/."""
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(
            download_mod,
            "_download_url_to_file",
            lambda url, dst, *a, **k: torch.save({"weight": torch.zeros(1)}, dst),
        )

    def test_single_url_success(self) -> None:
        with patch(self._MOCK_TARGET, return_value=self._SD) as mock:
            result = load_state_dict_from_url("http://example.com/model.pth")
        assert result == self._SD
        mock.assert_called_once_with("http://example.com/model.pth", weights_only=True)

    def test_list_single_url_success(self) -> None:
        # A single-element list behaves like a plain str — no file_name injection
        with patch(self._MOCK_TARGET, return_value=self._SD) as mock:
            result = load_state_dict_from_url(["http://example.com/model.pth"])
        assert result == self._SD
        mock.assert_called_once_with("http://example.com/model.pth", weights_only=True)

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
        assert fallback_call == call(fallback, file_name="weights-abc123.pth", weights_only=True)

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
        mock.assert_called_once_with("http://example.com/model.pth", map_location="cpu", weights_only=True)


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

        def fake_download(url, dst, hash_prefix=None, progress=True, timeout=None):
            transfers.append(url)
            if on_download is not None:
                on_download()
            torch.save({"weight": torch.zeros(1)}, dst)

        monkeypatch.setattr(download_mod, "_download_url_to_file", fake_download)
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

        def fake_download(url, dst, hash_prefix=None, progress=True, timeout=None):
            hook(url)
            torch.save({"weight": torch.zeros(1)}, dst)

        monkeypatch.setattr(download_mod, "_download_url_to_file", fake_download)

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

        def fake_download(url, dst, hash_prefix=None, progress=True, timeout=None):
            transfers.append(url)
            if url in bad_urls:
                # A rate-limit page served with a 200, or a truncated transfer.
                with open(dst, "wb") as f:
                    f.write(b"<html>429 Too Many Requests</html>")
            else:
                torch.save(TestPoisonedCacheEntry._GOOD, dst)

        monkeypatch.setattr(download_mod, "_download_url_to_file", fake_download)
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

        def offline(url, dst, hash_prefix=None, progress=True, timeout=None):
            transfers.append(url)
            raise URLError("network is unreachable")

        monkeypatch.setattr(download_mod, "_download_url_to_file", offline)

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

        # The fallback really was tried, and each of its transient failures retried;
        # then, since nothing had replaced the path, the discarded primary was given
        # the fetch the entry had denied it. Both are bounded by the sleep budget.
        assert transfers == [self._FALLBACK] * download_mod._MAX_ATTEMPTS + [self._PRIMARY] * download_mod._MAX_ATTEMPTS
        assert cached.read_bytes() == before  # and the caller kept its checkpoint
        assert not list(cached.parent.glob("*" + download_mod._QUARANTINE_SUFFIX))

    def test_a_failed_refetch_does_not_hide_the_load_failure(self, monkeypatch, tmp_path) -> None:
        """The refetch error is context; the failure that fired the discard is the cause.

        The re-attempt pass writes over ``last_exc``, so ``map_location='cuda'`` on
        a CPU-only build with the network down reported ``URLError`` and told the
        caller to delete an intact checkpoint -- up to 2.4 GB for ``sam.vit_h`` --
        while the one exception naming what actually went wrong was gone from both
        the message and ``__cause__``. Before the re-attempt existed this call
        raised the load error directly.
        """
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(download_mod, "time", _FakeTime())

        def offline(url, dst, hash_prefix=None, progress=True, timeout=None):
            raise URLError("network is unreachable")

        monkeypatch.setattr(download_mod, "_download_url_to_file", offline)

        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._GOOD, cached)

        def always_fails(url: str, **kwargs: object) -> dict:
            raise RuntimeError("Attempting to deserialize object on a CUDA device")

        with patch(self._MOCK_TARGET, side_effect=always_fails):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError) as excinfo:
                    load_state_dict_from_url(self._PRIMARY)

        message = str(excinfo.value)
        assert "Last error: RuntimeError: Attempting to deserialize object on a CUDA device" in message
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "CUDA device" in str(excinfo.value.__cause__)
        # The refetch failure is still reported, as context rather than as the cause.
        assert "URLError" in message
        assert cached.exists()  # and the file the message points at is the intact one

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

        def fake_download(url, dst, hash_prefix=None, progress=True, timeout=None):
            transfers.append(url)
            if offline:
                raise URLError("network is unreachable")
            torch.save(self._GOOD, dst)

        monkeypatch.setattr(download_mod, "_download_url_to_file", fake_download)

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

    def test_a_bad_fallback_download_does_not_destroy_the_healthy_entry(self, monkeypatch, tmp_path) -> None:
        """Settling on *path existence* kept whatever the network last wrote.

        The network being up but rate limited is this PR's own common case: the
        fallback answers with an HTML page and a 200, which lands at the shared
        cache path. Deciding by "is there a file there?" then dropped the intact
        quarantined checkpoint in favour of that page -- so a ``map_location``
        mistake, which every ``ModelBase.load_checkpoint`` caller can make, cost
        the caller a checkpoint of up to 2.4 GB *and* poisoned the entry, and the
        corrected call failed too. The call's outcome decides instead: nothing
        loaded, so nothing on disk is trusted and the original goes back.
        """
        self._cache(monkeypatch, tmp_path, bad_urls={self._FALLBACK})
        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._GOOD, cached)
        before = cached.read_bytes()

        def cuda_on_a_cpu_build(url: str, **kwargs: object) -> dict:
            raise RuntimeError("Attempting to deserialize object on a CUDA device")

        with patch(self._MOCK_TARGET, side_effect=cuda_on_a_cpu_build):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError):
                    load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        assert cached.read_bytes() == before
        assert not list(cached.parent.glob("*" + download_mod._QUARANTINE_SUFFIX))

        # And the corrected call -- the whole point of not destroying the file --
        # succeeds without touching the network.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert "weight" in load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

    def test_poisoned_entry_recovers_when_the_mirror_is_dead(self, monkeypatch, tmp_path) -> None:
        """The primary is refetched even though it is not the last URL.

        A poisoned entry short-circuits the primary's prefetch, so the primary is
        never actually fetched; the discard it triggers is meant to be paid for by
        a later source, and a dead mirror -- ``cmp.felk.cvut.cz`` and the
        ``raw.githubusercontent.com`` copies are exactly the sources that go away
        -- pays nothing. Pairing the refetch with the *last* URL alone left the
        entry poisoned in every call of every process.
        """
        transfers = self._cache(monkeypatch, tmp_path, bad_urls=set())
        cached = tmp_path / "checkpoints" / "model.pth"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"<html>429 Too Many Requests</html>")

        def fake_download(url, dst, hash_prefix=None, progress=True, timeout=None):
            transfers.append(url)
            if url == self._FALLBACK:
                raise HTTPError(url, 404, "Not Found", None, None)
            torch.save(self._GOOD, dst)

        monkeypatch.setattr(download_mod, "_download_url_to_file", fake_download)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        assert "weight" in result
        # The dead mirror was tried once (404 is permanent), then the primary was
        # fetched for real against the emptied path.
        assert transfers == [self._FALLBACK, self._PRIMARY]
        assert not list(cached.parent.glob("*" + download_mod._QUARANTINE_SUFFIX))

    def test_a_sources_own_bad_download_is_not_left_behind(self, monkeypatch, tmp_path) -> None:
        """A call must not leave a poisoned entry where it found none.

        Cold cache, the primary rate limited into serving an HTML page with a
        200, the mirror offline. Quarantining the page -- bytes this very source
        had just written -- meant :func:`_settle_quarantine` put it back, so the
        call ended with an entry it had created and the one allowed discard
        already spent on it: if the primary recovered while the mirror stayed
        dead, every later call in the process short-circuited on the page and
        failed. There was nothing on disk to protect, so there is nothing to move
        aside.
        """
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        healthy = False

        def fake_download(url, dst, hash_prefix=None, progress=True, timeout=None):
            if url == self._FALLBACK:
                raise URLError("network is unreachable")
            if healthy:
                torch.save(self._GOOD, dst)
            else:
                with open(dst, "wb") as f:
                    f.write(b"<html>429 Too Many Requests</html>")

        monkeypatch.setattr(download_mod, "_download_url_to_file", fake_download)
        cached = tmp_path / "checkpoints" / "model.pth"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError):
                load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

        assert not cached.exists()
        assert not list(cached.parent.glob("*" + download_mod._QUARANTINE_SUFFIX))
        # And the bound is unspent, since nothing was refetched to pay for it.
        assert not download_mod._DISCARDED_CACHE_PATHS

        # So the primary recovering is enough, with the mirror still dead.
        healthy = True
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert "weight" in load_state_dict_from_url([self._PRIMARY, self._FALLBACK])

    def test_a_cold_cache_load_side_failure_stays_bounded(self, monkeypatch, tmp_path) -> None:
        """Dropping every failed download unbounds the transfers it was meant to bound.

        On a cold cache a caller-side failure -- ``map_location='cuda'`` on a
        CPU-only build -- rejects a *healthy* file the source just fetched, and no
        exception type separates that from a corrupt one. Deleting whatever a
        source wrote would make every later call fetch it again to fail
        identically: 3 transfers over the three calls below rather than 2, and 6
        rather than 3 for a two-source list, once per test that builds the model,
        on checkpoints of up to 2.4 GB. After the last source the bytes stay, so
        the count is bounded per process however often the call is repeated.
        """
        transfers = self._cache(monkeypatch, tmp_path, bad_urls=set())
        cached = tmp_path / "checkpoints" / "model.pth"

        def always_fails(url: str, **kwargs: object) -> dict:
            raise RuntimeError("Attempting to deserialize object on a CUDA device")

        with patch(self._MOCK_TARGET, side_effect=always_fails):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                for _ in range(3):
                    with pytest.raises(RuntimeError):
                        load_state_dict_from_url(self._PRIMARY)

        # The first call's transfer, then the one discard this path is allowed.
        assert transfers == [self._PRIMARY, self._PRIMARY]
        assert cached.exists()

    def test_a_single_sources_own_bad_download_is_still_discardable(self, monkeypatch, tmp_path) -> None:
        """Twenty-four checkpoints have no mirror, so the recovery has to work there too.

        Bytes the last source wrote are kept, but they must not cost the path its
        discard: quarantining them spent the bound on an entry the call had just
        created, and the process could then never clear it. Nothing is spent, so
        the source recovering is enough.
        """
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        healthy = False
        transfers: list[str] = []

        def fake_download(url, dst, hash_prefix=None, progress=True, timeout=None):
            transfers.append(url)
            if healthy:
                torch.save(self._GOOD, dst)
            else:
                with open(dst, "wb") as f:
                    f.write(b"<html>429 Too Many Requests</html>")

        monkeypatch.setattr(download_mod, "_download_url_to_file", fake_download)
        cached = tmp_path / "checkpoints" / "model.pth"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError):
                load_state_dict_from_url(self._PRIMARY)

            assert cached.read_bytes().startswith(b"<html>")
            assert not download_mod._DISCARDED_CACHE_PATHS

            # The rate limit lifts; the same process must be able to recover.
            healthy = True
            assert "weight" in load_state_dict_from_url(self._PRIMARY)

        # The poisoned write, then the one refetch this path is allowed.
        assert transfers == [self._PRIMARY, self._PRIMARY]

    def test_explicit_file_name_none_still_shares_one_cache_path(self, monkeypatch, tmp_path) -> None:
        """``file_name=None`` must mean the same thing as omitting it.

        Pinning on ``"file_name" not in kwargs`` while resolving the path with
        ``kwargs.get("file_name")`` split the two: each URL got its own cache
        path, but the single quarantine still covered only the primary's, so a
        failed call left the primary's slot holding the *fallback's* bytes and
        stranded a quarantine file next to it.
        """
        transfers = self._cache(monkeypatch, tmp_path, bad_urls={self._PRIMARY})
        other = "http://fallback.example.com/other.pth"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_state_dict_from_url([self._PRIMARY, other], file_name=None)

        assert "weight" in result
        assert transfers == [self._PRIMARY, other]
        checkpoints = tmp_path / "checkpoints"
        assert {p.name for p in checkpoints.iterdir()} == {"model.pth"}

    def test_unremovable_entry_is_reported(self, monkeypatch, tmp_path) -> None:
        """A cache the process cannot write to locks the fallback out silently."""
        self._cache(monkeypatch, tmp_path, bad_urls=set())
        # A *pre-existing* poisoned entry: only those are moved aside, and only a
        # rename that fails leaves the fallback unreachable.
        cached_path = tmp_path / "checkpoints" / "model.pth"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(b"<html>429 Too Many Requests</html>")

        # ``download_mod.os`` *is* the os module, so delegate everything that is
        # not this cache entry rather than break renaming process-wide.
        cached = str(cached_path)
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

        def fake_download(url, dst, hash_prefix=None, progress=True, timeout=None):
            attempts.append(url)
            outcome = side_effects.pop(0)
            if outcome is not None:
                raise outcome
            torch.save({"weight": torch.zeros(1)}, dst)

        monkeypatch.setattr(download_mod, "_download_url_to_file", fake_download)
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

    def test_an_unusable_retry_after_falls_through_to_the_reset_header(self, monkeypatch, tmp_path) -> None:
        """An unparsable ``Retry-After`` is no guidance, not an answer for the function.

        Returning its ``None`` outright skipped the ``X-RateLimit-Reset`` branch
        below it, so a rate limit that sent both headers was retried on the 1s/2s
        guess -- inside the window it had just been told to wait out.
        """
        clock = _FakeTime()
        monkeypatch.setattr(download_mod, "time", clock)
        headers = {
            "Retry-After": "garbage",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(_FakeTime.NOW) + 40),
        }
        attempts = self._cache(monkeypatch, tmp_path, [_http_error(self._URL, 403, headers), None])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_state_dict_from_url(self._URL)

        assert len(attempts) == 2
        assert clock.slept == [40.0]  # the window the host named, not the guess

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
            download_mod,
            "_download_url_to_file",
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
        # And the cache path, verbatim, so a caller stuck behind a corrupt entry can
        # paste it into ``rm``/``del`` -- which a repr-escaped Windows path is not.
        assert os.path.join(str(tmp_path), "checkpoints", "m.pth") in message
        # The original exception stays chained for a full traceback.
        assert isinstance(excinfo.value.__cause__, HTTPError)


@pytest.fixture
def local_server(monkeypatch):
    """Serve in-memory files over HTTP from a thread bound to 127.0.0.1.

    The transfer runs through ``kornia.core.download._download_url_to_file`` for real -- nothing is
    stubbed between the socket and ``torch.load`` -- and nothing leaves the host. Yields
    ``(files, url, hits)``: ``files`` maps a path such as ``"/model.pth"`` to the bytes
    served for it (any other path answers 404), ``url(path)`` builds the URL, and ``hits``
    counts the requests per path.
    """
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")

    files: dict[str, bytes] = {}
    hits: collections.Counter[str] = collections.Counter()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits[self.path] += 1
            body = files.get(self.path)
            self.send_response(404 if body is None else 200)
            payload = b"not found" if body is None else body
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield files, (lambda path: f"http://127.0.0.1:{port}{path}"), hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _checkpoint_bytes(obj: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


class TestWeightsOnly(BaseTester):
    """A checkpoint is loaded as data: tensors and plain containers, never pickled callables.

    ``torch.hub.load_state_dict_from_url`` defaults to ``weights_only=False`` on every
    supported torch, and ``torch.load`` itself did before torch 2.6, so a wrapper that
    only forwards its keyword arguments unpickles whatever a checkpoint names. The
    wrapper defaults to ``weights_only=True`` instead, and a caller that trusts a file
    that needs more opts out with ``weights_only=False`` explicitly.

    The payload is an ordinary pickled callable in the current ``torch.save`` format. These
    tests pin that the default reaches torch and that torch's restricted unpickler refuses
    such a payload; they do not exercise the bypasses PyTorch has published for torch before
    2.10, which the wrapper's docstring describes.
    """

    @pytest.mark.parametrize(
        "sources",
        [["/model.pth"], ["/missing.pth", "/model.pth"]],
        ids=["single_source", "fallback_after_a_dead_primary"],
    )
    def test_a_pickled_callable_is_refused_and_never_run(self, local_server, tmp_path, sources) -> None:
        files, url, _ = local_server
        marker = tmp_path / "marker"
        files["/model.pth"] = _checkpoint_bytes({"weight": torch.zeros(2), "extra": CreatesMarkerOnLoad(marker)})

        urls = [url(s) for s in sources]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the dead primary's "Trying next source"
            with pytest.raises(RuntimeError) as excinfo:
                load_without_running_payload(
                    marker, lambda: load_state_dict_from_url(urls, model_dir=str(tmp_path / "cache"))
                )

        assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)

    def test_a_poisoned_cache_entry_is_refused_and_refetched(self, local_server, tmp_path, dtype) -> None:
        """A cached file is loaded under the same rule as a downloaded one, and a refused entry is refetched."""
        files, url, hits = local_server
        marker = tmp_path / "marker"
        good = {"weight": torch.arange(6, dtype=dtype).reshape(2, 3)}
        files["/model.pth"] = _checkpoint_bytes(good)
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "model.pth").write_bytes(_checkpoint_bytes({"extra": CreatesMarkerOnLoad(marker)}))

        result = load_without_running_payload(
            marker, lambda: load_state_dict_from_url(url("/model.pth"), model_dir=str(cache))
        )

        self.assert_close(result["weight"], good["weight"])
        assert hits["/model.pth"] == 1

    def test_a_plain_state_dict_still_loads(self, local_server, tmp_path, dtype) -> None:
        """Control: what a checkpoint normally holds is allowed, container types included."""
        files, url, _ = local_server
        state_dict = collections.OrderedDict(
            [("weight", torch.arange(6, dtype=dtype).reshape(2, 3)), ("index", torch.tensor([3, 1, 2]))]
        )
        files["/model.pth"] = _checkpoint_bytes(
            {"state_dict": state_dict, "epoch": 7, "lr": 0.5, "arch": "net", "shape": (2, 3), "flags": [True, None]}
        )

        result = load_state_dict_from_url(url("/model.pth"), model_dir=str(tmp_path / "cache"), map_location="cpu")

        assert isinstance(result["state_dict"], collections.OrderedDict)
        assert list(result["state_dict"]) == ["weight", "index"]
        self.assert_close(result["state_dict"]["weight"], state_dict["weight"])
        assert result["state_dict"]["weight"].dtype == dtype
        assert torch.equal(result["state_dict"]["index"], state_dict["index"])
        assert (result["epoch"], result["lr"], result["arch"], result["shape"], result["flags"]) == (
            7,
            0.5,
            "net",
            (2, 3),
            [True, None],
        )

    @pytest.mark.parametrize(
        ("kwargs", "forwarded"),
        [({}, True), ({"weights_only": None}, True), ({"weights_only": True}, True), ({"weights_only": False}, False)],
        ids=["omitted", "none", "true", "false"],
    )
    def test_weights_only_is_true_unless_the_caller_passes_false(self, kwargs, forwarded) -> None:
        # ``None`` counts as omitted: torch 2.5 reads it as ``False``.
        with patch("kornia.core.download._prefetch_to_cache", return_value=False):
            with patch("kornia.core.download.torch.hub.load_state_dict_from_url", return_value={}) as mock:
                load_state_dict_from_url("http://example.com/model.pth", **kwargs)
        mock.assert_called_once_with("http://example.com/model.pth", weights_only=forwarded)


class TestDownloadFileFromUrl:
    """The path for checkpoints torch cannot unpickle -- a ``.safetensors`` file.

    The transfers here are real: a ``file://`` URL goes through the same
    ``_download_url_to_file`` a remote one does, so the cache path, the
    bytes on disk and the cache hit are all exercised end to end without a
    network or a stub standing in for the transfer.
    """

    @staticmethod
    def _serve(tmp_path, name: str = "model.safetensors", payload: bytes = b"weights") -> tuple[str, bytes]:
        """Write a file and return the ``file://`` URL that serves it."""
        source = tmp_path / "remote" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        return source.as_uri(), payload

    def test_downloads_into_model_dir(self, tmp_path) -> None:
        url, payload = self._serve(tmp_path)
        model_dir = tmp_path / "cache"

        path = download_file_from_url(url, model_dir=str(model_dir), progress=False)

        assert Path(path) == model_dir / "model.safetensors"
        assert Path(path).read_bytes() == payload

    def test_second_call_is_a_cache_hit(self, monkeypatch, tmp_path) -> None:
        url, _ = self._serve(tmp_path)
        model_dir = tmp_path / "cache"
        transfers: list[str] = []
        real = download_mod._download_url_to_file

        def counted(url_, dst, *args, **kwargs):
            transfers.append(url_)
            return real(url_, dst, *args, **kwargs)

        monkeypatch.setattr(download_mod, "_download_url_to_file", counted)

        first = download_file_from_url(url, model_dir=str(model_dir), progress=False)
        second = download_file_from_url(url, model_dir=str(model_dir), progress=False)

        assert second == first
        assert transfers == [url], "the cached file was fetched again"

    def test_defaults_to_the_torch_hub_cache(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
        url, _ = self._serve(tmp_path)

        path = download_file_from_url(url, progress=False)

        assert Path(path) == tmp_path / "hub" / "checkpoints" / "model.safetensors"

    def test_file_name_overrides_the_basename(self, tmp_path) -> None:
        """Two repositories publish a ``model.safetensors`` each; one cache slot is not enough."""
        first_url, first_payload = self._serve(tmp_path / "a", payload=b"first")
        second_url, second_payload = self._serve(tmp_path / "b", payload=b"second")
        model_dir = tmp_path / "cache"

        first = download_file_from_url(first_url, file_name="a--model.safetensors", model_dir=str(model_dir))
        second = download_file_from_url(second_url, file_name="b--model.safetensors", model_dir=str(model_dir))

        assert first != second
        assert Path(first).read_bytes() == first_payload
        assert Path(second).read_bytes() == second_payload

    def test_fallback_source_is_tried(self, monkeypatch, tmp_path) -> None:
        url, payload = self._serve(tmp_path)
        model_dir = tmp_path / "cache"
        dead = (tmp_path / "remote" / "missing.safetensors").as_uri()
        # A missing file:// URL raises URLError, which is transient, so the dead
        # source is retried before the fallback is reached; the waits are faked.
        monkeypatch.setattr(download_mod, "time", _FakeTime())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = download_file_from_url([dead, url], model_dir=str(model_dir), progress=False)

        # The cache name is pinned to the *first* URL, as in load_state_dict_from_url.
        assert Path(path) == model_dir / "missing.safetensors"
        assert Path(path).read_bytes() == payload
        assert any("Trying next source" in str(warning.message) for warning in caught)

    def test_all_sources_failing_names_the_cause_and_the_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(
            download_mod,
            "_download_url_to_file",
            lambda url, dst, *a, **k: (_ for _ in ()).throw(_http_error(url, 404)),
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError) as excinfo:
                download_file_from_url(["http://a.example.com/m.safetensors", "http://b.example.com/m.safetensors"])

        message = str(excinfo.value)
        assert "Failed to download the file from all 2 source" in message
        assert "HTTPError" in message and "404" in message
        assert os.path.join(str(tmp_path), "checkpoints", "m.safetensors") in message
        assert isinstance(excinfo.value.__cause__, HTTPError)

    def test_a_transient_failure_is_retried(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        attempts: list[str] = []

        def flaky(url, dst, *args, **kwargs):
            attempts.append(url)
            if len(attempts) < 3:
                raise _http_error(url, 429)
            Path(dst).write_bytes(b"weights")

        monkeypatch.setattr(download_mod, "_download_url_to_file", flaky)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            path = download_file_from_url("http://example.com/m.safetensors", progress=False)

        assert len(attempts) == 3
        assert Path(path).read_bytes() == b"weights"

    def test_a_partial_transfer_does_not_block_the_next_source(self, monkeypatch, tmp_path) -> None:
        """A source that writes and then fails must not be handed to the next one as a cache hit."""
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

        def half_written(url, dst, *args, **kwargs):
            if "primary" in url:
                Path(dst).write_bytes(b"truncated")
                raise OSError("connection reset")
            Path(dst).write_bytes(b"complete")

        monkeypatch.setattr(download_mod, "_download_url_to_file", half_written)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            path = download_file_from_url(
                ["http://primary.example.com/m.safetensors", "http://mirror.example.com/m.safetensors"],
                progress=False,
            )

        # ``b"truncated"`` here would mean the mirror was skipped: the two share
        # one cache path, so the primary's leftovers would have read as a hit.
        assert Path(path).read_bytes() == b"complete"

    def test_transfer_is_announced_on_stderr_only(self, capsys, tmp_path) -> None:
        url, _ = self._serve(tmp_path)

        download_file_from_url(url, model_dir=str(tmp_path / "cache"), progress=False)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert f'Downloading: "{url}"' in captured.err

    @pytest.mark.parametrize("file_name", ["../escaped.safetensors", "sub/model.safetensors", "", ".", ".."])
    def test_file_name_must_be_a_bare_filename(self, tmp_path, file_name) -> None:
        """The cache is one flat directory; a name with a path in it would write outside it."""
        url, _ = self._serve(tmp_path)
        model_dir = tmp_path / "cache"

        with pytest.raises(ValueError, match="bare filename"):
            download_file_from_url(url, file_name=file_name, model_dir=str(model_dir), progress=False)

        assert not (tmp_path / "escaped.safetensors").exists()
        assert not model_dir.exists(), "nothing was transferred"

    def test_an_absolute_file_name_is_rejected(self, tmp_path) -> None:
        url, _ = self._serve(tmp_path)

        with pytest.raises(ValueError, match="bare filename"):
            download_file_from_url(url, file_name=str(tmp_path / "abs.safetensors"), model_dir=str(tmp_path / "cache"))

        assert not (tmp_path / "abs.safetensors").exists()


class TestDownloadHfFile:
    """The Hub wrapper: the ``resolve/main`` URL and the collision-free cache name in one call."""

    @staticmethod
    def _capture(monkeypatch) -> list[tuple]:
        """Record what reaches ``download_file_from_url`` instead of downloading."""
        calls: list[tuple] = []

        def fake(url, **kwargs):
            calls.append((url, kwargs))
            return "cached"

        monkeypatch.setattr(download_mod, "download_file_from_url", fake)
        return calls

    def test_full_repo_id(self, monkeypatch) -> None:
        calls = self._capture(monkeypatch)

        assert download_hf_file("google/siglip2-base-patch16-224", "model.safetensors") == "cached"

        url, kwargs = calls[0]
        assert url == "https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/model.safetensors"
        assert kwargs["file_name"] == "google--siglip2-base-patch16-224--model.safetensors"
        assert kwargs["model_dir"] is None

    def test_bare_repo_name_resolves_under_the_kornia_org(self, monkeypatch, tmp_path) -> None:
        calls = self._capture(monkeypatch)

        download_hf_file("kimi-vl-a3b-instruct-vision", "model.safetensors", model_dir=str(tmp_path), progress=False)

        url, kwargs = calls[0]
        assert url == "https://huggingface.co/kornia/kimi-vl-a3b-instruct-vision/resolve/main/model.safetensors"
        # The org is in the cache name too, so the bare and full spellings of the
        # same repo cannot end up in two cache entries.
        assert kwargs["file_name"] == "kornia--kimi-vl-a3b-instruct-vision--model.safetensors"
        assert kwargs["model_dir"] == str(tmp_path)
        assert kwargs["progress"] is False

    def test_two_repos_publishing_the_same_filename_do_not_collide(self, tmp_path) -> None:
        """End to end, through a real transfer: one cache directory, two entries."""
        model_dir = tmp_path / "cache"
        paths = []
        for owner in ("kornia", "google"):
            source = tmp_path / owner / "model.safetensors"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(owner.encode())
            with patch.object(download_mod, "hf_url", return_value=source.as_uri()):
                paths.append(download_hf_file(f"{owner}/a-model", "model.safetensors", model_dir=str(model_dir)))

        assert paths[0] != paths[1]
        assert Path(paths[0]).read_bytes() == b"kornia"
        assert Path(paths[1]).read_bytes() == b"google"


class TestDownloadValidate:
    """``validate=`` gives a download-only call the quarantine a load gets.

    Without it a transfer cut short after a 2xx leaves a truncated file in the
    cache, and every later call returns it as a hit -- the caller fails on it
    forever, until someone deletes the file by hand.
    """

    @staticmethod
    def _serve(tmp_path, payload: bytes) -> str:
        source = tmp_path / "remote" / "model.safetensors"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        return source.as_uri()

    @staticmethod
    def _reject_truncated(expected_size: int):
        """A stand-in for a header parse: rejects anything short."""

        def validate(path: str) -> None:
            if os.path.getsize(path) != expected_size:
                raise ValueError(f"{path}: truncated")

        return validate

    def test_a_poisoned_cache_entry_is_refetched(self, tmp_path) -> None:
        payload = b"the-whole-file"
        url = self._serve(tmp_path, payload)
        model_dir = tmp_path / "cache"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(payload[:4])

        path = download_file_from_url(
            url,
            model_dir=str(model_dir),
            progress=False,
            validate=self._reject_truncated(len(payload)),
        )

        assert Path(path).read_bytes() == payload

    def test_without_validate_the_poisoned_entry_is_returned(self, tmp_path) -> None:
        """The behaviour ``validate`` opts out of, pinned so it stays a choice."""
        payload = b"the-whole-file"
        url = self._serve(tmp_path, payload)
        model_dir = tmp_path / "cache"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(payload[:4])

        path = download_file_from_url(url, model_dir=str(model_dir), progress=False)

        assert Path(path).read_bytes() == payload[:4]

    def test_a_good_cache_entry_is_not_refetched(self, monkeypatch, tmp_path) -> None:
        """``validate`` runs on hits, so it must not cost a transfer when it passes."""
        payload = b"the-whole-file"
        url = self._serve(tmp_path, payload)
        model_dir = tmp_path / "cache"
        transfers: list[str] = []
        real = download_mod._download_url_to_file

        def counted(url_, dst, *args, **kwargs):
            transfers.append(url_)
            return real(url_, dst, *args, **kwargs)

        monkeypatch.setattr(download_mod, "_download_url_to_file", counted)
        validate = self._reject_truncated(len(payload))

        first = download_file_from_url(url, model_dir=str(model_dir), progress=False, validate=validate)
        second = download_file_from_url(url, model_dir=str(model_dir), progress=False, validate=validate)

        assert second == first
        assert transfers == [url], "a valid cache entry was fetched again"

    def test_a_source_that_serves_a_bad_file_falls_through_to_the_next(self, tmp_path) -> None:
        payload = b"the-whole-file"
        bad = tmp_path / "remote" / "bad.safetensors"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(payload[:4])
        good_url = self._serve(tmp_path, payload)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = download_file_from_url(
                [bad.as_uri(), good_url],
                file_name="model.safetensors",
                model_dir=str(tmp_path / "cache"),
                progress=False,
                validate=self._reject_truncated(len(payload)),
            )

        assert Path(path).read_bytes() == payload

    def test_a_rejected_fresh_transfer_is_not_left_behind(self, tmp_path) -> None:
        """A cold cache that fails must not end the call poisoned.

        One URL -- which is what ``download_hf_file`` passes -- nothing cached, and
        the transfer arrives truncated. Those bytes are this call's own and
        ``validate`` has refused them, so they go rather than waiting for a later
        call to find them.
        """
        url = self._serve(tmp_path, b"trunc")
        model_dir = tmp_path / "cache"

        with pytest.raises(RuntimeError, match="Failed to download the file"):
            download_file_from_url(url, model_dir=str(model_dir), progress=False, validate=self._reject_truncated(14))

        assert not (model_dir / "model.safetensors").exists(), "a refused transfer was left cached"

    def test_the_rejection_is_what_the_failure_reports(self, monkeypatch, tmp_path) -> None:
        """Offline with a poisoned entry: name the file, not the network.

        The re-attempt fails on the network, so the last exception is a
        ``URLError`` sitting on top of the rejection that actually explains the
        failure. Reporting the network points the caller at an entry that is
        intact and has just been restored.
        """
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        model_dir = tmp_path / "cache"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"trunc")
        dead = (tmp_path / "missing" / "model.safetensors").as_uri()

        with pytest.raises(RuntimeError) as excinfo:
            download_file_from_url(dead, model_dir=str(model_dir), progress=False, validate=self._reject_truncated(14))

        message = str(excinfo.value)
        assert "truncated" in message, "the rejection that explains the failure is missing"
        assert "refetching it from that same source failed too" in message

    def test_a_file_that_never_validates_raises_and_keeps_the_original(self, tmp_path) -> None:
        """Nothing on disk is known-good, so the pre-call entry is put back.

        Deleting instead would destroy a multi-gigabyte checkpoint over a
        validator that was wrong, which is why the quarantine renames.
        """
        payload = b"the-whole-file"
        url = self._serve(tmp_path, payload)
        model_dir = tmp_path / "cache"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"original")

        def always_reject(path: str) -> None:
            raise ValueError("never valid")

        with pytest.raises(RuntimeError, match="Failed to download the file"):
            download_file_from_url(url, model_dir=str(model_dir), progress=False, validate=always_reject)

        assert (model_dir / "model.safetensors").read_bytes() == b"original"

    def test_check_safetensors_rejects_a_truncated_checkpoint(self, tmp_path) -> None:
        """The validator the builders actually pass."""
        import json
        import struct

        from kornia.core.safetensors import check_safetensors

        data = torch.arange(8, dtype=torch.float32).numpy().tobytes()
        header = json.dumps({"a": {"dtype": "F32", "shape": [8], "data_offsets": [0, len(data)]}}).encode()
        whole = struct.pack("<Q", len(header)) + header + data

        good = tmp_path / "good.safetensors"
        good.write_bytes(whole)
        check_safetensors(good)  # must not raise

        truncated = tmp_path / "bad.safetensors"
        truncated.write_bytes(whole[:-4])
        with pytest.raises(ValueError):
            check_safetensors(truncated)


class _Chunked:
    """A ``scripted_server`` response sent with ``Transfer-Encoding: chunked``.

    ``content_length``, when given, adds a ``Content-Length`` header as well, which HTTP
    says the chunked framing overrides.
    """

    def __init__(self, body: bytes, content_length: int | None = None) -> None:
        self.body = body
        self.content_length = content_length


def _bypass_proxies(monkeypatch) -> None:
    """Send requests for 127.0.0.1 straight to the test server, whatever proxy the shell sets.

    A lowercase ``http_proxy`` pointing at a closed port -- one way to block the internet
    for a test run -- would otherwise take the request and fail it with ``URLError``.
    """
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")


@pytest.fixture
def scripted_server(monkeypatch):
    """Serve scripted responses over HTTP from a thread bound to 127.0.0.1.

    Like :func:`local_server`, and it shuts down in a tenth of the time, which the many
    short tests below add up. ``responses`` maps a path to a response, or to a list of
    them consumed one per request with the last one repeating: ``bytes`` is a 200 with
    that body, an ``int`` is that status with an empty body and ``Retry-After: 0``, an
    ``(announced, body)`` tuple is a 200 whose ``Content-Length`` header says ``announced``
    (omitted when ``None``) while ``body`` is sent and the connection closes, and a
    :class:`_Chunked` is a 200 with a chunked body. Yields ``(responses, url, hits)``;
    any other path answers 404.
    """
    _bypass_proxies(monkeypatch)

    responses: dict[str, object] = {}
    hits: collections.Counter[str] = collections.Counter()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits[self.path] += 1
            script = responses.get(self.path, 404)
            if not isinstance(script, list):
                script = [script]
            outcome = script[min(hits[self.path], len(script)) - 1]
            if isinstance(outcome, int):
                self.send_response(outcome)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if isinstance(outcome, _Chunked):
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                if outcome.content_length is not None:
                    self.send_header("Content-Length", str(outcome.content_length))
                self.end_headers()
                for start in range(0, len(outcome.body), 1000):
                    piece = outcome.body[start : start + 1000]
                    self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                return
            announced, body = outcome if isinstance(outcome, tuple) else (len(outcome), outcome)
            self.send_response(200)
            if announced is not None:
                self.send_header("Content-Length", str(announced))
            self.end_headers()
            self.wfile.write(body)  # HTTP/1.0: the connection closes when the handler returns

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield responses, (lambda path: f"http://127.0.0.1:{port}{path}"), hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class _StalledServer:
    """A 127.0.0.1 listener that never finishes a response.

    ``mode="silent"`` accepts each connection and never sends a byte; ``mode="mid_body"``
    reads the request, sends the status line, the headers and the first 100 of 1000
    announced body bytes, then goes quiet. Every connection is held open until
    :meth:`close`, so the only thing that can end a client's wait is its own timeout.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.connections: list[socket.socket] = []
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen()
        # Closing a listener does not wake a thread blocked in ``accept`` on Linux, so the
        # thread polls a stop flag instead and :meth:`close` never waits on it.
        self._listener.settimeout(0.05)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._listener.getsockname()[1]}{path}"

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:  # nobody connected in this poll interval
                continue
            except OSError:  # the listener was closed
                return
            self.connections.append(conn)
            if self.mode == "mid_body":
                conn.settimeout(5)
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    request += chunk
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n" + b"x" * 100)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._listener.close()
        for conn in self.connections:
            conn.close()


@pytest.fixture(params=["silent", "mid_body"])
def stalled_server(request, monkeypatch):
    _bypass_proxies(monkeypatch)
    server = _StalledServer(request.param)
    try:
        yield server
    finally:
        server.close()


def _outcome_within(fn, seconds: float) -> BaseException | object:
    """Run *fn* in a daemon thread; return what it returned or raised, failing if it is still running."""
    outcome: list[object] = []

    def target() -> None:
        try:
            outcome.append(fn())
        except BaseException as e:
            outcome.append(e)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"still waiting after {seconds} s: the transfer has no timeout"
    return outcome[0]


@pytest.fixture
def hub_dir_in_tmp(monkeypatch, tmp_path):
    """Point the default cache at *tmp_path* and prove it, before anything is fetched."""
    hub = tmp_path / "hub"
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))
    assert Path(torch.hub.get_dir()).is_relative_to(tmp_path)
    return hub


@pytest.mark.usefixtures("hub_dir_in_tmp")
class TestTransferTimeout:
    """A stalled server must end a download, not hang it (#5216).

    kornia used to fetch through ``torch.hub.download_url_to_file``, which opens the URL
    with no timeout, and set none either, so a server that accepted the connection and then
    went quiet held the call forever: the retry and fallback logic never got control,
    because nothing raised.
    """

    _TIMEOUT = 0.25
    _DEADLINE = 10.0

    @pytest.mark.parametrize("entry", ["load_state_dict_from_url", "download_file_from_url"])
    def test_a_stalled_server_fails_within_the_timeout(self, stalled_server, monkeypatch, tmp_path, entry) -> None:
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        cache = tmp_path / "cache"
        fn = {"load_state_dict_from_url": load_state_dict_from_url, "download_file_from_url": download_file_from_url}
        url = stalled_server.url("/w.pth")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outcome = _outcome_within(
                lambda: fn[entry](url, model_dir=str(cache), progress=False, timeout=self._TIMEOUT), self._DEADLINE
            )

        assert isinstance(outcome, RuntimeError), f"expected the documented RuntimeError, got {outcome!r}"
        assert isinstance(outcome.__cause__, TimeoutError)
        message = str(outcome)
        assert "TimeoutError" in message and f"{self._TIMEOUT:g} s" in message
        assert str(cache / "w.pth") in message
        # Where the bound can be raised, including for callers such as pretrained constructors that take no timeout=.
        assert "timeout=" in message and "KORNIA_DOWNLOAD_TIMEOUT" in message
        assert "_DOWNLOAD_TIMEOUT_SECONDS" not in message
        # A timeout is transient, so the retry logic got control on every attempt.
        assert len(stalled_server.connections) == download_mod._MAX_ATTEMPTS
        # And nothing partial is left behind, under the cache name or a temporary one.
        assert not cache.exists() or list(cache.iterdir()) == []

    def test_the_environment_bounds_a_call_that_passes_no_timeout(self, stalled_server, monkeypatch, tmp_path):
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", str(self._TIMEOUT))
        monkeypatch.setattr(download_mod, "_HF_BASE", stalled_server.url(""))
        cache = tmp_path / "cache"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outcome = _outcome_within(
                lambda: download_hf_file("r", "w.safetensors", model_dir=str(cache), progress=False), self._DEADLINE
            )

        assert isinstance(outcome, RuntimeError), f"expected the documented RuntimeError, got {outcome!r}"
        assert isinstance(outcome.__cause__, TimeoutError)
        assert f"{self._TIMEOUT:g} s" in str(outcome)
        assert not cache.exists() or list(cache.iterdir()) == []

    def test_a_timed_out_primary_falls_back_to_the_mirror(self, stalled_server, scripted_server, monkeypatch, tmp_path):
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        files, url, hits = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.arange(3.0)})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outcome = _outcome_within(
                lambda: load_state_dict_from_url(
                    [stalled_server.url("/w.pth"), url("/w.pth")],
                    model_dir=str(tmp_path / "cache"),
                    progress=False,
                    timeout=self._TIMEOUT,
                ),
                self._DEADLINE,
            )

        assert isinstance(outcome, dict), f"expected the mirror's state dict, got {outcome!r}"
        assert torch.equal(outcome["w"], torch.arange(3.0))
        assert hits["/w.pth"] == 1
        assert any("Trying next source" in str(w.message) for w in caught)

    @pytest.mark.parametrize(
        ("timeout", "error"),
        [
            (0, ValueError),
            (-1.0, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            # Finite but beyond threading.TIMEOUT_MAX: an OverflowError inside the socket otherwise (ruling C16).
            pytest.param(1e300, ValueError, id="1e300"),
            pytest.param(threading.TIMEOUT_MAX * 2, ValueError, id="twice_TIMEOUT_MAX"),
            # More digits than the 4300 that ``repr`` of an int allows by default, so the message cannot print it.
            pytest.param(10**5000, ValueError, id="int_beyond_float"),
            ("30", TypeError),
        ],
    )
    def test_an_unusable_timeout_is_rejected_before_any_request(self, scripted_server, tmp_path, timeout, error):
        files, url, hits = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.zeros(1)})

        for fn in (load_state_dict_from_url, download_file_from_url):
            with pytest.raises(error, match="timeout"):
                fn(url("/w.pth"), model_dir=str(tmp_path / "cache"), progress=False, timeout=timeout)

        assert sum(hits.values()) == 0
        assert not (tmp_path / "cache").exists()

    def test_torch_hub_itself_is_left_without_a_timeout(self, scripted_server, tmp_path) -> None:
        """kornia bounds its own transfers only: the socket default and a direct ``torch.hub`` call are unchanged."""
        files, url, _ = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.zeros(1)})
        load_state_dict_from_url(url("/w.pth"), model_dir=str(tmp_path / "cache"), progress=False, timeout=5.0)

        assert socket.getdefaulttimeout() is None
        server = _StalledServer("silent")

        ended: list[Exception] = []

        def direct_torch_call() -> None:
            try:
                torch.hub.download_url_to_file(server.url("/x.pth"), str(tmp_path / "direct.pth"), progress=False)
            except Exception as e:  # the server hanging up below is what ends it
                ended.append(e)

        worker = threading.Thread(target=direct_torch_call, daemon=True)
        try:
            worker.start()
            worker.join(4 * self._TIMEOUT)
            assert worker.is_alive(), f"a direct torch.hub call picked up kornia's timeout: {ended}"
        finally:
            server.close()
            worker.join(5)
        assert not [e for e in ended if isinstance(e, TimeoutError)]


class TestLoadStateDictFromUrlFileName:
    """``file_name`` names a cache entry; nothing may escape the cache through it (#5217).

    torch joins it onto ``model_dir`` with ``os.path.join``, so an absolute name or one with
    ``../`` pointed outside the cache, and kornia's cache repair then moved the file there
    aside, downloaded the checkpoint over it and deleted the original.
    """

    _VICTIM = b"precious user data, not a checkpoint"

    def test_an_absolute_file_name_is_rejected_and_the_file_is_untouched(self, scripted_server, tmp_path) -> None:
        files, url, hits = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.zeros(2)})
        victims = tmp_path / "user_files"
        victims.mkdir()
        victim = victims / "precious.txt"
        victim.write_bytes(self._VICTIM)

        with pytest.raises(ValueError, match="bare filename"):
            load_state_dict_from_url(
                url("/w.pth"), model_dir=str(tmp_path / "cache"), progress=False, file_name=str(victim)
            )

        assert victim.read_bytes() == self._VICTIM
        assert [p.name for p in victims.iterdir()] == ["precious.txt"]
        assert sum(hits.values()) == 0
        assert not (tmp_path / "cache").exists()

    @pytest.mark.parametrize("file_name", ["../escape.pth", "sub/w.pth", "", ".", ".."])
    @pytest.mark.parametrize("sources", [1, 2], ids=["one_url", "two_urls"])
    def test_file_name_must_be_a_bare_filename(self, scripted_server, tmp_path, file_name, sources) -> None:
        files, url, hits = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.zeros(2)})
        model_dir = tmp_path / "cache"

        with pytest.raises(ValueError, match="bare filename"):
            load_state_dict_from_url(
                [url("/w.pth")] * sources, model_dir=str(model_dir), progress=False, file_name=file_name
            )

        assert sum(hits.values()) == 0
        assert not (tmp_path / "escape.pth").exists()
        assert not model_dir.exists(), "nothing was written"


_ENTRY_POINTS = {"load_state_dict_from_url": load_state_dict_from_url, "download_file_from_url": download_file_from_url}


class TestUrlArguments:
    """A URL that names no file, or a value that is not a URL, is refused up front (#5219)."""

    @pytest.mark.parametrize("tail", ["/dir/..", "/dir/.", "/dir/"])
    @pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
    @pytest.mark.parametrize("as_list", [False, True], ids=["str", "list"])
    def test_a_url_without_a_file_name_is_rejected(self, scripted_server, tmp_path, tail, entry, as_list) -> None:
        """The derived cache path would be the cache directory or its parent."""
        files, url, hits = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.zeros(2)})
        model_dir = tmp_path / "parent" / "cache"
        model_dir.mkdir(parents=True)
        source = [url(tail), url("/w.pth")] if as_list else url(tail)

        with pytest.raises(ValueError, match="file_name="):
            _ENTRY_POINTS[entry](source, model_dir=str(model_dir), progress=False)

        assert sum(hits.values()) == 0
        assert list(model_dir.iterdir()) == []
        assert [p.name for p in model_dir.parent.iterdir()] == ["cache"]

    @pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
    def test_an_explicit_file_name_makes_such_a_url_usable(self, scripted_server, tmp_path, entry) -> None:
        """The error says to pass ``file_name=``; doing so must work."""
        files, url, _ = scripted_server
        files["/dir/"] = _checkpoint_bytes({"w": torch.zeros(2)})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ENTRY_POINTS[entry](url("/dir/"), model_dir=str(tmp_path / "cache"), progress=False, file_name="w.pth")

        assert (tmp_path / "cache" / "w.pth").is_file()

    @pytest.mark.parametrize(
        ("url", "error", "match"),
        [
            ([], ValueError, "at least one URL"),
            ("", ValueError, "empty"),
            (["http://127.0.0.1:9/w.pth", ""], ValueError, "empty"),
            (Path("w.pth"), TypeError, "as_uri"),
            ([Path("w.pth")], TypeError, "as_uri"),
            (None, TypeError, "URL string"),
            (b"http://127.0.0.1:9/w.pth", TypeError, "URL string"),
            ([b"http://127.0.0.1:9/w.pth"], TypeError, "URL string"),
        ],
        ids=["empty_list", "empty_str", "empty_str_in_list", "path", "path_in_list", "none", "bytes", "bytes_in_list"],
    )
    @pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
    def test_a_value_that_is_not_a_url_is_refused(self, hub_dir_in_tmp, url, error, match, entry) -> None:
        with pytest.raises(error, match=match):
            _ENTRY_POINTS[entry](url, progress=False)

        assert not hub_dir_in_tmp.exists()


class TestHfUrlEscaping:
    """``hf_url`` builds a URL, so its arguments are path segments, not URL syntax (#5219)."""

    @pytest.mark.parametrize(
        ("filename", "tail"),
        [("a#b.pth", "a%23b.pth"), ("a?b.pth", "a%3Fb.pth"), ("my model.pth", "my%20model.pth"), ("a%b", "a%25b")],
    )
    def test_a_reserved_character_is_percent_encoded(self, filename, tail) -> None:
        assert hf_url("r", filename) == f"https://huggingface.co/kornia/r/resolve/main/{tail}"

    def test_a_subdirectory_keeps_its_slashes(self) -> None:
        assert hf_url("loftr", "weights/outdoor.ckpt") == (
            "https://huggingface.co/kornia/loftr/resolve/main/weights/outdoor.ckpt"
        )

    def test_the_repo_id_is_encoded_too(self) -> None:
        assert hf_url("owner/my repo", "f.pth") == "https://huggingface.co/owner/my%20repo/resolve/main/f.pth"

    def test_download_hf_file_fetches_the_file_it_names(self, scripted_server, monkeypatch, tmp_path) -> None:
        """Unescaped, ``#b.pth`` was a fragment: the file ``a`` was fetched and cached as ``a#b.pth``."""
        files, url, hits = scripted_server
        files["/kornia/r/resolve/main/a%23b.pth"] = b"the file a#b.pth"
        files["/kornia/r/resolve/main/a"] = b"the file a"
        monkeypatch.setattr(download_mod, "_HF_BASE", url(""))

        path = download_hf_file("r", "a#b.pth", model_dir=str(tmp_path / "cache"), progress=False)

        assert Path(path).name == "kornia--r--a#b.pth"
        assert Path(path).read_bytes() == b"the file a#b.pth"
        assert hits == {"/kornia/r/resolve/main/a%23b.pth": 1}


class TestWarningAttribution:
    """A warning names the caller's line through every public entry point (#5219)."""

    @pytest.mark.parametrize("entry", ["load_state_dict_from_url", "download_file_from_url", "download_hf_file"])
    def test_a_retry_warning_points_at_the_caller(self, scripted_server, monkeypatch, tmp_path, entry) -> None:
        responses, url, _ = scripted_server
        body = _checkpoint_bytes({"w": torch.zeros(1)})
        responses["/flaky/w.pth"] = [503, body]
        responses["/kornia/flaky/resolve/main/w.pth"] = [503, body]
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        monkeypatch.setattr(download_mod, "_HF_BASE", url(""))
        cache = str(tmp_path / "cache")
        calls = {
            "load_state_dict_from_url": lambda: load_state_dict_from_url(
                url("/flaky/w.pth"), model_dir=cache, progress=False
            ),
            "download_file_from_url": lambda: download_file_from_url(
                url("/flaky/w.pth"), model_dir=cache, progress=False
            ),
            "download_hf_file": lambda: download_hf_file("flaky", "w.pth", model_dir=cache, progress=False),
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calls[entry]()

        retries = [w for w in caught if "Transient failure" in str(w.message)]
        assert len(retries) == 1
        assert Path(retries[0].filename).resolve() == Path(__file__).resolve(), retries[0].filename


class TestKeywordArguments:
    """A keyword torch does not accept is a caller error, not a corrupt cache entry (#5219)."""

    def test_a_mistyped_keyword_on_a_warm_cache_raises_at_once(self, scripted_server, tmp_path) -> None:
        """It used to quarantine the entry, download it again and raise ``RuntimeError``."""
        files, url, hits = scripted_server
        body = _checkpoint_bytes({"w": torch.zeros(2)})
        files["/w.pth"] = body
        cache = tmp_path / "cache"
        load_state_dict_from_url(url("/w.pth"), model_dir=str(cache), progress=False)
        assert hits["/w.pth"] == 1

        with pytest.raises(TypeError, match="map_locaton"):
            load_state_dict_from_url(url("/w.pth"), model_dir=str(cache), progress=False, map_locaton="cpu")

        assert hits["/w.pth"] == 1, "the checkpoint was downloaded again"
        assert [p.name for p in cache.iterdir()] == ["w.pth"]
        assert (cache / "w.pth").read_bytes() == body

    def test_a_mistyped_keyword_on_a_cold_cache_makes_no_request(self, scripted_server, tmp_path) -> None:
        files, url, hits = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.zeros(2)})

        with pytest.raises(TypeError, match="map_locaton"):
            load_state_dict_from_url(url("/w.pth"), model_dir=str(tmp_path / "cache"), map_locaton="cpu")

        assert sum(hits.values()) == 0
        assert not (tmp_path / "cache").exists()

    def test_every_keyword_torch_accepts_is_still_forwarded(self, scripted_server, tmp_path) -> None:
        files, url, _ = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.zeros(2)})

        result = load_state_dict_from_url(
            url("/w.pth"),
            model_dir=str(tmp_path / "cache"),
            map_location="cpu",
            progress=False,
            check_hash=False,
            file_name="w.pth",
            weights_only=True,
        )

        assert torch.equal(result["w"], torch.zeros(2))


class TestModelDirExpandsUser:
    """``model_dir="~/..."`` means the home directory, as it does everywhere else (#5219)."""

    @pytest.fixture
    def home(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        # ``~`` must resolve inside the temporary directory before anything is fetched. Compared as paths: on
        # Windows ``expanduser("~/kc")`` keeps the ``/`` and gives ``...\home/kc``, the same directory.
        assert Path(os.path.expanduser("~/kc")) == home / "kc"
        return home, cwd

    def test_download_file_from_url(self, scripted_server, home) -> None:
        home_dir, cwd = home
        files, url, _ = scripted_server
        files["/t.bin"] = b"payload"

        path = download_file_from_url(url("/t.bin"), model_dir="~/kc", progress=False)

        assert Path(path) == home_dir / "kc" / "t.bin"
        assert Path(path).read_bytes() == b"payload"
        assert list(cwd.iterdir()) == [], "a literal '~' directory was created"

    def test_load_state_dict_from_url(self, scripted_server, home) -> None:
        home_dir, cwd = home
        files, url, _ = scripted_server
        files["/w.pth"] = _checkpoint_bytes({"w": torch.ones(2)})

        result = load_state_dict_from_url(url("/w.pth"), model_dir="~/kc", progress=False)

        assert torch.equal(result["w"], torch.ones(2))
        assert (home_dir / "kc" / "w.pth").is_file()
        assert list(cwd.iterdir()) == [], "a literal '~' directory was created"


class TestKorniaTransfer:
    """``_download_url_to_file`` stands in for ``torch.hub.download_url_to_file``; it keeps torch's contract."""

    def test_check_hash_accepts_the_right_prefix_and_rejects_a_wrong_one(self, scripted_server, tmp_path) -> None:
        responses, url, _ = scripted_server
        body = _checkpoint_bytes({"w": torch.ones(3)})
        prefix = hashlib.sha256(body).hexdigest()[:8]
        responses[f"/w-{prefix}.pth"] = body
        responses["/w-deadbeef.pth"] = body
        cache = tmp_path / "cache"

        result = load_state_dict_from_url(
            url(f"/w-{prefix}.pth"), model_dir=str(cache), progress=False, check_hash=True
        )
        assert torch.equal(result["w"], torch.ones(3))

        with pytest.raises(RuntimeError, match="invalid hash value") as excinfo:
            load_state_dict_from_url(url("/w-deadbeef.pth"), model_dir=str(cache), progress=False, check_hash=True)
        assert f'got "{prefix}' in str(excinfo.value)
        # Nothing of the rejected transfer is kept, under its name or a temporary one.
        assert [p.name for p in cache.iterdir()] == [f"w-{prefix}.pth"]

    def test_the_progress_bar_stays_off_stdout(self, scripted_server, tmp_path, capsys) -> None:
        responses, url, _ = scripted_server
        responses["/t.bin"] = b"x" * 300_000

        path = download_file_from_url(url("/t.bin"), model_dir=str(tmp_path / "cache"), progress=True)

        assert Path(path).read_bytes() == b"x" * 300_000
        captured = capsys.readouterr()
        assert captured.out == ""
        assert 'Downloading: "' in captured.err

    def test_an_existing_file_is_replaced_only_by_a_complete_transfer(self, scripted_server, tmp_path) -> None:
        """The transfer writes beside the destination and swaps it in only once complete."""
        responses, url, _ = scripted_server
        responses["/t.bin"] = 503
        dst = tmp_path / "t.bin"
        dst.write_bytes(b"previous")

        with pytest.raises(HTTPError):
            download_mod._download_url_to_file(url("/t.bin"), str(dst), progress=False, timeout=5.0)
        assert dst.read_bytes() == b"previous"
        assert [p.name for p in tmp_path.iterdir()] == ["t.bin"]

        responses["/t.bin"] = b"complete"
        download_mod._download_url_to_file(url("/t.bin"), str(dst), progress=False, timeout=5.0)
        assert dst.read_bytes() == b"complete"
        assert [p.name for p in tmp_path.iterdir()] == ["t.bin"]


class TestTruncatedTransfer:
    """A body shorter than its ``Content-Length`` is a failed transfer, not a file (#5216).

    urllib ends a ``read(n)`` loop on the early close without an error, so a server that
    announced 1577 bytes and sent 50 left a 50-byte file in the cache, which every later
    call returned as a hit.
    """

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch, hub_dir_in_tmp):
        monkeypatch.setattr(download_mod, "time", _FakeTime())

    def test_a_short_body_is_retried_and_never_cached(self, scripted_server, tmp_path) -> None:
        responses, url, hits = scripted_server
        body = _checkpoint_bytes({"w": torch.ones(4)})
        responses["/t.bin"] = (len(body), body[:50])
        cache = tmp_path / "cache"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError) as excinfo:
                download_file_from_url(url("/t.bin"), model_dir=str(cache), progress=False)

        assert isinstance(excinfo.value.__cause__, http.client.IncompleteRead)
        message = str(excinfo.value)
        assert f"Last error: transfer truncated: the server sent 50 of the {len(body)} bytes" in message
        assert "_TruncatedTransfer" not in message
        assert hits["/t.bin"] == download_mod._MAX_ATTEMPTS  # retried as the transient failure it is
        assert not cache.exists() or list(cache.iterdir()) == []

    @pytest.mark.parametrize("entry", ["load_state_dict_from_url", "download_file_from_url"])
    def test_a_short_primary_hands_over_to_the_mirror(self, scripted_server, tmp_path, entry) -> None:
        responses, url, _ = scripted_server
        body = _checkpoint_bytes({"w": torch.ones(4)})
        responses["/primary/w.pth"] = (len(body), body[:50])
        responses["/mirror/w.pth"] = body
        cache = tmp_path / "cache"
        fn = {"load_state_dict_from_url": load_state_dict_from_url, "download_file_from_url": download_file_from_url}

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fn[entry]([url("/primary/w.pth"), url("/mirror/w.pth")], model_dir=str(cache), progress=False)

        assert [p.name for p in cache.iterdir()] == ["w.pth"]
        assert (cache / "w.pth").read_bytes() == body

    def test_a_retry_after_a_short_body_completes_the_file(self, scripted_server, tmp_path) -> None:
        responses, url, hits = scripted_server
        body = b"y" * 5000
        responses["/t.bin"] = [(len(body), body[:50]), body]

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            path = download_file_from_url(url("/t.bin"), model_dir=str(tmp_path / "cache"), progress=False)

        assert Path(path).read_bytes() == body
        assert hits["/t.bin"] == 2

    def test_a_short_body_leaves_an_existing_destination_untouched(self, scripted_server, tmp_path) -> None:
        responses, url, _ = scripted_server
        responses["/t.bin"] = (1000, b"z" * 10)
        dst = tmp_path / "t.bin"
        dst.write_bytes(b"previous")

        with pytest.raises(http.client.IncompleteRead):
            download_mod._download_url_to_file(url("/t.bin"), str(dst), progress=False, timeout=5.0)

        assert dst.read_bytes() == b"previous"
        assert [p.name for p in tmp_path.iterdir()] == ["t.bin"]


class TestCacheNameEdgeCases:
    """Review follow-ups on the cache-name checks (#5217, #5219)."""

    @pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
    def test_a_drive_relative_url_name_is_rejected_under_windows_path_rules(
        self, monkeypatch, hub_dir_in_tmp, entry
    ) -> None:
        """``d:x.pth`` is one segment of a URL path, but a drive-relative path to ``ntpath.join``."""
        monkeypatch.setattr(download_mod, "time", _FakeTime())
        monkeypatch.setattr(download_mod.os.path, "basename", ntpath.basename)
        # A closed local port: if the name were accepted, the call would fail on the transfer instead.
        with pytest.raises(ValueError, match="file_name="):
            _ENTRY_POINTS[entry]("http://127.0.0.1:9/d:x.pth", progress=False)

    @pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
    def test_a_path_file_name_is_accepted(self, scripted_server, tmp_path, entry) -> None:
        """``file_name=Path("w.pth")`` worked before the check existed; it is a bare name."""
        responses, url, _ = scripted_server
        body = _checkpoint_bytes({"w": torch.ones(2)})
        responses["/other.pth"] = body
        cache = tmp_path / "cache"

        _ENTRY_POINTS[entry](url("/other.pth"), model_dir=str(cache), progress=False, file_name=Path("w.pth"))

        assert [p.name for p in cache.iterdir()] == ["w.pth"]
        assert (cache / "w.pth").read_bytes() == body


class TestTruncatedTransferFraming:
    """The short-read check follows the framing http.client used, and the error travels (#5216 review)."""

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch, hub_dir_in_tmp):
        monkeypatch.setattr(download_mod, "time", _FakeTime())

    @pytest.mark.parametrize(
        "response",
        [(None, b"q" * 5000), _Chunked(b"q" * 5000), _Chunked(b"q" * 5000, content_length=10), (-5, b"q" * 5000)],
        ids=["close_delimited", "chunked", "chunked_with_a_mismatched_content_length", "negative_content_length"],
    )
    def test_a_body_not_framed_by_content_length_downloads_completely(self, scripted_server, tmp_path, response):
        """Without a valid ``Content-Length`` framing the body there is nothing to count against, as in torch."""
        responses, url, hits = scripted_server
        responses["/t.bin"] = response

        path = download_file_from_url(url("/t.bin"), model_dir=str(tmp_path / "cache"), progress=False)

        assert Path(path).read_bytes() == b"q" * 5000
        assert hits["/t.bin"] == 1

    def test_the_truncation_error_pickles_and_copies(self) -> None:
        error = download_mod._TruncatedTransfer(50, 1577)
        for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error)):  # noqa: S301 - our own bytes
            assert isinstance(clone, http.client.IncompleteRead)
            assert str(clone) == str(error)
            assert (clone.received, clone.announced) == (50, 1577)
        assert str(error) == "transfer truncated: the server sent 50 of the 1577 bytes its Content-Length announced"


class TestDownloadTimeoutEnvironment:
    """``KORNIA_DOWNLOAD_TIMEOUT`` sets the default for calls that pass no ``timeout=`` (ruling C14).

    Pretrained model constructors take no ``timeout=``, so the variable is how their users
    raise the bound on a slow link.
    """

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch, hub_dir_in_tmp):
        monkeypatch.setattr(download_mod, "time", _FakeTime())

    @pytest.fixture(autouse=True)
    def _no_proxy(self, monkeypatch):
        """Three tests below build a ``_StalledServer`` without the ``stalled_server`` fixture and its proxy scrub."""
        _bypass_proxies(monkeypatch)

    @pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
    def test_the_variable_bounds_a_call_without_timeout(self, monkeypatch, tmp_path, entry) -> None:
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", "0.25")
        server = _StalledServer("silent")
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                outcome = _outcome_within(
                    lambda: _ENTRY_POINTS[entry](server.url("/w.pth"), model_dir=str(tmp_path / "c"), progress=False),
                    10.0,
                )
        finally:
            server.close()

        assert isinstance(outcome, RuntimeError), f"expected the documented RuntimeError, got {outcome!r}"
        assert isinstance(outcome.__cause__, TimeoutError)
        assert "0.25 s" in str(outcome)

    def test_an_explicit_timeout_wins_over_the_variable(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", "600")
        server = _StalledServer("silent")
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                outcome = _outcome_within(
                    lambda: download_file_from_url(
                        server.url("/w.pth"), model_dir=str(tmp_path / "c"), progress=False, timeout=0.25
                    ),
                    10.0,
                )
        finally:
            server.close()

        assert isinstance(outcome, RuntimeError) and isinstance(outcome.__cause__, TimeoutError)
        assert "0.25 s" in str(outcome)

    def test_an_explicit_timeout_does_not_read_the_variable(self, scripted_server, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", "abc")
        responses, url, _ = scripted_server
        responses["/t.bin"] = b"payload"

        path = download_file_from_url(url("/t.bin"), model_dir=str(tmp_path / "c"), progress=False, timeout=5.0)

        assert Path(path).read_bytes() == b"payload"

    @pytest.mark.parametrize(
        "value",
        ["0", "-1", "abc", "inf", "nan", "1e300", pytest.param(str(threading.TIMEOUT_MAX * 2), id="twice_TIMEOUT_MAX")],
    )
    @pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
    def test_an_invalid_value_is_named_before_any_request(self, scripted_server, monkeypatch, tmp_path, value, entry):
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", value)
        responses, url, hits = scripted_server
        responses["/t.bin"] = b"payload"

        with pytest.raises(ValueError, match="KORNIA_DOWNLOAD_TIMEOUT"):
            _ENTRY_POINTS[entry](url("/t.bin"), model_dir=str(tmp_path / "c"), progress=False)

        assert sum(hits.values()) == 0
        assert not (tmp_path / "c").exists()

    @pytest.mark.parametrize("value", [None, "", "  "], ids=["unset", "empty", "blank"])
    def test_unset_gives_the_default(self, monkeypatch, value) -> None:
        if value is None:
            monkeypatch.delenv("KORNIA_DOWNLOAD_TIMEOUT", raising=False)
        else:
            monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", value)

        assert download_mod._resolve_timeout(None) == download_mod._DOWNLOAD_TIMEOUT_SECONDS == 30.0

    def test_a_valid_value_is_read_at_call_time(self, monkeypatch) -> None:
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", " 12.5 ")
        assert download_mod._resolve_timeout(None) == 12.5
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", "3")
        assert download_mod._resolve_timeout(None) == 3.0

    def test_the_stdlib_ceiling_itself_is_accepted(self, monkeypatch) -> None:
        """``threading.TIMEOUT_MAX`` is the largest blocking timeout the stdlib takes; validation lets it through."""
        assert download_mod._resolve_timeout(threading.TIMEOUT_MAX) == threading.TIMEOUT_MAX
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", repr(threading.TIMEOUT_MAX))
        assert download_mod._resolve_timeout(None) == threading.TIMEOUT_MAX

    def test_the_message_states_the_bound_exactly(self, monkeypatch) -> None:
        """``:g`` printed 9223372036.0 as 9.22337e+09, and Windows' 4294967.0 as 4.29497e+06, above the bound."""
        bound = f"at most {threading.TIMEOUT_MAX!r} (threading.TIMEOUT_MAX)"
        with pytest.raises(ValueError, match="timeout") as caught:
            download_mod._resolve_timeout(threading.TIMEOUT_MAX * 2)
        assert bound in str(caught.value)
        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", "1e300")
        with pytest.raises(ValueError, match="KORNIA_DOWNLOAD_TIMEOUT") as caught:
            download_mod._resolve_timeout(None)
        assert bound in str(caught.value)

    @pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
    def test_a_warm_cache_call_still_checks_the_variable(self, scripted_server, monkeypatch, tmp_path, entry) -> None:
        """The variable is checked when the function is called, before the cache; ``timeout=`` skips it."""
        responses, url, hits = scripted_server
        body = _checkpoint_bytes({"w": torch.ones(2)})
        responses["/w.pth"] = body
        cache = str(tmp_path / "c")
        _ENTRY_POINTS[entry](url("/w.pth"), model_dir=cache, progress=False)
        assert hits["/w.pth"] == 1

        monkeypatch.setenv("KORNIA_DOWNLOAD_TIMEOUT", "abc")
        with pytest.raises(ValueError, match="KORNIA_DOWNLOAD_TIMEOUT"):
            _ENTRY_POINTS[entry](url("/w.pth"), model_dir=cache, progress=False)
        _ENTRY_POINTS[entry](url("/w.pth"), model_dir=cache, progress=False, timeout=5)

        assert hits["/w.pth"] == 1, "the cached file was fetched again"
        assert (tmp_path / "c" / "w.pth").read_bytes() == body
