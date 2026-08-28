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

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

import torch

_HF_KORNIA_BASE = "https://huggingface.co/kornia"


def hf_url(repo: str, filename: str) -> str:
    """Return the HuggingFace URL for a file in a kornia model repo.

    Args:
        repo: repository name under the ``kornia`` HF org (e.g. ``"hardnet"``).
        filename: file at the root of that repo (e.g. ``"HardNetPP.pth"``).

    Returns:
        A ``resolve/main`` URL that can be passed directly to
        :func:`load_state_dict_from_url`.

    Example:
        >>> hf_url("hardnet", "HardNetPP.pth")
        'https://huggingface.co/kornia/hardnet/resolve/main/HardNetPP.pth'
    """
    return f"{_HF_KORNIA_BASE}/{repo}/resolve/main/{filename}"


_TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
"""HTTP statuses worth another attempt: request timeout, too early, rate limit, server errors."""

_MAX_ATTEMPTS = 3
"""Total attempts per URL, including the first."""

_BACKOFF_SECONDS = 1.0
"""Delay before the second attempt; doubled for each further one."""


def _is_transient(exc: BaseException) -> bool:
    """Return whether *exc* is a temporary network condition worth retrying.

    Rate limiting is the motivating case. An unauthenticated CI matrix fans many
    jobs out at once, which regularly trips the anonymous request limits of
    huggingface.co and github.com even though the checkpoint is served normally a
    moment later.

    Args:
        exc: the exception raised by a download attempt.

    Returns:
        ``True`` if another attempt could plausibly succeed.
    """
    if isinstance(exc, HTTPError):  # a subclass of URLError, so it must be tested first
        return exc.code in _TRANSIENT_HTTP_STATUS
    return isinstance(exc, (URLError, TimeoutError, ConnectionError))


def _cached_file_path(url: str, kwargs: dict[str, Any]) -> str:
    """Return the path :func:`torch.hub.load_state_dict_from_url` would cache *url* at.

    Mirrors torch's own resolution: ``<model_dir>/<file_name or basename(url)>``,
    where ``model_dir`` defaults to ``<torch.hub.get_dir()>/checkpoints``.

    Args:
        url: the URL that would be downloaded.
        kwargs: the keyword arguments destined for the torch function; only
            ``model_dir`` and ``file_name`` are consulted.

    Returns:
        The absolute path of the cache entry for *url*.
    """
    model_dir = kwargs.get("model_dir")
    if model_dir is None:
        model_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    filename = kwargs.get("file_name") or os.path.basename(urlparse(url).path)
    return os.path.join(model_dir, filename)


def _prefetch_to_cache(url: str, kwargs: dict[str, Any]) -> None:
    """Download *url* into the torch hub cache if it is not already there.

    Doing the fetch here rather than letting :func:`torch.hub.load_state_dict_from_url`
    do it means torch finds the file present and never writes its status line to
    stdout. A no-op when the file is already cached, which is the common case.

    If the path computed here ever disagreed with torch's, the only consequence
    is that torch downloads the file again -- the result stays correct.

    Args:
        url: the URL to fetch.
        kwargs: the keyword arguments destined for the torch function;
            ``model_dir``, ``file_name``, ``check_hash`` and ``progress`` are
            honoured so the cache entry is identical to torch's.
    """
    cached_file = _cached_file_path(url, kwargs)
    if os.path.exists(cached_file):
        return

    os.makedirs(os.path.dirname(cached_file), exist_ok=True)

    hash_prefix = None
    if kwargs.get("check_hash"):
        match = torch.hub.HASH_REGEX.search(os.path.basename(cached_file))
        hash_prefix = match.group(1) if match else None

    # torch writes this to stdout; status output belongs on stderr.
    sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
    torch.hub.download_url_to_file(url, cached_file, hash_prefix, progress=kwargs.get("progress", True))


_DISCARDED_CACHE_PATHS: set[str] = set()
"""Cache paths already dropped once in this process; see :func:`_discard_cache_entry`."""


def _discard_cache_entry(url: str, kwargs: dict[str, Any]) -> None:
    """Delete the cache entry for *url*, at most once per path per process.

    Every URL in a fallback list resolves to the same path, because the cache
    filename is pinned to the primary URL (see :func:`load_state_dict_from_url`).
    A single bad write -- a truncated transfer, or an HTML rate-limit page served
    with a 200 -- would therefore make :func:`_prefetch_to_cache` short-circuit on
    each remaining source and hand torch the same broken file every time. The
    fallback URLs could never take effect, and because the bad file survives the
    process, the failure would repeat on every later run until the cache was
    cleared by hand.

    A load failure cannot tell a corrupt cache entry from a healthy one that
    failed for an unrelated reason -- a bad ``map_location``, a ``weights_only``
    rejection -- so the discard is bounded instead of conditional. One discard
    per path is all a poisoned entry ever needs: the next source refetches it.
    Anything still failing after that refetch is not the file's fault, and
    dropping it again would re-download the checkpoint on every later call, once
    per test that builds the model. That is the download storm this module exists
    to prevent, reached from the other side.

    Bounding it rather than gating it on the exception type is deliberate. The
    types that would make up such a gate do not separate the two classes (torch
    2.9.1): an HTML rate-limit page served with a 200 -- the case that motivated
    this function -- and a ``weights_only`` rejection of a perfectly good file
    both raise ``UnpicklingError``, while a truncated zip checkpoint raises a
    bare ``RuntimeError``. A gate would therefore still discard the healthy entry
    it was meant to spare, and stop discarding the commonest corruption there is.
    The bound caps the cost instead: one extra fetch per path per process, spent
    only inside a call that raises either way.

    The bound is what makes a corrupt entry outlive a run in which *every* source
    failed. It cannot outlive more than that: the ledger is per-process, so the
    next run spends its one discard on the same path and refetches.

    Args:
        url: the URL whose cache entry should be dropped.
        kwargs: the keyword arguments destined for the torch function; only
            ``model_dir`` and ``file_name`` are consulted.
    """
    path = _cached_file_path(url, kwargs)
    if path in _DISCARDED_CACHE_PATHS:
        return

    try:
        os.remove(path)
    except FileNotFoundError:
        # The transfer itself failed, so there is nothing to clean up -- and
        # nothing was refetched either, so the one allowed discard stays unspent.
        return
    except OSError as e:
        # A read-only cache, or a Windows reader holding the file open. Removing
        # it again would fail the same way, so the ledger is marked either way,
        # but a fallback that can never be reached is worth saying out loud.
        warnings.warn(
            f"Could not discard the cache entry at {path!r}: {e}. If that file is corrupt, "
            f"the fallback sources cannot take effect until it is deleted by hand.",
            stacklevel=3,
        )

    _DISCARDED_CACHE_PATHS.add(path)


def _prefetch_with_retry(url: str, kwargs: dict[str, Any]) -> None:
    """Run :func:`_prefetch_to_cache`, retrying transient failures with backoff.

    Args:
        url: the URL to fetch.
        kwargs: the keyword arguments destined for the torch function.

    Raises:
        Exception: the last failure, once the attempts are exhausted or the
            failure is not transient.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            _prefetch_to_cache(url, kwargs)
            return
        except Exception as e:
            if attempt == _MAX_ATTEMPTS or not _is_transient(e):
                raise
            delay = _BACKOFF_SECONDS * 2 ** (attempt - 1)
            warnings.warn(
                f"Transient failure fetching {url!r}: {e}. Retrying in {delay:.0f}s "
                f"(attempt {attempt + 1} of {_MAX_ATTEMPTS}).",
                # 1 = here, 2 = load_state_dict_from_url, 3 = its caller, which is
                # the frame the sibling warning below also points at.
                stacklevel=3,
            )
            time.sleep(delay)


def load_state_dict_from_url(url: str | list[str], **kwargs: Any) -> dict[str, Any]:
    """Load a state dict from a URL, trying fallback URLs on failure.

    Drop-in replacement for :func:`torch.hub.load_state_dict_from_url` that
    accepts either a single URL string or an ordered list of URLs. Each URL is
    tried in turn; a :mod:`warnings` message is emitted for every failed
    attempt before the next source is tried.

    Progress reporting is written to :data:`sys.stderr`. This is the one
    deliberate deviation from the torch function, which since torch 2.x writes
    its ``Downloading: "<url>" to <path>`` line to :data:`sys.stdout` (the
    accompanying progress bar already goes to stderr). Status output on stdout
    corrupts any caller that treats stdout as data -- most visibly doctests,
    where the line is captured as unexpected example output and fails an
    example that downloads on a cold cache.

    That line is reached only when the file is absent from the cache, so this
    function fetches a missing file itself -- announcing it on stderr -- and
    leaves torch with nothing to report. Nothing process-global is touched:
    redirecting :data:`sys.stdout` around the call would divert unrelated
    threads' output for the whole transfer, and concurrent calls restoring out
    of order would leave stdout permanently pointing at stderr. This mirrors
    :func:`kornia.feature.lightglue_onnx.utils.download.download_onnx_from_url`,
    which already reimplements torch's caching for the same reason.

    When multiple URLs are given and ``file_name`` is not already in *kwargs*,
    the basename of the **first** URL is used as the local cache filename for
    all attempts. This guarantees that:

    * a file successfully downloaded from the primary source is found on the
      next call without re-downloading;
    * hash validation (``check_hash=True``) uses the filename — and therefore
      the hash embedded in it — of the primary URL consistently across all
      fallback attempts.

    Because that one path is shared, a failed attempt drops the cache entry
    before the next URL is tried. Otherwise a single bad write would be handed
    straight back to torch by every remaining source -- and by every later
    process -- making the fallback URLs unreachable. That discard happens at most
    once per cache path per process, so a failure the cache cannot fix costs one
    refetch rather than one per call; see :func:`_discard_cache_entry`.

    Each URL is attempted up to :data:`_MAX_ATTEMPTS` times, with exponential
    backoff, when it fails with a transient network condition such as an HTTP
    429. Unauthenticated CI matrices trip the rate limits of huggingface.co and
    github.com routinely, and a retry is far cheaper than a failed job.

    Args:
        url: a URL string, or a list of URL strings tried left-to-right.
        **kwargs: forwarded verbatim to
            :func:`torch.hub.load_state_dict_from_url`
            (``map_location``, ``check_hash``, ``file_name``, …).

    Returns:
        The loaded state dict.

    Raises:
        RuntimeError: if every URL fails. The message carries the last
            exception's type and text, and the exception itself is chained;
            without it a rate limit is indistinguishable from a dead link in a
            CI failure summary.

    Example:
        >>> sd = load_state_dict_from_url([          # doctest: +SKIP
        ...     hf_url("hardnet", "HardNetPP.pth"),  # primary  (HF mirror)
        ...     "https://github.com/DagnyT/hardnet/raw/master/"
        ...     "pretrained/pretrained_all_datasets/HardNet%2B%2B.pth",  # fallback
        ... ])
    """
    urls = [url] if isinstance(url, str) else list(url)

    # Pin the cache filename to the primary URL's basename so that all
    # attempts share one cache slot and hash validation stays consistent.
    if len(urls) > 1 and "file_name" not in kwargs:
        kwargs["file_name"] = Path(urlparse(urls[0]).path).name

    last_exc: Exception | None = None
    for i, u in enumerate(urls):
        try:
            # Populate the cache ourselves so torch's stdout line is never reached.
            _prefetch_with_retry(u, kwargs)
            return torch.hub.load_state_dict_from_url(u, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            # Whatever landed in the cache is not loadable, and every remaining
            # URL shares its path. Drop it so the next source is really fetched.
            _discard_cache_entry(u, kwargs)
            if i < len(urls) - 1:
                warnings.warn(
                    f"Failed to load weights from {u!r}: {e}. Trying next source.",
                    stacklevel=2,
                )

    raise RuntimeError(
        f"Failed to load weights from all {len(urls)} source(s). "
        f"Last URL tried: {urls[-1]!r}. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc
