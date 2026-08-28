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
import math
import os
import sys
import time
import warnings
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
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

_RATE_LIMIT_HTTP_STATUS = frozenset({403, 429})
"""Statuses a host may use to signal rate limiting.

429 is unambiguous and already transient above. 403 is not: GitHub documents it
as the *other* status it exceeds a rate limit with, while it is also the plain
"you may not have this file" answer that must never be retried. Only the
rate-limit headers separate the two, so 403 is transient exactly when they say
so -- see :func:`_rate_limited`.
"""

_MAX_ATTEMPTS = 3
"""Total attempts per URL, including the first."""

_BACKOFF_SECONDS = 1.0
"""Delay before the second attempt; doubled for each further one."""

_MAX_BACKOFF_SECONDS = 60.0
"""Ceiling on a single wait, including one a server asks for.

A host may name a delay of many minutes. Honouring it literally would hold a CI
job open far longer than refetching the checkpoint costs, so the request is
clamped: the wait is capped and the attempt made anyway.
"""

_MAX_CALL_SLEEP_SECONDS = 60.0
"""Ceiling on the *total* time one call may spend waiting between attempts.

Clamping each wait individually does not bound the call: two retries per URL
across a two-source list is four waits, so a host asking for the maximum on each
would hold a single :func:`load_state_dict_from_url` call open for
4 x :data:`_MAX_BACKOFF_SECONDS`, and a test module that builds the same model
five times would multiply that again. Retrying exists to ride out a *brief*
limit; once a call has spent this much of its life asleep the limit is not
brief, and failing over to the next source -- or out to the caller with the
cause named -- beats waiting longer. The budget is per call rather than per
process so that a long-lived process is never left permanently unable to retry.
"""


class _SleepBudget:
    """The time a single call may spend waiting between attempts."""

    def __init__(self, seconds: float) -> None:
        self.remaining = seconds

    def sleep(self, delay: float) -> None:
        """Wait *delay* seconds and charge it to the budget."""
        self.remaining -= delay
        time.sleep(delay)


def _rate_limited(exc: HTTPError) -> bool:
    """Return whether *exc*'s headers mark it as a rate-limit response.

    Args:
        exc: the HTTP error raised by a download attempt.

    Returns:
        ``True`` if the status is one hosts use for rate limiting *and* the
        response carries a header that only a rate limit sets.
    """
    if exc.code not in _RATE_LIMIT_HTTP_STATUS:
        return False
    headers = getattr(exc, "headers", None)
    if headers is None:
        return False
    if headers.get("Retry-After") is not None:
        return True
    remaining = headers.get("X-RateLimit-Remaining")
    return remaining is not None and remaining.strip() == "0"


def _retry_after_seconds(value: str) -> float | None:
    """Parse a ``Retry-After`` header value into a delay in seconds.

    Args:
        value: the header value, either a number of seconds or an HTTP date.

    Returns:
        The delay in seconds, or ``None`` if the value parses as neither form.
    """
    value = value.strip()
    try:
        return _usable_delay(float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)  # raises on a malformed date since 3.10
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:  # an HTTP date is GMT even when it omits the offset
        when = when.replace(tzinfo=UTC)
    return _usable_delay((when - datetime.now(UTC)).total_seconds())


def _usable_delay(seconds: float) -> float | None:
    """Return *seconds* if it is a delay worth waiting, else ``None``.

    ``Retry-After: nan`` parses as a float and would reach :func:`time.sleep`,
    which raises ``ValueError`` on a NaN -- from inside the retry handler, so the
    rate limit that sent the header would never be retried at all. A non-positive
    result is no guidance either: a date already in the past, or an
    ``X-RateLimit-Reset`` a host expressed as a delta rather than the epoch
    GitHub uses, would otherwise fire every remaining attempt back to back.
    """
    if not math.isfinite(seconds) or seconds <= 0.0:
        return None
    return seconds


def _server_requested_delay(exc: BaseException) -> float | None:
    """Return the delay *exc*'s host asked for before the next request.

    Args:
        exc: the exception raised by a download attempt.

    Returns:
        The requested delay in seconds, or ``None`` if the response carried no
        usable guidance.
    """
    if not isinstance(exc, HTTPError):
        return None
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None

    # ``Retry-After`` is defined for every status that can carry it -- 503 and 408
    # are its canonical uses, not just the rate limits -- so it is read whenever
    # the host sends one.
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        return _retry_after_seconds(retry_after)

    # ``X-RateLimit-Reset`` is different: GitHub attaches it to *every* response,
    # including ones nothing is limiting, and it points at the end of the current
    # window -- up to an hour out. Reading it off an unrelated 500 would turn a
    # one-second retry into the full clamp, so it speaks only for a response the
    # limit headers themselves mark as rate limited.
    if not _rate_limited(exc):
        return None
    reset = headers.get("X-RateLimit-Reset")  # an epoch timestamp, GitHub's form
    if reset is None:
        return None
    try:
        return _usable_delay(float(reset.strip()) - time.time())
    except ValueError:
        return None


def _retry_delay(exc: BaseException, attempt: int) -> float:
    """Return how long to wait before *attempt* + 1, honouring the server when it says.

    Exponential backoff is a guess; ``Retry-After`` and ``X-RateLimit-Reset`` are
    the host telling us when it will serve again. Retrying before then only
    spends another request against the same limit, and waiting far longer than
    asked wastes the job -- so the server's number wins where it is given,
    clamped to :data:`_MAX_BACKOFF_SECONDS`.

    Args:
        exc: the failure that is about to be retried.
        attempt: the 1-based number of the attempt that just failed.

    Returns:
        The delay in seconds.
    """
    requested = _server_requested_delay(exc)
    if requested is None:
        return _BACKOFF_SECONDS * 2 ** (attempt - 1)
    return min(requested, _MAX_BACKOFF_SECONDS)


def _is_transient(exc: BaseException) -> bool:
    """Return whether *exc* is a temporary network condition worth retrying.

    Rate limiting is the motivating case. An unauthenticated CI matrix fans many
    jobs out at once, which regularly trips the anonymous request limits of
    huggingface.co and github.com even though the checkpoint is served normally a
    moment later.

    Every :class:`~urllib.error.URLError` counts, ``exc.reason`` included: DNS
    resolution and "network is unreachable" fail this way and are as often a
    momentary condition as a permanent one. The cost of not separating them is
    that an offline run with a cold cache waits out its retries before reporting
    the same error, which the per-call sleep budget bounds.

    Args:
        exc: the exception raised by a download attempt.

    Returns:
        ``True`` if another attempt could plausibly succeed.
    """
    if isinstance(exc, HTTPError):  # a subclass of URLError, so it must be tested first
        return exc.code in _TRANSIENT_HTTP_STATUS or _rate_limited(exc)
    # IncompleteRead is a mid-transfer truncation of a chunked response. It
    # inherits from HTTPException alone -- neither URLError nor ConnectionError --
    # so it needs naming, or a truncation burns the fallback instead of retrying
    # the source that was already serving.
    return isinstance(exc, (URLError, TimeoutError, ConnectionError, http.client.IncompleteRead))


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
    # ``is not None``, not truthiness: this must agree with torch, which pins the
    # name whenever ``file_name`` is not None, and with the pin in
    # :func:`load_state_dict_from_url`, which decides on the same test.
    file_name = kwargs.get("file_name")
    filename = file_name if file_name is not None else os.path.basename(urlparse(url).path)
    return os.path.join(model_dir, filename)


def _prefetch_to_cache(url: str, kwargs: dict[str, Any]) -> bool:
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

    Returns:
        Whether a transfer happened. A cache hit returns ``False``, which is how
        :func:`load_state_dict_from_url` tells a source that was really fetched
        from one that was handed the file already on disk.
    """
    cached_file = _cached_file_path(url, kwargs)
    if os.path.exists(cached_file):
        return False

    os.makedirs(os.path.dirname(cached_file), exist_ok=True)

    hash_prefix = None
    if kwargs.get("check_hash"):
        match = torch.hub.HASH_REGEX.search(os.path.basename(cached_file))
        hash_prefix = match.group(1) if match else None

    # torch writes this to stdout; status output belongs on stderr.
    sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
    torch.hub.download_url_to_file(url, cached_file, hash_prefix, progress=kwargs.get("progress", True))
    return True


_DISCARDED_CACHE_PATHS: set[str] = set()
"""Cache paths already refetched once in this process; see :func:`_discard_cache_entry`."""

_QUARANTINE_SUFFIX = ".kornia-discarded"
"""Suffix a discarded cache entry is renamed with while its replacement is fetched.

Nothing ever reads a file under this name -- :func:`_prefetch_to_cache` looks
only at the real cache path -- so a process killed between the rename and
:func:`_settle_quarantine` leaves a stray file rather than a broken cache, and
the next discard of the same path overwrites it.
"""


def _discard_cache_entry(url: str, kwargs: dict[str, Any]) -> str | None:
    """Move the cache entry for *url* aside, at most once per path per process.

    Every URL in a fallback list resolves to the same path, because the cache
    filename is pinned to the primary URL (see :func:`load_state_dict_from_url`).
    A single bad write -- a truncated transfer, or an HTML rate-limit page served
    with a 200 -- would therefore make :func:`_prefetch_to_cache` short-circuit on
    each remaining source and hand torch the same broken file every time. The
    fallback URLs could never take effect, and because the bad file survives the
    process, the failure would repeat on every later run until the cache was
    cleared by hand.

    The entry is *renamed*, not deleted, because a load failure cannot tell a
    corrupt cache entry from a healthy one that failed for an unrelated reason --
    a bad ``map_location``, a ``weights_only`` rejection -- and a delete would
    sometimes destroy an intact checkpoint, several of which are over a gigabyte.
    Renaming makes the discard reversible; :func:`_settle_quarantine` owns the
    decision and documents it. Within the same directory the rename is a metadata
    operation, so the size of the checkpoint does not matter.

    Gating the discard on the exception type instead was considered and does not
    work. The types do not separate the two classes (torch 2.9.1): an HTML
    rate-limit page served with a 200 -- the case that motivated this function --
    and a ``weights_only`` rejection of a perfectly good file both raise
    ``UnpicklingError``, while a truncated zip checkpoint raises a bare
    ``RuntimeError``. A gate would discard the healthy entry it was meant to
    spare and keep the commonest corruption there is.

    What is bounded instead is the *refetch*: one per cache path per process,
    tracked in :data:`_DISCARDED_CACHE_PATHS`. Without that, a failure the cache
    cannot fix would re-download the checkpoint on every call, once per test that
    builds the model -- the download storm this module exists to prevent, reached
    from the other side.

    Neither the ledger nor the rename is synchronised: two threads loading the
    same checkpoint through one poisoned entry can have one of them find the path
    already moved aside and fail where a single thread would have recovered. The
    entry itself is never left in a worse state, and nothing in kornia loads one
    checkpoint concurrently, so a lock is not worth its deadlock surface here.

    Args:
        url: the URL whose cache entry should be moved aside.
        kwargs: the keyword arguments destined for the torch function; only
            ``model_dir`` and ``file_name`` are consulted.

    Returns:
        The path the entry was moved to, or ``None`` if there was nothing to move
        or the bound is already spent. The caller needs this to hand to
        :func:`_settle_quarantine`, and to know which source was denied a real
        fetch by the entry that is now out of the way.
    """
    path = _cached_file_path(url, kwargs)
    if path in _DISCARDED_CACHE_PATHS:
        return None

    quarantine = f"{path}{_QUARANTINE_SUFFIX}"
    try:
        os.replace(path, quarantine)
    except FileNotFoundError:
        # The transfer itself failed, so there is nothing to set aside -- and
        # nothing was refetched either, so the one allowed discard stays unspent.
        return None
    except OSError as e:
        # A read-only cache, or a Windows reader holding the file open. Renaming
        # it again would fail the same way, so the ledger is marked either way,
        # but a fallback that can never be reached is worth saying out loud.
        warnings.warn(
            f"Could not discard the cache entry at {path!r}: {e}. If that file is corrupt, "
            f"the fallback sources cannot take effect until it is deleted by hand.",
            stacklevel=3,
        )
        # The entry is still there and still unloadable. Marking the ledger stops
        # a rename that can only fail the same way from being tried again, and a
        # re-attempt would just reload the very file that failed.
        _DISCARDED_CACHE_PATHS.add(path)
        return None

    _DISCARDED_CACHE_PATHS.add(path)
    return quarantine


def _settle_quarantine(path: str, quarantine: str, *, loaded: bool, downloaded: bool) -> None:
    """Resolve a quarantined cache entry once every source has had its turn.

    The decision is the *outcome of the call*, never what happens to be sitting
    at *path*. Whether a file is there says only that some source wrote one, not
    that it is any good: a rate-limited mirror serving an HTML page with a 200
    puts a file there, and settling on its presence would drop the quarantined
    original in its favour -- destroying an intact multi-gigabyte checkpoint
    behind a ``map_location='cuda'`` failure on a CPU-only build, which is the
    very case renaming rather than deleting exists to survive.

    So:

    * *loaded*: whatever is at *path* is what just loaded, so the quarantined
      copy is the one that failed and is dropped.
    * not *loaded*: everything tried failed, including anything a source wrote,
      so nothing on disk is known-good and the pre-call state is the safest one
      to leave behind. The original is moved back over whatever is there, and
      the caller ends the call with the cache it started with.

    The bound in :data:`_DISCARDED_CACHE_PATHS` counts *successful refetches*, so
    a failed call releases it again unless a source really did transfer the file.
    Otherwise the first offline call in a process would spend the single allowed
    discard on a path it could not repair, and a later call -- with the network
    back -- could never clear a genuinely poisoned entry.

    Args:
        path: the cache path the entry was moved out of.
        quarantine: the path :func:`_discard_cache_entry` moved it to.
        loaded: whether the call is returning a state dict.
        downloaded: whether any source transferred the file during the call.
    """
    if loaded:
        try:
            os.remove(quarantine)
        except OSError as e:  # pragma: no cover - a cache that cannot be written to
            warnings.warn(f"Could not remove the discarded cache entry at {quarantine!r}: {e}.", stacklevel=3)
        return

    try:
        os.replace(quarantine, path)  # atomically overwrites whatever a source left there
    except OSError as e:  # pragma: no cover - a cache that cannot be written to
        warnings.warn(
            f"Could not restore the cache entry at {path!r} from {quarantine!r}: {e}. "
            f"Move it back by hand to avoid re-downloading the checkpoint.",
            stacklevel=3,
        )
        return
    if not downloaded:
        _DISCARDED_CACHE_PATHS.discard(path)


def _prefetch_with_retry(url: str, kwargs: dict[str, Any], budget: _SleepBudget) -> bool:
    """Run :func:`_prefetch_to_cache`, retrying transient failures with backoff.

    Args:
        url: the URL to fetch.
        kwargs: the keyword arguments destined for the torch function.
        budget: the waiting time the whole call has left; a retry that would
            exceed it is not taken.

    Returns:
        Whether a transfer happened; see :func:`_prefetch_to_cache`.

    Raises:
        Exception: the last failure, once the attempts are exhausted, the failure
            is not transient, or the call has no waiting time left.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return _prefetch_to_cache(url, kwargs)
        except Exception as e:
            if attempt == _MAX_ATTEMPTS or not _is_transient(e):
                raise
            delay = _retry_delay(e, attempt)
            if delay > budget.remaining:
                raise
            warnings.warn(
                f"Transient failure fetching {url!r}: {e}. Retrying in {delay:.0f}s "
                f"(attempt {attempt + 1} of {_MAX_ATTEMPTS}).",
                # 1 = here, 2 = load_state_dict_from_url, 3 = its caller, which is
                # the frame the sibling warning below also points at.
                stacklevel=3,
            )
            budget.sleep(delay)
    raise AssertionError("unreachable: the loop above always returns or raises")  # pragma: no cover


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

    Because that one path is shared, a failed attempt moves the cache entry aside
    before the next URL is tried. Otherwise a single bad write would be handed
    straight back to torch by every remaining source -- and by every later
    process -- making the fallback URLs unreachable. If no source ends up
    transferring the file, the source the entry was moved aside for is tried once
    more, this time against an empty path, so a poisoned entry is repaired inside
    the call rather than after one guaranteed spurious failure -- which matters
    most for the single-source checkpoints, where there is no fallback to do it.
    Should the call still fail, the entry is put back, so a failure the cache
    could not have fixed never costs the caller a checkpoint it already had. A
    path is refetched at most once per process this way; see
    :func:`_discard_cache_entry` and :func:`_settle_quarantine`.

    Each URL is attempted up to :data:`_MAX_ATTEMPTS` times, with exponential
    backoff, when it fails with a transient network condition such as an HTTP
    429. Unauthenticated CI matrices trip the rate limits of huggingface.co and
    github.com routinely, and a retry is far cheaper than a failed job. A
    response that names its own delay in ``Retry-After`` -- or, on a rate-limit
    response, ``X-RateLimit-Reset`` -- is honoured instead of the guess, clamped
    to :data:`_MAX_BACKOFF_SECONDS`, and the call as a whole never waits longer
    than :data:`_MAX_CALL_SLEEP_SECONDS`.

    Args:
        url: a URL string, or a list of URL strings tried left-to-right.
        **kwargs: forwarded verbatim to
            :func:`torch.hub.load_state_dict_from_url`
            (``map_location``, ``check_hash``, ``file_name``, …).

    Returns:
        The loaded state dict.

    Raises:
        RuntimeError: if every URL fails. The message carries the last
            exception's type and text, the source it came from and the cache path in play, and the
            exception itself is chained; without them a rate limit is
            indistinguishable from a dead link in a CI failure summary, and a
            caller stuck behind a corrupt entry has no file to delete.

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
    # ``is None`` rather than ``not in``: an explicit ``file_name=None`` means
    # "use the basename", which for a list of URLs is a *different* path per
    # source, while the quarantine below can only cover one. Pinning on the same
    # test :func:`_cached_file_path` uses is what makes the one-path claim true.
    if len(urls) > 1 and kwargs.get("file_name") is None:
        kwargs["file_name"] = Path(urlparse(urls[0]).path).name

    # Every URL resolves to this one path, so a single quarantine covers the call.
    cache_path = _cached_file_path(urls[0], kwargs)
    budget = _SleepBudget(_MAX_CALL_SLEEP_SECONDS)
    quarantine: str | None = None
    discarded_url: str | None = None
    downloaded = False
    last_exc: Exception | None = None
    last_url: str | None = None
    sources = urls
    re_attempted = False
    try:
        while True:
            for i, u in enumerate(sources):
                try:
                    # Populate the cache ourselves so torch's stdout line is never reached.
                    downloaded |= _prefetch_with_retry(u, kwargs, budget)
                    state_dict = torch.hub.load_state_dict_from_url(u, **kwargs)
                except Exception as e:  # noqa: BLE001
                    last_exc, last_url = e, u
                    # Whatever is in the cache is not loadable, and every remaining
                    # URL shares its path. Move it aside so the next source is really
                    # fetched; it comes back below if none of them manages that.
                    moved = _discard_cache_entry(u, kwargs)
                    if moved is not None:
                        quarantine, discarded_url = moved, u
                    if i < len(sources) - 1:
                        warnings.warn(
                            f"Failed to load weights from {u!r}: {e}. Trying next source.",
                            stacklevel=2,
                        )
                    continue
                if quarantine is not None:
                    _settle_quarantine(cache_path, quarantine, loaded=True, downloaded=downloaded)
                    quarantine = None
                return state_dict

            # A discard that nothing refetched is pure loss: the source it fired
            # for was handed the bad file instead of being fetched, and no later
            # source replaced it either -- the last URL of a list has nothing
            # after it, and a mirror that is dead or offline writes nothing. Give
            # that one source the fetch it never got. The ledger is already spent
            # on this path, so this stays one extra pass per path per process, and
            # a poisoned entry recovers inside the call rather than after one
            # guaranteed spurious failure.
            if re_attempted or discarded_url is None or downloaded:
                break
            sources = [discarded_url]
            re_attempted = True
    finally:
        if quarantine is not None:
            _settle_quarantine(cache_path, quarantine, loaded=False, downloaded=downloaded)

    raise RuntimeError(
        f"Failed to load weights from all {len(urls)} source(s). "
        f"Last URL tried: {last_url!r}. "
        f"Last error: {type(last_exc).__name__}: {last_exc}. "
        f"Cache path: {cache_path!r} -- delete it if it is corrupt and this repeats."
    ) from last_exc
