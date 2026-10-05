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

import contextlib
import errno
import hashlib
import http.client
import inspect
import math
import os
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import torch

_HF_BASE = "https://huggingface.co"

_HF_KORNIA_ORG = "kornia"


def _hf_repo_id(repo: str) -> str:
    """Return the full ``owner/name`` id *repo* names.

    A repository *name* cannot contain a ``/``, so one marks a spelling that
    already carries its owner; anything else is a repository in the ``kornia``
    org. Both spellings of the same kornia repo must resolve identically, or the
    URL and the cache name derived from them could disagree.

    Example:
        >>> _hf_repo_id("hardnet"), _hf_repo_id("kornia/hardnet")
        ('kornia/hardnet', 'kornia/hardnet')
    """
    return repo if "/" in repo else f"{_HF_KORNIA_ORG}/{repo}"


def hf_url(repo: str, filename: str) -> str:
    """Return the HuggingFace URL for a file in a model repo.

    Both arguments are percent-encoded, ``/`` excepted, because they are path
    segments rather than URL syntax: unencoded, the ``#`` of ``"a#b.pth"`` would
    start a fragment and fetch the file ``a``, and a space would make the URL
    invalid. Names made of letters, digits, ``-``, ``_``, ``.`` and ``~`` are
    unchanged. Pass the plain name: one that is already percent-encoded, such as
    ``"a%20b.pth"``, is encoded again.

    Args:
        repo: repository name under the ``kornia`` HF org (e.g. ``"hardnet"``),
            or a full ``owner/name`` repository id for a repo owned by anyone
            else (e.g. ``"google/siglip2-base-patch16-224"``). The two are told
            apart by the ``/``, which a repository *name* cannot contain.
        filename: path of the file in that repo (e.g. ``"HardNetPP.pth"``); a
            ``/`` in it reaches into a subdirectory.

    Returns:
        A ``resolve/main`` URL that can be passed directly to
        :func:`load_state_dict_from_url` or :func:`download_file_from_url`.

    Example:
        >>> hf_url("hardnet", "HardNetPP.pth")
        'https://huggingface.co/kornia/hardnet/resolve/main/HardNetPP.pth'
        >>> hf_url("google/siglip2-base-patch16-224", "model.safetensors")
        'https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/model.safetensors'
    """
    return f"{_HF_BASE}/{quote(_hf_repo_id(repo), safe='/')}/resolve/main/{quote(filename, safe='/')}"


def _hf_cache_file_name(repo: str, filename: str) -> str:
    """Return a cache filename that cannot collide with another repo's.

    The download cache is one flat directory keyed by filename, which is fine
    while every checkpoint has a name of its own -- and wrong the moment two
    repositories publish the same one. Every safetensors repository on the Hub
    calls its single-shard checkpoint ``model.safetensors``, so caching by the
    URL basename would hand the second model whichever one was fetched first,
    silently and forever.

    The repository id is folded into the name with ``--`` for the same reason
    the Hub's own cache layout does: it is not a path separator, so the result
    stays one filename in one directory.

    Args:
        repo: the repository, in either spelling :func:`hf_url` accepts. Both
            resolve to the same cache name, because they resolve to the same URL.
        filename: the file's name within that repository.

    Returns:
        The cache filename to pass as ``file_name``.

    Example:
        >>> _hf_cache_file_name("kornia/kimi-vl-a3b-instruct-vision", "model.safetensors")
        'kornia--kimi-vl-a3b-instruct-vision--model.safetensors'
        >>> _hf_cache_file_name("hardnet", "HardNetPP.pth")
        'kornia--hardnet--HardNetPP.pth'
    """
    return f"{_hf_repo_id(repo).replace('/', '--')}--{filename}"


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

_DOWNLOAD_TIMEOUT_SECONDS = 30.0
"""Default bound, in seconds, on each wait of a transfer: connecting, and every read.

The fallback when neither ``timeout=`` nor :data:`_DOWNLOAD_TIMEOUT_ENV_VAR` is set.

A server that accepts the connection and then sends nothing -- overloaded, or a
half-open connection -- would otherwise hold the call forever, and the retry and
fallback logic never got control, because nothing raised. The bound applies to
each wait, not to the transfer as a whole: a multi-gigabyte checkpoint on a slow
but live link takes as long as it takes. A timeout is a transient failure, so it
is retried like a 503 and then hands over to the next source; a source that never
answers therefore costs up to :data:`_MAX_ATTEMPTS` times this, plus the backoff.
The public functions take ``timeout=`` to override it for one call.
"""

_DOWNLOAD_TIMEOUT_ENV_VAR = "KORNIA_DOWNLOAD_TIMEOUT"
"""Environment variable holding the default download timeout, in seconds.

Read at call time, whenever a download function is called without ``timeout=``, so
it reaches callers that take no timeout -- a ``pretrained=True`` constructor -- and
can be changed without restarting the process. Unset or blank means 30 s; anything
that :func:`_usable_timeout` refuses raises :class:`ValueError` naming the
variable, at the call rather than at import.
"""


def _usable_timeout(seconds: float) -> bool:
    """Return whether *seconds* is more than 0 and at most ``threading.TIMEOUT_MAX``.

    ``threading.TIMEOUT_MAX`` is the stdlib's documented ceiling on a blocking timeout. A
    finite value far beyond it, such as ``1e300``, makes the socket raise ``OverflowError``
    once the transfer starts, which the retry logic would then report as a failed source.
    The bound does not promise that every value up to it is honoured: some platforms
    truncate a wait longer than about 49 days (2**32 milliseconds), so such a timeout can
    expire much sooner than asked. NaN and infinity fail the comparison, and an ``int`` too
    large for a ``float`` is compared exactly rather than converted.
    """
    return 0 < seconds <= threading.TIMEOUT_MAX


_READ_CHUNK_BYTES = 128 * 1024
"""Bytes read per call during a transfer, the chunk :func:`torch.hub.download_url_to_file` uses."""


class _TruncatedTransfer(http.client.IncompleteRead):
    """A response body that ended before the ``Content-Length`` its headers announced.

    urllib ends a ``read(n)`` loop on an early close without an error, so without this
    check a server that dropped the connection part-way left a short file in the cache,
    returned as a hit by every later call (the root cause of #4309). It is an
    :class:`~http.client.IncompleteRead`, which :func:`_is_transient` retries before the
    next source is tried. Its message counts the bytes on disk, which the parent class
    would report as 0 because it counts only a partial body held in memory, and reads
    as a sentence, because the failure summaries print it without the class name (see
    :func:`_describe`).
    """

    def __init__(self, received: int, announced: int) -> None:
        super().__init__(b"", announced - received)
        # ``args`` is what pickle and copy rebuild an exception from.
        self.args = (received, announced)
        self.received = received
        self.announced = announced

    def __repr__(self) -> str:
        return (
            f"transfer truncated: the server sent {self.received} of the {self.announced} bytes "
            f"its Content-Length announced"
        )


def _describe(exc: BaseException) -> str:
    """Return ``"<type>: <message>"`` for a failure summary; a truncated transfer describes itself."""
    if isinstance(exc, _TruncatedTransfer):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _resolve_timeout(timeout: float | None) -> float:
    """Return the timeout a call runs with, validated.

    Args:
        timeout: the caller's value, or ``None`` for the ``KORNIA_DOWNLOAD_TIMEOUT``
            environment variable, or 30 s when that is unset or blank. An explicit value
            wins, and the variable is then not read at all.

    Returns:
        The timeout in seconds.

    Raises:
        TypeError: if *timeout* is not a number.
        ValueError: if *timeout*, or the variable it falls back to, is not greater than
            0 and at most ``threading.TIMEOUT_MAX`` (see :func:`_usable_timeout`); the
            message names which of the two it came from.
    """
    if timeout is None:
        raw = os.environ.get(_DOWNLOAD_TIMEOUT_ENV_VAR, "").strip()
        if not raw:
            return _DOWNLOAD_TIMEOUT_SECONDS
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not _usable_timeout(value):
            raise ValueError(
                f"{_DOWNLOAD_TIMEOUT_ENV_VAR} must be a number of seconds greater than 0 and at most "
                f"{threading.TIMEOUT_MAX!r} (threading.TIMEOUT_MAX), got {raw!r}. "
                f"Unset it to use the default of {_DOWNLOAD_TIMEOUT_SECONDS:g} s."
            )
        return value
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"timeout must be a number of seconds or None, got {type(timeout).__name__}.")
    if not _usable_timeout(timeout):
        # An int too large for a float could also be too long to print.
        huge = isinstance(timeout, int) and timeout.bit_length() > 64
        shown = f"an int of {timeout.bit_length()} bits" if huge else repr(timeout)
        raise ValueError(
            f"timeout must be a number of seconds greater than 0 and at most {threading.TIMEOUT_MAX!r} "
            f"(threading.TIMEOUT_MAX), got {shown}."
        )
    return float(timeout)


def _is_timeout(exc: BaseException) -> bool:
    """Return whether *exc* is a socket timeout, raised directly or wrapped by urllib."""
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, URLError) and not isinstance(exc, HTTPError) and isinstance(exc.reason, TimeoutError)


def _warn(message: str) -> None:
    """Warn, attributed to the first frame outside this module.

    The public functions call each other -- :func:`download_hf_file` goes through
    :func:`download_file_from_url` -- so a fixed ``stacklevel`` that is right for one
    entry point points into this file for another. Walking out of the module instead
    names the caller's line whichever way it came in.

    Args:
        message: the warning text.
    """
    level = 1
    frame = inspect.currentframe()
    while frame is not None and frame.f_globals is globals():
        frame = frame.f_back
        level += 1
    warnings.warn(message, stacklevel=level)


def _url_list(url: object) -> list[str]:
    """Return *url* as a non-empty list of URL strings, or raise naming what is wrong.

    Args:
        url: what the caller passed as ``url``.

    Returns:
        The URLs, in order.

    Raises:
        TypeError: if *url* is not a string or an iterable of strings.
        ValueError: if there is no URL, or one of them is empty or has no scheme.
    """
    expected = "url must be a URL string or a list of them"
    if isinstance(url, (str, bytes, os.PathLike)):
        urls: list[object] = [url]
    else:
        try:
            urls = list(url)
        except TypeError:
            raise TypeError(f"{expected}, got {type(url).__name__}.") from None
        if not urls:
            raise ValueError("url is an empty list; pass at least one URL.")
    checked: list[str] = []
    for u in urls:
        if isinstance(u, os.PathLike):
            raise TypeError(f"{expected}, got a {type(u).__name__}; for a local file pass its path.as_uri().")
        if not isinstance(u, str):
            raise TypeError(f"{expected}, got {type(u).__name__}.")
        if not u:
            raise ValueError(f"url must not be empty, got {url!r}.")
        # The cache is looked up by the URL's base name before anything is
        # fetched, so a local path here would load whichever cached file shares
        # its name. A one-letter scheme is a Windows drive letter, not a URL.
        if len(urlparse(u).scheme) <= 1:
            raise ValueError(f"url must be a URL with a scheme, got {u!r}; to load a local file use torch.load.")
        checked.append(u)
    return checked


_NOT_A_FILE_NAME = frozenset({"", ".", ".."})
"""Names that would make the cache path the cache directory itself, or its parent."""


def _check_cache_file_name(urls: list[str], file_name: str | os.PathLike[str] | None) -> str | None:
    """Refuse a cache entry name that is not a single file inside the cache directory.

    The cache is one flat directory and the entry is ``<model_dir>/<name>``. A
    ``file_name`` carrying a separator, an absolute path or ``..`` would put it
    outside, where the cache repair of :func:`load_state_dict_from_url` would move
    a file it never wrote aside, write over it and delete the original. A URL whose
    path ends in ``/``, ``/.`` or ``/..`` names no file, and the name derived from it
    would make the entry the cache directory or its parent. The name derived from a
    URL is held to the same bare-name test as ``file_name``: on Windows a last
    segment such as ``d:x.pth`` is a drive-relative path, not a name in the cache.

    Args:
        urls: the URLs of the call; without ``file_name`` the first one names the entry.
        file_name: the caller's ``file_name``, a ``str`` or an :class:`os.PathLike`, or ``None``.

    Returns:
        ``file_name`` as a ``str``, or ``None`` when it was not given.

    Raises:
        TypeError: if ``file_name`` is neither a ``str`` nor a path-like naming one.
        ValueError: if the name is not a bare file name.
    """
    if file_name is not None:
        name = os.fspath(file_name)
        if not isinstance(name, str):
            raise TypeError(f"file_name must be a str or a path-like object, got {type(file_name).__name__}.")
        if name in _NOT_A_FILE_NAME or os.path.basename(name) != name:
            raise ValueError(f"file_name must be a bare filename inside the cache directory, got {file_name!r}.")
        return name
    name = os.path.basename(urlparse(urls[0]).path)
    if name in _NOT_A_FILE_NAME or os.path.basename(name) != name:
        raise ValueError(
            f"{urls[0]!r} does not end in a name that can be a file in the cache directory (its last path "
            f"segment is {name!r}); pass file_name= to name the cache entry."
        )
    return None


def _expand_model_dir(model_dir: Any) -> Any:
    """Expand a leading ``~`` in *model_dir*, as a shell would; ``None`` passes through.

    Neither the ``makedirs`` nor the existence check expands it, so ``"~/weights"``
    used to create a directory literally named ``~`` in the working directory.
    """
    if model_dir is None:
        return None
    expanded = os.path.expanduser(model_dir)
    return model_dir if expanded == os.fspath(model_dir) else expanded


def _check_torch_keywords(kwargs: dict[str, Any]) -> None:
    """Raise :class:`TypeError` for a keyword :func:`torch.hub.load_state_dict_from_url` does not take.

    torch raises the same error, but only when it is called -- after the cache has been
    consulted -- so the call's cache repair took a caller's typo for a corrupt entry: it
    set the entry aside, downloaded the checkpoint again and reported ``RuntimeError``.

    Args:
        kwargs: the keyword arguments destined for the torch function.

    Raises:
        TypeError: naming the first keyword torch does not accept.
    """
    try:
        parameters = inspect.signature(torch.hub.load_state_dict_from_url).parameters
    except (TypeError, ValueError):  # pragma: no cover - a signature inspect cannot read
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return  # a stand-in that takes anything, such as a test double
    accepted = {
        name
        for name, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and name != "url"
    }
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise TypeError(
            f"load_state_dict_from_url() got an unexpected keyword argument {unknown[0]!r}. "
            f"It accepts timeout and the keywords of torch.hub.load_state_dict_from_url: {', '.join(sorted(accepted))}."
        )


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
    # the host sends one. A header that does not parse, or whose date has already
    # passed, is no guidance at all, so it falls through to the branch below
    # rather than answering ``None`` for the whole function.
    retry_after = headers.get("Retry-After")
    if retry_after is not None and (delay := _retry_after_seconds(retry_after)) is not None:
        return delay

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
    # IncompleteRead is a truncated body: http.client raises it when a chunked
    # response is cut mid-chunk, and _download_url_to_file raises its subclass
    # _TruncatedTransfer when a body ends short of its Content-Length. It inherits
    # from HTTPException alone -- neither URLError nor ConnectionError -- so it needs
    # naming, or a truncation burns the fallback instead of retrying the source
    # that was already serving.
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


def _download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: str | None = None,
    progress: bool = True,
    timeout: float | None = None,
) -> None:
    """Download *url* to *dst*, failing an attempt that stalls for *timeout* seconds.

    :func:`torch.hub.download_url_to_file` with a timeout. torch's function opens
    the URL with no timeout and has no parameter for one, so a server that accepted
    the connection and then went quiet held the call forever. Giving it one from
    outside would mean replacing ``torch.hub.urlopen`` or the process-wide socket
    default, which changes torch or every other thread for everyone, so kornia runs
    the transfer itself. Everything else follows torch: the ``User-Agent``, the chunk
    size, the progress bar (torch's own ``tqdm``, or its stand-in without tqdm), the
    hash check and its message, and writing to a temporary file beside *dst* that
    replaces it only once complete, so a failed or interrupted transfer never leaves
    a partial file at *dst*. Unlike torch, a body shorter than the ``Content-Length``
    that framed it counts as failed rather than complete; a chunked or close-delimited
    body, or one whose header http.client ignores, is taken as torch takes it.

    Args:
        url: the URL to fetch; ``file://`` URLs work too.
        dst: the destination path; a leading ``~`` is expanded.
        hash_prefix: if given, the SHA256 of the download must start with it.
        progress: whether to show a progress bar on stderr.
        timeout: the bound on connecting and on each read, in seconds; ``None``
            uses the ``KORNIA_DOWNLOAD_TIMEOUT`` environment variable, or 30 s. It
            bounds each wait, not the whole transfer.

    Raises:
        TimeoutError: if the server went *timeout* seconds without answering.
        http.client.IncompleteRead: if the body ended before its ``Content-Length``.
        RuntimeError: if the download does not match *hash_prefix*.
    """
    timeout = _resolve_timeout(timeout)
    dst = os.path.expanduser(dst)
    for _ in range(tempfile.TMP_MAX):
        partial = f"{dst}.{uuid.uuid4().hex}.partial"
        try:
            # Opened outside ``with`` so a taken name can be retried; ``with f:`` below closes it.
            f = open(partial, "xb")  # noqa: SIM115
        except FileExistsError:
            continue
        break
    else:  # pragma: no cover - every uuid4 name already taken
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")

    sha256 = hashlib.sha256() if hash_prefix is not None else None
    bar_type = getattr(torch.hub, "tqdm", None)  # tqdm, or torch's stand-in when it is absent
    try:
        with f:
            # ``file://`` is deliberately accepted, as torch accepts it.
            request = Request(url, headers={"User-Agent": "torch.hub"})  # noqa: S310
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                # The length http.client frames the body by: None for a chunked body and for
                # a Content-Length that is missing, negative or not an integer, 0 for a
                # bodiless status. A ``file://`` response has no such attribute.
                framed = getattr(response, "length", None)
                lengths = response.info().get_all("Content-Length")
                total = int(lengths[0]) if lengths else None
                bar = (
                    bar_type(total=total, disable=not progress, unit="B", unit_scale=True, unit_divisor=1024)
                    if bar_type is not None
                    else contextlib.nullcontext()
                )
                received = 0
                with bar:
                    while chunk := response.read(_READ_CHUNK_BYTES):
                        f.write(chunk)
                        received += len(chunk)
                        if sha256 is not None:
                            sha256.update(chunk)
                        if bar_type is not None:
                            bar.update(len(chunk))
        # Checked only when Content-Length framed the body, as torch has no check at all
        # elsewhere; http.client clips a longer body there, so only a short one can occur.
        # urllib does not decode a Content-Encoding, so the raw byte count is what it counts.
        if isinstance(framed, int) and received < framed:
            raise _TruncatedTransfer(received, framed)
        if sha256 is not None and hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{digest}")')
        os.replace(partial, dst)
    except (TimeoutError, URLError) as e:
        if not _is_timeout(e):
            raise
        raise TimeoutError(
            f"the server went {timeout:g} s without answering (the download timeout, which bounds connecting "
            f"and each read). If the link is slow rather than stalled, pass a larger timeout= to "
            f"load_state_dict_from_url, download_file_from_url or download_hf_file, or set the "
            f"{_DOWNLOAD_TIMEOUT_ENV_VAR} environment variable (in seconds), which also reaches callers "
            f"that take no timeout, such as pretrained model constructors"
        ) from e
    finally:
        try:
            os.remove(partial)
        except FileNotFoundError:
            pass
        except OSError as e:
            # Another process (an antivirus scanner on Windows, say) can still hold the file open. Raising here
            # would replace the error that ended the transfer, which decides whether the download is retried.
            _warn(
                f"Could not remove the temporary download file {partial}: {e}. "
                f"Delete it by hand once no other process holds it."
            )


def _prefetch_to_cache(url: str, kwargs: dict[str, Any], timeout: float) -> bool:
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
        timeout: the bound on connecting and on each read, in seconds, already
            resolved by :func:`_resolve_timeout`.

    Returns:
        Whether a transfer happened. A cache hit returns ``False``, which is how
        :func:`load_state_dict_from_url` tells a source that was really fetched
        from one that was handed the file already on disk.

    Raises:
        TimeoutError: if the server stalled for *timeout* seconds; see
            :func:`_download_url_to_file`, which leaves nothing at the cache path.
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
    _download_url_to_file(url, cached_file, hash_prefix, progress=kwargs.get("progress", True), timeout=timeout)
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
        _warn(
            f"Could not discard the cache entry at {path}: {e}. If that file is corrupt, "
            f"the fallback sources cannot take effect until it is deleted by hand."
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
            _warn(f"Could not remove the discarded cache entry at {quarantine}: {e}.")
        return

    try:
        os.replace(quarantine, path)  # atomically overwrites whatever a source left there
    except OSError as e:  # pragma: no cover - a cache that cannot be written to
        _warn(
            f"Could not restore the cache entry at {path} from {quarantine}: {e}. "
            f"Move it back by hand to avoid re-downloading the checkpoint."
        )
        return
    if not downloaded:
        _DISCARDED_CACHE_PATHS.discard(path)


def _drop_failed_download(path: str) -> None:
    """Delete bytes the current source transferred and then failed to load.

    Nothing is quarantined here: the file did not exist before this source wrote
    it -- :func:`_prefetch_to_cache` only transfers into an empty path -- so there
    is no earlier state to preserve and nothing to weigh it against. Setting it
    aside instead would put it *back* in :func:`_settle_quarantine`, leaving the
    caller a poisoned entry where the call found none, with the one allowed
    discard spent on it, so no later call in the process could clear it. Deleting
    returns the path to the state the call found, which is also the empty path the
    next source needs in order to be fetched at all.

    :func:`load_state_dict_from_url` calls this only while a later source remains,
    because there a load failure does not establish that the bytes are bad -- a
    ``map_location`` a build cannot satisfy fails an intact checkpoint -- so the
    final source keeps what it wrote. :func:`download_file_from_url` calls it for
    every ``validate`` rejection of a fresh transfer, last source or not, because
    that rejection *is* a verdict on the file.

    Args:
        path: the cache path this source just wrote.
    """
    try:
        os.remove(path)
    except FileNotFoundError:  # pragma: no cover - torch removed it itself
        pass
    except OSError as e:  # pragma: no cover - a cache that cannot be written to
        _warn(
            f"Could not remove the failed download at {path}: {e}. "
            f"Delete it by hand if the fallback sources stop taking effect."
        )


def _prefetch_with_retry(url: str, kwargs: dict[str, Any], budget: _SleepBudget, timeout: float) -> bool:
    """Run :func:`_prefetch_to_cache`, retrying transient failures with backoff.

    Args:
        url: the URL to fetch.
        kwargs: the keyword arguments destined for the torch function.
        budget: the waiting time the whole call has left; a retry that would
            exceed it is not taken.
        timeout: forwarded to :func:`_prefetch_to_cache`.

    Returns:
        Whether a transfer happened; see :func:`_prefetch_to_cache`.

    Raises:
        Exception: the last failure, once the attempts are exhausted, the failure
            is not transient, or the call has no waiting time left.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return _prefetch_to_cache(url, kwargs, timeout)
        except Exception as e:
            if attempt == _MAX_ATTEMPTS or not _is_transient(e):
                raise
            delay = _retry_delay(e, attempt)
            if delay > budget.remaining:
                raise
            _warn(
                f"Transient failure fetching {url!r}: {e}. Retrying in {delay:.0f}s "
                f"(attempt {attempt + 1} of {_MAX_ATTEMPTS})."
            )
            budget.sleep(delay)
    raise AssertionError("unreachable: the loop above always returns or raises")  # pragma: no cover


def load_state_dict_from_url(url: str | list[str], *, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
    """Load a state dict from a URL, trying fallback URLs on failure.

    Replacement for :func:`torch.hub.load_state_dict_from_url` that also accepts
    an ordered list of URLs. Each URL is tried in turn; a :mod:`warnings` message
    is emitted for every failed attempt before the next source is tried. It
    deliberately differs from the torch function in two ways, both described
    below: the ``weights_only`` default and where progress is reported.

    The checkpoint is loaded with ``weights_only=True`` unless the caller passes
    ``weights_only=False``, whereas the torch function defaults to ``False`` on
    every torch version kornia supports; ``weights_only=True`` is
    ``torch.load``'s own default since torch 2.6. ``torch.load`` then unpickles
    only tensors, primitive types and plain containers, and refuses a pickled
    callable instead of running it. A checkpoint that stores any other type
    fails with a :class:`RuntimeError` chained to the
    :class:`pickle.UnpicklingError` that names the type. Allowlist the type for
    the call with ``torch.serialization.safe_globals([...])``, or pass
    ``weights_only=False``, but only for a file you trust, because unpickling it
    can run arbitrary code.

    On older torch, ``weights_only=True`` narrows what a checkpoint can do but
    does not guarantee that it runs no code. PyTorch's advisories report
    checkpoints crafted to run code despite it before torch 2.6
    (`GHSA-53q9-r3pm-6pq6
    <https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6>`__)
    and to corrupt memory, potentially running code, before torch 2.10
    (`GHSA-63cw-57p8-fm3p
    <https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p>`__).
    kornia supports torch 2.5.1 and later, so load a checkpoint from a source
    you do not trust only on torch 2.10 or later, which fixes both.

    Progress reporting is written to :data:`sys.stderr`. This is the second
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

    Only an entry that *predates* the call is ever quarantined. Bytes a source
    transferred itself and then could not load are not something the caller had,
    so there is nothing to make reversible: they are deleted outright while a
    later source remains, which is what hands that source an empty path to fetch
    into (see :func:`_drop_failed_download`). After the last source they are left
    where they are -- deleting them would make every later call in the process
    transfer the file again, unbounded and up to 2.4 GB a time, to fail in the
    same way -- but no discard is spent on them either, so the next call can still
    move them aside and reach the sources behind them.

    Each URL is attempted up to :data:`_MAX_ATTEMPTS` times, with exponential
    backoff, when it fails with a transient network condition such as an HTTP
    429. Unauthenticated CI matrices trip the rate limits of huggingface.co and
    github.com routinely, and a retry is far cheaper than a failed job. A
    response that names its own delay in ``Retry-After`` -- or, on a rate-limit
    response, ``X-RateLimit-Reset`` -- is honoured instead of the guess, clamped
    to :data:`_MAX_BACKOFF_SECONDS`, and the call as a whole never waits longer
    than :data:`_MAX_CALL_SLEEP_SECONDS`. A server that stalls -- accepts the
    connection and then sends nothing -- fails the attempt after ``timeout``
    seconds and counts as transient too.

    The arguments are checked before the cache is consulted or a request made:
    ``file_name`` must be a bare file name, as it names an entry in the flat cache
    directory, and without it the first URL's path must end in one; a keyword
    the torch function does not take raises :class:`TypeError` there and then,
    rather than being mistaken for a corrupt cache entry. A leading ``~`` in
    ``model_dir`` is expanded.

    Args:
        url: a URL string, or a list of URL strings tried left-to-right.
        timeout: seconds a connection attempt or a single read may stall before the
            attempt fails; it bounds each wait, not the whole transfer. ``None``
            uses the ``KORNIA_DOWNLOAD_TIMEOUT`` environment variable, read at
            each call, or 30 s when it is unset. The variable is how to raise the
            bound for callers that take no ``timeout``, such as ``pretrained=True``
            model constructors.
        **kwargs: forwarded to :func:`torch.hub.load_state_dict_from_url`
            (``map_location``, ``check_hash``, ``file_name``, …), with
            ``weights_only`` set to ``True`` unless it is passed as ``False``.

    Returns:
        The loaded state dict.

    Raises:
        TypeError: if ``url`` is not a URL string or a list of them, ``timeout``
            is not a number, or a keyword is not one the torch function takes.
        ValueError: if ``url`` is empty or has no scheme (a local path is not a
            URL); if ``timeout`` is not greater than 0 and
            at most ``threading.TIMEOUT_MAX``, or it is omitted and
            ``KORNIA_DOWNLOAD_TIMEOUT`` holds such a value or no number (the message
            names which); if ``file_name`` is not a bare file name, or it is omitted
            and the first URL does not end in one.
        RuntimeError: if every URL fails. The message carries the failing
            exception's type and text, the source it came from and the cache path in play, and the
            exception itself is chained; without them a rate limit is
            indistinguishable from a dead link in a CI failure summary, and a
            caller stuck behind a corrupt entry has no file to delete. Where a
            load failure set the cache entry aside and the refetch then failed
            too, the load failure is the one reported and chained, and the
            refetch failure rides along as context.

    Example:
        >>> sd = load_state_dict_from_url([          # doctest: +SKIP
        ...     hf_url("hardnet", "HardNetPP.pth"),  # primary  (HF mirror)
        ...     "https://github.com/DagnyT/hardnet/raw/master/"
        ...     "pretrained/pretrained_all_datasets/HardNet%2B%2B.pth",  # fallback
        ... ])
    """
    urls = _url_list(url)
    timeout = _resolve_timeout(timeout)
    _check_torch_keywords(kwargs)
    file_name = _check_cache_file_name(urls, kwargs.get("file_name"))
    if file_name is not None:
        kwargs["file_name"] = file_name
    if kwargs.get("model_dir") is not None:
        kwargs["model_dir"] = _expand_model_dir(kwargs["model_dir"])

    # The one torch call below loads from every source, the fallbacks and the
    # refetch after a quarantine alike, so setting this once covers them all.
    # ``None`` is included because torch 2.5 reads it as ``False``.
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = True

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
    discard_exc: Exception | None = None
    downloaded = False
    last_exc: Exception | None = None
    last_url: str | None = None
    sources = urls
    re_attempted = False
    try:
        while True:
            for i, u in enumerate(sources):
                fetched = False
                more_sources = i < len(sources) - 1
                try:
                    # Populate the cache ourselves so torch's stdout line is never reached.
                    fetched = _prefetch_with_retry(u, kwargs, budget, timeout)
                    downloaded |= fetched
                    state_dict = torch.hub.load_state_dict_from_url(u, **kwargs)
                except Exception as e:  # noqa: BLE001
                    last_exc, last_url = e, u
                    if fetched:
                        # These bytes are this source's own transfer, not an entry
                        # the call found, so there is nothing to preserve and the
                        # quarantine -- which exists to make a discard reversible --
                        # does not apply. Drop them when a later source can use the
                        # emptied path; otherwise leave them, and leave the bound
                        # unspent so a later call can still discard them.
                        if more_sources:
                            _drop_failed_download(cache_path)
                    else:
                        # Whatever is in the cache is not loadable, and every remaining
                        # URL shares its path. Move it aside so the next source is really
                        # fetched; it comes back below if none of them manages that.
                        moved = _discard_cache_entry(u, kwargs)
                        if moved is not None:
                            quarantine, discarded_url, discard_exc = moved, u, e
                    if more_sources:
                        _warn(f"Failed to load weights from {u!r}: {e}. Trying next source.")
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

    # The re-attempt pass runs only when nothing transferred, so if it also failed
    # without transferring, ``last_exc`` is a refetch failure sitting on top of the
    # load failure that fired the discard -- and that load failure is the one thing
    # naming what actually went wrong. Reporting the network instead points the
    # caller at a checkpoint that is intact and, offline, restored by the ``finally``
    # above. The refetch failure is still worth stating, as context.
    refetch_note = ""
    if re_attempted and not downloaded and discard_exc is not None:
        refetch_note = (
            f" (the cache entry was set aside and refetching it from that same source "
            f"failed too: {_describe(last_exc)})"
        )
        last_exc, last_url = discard_exc, discarded_url

    raise RuntimeError(
        f"Failed to load weights from all {len(urls)} source(s). "
        f"Last URL tried: {last_url!r}. "
        f"Last error: {_describe(last_exc)}{refetch_note}. "
        # Unquoted: the point of naming the path is that it can be pasted into
        # ``rm``/``del``, and ``repr`` doubles every backslash of a Windows path.
        f"Cache path: {cache_path} -- delete it if it is corrupt and this repeats."
    ) from last_exc


def download_file_from_url(
    url: str | list[str],
    *,
    file_name: str | None = None,
    model_dir: str | None = None,
    progress: bool = True,
    validate: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> str:
    """Download a file into the torch hub cache and return its path, without loading it.

    The sibling of :func:`load_state_dict_from_url` for a checkpoint torch
    cannot unpickle -- a ``.safetensors`` file, read afterwards with
    :func:`kornia.core.load_safetensors`. It shares that function's cache, its
    fallback-URL handling, its retry-with-backoff on transient failures and rate
    limits (see :data:`_MAX_ATTEMPTS` and :func:`_retry_delay`), its bound on a
    stalled server (see ``timeout``), and its habit of
    announcing a transfer on :data:`sys.stderr` rather than stdout. A file
    already in the cache is returned as it is, with no request made.

    Pass ``validate`` to get :func:`load_state_dict_from_url`'s quarantine as
    well. That function hooks its quarantine on the *load* step, which a
    download-only function does not have: without one, nothing here can tell a
    truncated cache entry from an intact one, so a bad file is handed back as a
    cache hit on every later call and the caller keeps failing until it is
    deleted by hand.

    ``validate`` supplies the missing step: it is called with the cache path
    after each attempt, and raising from it rejects the file. What follows
    depends on where the file came from.

    A rejected *cache entry* -- one the call found rather than fetched -- is
    quarantined as :func:`load_state_dict_from_url` quarantines a checkpoint that
    fails to load: it is moved aside so the remaining sources see an empty path,
    the source that was handed it is re-fetched once, and if nothing usable turns
    up the original is put back.

    A rejected *fresh transfer* is deleted outright, whether or not a later
    source could have used the emptied path. Those bytes arrived during this call
    and ``validate`` refused them, which is a verdict on the file itself, unlike
    the ambiguous load failures that function has to weigh; keeping them would end
    the call having added a poisoned entry to a cache that had none. Nothing is
    re-fetched in that case, so a one-URL call -- what :func:`download_hf_file`
    makes -- raises with the cache left empty rather than poisoned.

    Without ``validate`` nothing is quarantined, as before. The path is returned
    to the caller and named in the failure message so that a file which turns
    out to be unreadable can be deleted; :func:`kornia.core.load_safetensors`
    names it in every error it raises for the same reason.

    Args:
        url: a URL string, or a list of URL strings tried left-to-right.
        file_name: name to cache the file under. Defaults to the basename of the
            URL -- of the *first* URL when several are given, so that every
            source shares one cache slot. Pass it explicitly whenever that
            basename is not unique to this file: two repositories publishing a
            ``model.safetensors`` each would otherwise share one cache entry, and
            the second model would silently load the first one's weights (see
            :func:`_hf_cache_file_name`). Must be a bare filename: the cache is
            one flat directory, so a value carrying a path separator or naming
            ``.``/``..`` would write outside it. Without it, the first URL's path
            must end in a file name.
        model_dir: directory to cache the file in. Defaults to torch's
            ``<hub dir>/checkpoints``, which is the cache CI restores. A leading
            ``~`` is expanded.
        progress: whether to display a progress bar during a transfer.
        validate: called with the cache path after each attempt, to decide
            whether what is there is usable. Raising rejects the entry. Keep it
            cheap -- a header parse, not a full read -- since it runs on cache
            hits too.
        timeout: seconds a connection attempt or a single read may stall before the
            attempt fails, which then counts as transient; it bounds each wait, not
            the whole transfer. ``None`` uses the ``KORNIA_DOWNLOAD_TIMEOUT``
            environment variable, read at each call, or 30 s when it is unset.

    Returns:
        The path of the cached file.

    Raises:
        TypeError: if ``url`` is not a URL string or a list of them, or ``timeout``
            is not a number.
        ValueError: if ``file_name`` is not a single path component, or it is
            omitted and the first URL's path does not end in a file name; if
            ``url`` is empty or has no scheme (a local path is not a URL); or if
            ``timeout`` is not greater than 0 and at most
            ``threading.TIMEOUT_MAX``, or it is omitted and
            ``KORNIA_DOWNLOAD_TIMEOUT`` holds such a value or no number (the message
            names which).
        RuntimeError: if every URL fails. The message carries the last failure's
            type and text, the source it came from and the cache path in play,
            and the exception itself is chained.

    Example:
        >>> path = download_file_from_url(                      # doctest: +SKIP
        ...     hf_url("kimi-vl-a3b-instruct-vision", "model.safetensors"),
        ...     file_name="kornia--kimi-vl-a3b-instruct-vision--model.safetensors",
        ... )
    """
    urls = _url_list(url)
    file_name = _check_cache_file_name(urls, file_name)
    timeout = _resolve_timeout(timeout)
    model_dir = _expand_model_dir(model_dir)

    # Pin the cache filename to the primary URL's basename so that all attempts
    # share one cache slot, exactly as :func:`load_state_dict_from_url` does.
    if len(urls) > 1 and file_name is None:
        file_name = Path(urlparse(urls[0]).path).name
    kwargs: dict[str, Any] = {"model_dir": model_dir, "file_name": file_name, "progress": progress}

    # Every URL resolves to this one path, so a single quarantine covers the call.
    cache_path = _cached_file_path(urls[0], kwargs)
    budget = _SleepBudget(_MAX_CALL_SLEEP_SECONDS)
    quarantine: str | None = None
    discarded_url: str | None = None
    discard_exc: Exception | None = None
    downloaded = False
    last_exc: Exception | None = None
    last_url: str | None = None
    sources = urls
    re_attempted = False
    try:
        while True:
            for i, u in enumerate(sources):
                fetched = False
                more_sources = i < len(sources) - 1
                try:
                    fetched = _prefetch_with_retry(u, kwargs, budget, timeout)
                    downloaded |= fetched
                    if validate is not None:
                        validate(cache_path)
                except Exception as e:  # noqa: BLE001
                    last_exc, last_url = e, u
                    if fetched:
                        # These bytes are this source's own transfer, not an entry
                        # the call found, so there is nothing to preserve and the
                        # quarantine does not apply.
                        #
                        # They go whether or not a later source can use the emptied
                        # path, which is where this differs from
                        # :func:`load_state_dict_from_url`. Reaching here with
                        # ``fetched`` true means the transfer succeeded and
                        # ``validate`` refused what it wrote -- a verdict on the file
                        # itself. The load failures that function has to weigh are
                        # ambiguous, so it keeps the bytes rather than risk deleting
                        # an intact checkpoint behind a bad ``map_location``; a
                        # rejection here is not, and keeping them would end the call
                        # having *added* a poisoned entry to a cache that had none.
                        # ``download_hf_file`` passes a single URL, so this is the
                        # ordinary cold-cache path, not an edge case.
                        _drop_failed_download(cache_path)
                    else:
                        # Either nothing transferred, or -- the case this whole
                        # branch exists for -- a cache hit was handed back and
                        # ``validate`` rejected it. Move it aside so the next
                        # source really fetches; it comes back below if none does.
                        moved = _discard_cache_entry(u, kwargs)
                        if moved is not None:
                            quarantine, discarded_url, discard_exc = moved, u, e
                    if more_sources:
                        _warn(f"Failed to download {u!r}: {e}. Trying next source.")
                    continue
                if quarantine is not None:
                    _settle_quarantine(cache_path, quarantine, loaded=True, downloaded=downloaded)
                    quarantine = None
                return cache_path

            # A discard that nothing refetched is pure loss: the source it fired
            # for was handed the bad file instead of being fetched, and no later
            # source replaced it. Give that one source the fetch it never got.
            if re_attempted or discarded_url is None or downloaded:
                break
            sources = [discarded_url]
            re_attempted = True
    finally:
        if quarantine is not None:
            _settle_quarantine(cache_path, quarantine, loaded=False, downloaded=downloaded)

    # The re-attempt pass runs only when nothing transferred, so if it also failed
    # without transferring, ``last_exc`` is a refetch failure sitting on top of the
    # rejection that fired the discard -- and that rejection is the one thing naming
    # what is wrong with the file. Reporting the network instead points the caller at
    # an entry that is intact and, offline, has just been restored by the ``finally``
    # above. :func:`load_state_dict_from_url` makes the same swap.
    refetch_note = ""
    if re_attempted and not downloaded and discard_exc is not None:
        refetch_note = (
            f" (the cache entry was set aside and refetching it from that same source "
            f"failed too: {_describe(last_exc)})"
        )
        last_exc, last_url = discard_exc, discarded_url

    raise RuntimeError(
        f"Failed to download the file from all {len(urls)} source(s). "
        f"Last URL tried: {last_url!r}. "
        f"Last error: {_describe(last_exc)}{refetch_note}. "
        # Unquoted: the point of naming the path is that it can be pasted into
        # ``rm``/``del``, and ``repr`` doubles every backslash of a Windows path.
        f"Cache path: {cache_path} -- delete it if it is corrupt and this repeats."
    ) from last_exc


def download_hf_file(
    repo: str,
    filename: str,
    *,
    model_dir: str | None = None,
    progress: bool = True,
    validate: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> str:
    """Download one file from a HuggingFace repo and return the path it is cached at.

    :func:`download_file_from_url` with the two decisions a Hub file needs
    already made: the ``resolve/main`` URL, and a cache name that carries the
    repository id. The second one is not optional -- every repository publishing
    a single-shard checkpoint calls it ``model.safetensors``, and the cache is
    one flat directory, so caching under the URL basename would serve one
    repository's weights for another's, silently and on every later call.

    Args:
        repo: repository name under the ``kornia`` HF org, or a full
            ``owner/name`` id; see :func:`hf_url`.
        filename: file at the root of that repo.
        model_dir: directory to cache the file in. Defaults to torch's
            ``<hub dir>/checkpoints``, which is the cache CI restores.
        progress: whether to display a progress bar during a transfer.
        validate: forwarded to :func:`download_file_from_url`, which documents
            it. Pass :func:`kornia.core.check_safetensors` for a checkpoint, so
            that a truncated cache entry is re-fetched rather than handed back
            on every later call.
        timeout: seconds a connection attempt or a single read may stall before the
            attempt fails; it bounds each wait, not the whole transfer. ``None``
            uses the ``KORNIA_DOWNLOAD_TIMEOUT`` environment variable, read at
            each call, or 30 s when it is unset. Forwarded to
            :func:`download_file_from_url`.

    Returns:
        The path of the cached file. Read a ``.safetensors`` one with
        :func:`kornia.core.load_safetensors`.

    Raises:
        TypeError: if ``timeout`` is not a number.
        ValueError: if ``filename`` carries a path separator; only files at the
            repository root are supported, because the cache is one flat directory.
            Also if ``timeout`` is not greater than 0 and at most
            ``threading.TIMEOUT_MAX``, or it is omitted and ``KORNIA_DOWNLOAD_TIMEOUT``
            holds such a value or no number (the message names which).
        RuntimeError: if the download fails; see :func:`download_file_from_url`.

    Example:
        >>> path = download_hf_file(                        # doctest: +SKIP
        ...     "kimi-vl-a3b-instruct-vision", "model.safetensors"
        ... )
    """
    return download_file_from_url(
        hf_url(repo, filename),
        file_name=_hf_cache_file_name(repo, filename),
        model_dir=model_dir,
        progress=progress,
        validate=validate,
        timeout=timeout,
    )
