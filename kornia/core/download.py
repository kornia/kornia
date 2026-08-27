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
import warnings
from pathlib import Path
from typing import Any
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

    Args:
        url: a URL string, or a list of URL strings tried left-to-right.
        **kwargs: forwarded verbatim to
            :func:`torch.hub.load_state_dict_from_url`
            (``map_location``, ``check_hash``, ``file_name``, …).

    Returns:
        The loaded state dict.

    Raises:
        RuntimeError: if every URL fails, chaining the last exception.

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
            _prefetch_to_cache(u, kwargs)
            return torch.hub.load_state_dict_from_url(u, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if i < len(urls) - 1:
                warnings.warn(
                    f"Failed to load weights from {u!r}: {e}. Trying next source.",
                    stacklevel=2,
                )

    raise RuntimeError(
        f"Failed to load weights from all {len(urls)} source(s). Last URL tried: {urls[-1]!r}"
    ) from last_exc
