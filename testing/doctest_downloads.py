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

"""Keep the doctest run off the network.

``pytest --doctest-modules kornia/`` executes the examples in every public
docstring, and a dozen of those construct pretrained models. On a cold cache
that turns the nominal documentation check into an ~838 MB download.

This module patches the download *primitives* -- the functions that run only
when a weight file is missing from the cache -- so that a doctest which would
download instead reports as skipped. Doctests whose weights are already cached
still run for real, so a warm cache (CI, or a developer who has run the model
tests) keeps full coverage with no list of exempted examples to maintain.

Set ``KORNIA_DOCTEST_DOWNLOAD=1`` (or pass ``--doctest-download``) to allow the
downloads, which is what the scheduled main-branch documentation job does.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, NoReturn, Sequence

DOWNLOAD_ENV_VAR = "KORNIA_DOCTEST_DOWNLOAD"

# Entry points that reach the network only on a cache miss, as
# ``(module, attribute)`` pairs resolved lazily at patch time.
#
# ``kornia.feature.lightglue_onnx.utils.download`` binds ``download_url_to_file``
# at import time, so patching ``torch.hub`` alone would not cover it.
_DOWNLOAD_PRIMITIVES: tuple[tuple[str, str], ...] = (
    ("torch.hub", "download_url_to_file"),
    ("kornia.feature.lightglue_onnx.utils.download", "download_url_to_file"),
    # kornia.onnx.download.CachedDownloader.download reaches the network here.
    ("urllib.request", "urlretrieve"),
)


def downloads_allowed(env: dict[str, str]) -> bool:
    """Return whether doctests may download model weights.

    Args:
        env: environment mapping to read ``KORNIA_DOCTEST_DOWNLOAD`` from.

    Returns:
        ``True`` when the variable is set to ``1``, ``true`` or ``yes``
        (case-insensitive), ``False`` otherwise.

    Example:
        >>> downloads_allowed({})
        False
        >>> downloads_allowed({"KORNIA_DOCTEST_DOWNLOAD": "1"})
        True
        >>> downloads_allowed({"KORNIA_DOCTEST_DOWNLOAD": "false"})
        False
    """
    return env.get(DOWNLOAD_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def install_download_guard(
    setattr_fn: Callable[[Any, str, Any], None],
    on_download: Callable[[str], NoReturn],
    primitives: Sequence[tuple[str, str]] = _DOWNLOAD_PRIMITIVES,
) -> list[tuple[str, str]]:
    """Redirect every kornia weight-download primitive to *on_download*.

    Args:
        setattr_fn: ``monkeypatch.setattr``-style callable, so the patches are
            undone when the surrounding test finishes.
        on_download: called with the requested URL instead of downloading it.
            It must not return -- typically it raises, e.g. ``pytest.skip``.
        primitives: ``(module, attribute)`` pairs to patch. Pairs whose module
            is not importable are skipped, which keeps optional subpackages
            from turning into a hard dependency of the test run.

    Returns:
        The pairs that were actually patched, in the order given.

    Example:
        >>> patched = {}
        >>> def fake_setattr(obj, name, value):
        ...     patched[name] = value
        >>> def refuse(url):
        ...     raise RuntimeError(url)
        >>> install_download_guard(fake_setattr, refuse, [("torch.hub", "download_url_to_file")])
        [('torch.hub', 'download_url_to_file')]
        >>> patched["download_url_to_file"]("https://example.com/w.pth", "/tmp/w.pth")
        Traceback (most recent call last):
        ...
        RuntimeError: https://example.com/w.pth
    """
    patched: list[tuple[str, str]] = []

    for module_name, attribute in primitives:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if not hasattr(module, attribute):
            continue

        def _guard(url: str, *args: Any, **kwargs: Any) -> NoReturn:
            on_download(url)
            # on_download is documented as non-returning; be explicit if it is not.
            raise AssertionError(f"download guard callback returned instead of raising for {url!r}")

        setattr_fn(module, attribute, _guard)
        patched.append((module_name, attribute))

    return patched


def skip_reason(url: str) -> str:
    """Return the message shown when a doctest is skipped for needing a download.

    Args:
        url: the weight URL the doctest tried to fetch.

    Returns:
        A message naming the URL and how to opt in.

    Example:
        >>> skip_reason("https://example.com/w.pth").split(";")[0]
        'doctest needs to download model weights from https://example.com/w.pth'
    """
    return (
        f"doctest needs to download model weights from {url}; "
        f"set {DOWNLOAD_ENV_VAR}=1 to allow it, or pre-populate the cache "
        "with `python .github/download-models-weights.py`"
    )
