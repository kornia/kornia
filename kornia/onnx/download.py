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

import logging
import os
import urllib.error
from typing import Optional

from kornia.config import kornia_config
from kornia.core import download as _core_download

__all__ = ["CachedDownloader"]

_logger = logging.getLogger(__name__)


def _relative_path_components(name: str, label: str = "name") -> list[str]:
    """Split ``name``, a relative path inside a cache directory, into its components, separated by ``/`` or ``os.sep``.

    Args:
        name: The relative path.
        label: What ``name`` is, for the error message.

    Raises:
        ValueError: If ``name`` is absolute, has a drive (on Windows) or has an empty, ``.`` or ``..`` component, which
            would put the path outside the cache directory or name no file.
    """
    # the component test of kornia.core.download._check_cache_file_name; os.path.basename also catches a Windows
    # drive ("C:", "d:x"), which os.path.join would not keep under the cache directory
    components = name.replace(os.sep, "/").split("/")
    if any(c in _core_download._NOT_A_FILE_NAME or os.path.basename(c) != c for c in components):
        raise ValueError(
            f"{label} must be a relative path inside the cache directory, without a drive or an empty, '.' or '..' "
            f"component; got {name!r}."
        )
    return components


class CachedDownloader:
    """Downloads files from URLs to the local cache or .kornia_hub directory."""

    @classmethod
    def _get_file_path(cls, model_name: str, cache_dir: Optional[str], suffix: Optional[str] = None) -> str:
        """Construct the file path for the ONNX model based on the model name and cache directory.

        Args:
            model_name: The name of the model or operator, typically in the format 'operators/model_name': a relative
                path whose components are separated by ``/`` (or ``os.sep``).
            cache_dir: The directory where the model should be cached.
                Defaults to None, which will use the default ``kornia.config.kornia_config.hub_cache_dir`` directory.
            suffix: Optional file suffix when the filename is the model name.

        Returns:
            str: The full local path where the model should be stored or loaded from.

        Raises:
            ValueError: If ``model_name`` is absolute, has a drive (on Windows) or has an empty, ``.`` or ``..``
                component, which would put the file outside ``cache_dir`` or name no file, or if ``suffix`` adds a
                directory or a drive to the file name.

        """
        components = _relative_path_components(model_name)

        # Determine the local file path
        if cache_dir is None:
            cache_dir = kornia_config.hub_cache_dir
        cache_dir = os.path.expanduser(cache_dir)

        # The filename is the model name (without directory path)
        file_name = components[-1]
        if suffix is not None and not file_name.endswith(suffix):
            file_name = f"{file_name}{suffix}"
            # the suffix joins the name after the check above, so it must not add a directory or a drive either
            if os.path.basename(file_name) != file_name:
                raise ValueError(f"suffix must not add a directory or a drive to the file name; got {suffix!r}.")

        return os.path.join(cache_dir, *components[:-1], file_name)

    @classmethod
    def download_to_cache(
        cls,
        url: str,
        name: str,
        download: bool = True,
        *,
        cache_dir: Optional[str] = None,
        suffix: Optional[str] = None,
        force_download: bool = False,
    ) -> str:
        """Resolve a remote file into Kornia's local cache and download it when needed.

        A file that is in the cache is used as it is, for as long as it exists: a model updated on the server is not
        fetched again, and a cached file that is damaged, or was replaced by hand, stays so. Pass
        ``force_download=True`` to fetch the file again, or delete the cached file.

        Args:
            url: HTTP or HTTPS URL of the file to cache.
            name: Logical model or operator name used to construct the cache
                path. Directory components in ``name``, separated by ``/``, are
                preserved under the cache directory.
            download: If ``True``, download the file when it is not already
                present in the cache. If ``False``, require the cached file to
                exist.
            cache_dir: The cache root. Defaults to ``None``, which uses
                ``kornia.config.kornia_config.hub_cache_dir``; ``""`` is the
                current working directory.
            suffix: Appended to the file name unless it already ends with it.
            force_download: If ``True``, fetch the file from ``url`` even when it is already cached, and replace the
                cached file with it. The old file is replaced only once the new one is complete, so a fetch that
                fails leaves it as it was. Needs ``download=True``. Defaults to ``False``.

        Returns:
            Local filesystem path to the cached file.

        Raises:
            ValueError: If ``url`` is not an HTTP or HTTPS URL, if ``name`` is absolute or has a drive or an empty,
                ``.`` or ``..`` component, if ``suffix`` adds a directory to the file name, if download is disabled
                and the file is missing, if ``force_download`` is ``True`` while ``download`` is ``False``, if the
                server answers with an HTTP error, or if ``KORNIA_DOWNLOAD_TIMEOUT`` is set to anything but a
                positive number of seconds.
            TimeoutError: If the server goes ``KORNIA_DOWNLOAD_TIMEOUT`` seconds without answering (see ``download``).
            urllib.error.URLError: If the server cannot be reached, or ``urllib.error.ContentTooShortError``
                if the body is shorter than its ``Content-Length`` (see ``download``).
            http.client.IncompleteRead: If a chunked body is cut mid-chunk (see ``download``).
        """
        if url.startswith(("http:", "https:")):
            file_path = cls._get_file_path(name, cache_dir, suffix=suffix)
            # passed only when set, so that a subclass whose ``download`` predates the keyword keeps working
            extra = {"force_download": True} if force_download else {}
            cls.download(url, file_path, download_if_not_exists=download, **extra)
            return file_path
        raise ValueError(f"URL must start with 'http:' or 'https:'. Got {url}")

    @classmethod
    def download(
        cls,
        url: str,
        file_path: str,
        download_if_not_exists: bool = True,
        *,
        force_download: bool = False,
    ) -> None:
        """Download an ONNX model from the specified URL and save it to the specified file path.

        Args:
            url: The URL of the ONNX model to download.
            file_path: The local path where the downloaded model should be saved.
            download_if_not_exists: If True, the file will be downloaded if it's not already downloaded.
            force_download: If True, download the file even when ``file_path`` exists, and replace it. The old file
                is replaced only once the new one is complete, so a download that fails leaves it as it was. Needs
                ``download_if_not_exists=True``.

        Raises:
            ValueError: If the file is missing and ``download_if_not_exists`` is ``False``, if ``force_download`` is
                ``True`` while ``download_if_not_exists`` is ``False``, if ``url`` is not an HTTP or HTTPS URL, if
                the server answers with an HTTP error, or if ``KORNIA_DOWNLOAD_TIMEOUT`` is set to anything but a
                positive number of seconds.
            TimeoutError: If the server goes ``KORNIA_DOWNLOAD_TIMEOUT`` seconds (30 s when it is unset) without
                answering. An existing cached file is left unchanged.
            urllib.error.ContentTooShortError: If the body is shorter than its ``Content-Length``. An existing cached
                file is left unchanged.
            http.client.IncompleteRead: If a chunked body is cut mid-chunk. An existing cached file is left unchanged.
            urllib.error.URLError: If the server cannot be reached.

        """
        file_path = os.path.expanduser(file_path)
        if force_download and not download_if_not_exists:
            # the two flags ask for opposite things, and ignoring either would hide a mistake in the call
            raise ValueError(
                "`force_download=True` needs `download=True`: it cannot fetch a file that may not be downloaded."
            )
        if os.path.exists(file_path) and not force_download:
            _logger.info(f"Loading `{url}` from `{file_path}`.")
            return

        if not download_if_not_exists:
            raise ValueError(f"`{file_path}` not found. You may set `download=True`.")

        directory = os.path.dirname(file_path)
        if directory:  # a path without a directory part is a file in the working directory
            os.makedirs(directory, exist_ok=True)

        if url.startswith(("http:", "https:")):
            try:
                _logger.info(f"Downloading `{url}` to `{file_path}`.")
                _core_download._download_url_to_file(url, file_path, progress=False)
            except urllib.error.HTTPError as exc:
                raise ValueError(f"Error in resolving `{url}`.") from exc
            except _core_download._TruncatedTransfer as exc:
                # urlretrieve's type and message for a body shorter than its Content-Length; any other
                # IncompleteRead (a chunked body cut mid-chunk) propagates unchanged, as it did from urlretrieve
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got only {exc.received} out of {exc.announced} bytes", b""
                ) from exc
        else:
            raise ValueError("URL must start with 'http:' or 'https:'")
