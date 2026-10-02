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
import urllib.request
import uuid
from contextlib import suppress
from typing import Any, Optional

from kornia.config import kornia_config

__all__ = ["CachedDownloader"]

_logger = logging.getLogger(__name__)


class CachedDownloader:
    """Downloads files from URLs to the local cache or .kornia_hub directory."""

    @classmethod
    def _get_file_path(cls, model_name: str, cache_dir: Optional[str], suffix: Optional[str] = None) -> str:
        """Construct the file path for the ONNX model based on the model name and cache directory.

        Args:
            model_name: The name of the model or operator, typically in the format 'operators/model_name'.
            cache_dir: The directory where the model should be cached.
                Defaults to None, which will use a default `kornia.config.hub_onnx_dir` directory.
            suffix: Optional file suffix when the filename is the model name.

        Returns:
            str: The full local path where the model should be stored or loaded from.

        """
        # Determine the local file path
        if cache_dir is None:
            cache_dir = kornia_config.hub_cache_dir

        # The filename is the model name (without directory path)
        if suffix is not None and not model_name.endswith(suffix):
            file_name = f"{os.path.split(model_name)[-1]}{suffix}"
        else:
            file_name = os.path.split(model_name)[-1]

        return os.path.join(cache_dir, *model_name.split(os.sep)[:-1], file_name)

    @classmethod
    def download_to_cache(cls, url: str, name: str, download: bool = True, **kwargs: Any) -> str:
        """Resolve a remote file into Kornia's local cache and download it when needed.

        Args:
            url: HTTP or HTTPS URL of the file to cache.
            name: Logical model or operator name used to construct the cache
                path. Directory components in ``name`` are preserved under the
                cache directory.
            download: If ``True``, download the file when it is not already
                present in the cache. If ``False``, require the cached file to
                exist.
            kwargs: Optional cache parameters. Supported keys include
                ``cache_dir`` for overriding the default cache root and
                ``suffix`` for appending a filename suffix when needed.

        Returns:
            Local filesystem path to the cached file.

        Raises:
            ValueError: If ``url`` is not an HTTP or HTTPS URL, if download is
                disabled and the file is missing, or if the server answers with an HTTP error.
            urllib.error.URLError: If the server cannot be reached, or ``urllib.error.ContentTooShortError``
                if the body is shorter than its ``Content-Length`` (see ``download``).
        """
        if url.startswith(("http:", "https:")):
            cache_dir = kwargs.get("cache_dir")
            suffix = kwargs.get("suffix")
            file_path = cls._get_file_path(name, cache_dir, suffix=suffix)
            cls.download(url, file_path, download_if_not_exists=download)
            return file_path
        raise ValueError(f"URL must start with 'http:' or 'https:'. Got {url}")

    @classmethod
    def download(
        cls,
        url: str,
        file_path: str,
        download_if_not_exists: bool = True,
    ) -> None:
        """Download an ONNX model from the specified URL and save it to the specified file path.

        Args:
            url: The URL of the ONNX model to download.
            file_path: The local path where the downloaded model should be saved.
            download_if_not_exists: If True, the file will be downloaded if it's not already downloaded.

        Raises:
            ValueError: If the file is missing and ``download_if_not_exists`` is ``False``, if ``url`` is not an
                HTTP or HTTPS URL, or if the server answers with an HTTP error.
            urllib.error.ContentTooShortError: If the body is shorter than its ``Content-Length``. Nothing is
                left at ``file_path``, so the next call downloads again.
            urllib.error.URLError: If the server cannot be reached.

        """
        if os.path.exists(file_path):
            _logger.info(f"Loading `{url}` from `{file_path}`.")
            return

        if not download_if_not_exists:
            raise ValueError(f"`{file_path}` not found. You may set `download=True`.")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create the cache directory if it doesn't exist

        if url.startswith(("http:", "https:")):
            # Keep incomplete transfers out of the cache, including while another caller is checking it: write
            # beside the destination and publish with os.replace. Let urlretrieve create the file, so it gets the
            # permissions of any new file (umask applied); a tempfile would be 0o600 and lock other users out of a
            # shared cache.
            temporary_path = f"{file_path}.{uuid.uuid4().hex}.partial"
            try:
                _logger.info(f"Downloading `{url}` to `{file_path}`.")
                urllib.request.urlretrieve(url, temporary_path)  # noqa: S310
                os.replace(temporary_path, file_path)
            except urllib.error.HTTPError as e:
                raise ValueError(f"Error in resolving `{url}`.") from e
            finally:
                with suppress(FileNotFoundError):
                    os.remove(temporary_path)
        else:
            raise ValueError("URL must start with 'http:' or 'https:'")
