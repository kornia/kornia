from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any, Optional

from kornia.config import kornia_config

__all__ = ["CachedDownloader"]

logger = logging.getLogger(__name__)


class CachedDownloader:
    """Downloads files from URLs to the local cache or .kornia_hub directory."""

    @classmethod
    def _get_file_path(cls, model_name: str, cache_dir: Optional[str], suffix: Optional[str] = None) -> str:
        f"""Constructs the file path for the ONNX model based on the model name and cache directory.

        Args:
            model_name: The name of the model or operator, typically in the format 'operators/model_name'.
            cache_dir: The directory where the model should be cached.
                Defaults to None, which will use a default `{kornia_config.hub_onnx_dir}` directory.

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
        file_path = os.path.join(*cache_dir.split(os.sep), *model_name.split(os.sep)[:-1], file_name)
        return file_path

    @classmethod
    def download_to_cache(cls, url: str, name: str, download: bool = True, **kwargs: Any) -> str:
        if url.startswith(("http:", "https:")):
            cache_dir = kwargs.get("cache_dir", None)
            suffix = kwargs.get("suffix", None)
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
        """Downloads an ONNX model from the specified URL and saves it to the specified file path.

        Args:
            url: The URL of the ONNX model to download.
            file_path: The local path where the downloaded model should be saved.
            download_if_not_exists: If True, the file will be downloaded if it's not already downloaded.
        """
        if os.path.exists(file_path):
            logger.info(f"Loading `{url}` from `{file_path}`.")
            return

        if not download_if_not_exists:
            raise ValueError(f"`{file_path}` not found. You may set `download=True`.")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create the cache directory if it doesn't exist

        if url.startswith(("http:", "https:")):
            try:
                logger.info(f"Downloading `{url}` to `{file_path}`.")
                urllib.request.urlretrieve(url, file_path)  # noqa: S310
            except urllib.error.HTTPError as e:
                raise ValueError(f"Error in resolving `{url}`. {e}.")
        else:
            raise ValueError("URL must start with 'http:' or 'https:'")
