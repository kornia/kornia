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
import logging
import os
import uuid
from typing import Any, Callable, List, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import torch

from kornia.config import kornia_config
from kornia.core.download import _resolve_timeout

from .io import load_image

__all__ = [
    "get_sample_images",
]

_IMAGE_URLS: List[str] = [
    "https://raw.githubusercontent.com/kornia/data/main/panda.jpg",
    "https://raw.githubusercontent.com/kornia/data/main/simba.png",
    "https://raw.githubusercontent.com/kornia/data/main/girona.png",
    "https://raw.githubusercontent.com/kornia/data/main/baby_giraffe.png",
    "https://raw.githubusercontent.com/kornia/data/main/persistencia_memoria.jpg",
    "https://raw.githubusercontent.com/kornia/data/main/delorean.png",
]


def download_image(url: str, save_to: str) -> None:
    """Download a file and save its bytes unchanged, so a JPEG is not decoded and re-encoded.

    Each call writes the bytes to a temporary file of its own next to ``save_to`` and renames it onto ``save_to`` only
    once the write is complete, removing it if the write fails. An interrupted download therefore never leaves a
    partial file at ``save_to``, and concurrent downloads of the same file each leave a complete copy there. Windows
    cannot replace a file that is open, for example one that another call is reading; a call whose replacement is
    refused keeps the copy already there only if its bytes match the download, and removes its own temporary file. A
    refused replacement with a missing, unreadable, or different destination still raises the replacement error.

    The download fails once the server goes ``KORNIA_DOWNLOAD_TIMEOUT`` seconds (30 s when it is unset) without
    answering, as kornia's other downloads do.

    Args:
        url: The URL of the image to download.
        save_to: The file path where the downloaded image will be saved.

    Raises:
        ValueError: If ``KORNIA_DOWNLOAD_TIMEOUT`` is set to anything but a positive number of seconds.

    """
    _download(url, save_to)


def _download(url: str, save_to: str, load: Optional[Callable[[str], torch.Tensor]] = None) -> Optional[torch.Tensor]:
    """Download ``url`` to ``save_to`` as :func:`download_image` does; return ``load`` of the download, or ``None``.

    ``load`` reads the call's own temporary file before it is renamed onto ``save_to``, so the call does not load
    ``save_to``, which a concurrent call may be replacing (Windows refuses to open a file while it is replaced). A body
    that ``load`` rejects never reaches ``save_to``: the temporary file is removed and the error propagates.
    """
    # get_sample_images only routes http(s) URLs here (default: the https:// literals above). The timeout is the one
    # kornia's other downloads use: KORNIA_DOWNLOAD_TIMEOUT, or 30 s when it is unset.
    with urlopen(url, timeout=_resolve_timeout(None)) as resp:  # noqa: S310
        data = resp.read()
    # a unique name in the destination directory keeps the rename atomic and concurrent calls apart; open() with "x"
    # creates the file with the usual permissions (tempfile.mkstemp would make the cached image private, 0o600).
    # A file to load ends with the destination's extension, which load_image picks its decoder by.
    suffix = os.path.splitext(save_to)[1] if load is not None else ""
    part = f"{save_to}.{uuid.uuid4().hex}.part{suffix}"
    try:
        with open(part, "xb") as f:
            f.write(data)
        image = None if load is None else load(part)
        try:
            os.replace(part, save_to)
        except PermissionError:
            # Windows may refuse to replace an open file. Only accept an identical completed download.
            matches = False
            try:
                with open(save_to, "rb") as existing:
                    matches = existing.read(len(data) + 1) == data
            except OSError:
                pass  # preserve the replacement error when the destination cannot be verified
            if not matches:
                raise
            os.remove(part)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(part)
        raise
    return image


def get_sample_images(
    resize: Optional[Tuple[int, int]] = None,
    paths: List[str] = _IMAGE_URLS,
    download: bool = True,
    cache_dir: Optional[str] = None,
    as_list: Optional[bool] = None,
    divisible_factor: Optional[int] = None,
    **kwargs: Any,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Load multiple images from the given paths or URLs.

    Optionally download them, resize them if specified, and return them as a batch of tensors or a list of tensors.

    Convention:
        - Each image loads with :func:`~kornia.io.load_image`'s default type, ``RGB32``: :math:`(3, H, W)`
          ``float32`` in [0, 1].
        - A path is a URL only when its scheme is ``http`` or ``https``. A URL is cached under ``cache_dir`` with the
          file name of its path, byte for byte as downloaded, and a cached file is loaded without downloading it
          again. With ``download=False``, a URL that is not cached logs an error and raises ``FileNotFoundError``.
          A download fails once the server goes ``KORNIA_DOWNLOAD_TIMEOUT`` seconds (30 s when it is unset) without
          answering, and an unusable value of that variable raises ``ValueError``.
        - ``resize`` is ``(height, width)`` and goes to :func:`~kornia.geometry.transform.resize`; ``divisible_factor``
          is applied after it by :func:`~kornia.geometry.transform.resize_to_be_divisible`, so the final size can
          differ from ``resize``. ``**kwargs`` reach both.
        - The result is one :math:`(B, 3, H, W)` tensor when ``resize`` is given and ``as_list`` is not ``True``, and a
          list of :math:`(3, H, W)` tensors otherwise; ``as_list=False`` without ``resize`` still gives a list.

    Args:
        paths: A list of path or URL from which to load or download images.
              Defaults to the pre-defined sample image URLs if not provided.
        resize: Optional target size ``(height, width)`` for every image. If not provided, the images keep their
            sizes unless ``divisible_factor`` is given.
        download: Whether to download the images if they are not already cached. Defaults to True.
        cache_dir: The directory where the downloaded images will be cached.
            Defaults to ``<kornia_config.hub_cache_dir>/images``, a per-user directory under ``torch.hub.get_dir()``
            unless configured otherwise. Created when an image has to be downloaded. A download is loaded from a
            temporary file of its own before it is renamed into the cache, so a download that does not load as an
            image never reaches the cache: its temporary file is removed and the error propagates.
        as_list: ``True`` to return a list even when ``resize`` is given.
        divisible_factor:
            If not None, each side of every image is resized to a multiple of ``divisible_factor``.
        **kwargs: Additional keyword arguments for both resizes.

    Returns:
        torch.Tensor | list[torch.Tensor]: the images, stacked or as a list.

    """
    import kornia.geometry  # local import to avoid a circular dependency at import time

    if cache_dir is None:
        cache_dir = os.path.join(kornia_config.hub_cache_dir, "images")
    tensors = []
    for path in paths:
        img_tensor: Optional[torch.Tensor] = None
        url = urlparse(path)
        if url.scheme in ("http", "https"):
            name = os.path.basename(url.path)  # the cache file is named without the URL's query or fragment
            if not name:
                raise ValueError(f"Cannot derive a cache file name from URL `{path}`: its path names no file.")
            fname = os.path.join(cache_dir, name)
            if not os.path.exists(fname) and download:
                os.makedirs(cache_dir, exist_ok=True)
                logging.info(f"Downloading `{path}` to `{fname}`.")
                img_tensor = _download(path, fname, load_image)
            elif not os.path.exists(fname) and not download:
                logging.error(
                    f"Image `{path}` not found at `{fname}`. You may want to set `download=True` to download it."
                )
        else:
            fname = path
        if img_tensor is None:
            img_tensor = load_image(fname)  # a local file or a cached copy; a failure here deletes nothing
        if resize is not None:
            img_tensor = kornia.geometry.resize(img_tensor, resize, **kwargs)
        if divisible_factor is not None:
            img_tensor = kornia.geometry.transform.resize_to_be_divisible(img_tensor, divisible_factor, **kwargs)
        tensors.append(img_tensor)

    if not as_list and resize is not None:
        return torch.stack(tensors)

    return tensors
