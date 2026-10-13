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

    Args:
        url: The URL of the image to download.
        save_to: The file path where the downloaded image will be saved.

    """
    _download(url, save_to)


def _download(url: str, save_to: str, load: Optional[Callable[[str], torch.Tensor]] = None) -> Optional[torch.Tensor]:
    """Download ``url`` to ``save_to`` as :func:`download_image` does; return ``load`` of the download, or ``None``.

    ``load`` reads the call's own temporary file before it is renamed onto ``save_to``, so the call does not load
    ``save_to``, which a concurrent call may be replacing (Windows refuses to open a file while it is replaced). A body
    that ``load`` rejects never reaches ``save_to``: the temporary file is removed and the error propagates.
    """
    # get_sample_images only routes http(s) URLs here (default: the https:// literals above).
    with urlopen(url, timeout=30) as resp:  # noqa: S310
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
    """Load multiple images from the given URLs.

    Optionally download them, resize them if specified, and return them as a batch of tensors or a list of tensors.

    Removed from ``kornia.utils.sample`` in 0.8.3 and restored here as public API — the docs and downstream
    demos rely on it.

    Args:
        paths: A list of path or URL from which to load or download images.
              Defaults to the pre-defined sample image URLs if not provided.
        resize: Optional target size for resizing all images as a tuple (height, width).
            If not provided, the images will not be resized, and their original sizes will be retained.
        download: Whether to download the images if they are not already cached. Defaults to True.
        cache_dir: The directory where the downloaded images will be cached.
            Defaults to ``<kornia_config.hub_cache_dir>/images``, relative to the working directory unless configured
            otherwise. Created when an image has to be downloaded. A download is loaded from a temporary file of its
            own before it is renamed into the cache, so a download that does not load as an image never reaches the
            cache: its temporary file is removed and the error propagates.
        as_list: if to keep the output as a list. If None and `resize` is None, the output will be a list.
            If None and `resize` is not None, the output will be a single tensor.
        divisible_factor:
            If not None, the images will be resized to the nearest multiple of `divisible_factor`.
        **kwargs: Additional keyword arguments to pass to `kornia.geometry.resize`.

    Returns:
        torch.Tensor | list[torch.Tensor]:
            If `resize` is provided, returns a single stacked tensor with shape (B, C, H, W).
            Otherwise, returns a list of tensors, each with its original shape (C, H, W).

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
