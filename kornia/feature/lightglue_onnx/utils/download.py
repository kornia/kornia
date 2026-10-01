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

import hashlib
import os
from typing import Optional
from urllib.parse import urlparse

from torch.hub import HASH_REGEX

from kornia.core import download as core_download


def download_onnx_from_url(
    url: str,
    model_dir: Optional[str] = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    r"""Load the ONNX model at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.
        timeout (float, optional): maximum seconds a connection or read may stall. Defaults to the
            ``KORNIA_DOWNLOAD_TIMEOUT`` environment variable or 30 seconds.

    Example:
        >>> model = download_onnx_from_url('https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/disk_lightglue_fused_fp16.onnx')

    """
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    validate = None
    if check_hash:
        match = HASH_REGEX.search(filename)
        if match is not None:
            hash_prefix = match.group(1)

            def validate(file_path: str) -> None:
                digest = hashlib.sha256()
                with open(file_path, "rb") as file:
                    for chunk in iter(lambda: file.read(128 * 1024), b""):
                        digest.update(chunk)
                actual_hash = digest.hexdigest()
                if actual_hash[: len(hash_prefix)] != hash_prefix:
                    raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{actual_hash}")')

    return core_download.download_file_from_url(
        url, file_name=filename, model_dir=model_dir, progress=progress, validate=validate, timeout=timeout
    )
