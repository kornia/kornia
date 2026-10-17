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
import json
import logging
import os
import posixpath
import pprint
import urllib.error
from typing import Any, Optional
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import urlopen

import kornia
from kornia.config import kornia_config
from kornia.core.external import numpy as np
from kornia.core.external import onnx
from kornia.onnx.download import CachedDownloader

__all__ = ["ONNXLoader", "add_metadata", "io_name_conversion"]

_logger = logging.getLogger(__name__)


# Characters a cache directory keeps besides letters, digits and "_.-~"; quote() percent-encodes every other one, "%"
# included, so no directory is refused on Windows (``:``, ``*``, ``|``, ...) and two different URL directories never
# become one cache directory
_SAFE_DIRECTORY_CHARACTERS = "!$&'()+,;=@[]~"


def _url_cache_name(url: str) -> str:
    """Return the name, relative to the cache directory, that :class:`ONNXLoader` caches ``url`` under.

    The name is ``<scheme>/<host>[%3A<port>]/<directories of the URL's path>[/%3F<hash>]/<file name>``, under an
    explicit ``cache_dir`` as under the default directory. Two URLs share a file only when they differ in nothing but
    credentials, the fragment, ``.``, ``..`` or empty directories (resolved as RFC 3986 resolves them, which most
    servers follow), or a ``.onnx`` / ``.json`` extension that one of them lacks. The scheme keeps a file fetched over
    ``http`` from being served for ``https``, and keeps URLs apart from the ``hf://`` folders in the same directory. A
    query can name the file (``?id=...``, ``?ref=...``), so a URL with one is stored in a directory named after the
    query's hash; the query itself is not part of any name, since ``?`` cannot be in a file name on Windows. The host
    (an IPv6 address in its brackets) and the directories are percent-encoded (see ``_SAFE_DIRECTORY_CHARACTERS``),
    and so are the ``:`` before the port and the ``?`` before the query's hash, which no host or directory of a URL
    can become, as ``%`` itself is encoded: ``%3A8080`` is never a host and ``%3F<hash>`` never a directory of the
    path. The file name is kept as it is, because a model refers to its ``.onnx_data`` weights by name.

    Raises:
        ValueError: If the URL has no host, or its path does not end in a file name.
    """
    parts = urlsplit(url)
    *directories, file_name = parts.path.split("/")
    if file_name in ("", ".", ".."):
        raise ValueError(f"{url!r} does not end in a file name to cache it under.")
    host = parts.hostname  # lower-cased, without any user name or password
    if not host:  # urlopen refuses it too; without the host, "https:///a/m" would be cached as "https://a/m"
        raise ValueError(f"{url!r} has no host to cache it under.")
    if ":" in host:  # an IPv6 address keeps its brackets, so that "[::1]:8000" stays apart from "[::1:8000]"
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    # the leading "/" makes the path absolute, so normpath drops every ".." that would climb above it
    directory = posixpath.normpath("/" + "/".join(directories))
    components = [parts.scheme, quote(host, safe=_SAFE_DIRECTORY_CHARACTERS)]
    components += [quote(c, safe=_SAFE_DIRECTORY_CHARACTERS) for c in directory.split("/") if c]
    if parts.query:
        components.append(quote("?") + hashlib.sha256(parts.query.encode()).hexdigest()[:16])
    return "/".join([*components, file_name])


def _check_keywords(method: str, kwargs: dict[str, Any]) -> None:
    """Raise Python's ``TypeError`` for any keyword of ``method`` but ``cache_dir``, whichever branch it takes.

    A local model, or a config URL that is not HTTP(S), never reaches ``download_to_cache``, which would raise it.
    """
    for key in kwargs:
        if key != "cache_dir":
            raise TypeError(f"{method}() got an unexpected keyword argument {key!r}")


def _onnx_data_name(name: str) -> str:
    """Return the name of the ``.onnx_data`` weights stored beside the model ``name``, with or without ``.onnx``."""
    return (name[: -len(".onnx")] if name.endswith(".onnx") else name) + ".onnx_data"


class ONNXLoader(CachedDownloader):
    """Manages ONNX models, handling local caching, downloading from Hugging Face, and loading models."""

    @classmethod
    def load_config(cls, url: str, download: bool = True, **kwargs: Any) -> dict[str, Any]:
        """Load JSON config from the specified URL.

        Args:
            url: The URL of the preprocessor config to load.
            download: If True, the config will be downloaded if it's not already in the local cache.
            kwargs: Keyword arguments of ``CachedDownloader.download_to_cache``. Only ``cache_dir`` applies: the root
                the config is stored under, at ``<scheme>/<host>/<directories of the URL's path>/<file name>``, with a
                ``%3F<hash>`` directory before the file name when the URL has a query, as for ``load_model``. ``None``
                or ``""`` means ``kornia_config.hub_cache_dir``. Any other keyword raises ``TypeError``, whatever
                ``url`` is.

        Returns:
            dict[str, Any]: The loaded preprocessor config.

        """
        _check_keywords("ONNXLoader.load_config", kwargs)
        if url.startswith(("http:", "https:")):
            kwargs["cache_dir"] = kwargs.get("cache_dir") or None  # "" is the default directory, as for load_model
            file_path = cls.download_to_cache(url, _url_cache_name(url), download=download, suffix=".json", **kwargs)
            with open(file_path) as f:
                return json.load(f)

        if not download:
            raise RuntimeError(f"File `{url}` not found. You may set `download=True`.")

        raise RuntimeError(f"File `{url}` not found.")

    @classmethod
    def load_model(cls, model_name: str, download: bool = True, with_data: bool = False, **kwargs) -> onnx.ModelProto:
        """Load an ONNX model from the local cache or downloads it from Hugging Face if necessary.

        Args:
            model_name: The name of the ONNX model or operator. For Hugging Face-hosted models,
                use the format 'hf://model_name'. Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models.
                Or a URL to the ONNX model.
            download: If True, the model will be downloaded from Hugging Face if it's not already in the local cache.
            cache_dir: The root of the cache. An ``hf://`` model is stored under it at the path of its name after
                the folder (``<cache_dir>/a/b.onnx`` for ``hf://models/a/b``): the kornia/ONNX_models names are fully
                qualified (``kornia.color.gray.RgbToGrayscale``), so the folders never share a name. A URL is stored at
                ``<cache_dir>/<scheme>/<host>/<directories of the URL's path>/<file name>``, with a ``%3F<hash>``
                directory before the file name when the URL has a query, so that two URLs share a file only when they
                differ in nothing but credentials, the fragment, ``.``, ``..`` or empty directories, or an ``.onnx``
                extension that one of them lacks. Defaults to None (``""`` too): an ``hf://`` model goes to
                ``<kornia_config.hub_onnx_dir>/<folder>/``, and a URL under ``kornia_config.hub_onnx_dir``.
                Limitations: on Windows, a URL whose file name is not a valid file name there, or whose cache path
                is longer than ``MAX_PATH`` (260 characters) while long paths are disabled, cannot be cached, and
                neither can a directory with a reserved name (``con``); on a case-insensitive file system, URLs that
                differ only in letter case share a file.
            with_data: If True, the model will be loaded with its `.onnx_data` weights, stored beside the model.
            **kwargs: Only ``cache_dir``; any other keyword raises ``TypeError``, whatever ``model_name`` is.

        Returns:
            onnx.ModelProto: The loaded ONNX model.

        """
        _check_keywords("ONNXLoader.load_model", kwargs)
        if model_name.startswith("hf://"):
            model_name = model_name[len("hf://") :]
            url = f"https://huggingface.co/kornia/ONNX_models/resolve/main/{model_name}.onnx"
            if kwargs.get("cache_dir"):
                name = model_name.partition("/")[2]  # under an explicit cache_dir, the path after the folder
            else:
                kwargs["cache_dir"] = kornia_config.hub_onnx_dir
                name = model_name  # "<folder>/<a>/<b>": the folder is checked as the other components are
            file_path = cls.download_to_cache(url, name, download=download, suffix=".onnx", **kwargs)
            if with_data:
                url_data = f"https://huggingface.co/kornia/ONNX_models/resolve/main/{model_name}.onnx_data"
                cls.download_to_cache(url_data, name, download=download, suffix=".onnx_data", **kwargs)
            return onnx.load(file_path)

        if model_name.startswith(("http://", "https://")):
            name = _url_cache_name(model_name)
            kwargs["cache_dir"] = kwargs.get("cache_dir") or kornia_config.hub_onnx_dir  # "" is the default too
            file_path = cls.download_to_cache(model_name, name, download=download, suffix=".onnx", **kwargs)
            if with_data:
                parts = urlsplit(model_name)
                url_data = urlunsplit(parts._replace(path=_onnx_data_name(parts.path)))
                cls.download_to_cache(url_data, _onnx_data_name(name), download=download, suffix=".onnx_data", **kwargs)
            return onnx.load(file_path)

        if os.path.exists(model_name):
            return onnx.load(model_name)

        raise ValueError(f"File {model_name} not found")

    @staticmethod
    def _fetch_repo_contents(folder: str) -> list[dict[str, Any]]:
        """Fetch the contents of the Hugging Face repository using the Hugging Face API.

        Returns:
            A list of all files in the repository as dictionaries containing file details.

        """
        url = f"https://huggingface.co/api/models/kornia/ONNX_models/tree/main/{folder}"

        try:
            # url is a fixed https:// literal built from a hardcoded Hugging Face API prefix.
            with urlopen(url, timeout=10) as response:
                return json.loads(response.read())  # Returns the JSON content of the repo
        except urllib.error.HTTPError as e:
            raise ValueError(f"Failed to fetch repository contents: {e.code}") from e

    @classmethod
    def list_operators(cls) -> None:
        """List all available ONNX operators in the 'operators' folder of the Hugging Face repository."""
        repo_contents = cls._fetch_repo_contents("operators")

        # Filter for operators in the 'operators' directory
        operators = [file["path"] for file in repo_contents]

        pprint.pp(operators)

    @classmethod
    def list_models(cls) -> None:
        """List all available ONNX models in the 'models' folder of the Hugging Face repository."""
        repo_contents = cls._fetch_repo_contents("models")

        # Filter for models in the 'models' directory
        models = [file["path"] for file in repo_contents]

        pprint.pp(models)


def io_name_conversion(
    onnx_model: onnx.ModelProto,
    io_name_mapping: dict[str, str],
) -> onnx.ModelProto:
    """Convert the input and output names of an ONNX model to 'input' and 'output'.

    Args:
        onnx_model: The ONNX model to convert.
        io_name_mapping: A dictionary mapping the original input and output names to the new ones.

    """
    # Convert I/O nodes
    for i in range(len(onnx_model.graph.input)):
        in_name = onnx_model.graph.input[i].name
        if in_name in io_name_mapping:
            onnx_model.graph.input[i].name = io_name_mapping[in_name]

    for i in range(len(onnx_model.graph.output)):
        out_name = onnx_model.graph.output[i].name
        if out_name in io_name_mapping:
            onnx_model.graph.output[i].name = io_name_mapping[out_name]

    # Convert intermediate nodes
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].input)):
            node_in_name = onnx_model.graph.node[i].input[j]
            if node_in_name in io_name_mapping:
                onnx_model.graph.node[i].input[j] = io_name_mapping[node_in_name]

        for j in range(len(onnx_model.graph.node[i].output)):
            node_out_name = onnx_model.graph.node[i].output[j]
            if node_out_name in io_name_mapping:
                onnx_model.graph.node[i].output[j] = io_name_mapping[node_out_name]

    return onnx_model


def add_metadata(
    onnx_model: onnx.ModelProto,
    additional_metadata: Optional[list[tuple[str, str]]] = None,
) -> onnx.ModelProto:
    """Add metadata to an ONNX model.

    The metadata includes the source library (set to "kornia"), the version of kornia,
    and any additional metadata provided as a list of key-value pairs. A key that is already
    present in the model (or earlier in the list) is overwritten rather than repeated, since
    ``onnx.checker.check_model`` rejects duplicate keys. Duplicate keys already in the model, as
    earlier kornia versions wrote them, are merged into one entry with the last value, the one
    onnxruntime reads.

    Args:
        onnx_model: The ONNX model to add metadata to.
        additional_metadata: A list of tuples, where each tuple contains a key and a value
            for the additional metadata to add to the ONNX model. Values are stored as strings.

    Returns:
        The ONNX model with the added metadata.

    """
    if additional_metadata is None:
        additional_metadata = []
    # Rebuilt from a mapping, which keeps each key at its first position with its last value.
    metadata = {metadata_props.key: metadata_props.value for metadata_props in onnx_model.metadata_props}
    for key, value in [
        ("source", "kornia"),
        ("version", kornia.__version__),
        *additional_metadata,
    ]:
        metadata[key] = str(value)
    del onnx_model.metadata_props[:]
    for key, value in metadata.items():
        metadata_props = onnx_model.metadata_props.add()
        metadata_props.key = key
        metadata_props.value = value
    return onnx_model


def onnx_type_to_numpy(onnx_type: str) -> Any:
    type_mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(bool)": np.bool_,
    }
    if onnx_type not in type_mapping:
        raise TypeError(f"ONNX type {onnx_type} not understood")
    return type_mapping[onnx_type]
