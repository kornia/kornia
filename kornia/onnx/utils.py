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
from typing import Any, Optional, Union
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


def _path_text(method: str, value: Any) -> str:
    """Return the text of ``value``, a ``str`` or an :class:`os.PathLike` naming a ``str`` path.

    Anything else raises ``TypeError``. ``bytes`` are not a path here: ONNX Runtime reads ``bytes`` as a serialized
    model, so the message names the call for that.
    """
    text = os.fspath(value) if isinstance(value, os.PathLike) else value
    if isinstance(text, str):
        return text
    hint = " For a serialized model, pass onnx.load_model_from_string(data)." if isinstance(text, bytes) else ""
    raise TypeError(f"{method} takes a str or an os.PathLike, got {type(value).__name__}.{hint}")


def _onnx_data_name(name: str) -> str:
    """Return the name of the ``.onnx_data`` weights stored beside the model ``name``, with or without ``.onnx``."""
    return (name[: -len(".onnx")] if name.endswith(".onnx") else name) + ".onnx_data"


def _cached_file_hint(file_path: str) -> str:
    """Return what to do when the cached file ``file_path`` does not load."""
    return (
        f"If the cached file `{file_path}` is damaged, or was replaced by hand, pass force_download=True to download "
        "it again, or delete the file."
    )


def _load_cached_onnx(file_path: str) -> onnx.ModelProto:
    """Load the ONNX model cached at ``file_path``; the error for a file that does not load names the way out.

    The error is raised as it is, so a caller that catches ``google.protobuf.message.DecodeError`` still does; the
    hint is a note on it (:meth:`BaseException.add_note`), which a traceback prints.
    """
    try:
        return onnx.load(file_path)
    except MemoryError:  # a file too large to load is not a damaged one
        raise
    except Exception as exc:
        exc.add_note(_cached_file_hint(file_path))
        raise


def _load_cached_json(file_path: str) -> Any:
    """Load the JSON cached at ``file_path``; the error for a file that does not parse names the way out.

    The error is raised as it is, so a caller that catches ``json.JSONDecodeError`` still does; the hint is a note on
    it (:meth:`BaseException.add_note`), which a traceback prints.
    """
    try:
        with open(file_path, "rb") as f:  # json.load detects UTF-8, -16 or -32 in bytes, whatever the locale
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:  # both are ValueErrors
        exc.add_note(_cached_file_hint(file_path))
        raise


class ONNXLoader(CachedDownloader):
    """Manages ONNX models, handling local caching, downloading from Hugging Face, and loading models."""

    @classmethod
    def load_config(
        cls, url: Union[str, os.PathLike], download: bool = True, *, force_download: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Load JSON config from the specified URL or local file.

        A config that is in the cache is used as it is, however old: pass ``force_download=True`` to fetch it again.

        Args:
            url: The HTTP(S) URL of the preprocessor config to load, or the path of a local JSON file, which is read
                as it is: a ``str``, or an :class:`os.PathLike`, which always names a local file. A local path may
                start with ``~``.
            download: If True, the config will be downloaded if it's not already in the local cache. A local file
                is read either way.
            force_download: If True, fetch an HTTP(S) config again and replace the cached file, even when it is
                cached. Needs ``download=True``. It has no effect on a local file. Defaults to False.
            kwargs: Keyword arguments of ``CachedDownloader.download_to_cache``. Only ``cache_dir`` applies: the root
                the config is stored under, at ``<scheme>/<host>/<directories of the URL's path>/<file name>``, with a
                ``%3F<hash>`` directory before the file name when the URL has a query, as for ``load_model``. ``None``
                or ``""`` means ``kornia_config.hub_cache_dir``. Any other keyword raises ``TypeError``, whatever
                ``url`` is.

        Returns:
            dict[str, Any]: The loaded preprocessor config.

        Raises:
            RuntimeError: If ``url`` is neither an HTTP(S) URL nor an existing local file.
            TypeError: If ``url`` is not a ``str`` or an :class:`os.PathLike`; ``bytes`` raise it too.

        """
        _check_keywords("ONNXLoader.load_config", kwargs)
        local = isinstance(url, os.PathLike)  # a path object names a local file, whatever its text
        url = _path_text("ONNXLoader.load_config", url)
        if not local and url.startswith(("http:", "https:")):
            kwargs["cache_dir"] = kwargs.get("cache_dir") or None  # "" is the default directory, as for load_model
            file_path = cls.download_to_cache(
                url, _url_cache_name(url), download=download, suffix=".json", force_download=force_download, **kwargs
            )
            return _load_cached_json(file_path)

        path = os.path.expanduser(url)
        if os.path.isfile(path):
            with open(path, "rb") as f:  # json.load decodes UTF-8, -16 or -32 bytes, whatever the locale
                return json.load(f)

        raise RuntimeError(
            f"`{url}` is neither an HTTP(S) URL nor an existing local file: ONNXLoader.load_config reads only those."
        )

    @classmethod
    def load_model(
        cls,
        model_name: Union[str, os.PathLike],
        download: bool = True,
        with_data: bool = False,
        *,
        force_download: bool = False,
        **kwargs,
    ) -> onnx.ModelProto:
        """Load an ONNX model from the local cache or downloads it from Hugging Face if necessary.

        A model that is in the cache is used as it is, however old, and a cached file that does not load fails every
        time: pass ``force_download=True`` to fetch it again, or delete the file.

        Args:
            model_name: The name of the ONNX model or operator. For Hugging Face-hosted models,
                use the format 'hf://model_name'. Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models.
                Or a URL to the ONNX model, or the path of a local file. A ``str`` can be any of those; an
                :class:`os.PathLike` always names a local file. A local path may start with ``~``.
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
            force_download: If True, fetch the model, and its `.onnx_data` weights with ``with_data``, again and
                replace the cached files, even when they are cached. A file is replaced only once its download is
                complete, so a download that fails leaves the cached one as it was. The model is replaced before the
                weights are fetched: if the weights then fail to download, the cache holds the new model with the old
                weights, so call again with ``force_download=True`` before loading it. Needs ``download=True``. It has
                no effect on a model given as a path to a local file, which is never downloaded. Defaults to False.
            **kwargs: Only ``cache_dir``; any other keyword raises ``TypeError``, whatever ``model_name`` is.

        Returns:
            onnx.ModelProto: The loaded ONNX model.

        Raises:
            TypeError: If ``model_name`` is not a ``str`` or an :class:`os.PathLike`; ``bytes`` raise it too.

        """
        _check_keywords("ONNXLoader.load_model", kwargs)
        local = isinstance(model_name, os.PathLike)  # a path object names a local file, whatever its text
        model_name = _path_text("ONNXLoader.load_model", model_name)
        if not local and model_name.startswith("hf://"):
            model_name = model_name[len("hf://") :]
            url = f"https://huggingface.co/kornia/ONNX_models/resolve/main/{model_name}.onnx"
            if kwargs.get("cache_dir"):
                name = model_name.partition("/")[2]  # under an explicit cache_dir, the path after the folder
            else:
                kwargs["cache_dir"] = kornia_config.hub_onnx_dir
                name = model_name  # "<folder>/<a>/<b>": the folder is checked as the other components are
            file_path = cls.download_to_cache(
                url, name, download=download, suffix=".onnx", force_download=force_download, **kwargs
            )
            if with_data:
                url_data = f"https://huggingface.co/kornia/ONNX_models/resolve/main/{model_name}.onnx_data"
                cls.download_to_cache(
                    url_data, name, download=download, suffix=".onnx_data", force_download=force_download, **kwargs
                )
            return _load_cached_onnx(file_path)

        if not local and model_name.startswith(("http://", "https://")):
            name = _url_cache_name(model_name)
            kwargs["cache_dir"] = kwargs.get("cache_dir") or kornia_config.hub_onnx_dir  # "" is the default too
            file_path = cls.download_to_cache(
                model_name, name, download=download, suffix=".onnx", force_download=force_download, **kwargs
            )
            if with_data:
                parts = urlsplit(model_name)
                url_data = urlunsplit(parts._replace(path=_onnx_data_name(parts.path)))
                cls.download_to_cache(
                    url_data,
                    _onnx_data_name(name),
                    download=download,
                    suffix=".onnx_data",
                    force_download=force_download,
                    **kwargs,
                )
            return _load_cached_onnx(file_path)

        path = os.path.expanduser(model_name)
        if os.path.exists(path):
            return onnx.load(path)

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
    """Rename the inputs and outputs of an ONNX model in place, as ``io_name_mapping`` maps them.

    Any mapping applies, such as ``{"pixel_values": "input", "predicted_depth": "output"}`` for the default names that
    :class:`~kornia.onnx.sequential.ONNXSequential` connects. The graph's inputs and outputs and every input and
    output of its nodes with a mapped name are renamed; names absent from the model are ignored. The initializers,
    the ``value_info`` entries and the nodes of subgraphs (the branches of an ``If``, the body of a ``Loop``) are not
    renamed, so a mapped name that is also an initializer leaves its consumers reading a name that no longer exists.

    Args:
        onnx_model: The ONNX model to rename, modified in place.
        io_name_mapping: A dictionary mapping the original input and output names to the new ones.

    Returns:
        The same ``onnx_model`` object, renamed.

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
        ``onnx_model`` itself, with the added metadata: the metadata is written in place.

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
