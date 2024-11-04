from __future__ import annotations

import json
import logging
import os
import pprint
from typing import Any

import kornia
from kornia.config import kornia_config
from kornia.core.external import numpy as np
from kornia.core.external import onnx, requests
from kornia.utils.download import CachedDownloader

__all__ = ["ONNXLoader", "io_name_conversion", "add_metadata"]

logger = logging.getLogger(__name__)


class ONNXLoader(CachedDownloader):
    """Manages ONNX models, handling local caching, downloading from Hugging Face, and loading models."""

    @classmethod
    def load_config(cls, url: str, download: bool = True, **kwargs: Any) -> dict[str, Any]:
        """Loads JSON config from the specified URL.

        Args:
            url: The URL of the preprocessor config to load.
            download: If True, the config will be downloaded if it's not already in the local cache.

        Returns:
            dict[str, Any]: The loaded preprocessor config.
        """
        if url.startswith(("http:", "https:")):
            file_path = cls.download_to_cache(
                url,
                os.path.split(url)[-1],
                download=download,
                suffix=".json",
                **kwargs,
            )
            with open(file_path) as f:
                json_data = json.load(f)
                return json_data

        if not download:
            raise RuntimeError(f"File `{url}` not found. You may set `download=True`.")

        raise RuntimeError(f"File `{file_path}` not found.")

    @classmethod
    def load_model(cls, model_name: str, download: bool = True, with_data: bool = False, **kwargs) -> onnx.ModelProto:  # type:ignore
        """Loads an ONNX model from the local cache or downloads it from Hugging Face if necessary.

        Args:
            model_name: The name of the ONNX model or operator. For Hugging Face-hosted models,
                use the format 'hf://model_name'. Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models.
                Or a URL to the ONNX model.
            download: If True, the model will be downloaded from Hugging Face if it's not already in the local cache.
            cache_dir: The directory where the model should be cached.
                Defaults to None, which will use a default `{kornia_config.hub_onnx_dir}` directory.
            with_data: If True, the model will be loaded with its `.onnx_data` weights.
            **kwargs: Additional arguments to pass to the download method, if needed.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        if model_name.startswith("hf://"):
            model_name = model_name[len("hf://") :]
            url = f"https://huggingface.co/kornia/ONNX_models/resolve/main/{model_name}.onnx"
            cache_dir = kwargs.get("cache_dir", None) or os.path.join(
                kornia_config.hub_onnx_dir, model_name.split("/")[0]
            )
            kwargs.update({"cache_dir": cache_dir})
            file_path = cls.download_to_cache(
                url, model_name.split("/")[1], download=download, suffix=".onnx", **kwargs
            )
            if with_data:
                url_data = f"https://huggingface.co/kornia/ONNX_models/resolve/main/{model_name}.onnx_data"
                cls.download_to_cache(
                    url_data, model_name.split("/")[1], download=download, suffix=".onnx_data", **kwargs
                )
            return onnx.load(file_path)  # type:ignore

        elif model_name.startswith("http://") or model_name.startswith("https://"):
            cache_dir = kwargs.get("cache_dir", None) or kornia_config.hub_onnx_dir
            kwargs.update({"cache_dir": cache_dir})
            file_path = cls.download_to_cache(
                model_name,
                os.path.split(model_name)[-1],
                download=download,
                suffix=".onnx",
                **kwargs,
            )
            if with_data:
                url_data = model_name[:-5] + ".onnx_data"
                cls.download_to_cache(
                    url_data,
                    os.path.split(model_name)[-1][:-5] + ".onnx_data",
                    download=download,
                    suffix=".onnx_data",
                    **kwargs,
                )
            return onnx.load(file_path)  # type:ignore

        elif os.path.exists(model_name):
            return onnx.load(model_name)  # type:ignore

        raise ValueError(f"File {model_name} not found")

    @staticmethod
    def _fetch_repo_contents(folder: str) -> list[dict[str, Any]]:
        """Fetches the contents of the Hugging Face repository using the Hugging Face API.

        Returns:
            A list of all files in the repository as dictionaries containing file details.
        """
        url = f"https://huggingface.co/api/models/kornia/ONNX_models/tree/main/{folder}"

        response = requests.get(url, timeout=10)  # type:ignore

        if response.status_code == 200:
            return response.json()  # Returns the JSON content of the repo
        else:
            raise ValueError(f"Failed to fetch repository contents: {response.status_code}")

    @classmethod
    def list_operators(cls) -> None:
        """Lists all available ONNX operators in the 'operators' folder of the Hugging Face repository."""
        repo_contents = cls._fetch_repo_contents("operators")

        # Filter for operators in the 'operators' directory
        operators = [file["path"] for file in repo_contents]

        pprint.pp(operators)

    @classmethod
    def list_models(cls) -> None:
        """Lists all available ONNX models in the 'models' folder of the Hugging Face repository."""
        repo_contents = cls._fetch_repo_contents("models")

        # Filter for models in the 'models' directory
        models = [file["path"] for file in repo_contents]

        pprint.pp(models)


def io_name_conversion(
    onnx_model: onnx.ModelProto,  # type:ignore
    io_name_mapping: dict[str, str],
) -> onnx.ModelProto:  # type:ignore
    """Converts the input and output names of an ONNX model to 'input' and 'output'.

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
            if onnx_model.graph.node[i].input[j] in io_name_mapping:
                onnx_model.graph.node[i].input[j] = io_name_mapping[in_name]

    for j in range(len(onnx_model.graph.node[i].output)):
        if onnx_model.graph.node[i].output[j] in io_name_mapping:
            onnx_model.graph.node[i].output[j] = io_name_mapping[out_name]

    return onnx_model


def add_metadata(
    onnx_model: onnx.ModelProto,  # type: ignore
    additional_metadata: list[tuple[str, str]] = [],
) -> onnx.ModelProto:  # type: ignore
    """Adds metadata to an ONNX model.

    The metadata includes the source library (set to "kornia"), the version of kornia,
    and any additional metadata provided as a list of key-value pairs.

    Args:
        onnx_model: The ONNX model to add metadata to.
        additional_metadata: A list of tuples, where each tuple contains a key and a value
            for the additional metadata to add to the ONNX model.

    Returns:
        The ONNX model with the added metadata.
    """
    for key, value in [
        ("source", "kornia"),
        ("version", kornia.__version__),
        *additional_metadata,
    ]:
        metadata_props = onnx_model.metadata_props.add()
        metadata_props.key = key
        metadata_props.value = str(value)
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
