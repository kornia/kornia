from __future__ import annotations

import json
import logging
import os
import pprint
import urllib.request
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import requests

import kornia
from kornia.config import kornia_config
from kornia.core.external import onnx

__all__ = ["ONNXLoader", "io_name_conversion", "add_metadata"]

logger = logging.getLogger(__name__)


class ONNXLoader:
    f"""Manages ONNX models, handling local caching, downloading from Hugging Face, and loading models.

    Attributes:
        cache_dir: The directory where ONNX models are cached locally.
            Defaults to None, which will use a default `{kornia_config.hub_onnx_dir}` directory.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def _get_file_path(self, model_name: str, cache_dir: Optional[str], suffix: str = ".onnx") -> str:
        """Constructs the file path for the ONNX model based on the model name and cache directory.

        Args:
            model_name: The name of the model or operator, typically in the format 'operators/model_name'.
            cache_dir: The directory where the model should be cached.

        Returns:
            str: The full local path where the model should be stored or loaded from.
        """
        # Determine the local file path
        if cache_dir is None:
            if self.cache_dir is not None:
                cache_dir = self.cache_dir
            else:
                cache_dir = kornia_config.hub_onnx_dir

        # The filename is the model name (without directory path)
        if not model_name.endswith(suffix):
            file_name = f"{os.path.split(model_name)[-1]}{suffix}"
        else:
            file_name = os.path.split(model_name)[-1]
        file_path = os.path.join(*cache_dir.split(os.sep), *model_name.split(os.sep)[:-1], file_name)
        return file_path

    def load_config(self, url: str, download: bool = True, **kwargs) -> dict[str, Any]:
        """Loads JSON config from the specified URL.

        Args:
            url: The URL of the preprocessor config to load.
            download: If True, the config will be downloaded if it's not already in the local cache.

        Returns:
            dict[str, Any]: The loaded preprocessor config.
        """
        if url.startswith(("http:", "https:")):
            cache_dir = kwargs.get("cache_dir", None) or self.cache_dir
            file_path = self._get_file_path(os.path.split(url)[-1], cache_dir, suffix=".json")
            self.download(url, file_path, download_if_not_exists=download)
            with open(file_path) as f:
                json_data = json.load(f)
                return json_data

        if not download:
            raise RuntimeError(f"File `{file_path}` not found. You may set `download=True`.")

        raise RuntimeError(f"File `{file_path}` not found.")

    def load_model(self, model_name: str, download: bool = True, **kwargs) -> onnx.ModelProto:  # type:ignore
        """Loads an ONNX model from the local cache or downloads it from Hugging Face if necessary.

        Args:
            model_name: The name of the ONNX model or operator. For Hugging Face-hosted models,
                use the format 'hf://model_name'. Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models.
                Or a URL to the ONNX model.
            download: If True, the model will be downloaded from Hugging Face if it's not already in the local cache.
            **kwargs: Additional arguments to pass to the download method, if needed.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        if model_name.startswith("hf://"):
            model_name = model_name[len("hf://") :]
            cache_dir = kwargs.get("cache_dir", None) or self.cache_dir
            file_path = self._get_file_path(model_name, cache_dir)
            url = f"https://huggingface.co/kornia/ONNX_models/resolve/main/{model_name}.onnx"
            self.download(url, file_path, download_if_not_exists=download)
            return onnx.load(file_path)  # type:ignore

        elif model_name.startswith("http://") or model_name.startswith("https://"):
            cache_dir = kwargs.get("cache_dir", None) or self.cache_dir
            file_path = self._get_file_path(os.path.split(model_name)[-1], cache_dir)
            self.download(model_name, file_path, download_if_not_exists=download)
            return onnx.load(file_path)  # type:ignore

        if os.path.exists(model_name):
            assert False, type(onnx.load(model_name))
            return onnx.load(model_name)  # type:ignore

        raise ValueError(f"File {model_name} not found")

    def download(
        self,
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

    @staticmethod
    def _fetch_repo_contents(folder: str) -> List[Dict[str, Any]]:
        """Fetches the contents of the Hugging Face repository using the Hugging Face API.

        Returns:
            List[dict]: A list of all files in the repository as dictionaries containing file details.
        """
        url = f"https://huggingface.co/api/models/kornia/ONNX_models/tree/main/{folder}"

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            return response.json()  # Returns the JSON content of the repo
        else:
            raise ValueError(f"Failed to fetch repository contents: {response.status_code}")

    @staticmethod
    def list_operators() -> None:
        """Lists all available ONNX operators in the 'operators' folder of the Hugging Face repository."""
        repo_contents = ONNXLoader._fetch_repo_contents("operators")

        # Filter for operators in the 'operators' directory
        operators = [file["path"] for file in repo_contents]

        pprint.pp(operators)

    @staticmethod
    def list_models() -> None:
        """Lists all available ONNX models in the 'models' folder of the Hugging Face repository."""
        repo_contents = ONNXLoader._fetch_repo_contents("models")

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
    additional_metadata: List[tuple[str, str]] = None,
) -> onnx.ModelProto:  # type: ignore
    for key, value in [
        ("source", "kornia"),
        ("version", kornia.__version__),
        *additional_metadata,
    ]:
        metadata_props = onnx_model.metadata_props.add()
        metadata_props.key = key
        metadata_props.value = str(value)
    return onnx_model
