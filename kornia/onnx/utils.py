import logging
import os
import pprint
import urllib.request
from typing import Any, Dict, List, Optional

import requests

from kornia.core.external import onnx
from kornia.config import kornia_config
__all__ = ["ONNXLoader"]

logger = logging.getLogger(__name__)


class ONNXLoader:
    f"""Manages ONNX models, handling local caching, downloading from Hugging Face, and loading models.

    Attributes:
        cache_dir: The directory where ONNX models are cached locally.
            Defaults to None, which will use a default `{kornia_config.hub_onnx_dir}` directory.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def _get_file_path(self, model_name: str, cache_dir: Optional[str]) -> str:
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
        file_name = f"{model_name.split('/')[-1]}.onnx"
        file_path = os.path.join(cache_dir, "/".join(model_name.split("/")[:-1]), file_name)
        return file_path

    def load_model(self, model_name: str, download: bool = True, **kwargs) -> "onnx.ModelProto":  # type:ignore
        """Loads an ONNX model from the local cache or downloads it from Hugging Face if necessary.

        Args:
            model_name: The name of the ONNX model or operator. For Hugging Face-hosted models,
                use the format 'hf://model_name'. Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models.
            download: If True, the model will be downloaded from Hugging Face if it's not already in the local cache.
            **kwargs: Additional arguments to pass to the download method, if needed.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        if model_name.startswith("hf://"):
            model_name = model_name[len("hf://") :]
            cache_dir = kwargs.get(kornia_config.hub_onnx_dir, None) or self.cache_dir
            file_path = self._get_file_path(model_name, cache_dir)
            if not os.path.exists(file_path):
                # Construct the raw URL for the ONNX file
                if download:
                    url = f"https://huggingface.co/kornia/ONNX_models/resolve/main/{model_name}.onnx"
                    self.download(url, file_path)
                else:
                    raise ValueError(f"`{model_name}` is not found in `{file_path}`. You may set `download=True`.")
            return onnx.load(file_path)  # type:ignore

        if os.path.exists(model_name):
            return onnx.load(model_name)  # type:ignore

        raise ValueError(f"File {model_name} not found")

    def download(
        self,
        url: str,
        file_path: str,
    ) -> None:
        """Downloads an ONNX model from the specified URL and saves it to the specified file path.

        Args:
            url: The URL of the ONNX model to download.
            file_path: The local path where the downloaded model should be saved.
            cache_dir: The directory to use for caching the file, defaults to the instance cache
                directory if not provided.
        """

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
