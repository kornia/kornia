from future import __annotations__

import os
from typing import Any, Optional

from kornia.core import ImageSequential
from kornia.onnx.utils import ONNXLoader
from kornia.onnx.sequential import ONNXSequential
from kornia.config import kornia_config

from kornia.core.external import onnx
from .preprocessor import PreprocessingLoader


class ONNXComunnityModelLoader:

    def __init__(
        self, model_name: str, model_type: str = "model", preprocessing: bool = True, cache_dir: Optional[str] = None
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.preprocessing = preprocessing
        
        if cache_dir is None:
            cache_dir = kornia_config.hub_onnx_dir
        self.loader = ONNXLoader(cache_dir=os.path.join(cache_dir, self.model_name))

    def load_model(self, download: bool = True, **kwargs: Any) -> "onnx.ModelProto":  # type:ignore
        url = f"https://huggingface.co/onnx-community/{self.model_name}/resolve/main/onnx/{self.model_type}.onnx"
        return self.loader.load_model(url, download=download, **kwargs)

    def load_preprocessing(self,) -> ImageSequential:
        url = f"https://huggingface.co/onnx-community/{self.model_name}/resolve/main/preprocessor_config.json"
        return PreprocessingLoader.from_url(url)

    def load(self, download: bool = True) -> ONNXSequential:
        if self.preprocessing:
            return ONNXSequential(
                self.load_model(download)
            )

        return ONNXSequential(
            self.load_preprocessing().to_onnx(save=False),
            self.load_model(download)
        )
