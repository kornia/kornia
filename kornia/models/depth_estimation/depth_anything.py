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

from typing import Optional

from kornia.core import ImageSequential
from kornia.models._hf_models import HFONNXComunnityModelLoader
from kornia.models._hf_models.preprocessor import _DPTResize

from .base import DepthEstimation

__all__ = ["DepthAnythingONNXBuilder"]


class DepthAnythingONNXBuilder:
    """Provide static methods to build DepthAnything models for ONNX inference."""

    @staticmethod
    def build(
        model_name: str = "depth-anything-v2-small",
        model_type: str = "model",
        cache_dir: Optional[str] = None,
        *,
        force_download: bool = False,
    ) -> DepthEstimation:
        """Build a ``DepthEstimation`` around the onnx-community graph (downloaded on first use).

        The returned model resizes each image to the size transformers' ``DPTImageProcessor`` computes from the
        repository's ``preprocessor_config.json``: it keeps the aspect ratio and makes each side a multiple of
        ``ensure_multiple_of`` when the config asks for it. The filter is the one of transformers' DPT image
        processor (its torchvision backend, ``torchvision.transforms.v2.functional.resize`` with ``antialias=True``),
        which approximates PIL's ``Image.resize``. The output size depends on the input size, so the resize runs in
        torch, as the model's ``resize`` stage. The rescale and normalize steps are exported to an ONNX graph that
        runs before the model's.

        Args:
            model_name: The name of the model to be loaded. Valid model names include:
                - `depth-anything-v2-small`
                - `depth-anything-v2-base`
                - `depth-anything-v2-large`
            model_type:
                The type of the model to be loaded. Valid model types include:
                    - `model`
                    - `model_bnb4`
                    - `model_fp16`
                    - `model_int8`
                    - `model_q4`
                    - `model_quantized`
                    - `model_uint8`
            cache_dir:
                The directory where the model should be cached.
            force_download:
                Whether to download the model and its ``preprocessor_config.json`` again and replace the cached
                files, even when they are cached. By default a cached file is used as it is, however old, and one
                that does not load fails every time until it is deleted.

        Returns:
            DepthEstimation: The depth estimation model.

        Note:
            To use this model, load an image tensor (shape: ``(B, 3, H, W)``) and call ``model.save(image)``.

        """
        if model_name not in [
            "depth-anything-v2-small",
            "depth-anything-v2-base",
            "depth-anything-v2-large",
        ]:
            raise ValueError(f"{model_name} is not a valid model name.")
        loader = HFONNXComunnityModelLoader(model_name, model_type=model_type, cache_dir=cache_dir)
        onnx_model = loader.load_model(
            download=True,
            io_name_mapping={"pixel_values": "input", "predicted_depth": "output"},
            force_download=force_download,
        )
        preprocessing = loader.load_preprocessing(force_download=force_download)
        resize = next((m for m in preprocessing if isinstance(m, _DPTResize)), None)
        others = [m for m in preprocessing if not isinstance(m, _DPTResize)]
        preproc = ImageSequential(*others).to_onnx(save=False) if others else None
        return DepthEstimation(onnx_model, pre_processor=preproc, name=f"{model_name}_{model_type}", resize=resize)
