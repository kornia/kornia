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

from kornia.models._hf_models import HFONNXComunnityModelLoader

from .base import DepthEstimation

__all__ = ["DepthAnythingONNXBuilder"]


class DepthAnythingONNXBuilder:
    @staticmethod
    def build(
        model_name: str = "depth-anything-v2-small", model_type: str = "model", cache_dir: Optional[str] = None
    ) -> DepthEstimation:
        """Export a DepthAnything model to an ONNX model file.

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

        Returns:
            str: The name of the output ONNX file.

        .. code-block:: python

            images = kornia.utils.sample.get_sample_images()
            model = DepthAnythingONNXBuilder.build()
            model.save(images)

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
        )
        preproc = loader.load_preprocessing().to_onnx(save=False)
        return DepthEstimation(onnx_model, pre_processor=preproc, name=f"{model_name}_{model_type}")
