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

from typing import Any

from kornia.core import ImageSequential, Module, Tensor, tensor
from kornia.enhance.normalize import Normalize
from kornia.enhance.rescale import Rescale
from kornia.geometry.transform import Resize


class PreprocessingLoader:
    @staticmethod
    def normalize(mean: Tensor, std: Tensor) -> Normalize:
        return Normalize(mean=mean, std=std)

    @staticmethod
    def rescale(rescale_factor: float) -> Rescale:
        return Rescale(factor=rescale_factor)

    @staticmethod
    def resize(width: int, height: int) -> Resize:
        return Resize((height, width))

    @staticmethod
    def from_json(req: dict[str, Any]) -> ImageSequential:
        if req["image_processor_type"] == "DPTImageProcessor":
            return DPTImageProcessor.from_json(req)
        raise RuntimeError(f"Unsupported image processor type: {req['image_processor_type']}")


class DPTImageProcessor(PreprocessingLoader):
    @staticmethod
    def from_json(json_data: dict[str, Any]) -> ImageSequential:
        preproc_list: list[Module] = []
        if json_data["do_pad"]:
            raise NotImplementedError
        if json_data["do_resize"]:
            # Missing some parameters such as `ensure_multiple_of`, `keep_aspect_ratio`
            preproc_list.append(
                PreprocessingLoader.resize(width=json_data["size"]["width"], height=json_data["size"]["height"])
            )
        if json_data["do_rescale"]:
            preproc_list.append(PreprocessingLoader.rescale(rescale_factor=json_data["rescale_factor"] * 255))
        if json_data["do_normalize"]:
            preproc_list.append(
                PreprocessingLoader.normalize(
                    mean=tensor([json_data["image_mean"]]), std=tensor([json_data["image_std"]])
                )
            )
        return ImageSequential(*preproc_list)
