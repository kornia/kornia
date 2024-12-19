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

from torch import nn

from kornia.core import tensor
from kornia.enhance.normalize import Normalize
from kornia.filters.dexined import DexiNed
from kornia.models.edge_detection.base import EdgeDetector
from kornia.models.utils import ResizePostProcessor, ResizePreProcessor

__all__ = ["DexiNedBuilder"]


class DexiNedBuilder:
    """DexiNedBuilder is a class that builds a DexiNed model.

    .. code-block:: python

        images = kornia.utils.sample.get_sample_images()
        model = DexiNedBuilder.build()
        model.save(images)
    """

    @staticmethod
    def build(model_name: str = "dexined", pretrained: bool = True, image_size: int = 352) -> EdgeDetector:
        if model_name.lower() == "dexined":
            # Normalize then scale to [0, 255]
            norm = Normalize(mean=tensor([[0.485, 0.456, 0.406]]), std=tensor([[1.0 / 255.0] * 3]))
            model = nn.Sequential(norm, DexiNed(pretrained=pretrained), nn.Sigmoid())
        else:
            raise ValueError(f"Model {model_name} not found. Please choose from 'DexiNed'.")

        return EdgeDetector(
            model,
            ResizePreProcessor(image_size, image_size),
            ResizePostProcessor(),
            name="dexined",
        )
