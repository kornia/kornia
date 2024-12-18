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

from typing import Any, Dict, Optional, Tuple

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.filters import StableDiffusionDissolving


class RandomDissolving(IntensityAugmentationBase2D):
    r"""Perform dissolving transformation using StableDiffusion models.

    Based on :cite:`shi2024dissolving`, the dissolving transformation is essentially applying one-step
    reverse diffusion. Our implementation currently supports HuggingFace implementations of SD 1.4, 1.5
    and 2.1. SD 1.X tends to remove more details than SD2.1.

    .. list-table:: Title
        :widths: 32 32 32
        :header-rows: 1

        * - SD 1.4
          - SD 1.5
          - SD 2.1
        * - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-1.4.png
          - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-1.5.png
          - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-2.1.png

    Args:
        p: probability of applying the transformation.
        version: the version of the stable diffusion model.
        step_range: the step range of the diffusion model steps. Higher the step, stronger
                    the dissolving effects.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
        **kwargs: additional arguments for `.from_pretrained` for HF StableDiffusionPipeline.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        - Output: :math:`(B, C, H, W)`

    """

    def __init__(
        self,
        step_range: Tuple[float, float] = (100, 500),
        version: str = "2.1",
        p: float = 0.5,
        keepdim: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(p=p, same_on_batch=True, keepdim=keepdim)
        self.step_range = step_range
        self._dslv = StableDiffusionDissolving(version, **kwargs)
        self._param_generator = rg.PlainUniformGenerator((self.step_range, "step_range_factor", None, None))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return self._dslv(input, params["step_range_factor"][0].long().item())
