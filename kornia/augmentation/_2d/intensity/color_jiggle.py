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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _range_bound
from kornia.constants import pi
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation


class ColorJiggle(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a torch.tensor image.

    .. image:: _static/img/ColorJiggle.png

    Args:
        p: probability of applying the transformation.
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_brightness`,
        :func:`kornia.enhance.adjust_contrast`. :func:`kornia.enhance.adjust_saturation`,
        :func:`kornia.enhance.adjust_hue`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 3, 3, 3)
        >>> aug = ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.)
        >>> aug(inputs)
        tensor([[[[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        brightness: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        self.brightness_range: torch.Tensor = _range_bound(brightness, "brightness", center=1.0)
        self.contrast_range: torch.Tensor = _range_bound(contrast, "contrast", center=1.0)
        self.saturation_range: torch.Tensor = _range_bound(saturation, "saturation", center=1.0)
        self.hue_range: torch.Tensor = _range_bound(hue, "hue", bounds=(-0.5, 0.5))

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]

        # Order must match original ColorJiggleGenerator: brightness, contrast, hue, saturation
        if self.same_on_batch:
            brightness_factor = (
                torch.empty(1, device=self.device, dtype=self.dtype)
                .uniform_(self.brightness_range[0].item(), self.brightness_range[1].item())
                .expand(batch_size)
            )
            contrast_factor = (
                torch.empty(1, device=self.device, dtype=self.dtype)
                .uniform_(self.contrast_range[0].item(), self.contrast_range[1].item())
                .expand(batch_size)
            )
            hue_factor = (
                torch.empty(1, device=self.device, dtype=self.dtype)
                .uniform_(self.hue_range[0].item(), self.hue_range[1].item())
                .expand(batch_size)
            )
            saturation_factor = (
                torch.empty(1, device=self.device, dtype=self.dtype)
                .uniform_(self.saturation_range[0].item(), self.saturation_range[1].item())
                .expand(batch_size)
            )
        else:
            brightness_factor = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                self.brightness_range[0].item(), self.brightness_range[1].item()
            )
            contrast_factor = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                self.contrast_range[0].item(), self.contrast_range[1].item()
            )
            hue_factor = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                self.hue_range[0].item(), self.hue_range[1].item()
            )
            saturation_factor = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                self.saturation_range[0].item(), self.saturation_range[1].item()
            )

        return {
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "hue_factor": hue_factor,
            "saturation_factor": saturation_factor,
            "order": torch.randperm(4, dtype=torch.long),
        }

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        transforms = [
            lambda img: adjust_brightness(img, params["brightness_factor"] - 1)
            if (params["brightness_factor"] - 1 != 0).any()
            else img,
            lambda img: adjust_contrast(img, params["contrast_factor"])
            if (params["contrast_factor"] != 1).any()
            else img,
            lambda img: adjust_saturation(img, params["saturation_factor"])
            if (params["saturation_factor"] != 1).any()
            else img,
            lambda img: adjust_hue(img, params["hue_factor"] * 2 * pi) if (params["hue_factor"] != 0).any() else img,
        ]

        jittered = input
        for idx in params["order"].tolist():
            t = transforms[idx]
            jittered = t(jittered)

        return jittered
