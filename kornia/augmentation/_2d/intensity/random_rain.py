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

from typing import Any, Dict, Optional, Tuple

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core.check import KORNIA_CHECK


class RandomRain(IntensityAugmentationBase2D):
    r"""Add Random Rain to the image.

    Args:
        p: probability of applying the transformation.
        number_of_drops: number of drops per image
        drop_height: height of the drop in image(same for each drops in one image)
        drop_width: width of the drop in image(same for each drops in one image)
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> rain = RandomRain(p=1,drop_height=(1,2),drop_width=(1,2),number_of_drops=(1,1))
        >>> rain(input)
        tensor([[[[0.4963, 0.7843, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])

    """

    def __init__(
        self,
        number_of_drops: Tuple[int, int] = (1000, 2000),
        drop_height: Tuple[int, int] = (5, 20),
        drop_width: Tuple[int, int] = (-5, 5),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.number_of_drops = number_of_drops
        self.drop_height = drop_height
        self.drop_width = drop_width

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _device, _dtype = self.device, self.dtype

        # Sample number of drops, drop height, drop width
        if self.same_on_batch:
            number_of_drops_factor = (
                torch.empty(1, device=_device)
                .uniform_(float(self.number_of_drops[0]), float(self.number_of_drops[1] + 1))
                .long()
                .expand(batch_size)
            )
            drop_height_factor = (
                torch.empty(1, device=_device)
                .uniform_(float(self.drop_height[0]), float(self.drop_height[1] + 1))
                .long()
                .expand(batch_size)
            )
            drop_width_factor = (
                torch.empty(1, device=_device)
                .uniform_(float(self.drop_width[0]), float(self.drop_width[1] + 1))
                .long()
                .expand(batch_size)
            )
        else:
            number_of_drops_factor = (
                torch.empty(batch_size, device=_device)
                .uniform_(float(self.number_of_drops[0]), float(self.number_of_drops[1] + 1))
                .long()
            )
            drop_height_factor = (
                torch.empty(batch_size, device=_device)
                .uniform_(float(self.drop_height[0]), float(self.drop_height[1] + 1))
                .long()
            )
            drop_width_factor = (
                torch.empty(batch_size, device=_device)
                .uniform_(float(self.drop_width[0]), float(self.drop_width[1] + 1))
                .long()
            )

        # Sample coordinates for drops
        max_drops = int(number_of_drops_factor.max().item()) if number_of_drops_factor.numel() > 0 else 0
        if self.same_on_batch:
            coordinates_factor = (
                torch.rand(1, max_drops, 2, device=_device, dtype=_dtype).expand(batch_size, -1, -1).contiguous()
            )
        else:
            coordinates_factor = torch.rand(batch_size, max_drops, 2, device=_device, dtype=_dtype)

        return {
            "number_of_drops_factor": number_of_drops_factor,
            "coordinates_factor": coordinates_factor,
            "drop_height_factor": drop_height_factor,
            "drop_width_factor": drop_width_factor,
        }

    def apply_transform(
        self,
        image: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Check array and drops size
        KORNIA_CHECK(image.shape[1] in {3, 1}, "Number of color channels should be 1 or 3.")
        KORNIA_CHECK(
            bool(
                torch.all(params["drop_height_factor"] <= image.shape[2])
                and torch.all(params["drop_height_factor"] > 0)
            ),
            "Height of drop should be greater than zero and less than image height.",
        )

        KORNIA_CHECK(
            bool(torch.all(torch.abs(params["drop_width_factor"]) <= image.shape[3])),
            "Width of drop should be less than image width.",
        )
        modeified_img = image.clone()
        for i in range(image.shape[0]):
            number_of_drops: int = int(params["number_of_drops_factor"][i])
            # We generate torch.tensor with maximum number of drops, and then remove unnecessary drops.

            coordinates_of_drops: torch.Tensor = params["coordinates_factor"][i][:number_of_drops]
            height_of_drop: int = int(params["drop_height_factor"][i])
            width_of_drop: int = int(params["drop_width_factor"][i])

            # Generate start coordinates for each drop
            random_y_coords = coordinates_of_drops[:, 0] * (image.shape[2] - height_of_drop - 1)
            if width_of_drop > 0:
                random_x_coords = coordinates_of_drops[:, 1] * (image.shape[3] - width_of_drop - 1)
            else:
                random_x_coords = coordinates_of_drops[:, 1] * (image.shape[3] + width_of_drop - 1) - width_of_drop

            coords = torch.cat([random_y_coords[None], random_x_coords[None]], dim=0).to(image.device, dtype=torch.long)

            # Generate how our drop will look like into the image
            size_of_line: int = max(height_of_drop, abs(width_of_drop))
            x = torch.linspace(start=0, end=height_of_drop, steps=size_of_line, dtype=torch.long).to(image.device)
            y = torch.linspace(start=0, end=width_of_drop, steps=size_of_line, dtype=torch.long).to(image.device)
            # Draw lines
            for k in range(x.shape[0]):
                modeified_img[i, :, coords[0] + x[k], coords[1] + y[k]] = 200 / 255
        return modeified_img
