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

from typing import Any, Optional

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.random_generator._2d import RainGenerator
from kornia.core.check import KORNIA_CHECK


class RandomRain(IntensityAugmentationBase2D):
    r"""Add Random Rain to the image.

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        p: probability of applying the transformation.
        number_of_drops: number of drops per image
        drop_height: Height of the drop in the image (same for each drop in one image). Sampled values must be
            greater than zero and strictly smaller than the input image height.
        drop_width: Width of the drop in the image (same for each drop in one image). The absolute sampled value
            must be strictly smaller than the input image width.
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Convention:
        - the input must have one or three channels; any other channel count raises on the forward pass.
        - ``drop_height`` runs down rows and ``drop_width`` along columns: both name image axes, not
          drop-local ones, and a negative ``drop_width`` slants the drop the other way across the columns.
        - a drop is written as the fixed value ``200 / 255``, not as a function of the image, so the rain is
          darker than every pixel above ``200 / 255`` that it falls on, inside ``[0, 1]`` or not. Every other
          pixel is carried through unclamped.
        - both sizes must be strictly smaller than the image on their own axis. Once the larger of the two
          sizes is at least ``2``, a drop of size ``h`` spans ``h + 1`` rows or columns end to end, so a size
          one short of the image already reaches from edge to edge; a drop whose sizes are both at most ``1``
          is a single pixel. ``span`` is an extent, not a count: the drop is a ``linspace`` of
          ``max(drop_height, abs(drop_width))`` steps truncated to integers, so when the two sizes differ the
          painted cells have gaps inside that span -- ``drop_height=5`` with ``drop_width=0`` on a ``6 x 10``
          image paints rows ``[0, 1, 2, 3, 5]``. A size as large as the image's, or a ``drop_height`` below
          ``1``, raises on the forward pass, where the image shape is known -- constructing it succeeds.
        - a drop's start is uniform over every position that keeps the whole drop inside the image, so every
          row and column, the last ones included, can be painted.
        - the three integer ranges are closed and uniform: every integer from the lower to the upper bound
          is drawn with the same probability, so the default ``drop_height=(5, 20)`` reaches ``20``, the
          default ``drop_width=(-5, 5)`` reaches ``-5`` and ``5``, and ``0`` carries no more weight than
          any other width. Both upper bounds are live against the size rule above, which they were not
          when they were practically never drawn: with the defaults an image 20 pixels tall, or 5 pixels
          wide, now raises on some seeds -- and on every seed once it is 5 pixels tall or shorter, where
          no drawable height is legal. A range that is reversed, fractional or non-finite raises
          ``ValueError`` at construction.
        - ``same_on_batch=True`` gives every sample of the batch the same drop count, the same drop size and
          the same coordinates; left at ``False`` each sample draws its own.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> rain = RandomRain(p=1,drop_height=(1,2),drop_width=(1,2),number_of_drops=(1,1))
        >>> rain(input)
        tensor([[[[0.4963, 0.7843, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.7843, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])

    """

    def __init__(
        self,
        number_of_drops: tuple[int, int] = (1000, 2000),
        drop_height: tuple[int, int] = (5, 20),
        drop_width: tuple[int, int] = (-5, 5),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = RainGenerator(number_of_drops, drop_height, drop_width)

    def apply_transform(
        self,
        image: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Check array and drops size
        KORNIA_CHECK(image.shape[1] in {3, 1}, "Number of color channels should be 1 or 3.")
        KORNIA_CHECK(
            bool(
                torch.all(params["drop_height_factor"] < image.shape[2]) and torch.all(params["drop_height_factor"] > 0)
            ),
            "Height of drop should be greater than zero and less than image height.",
        )

        KORNIA_CHECK(
            bool(torch.all(torch.abs(params["drop_width_factor"]) < image.shape[3])),
            "Width of drop should be less than image width.",
        )
        modeified_img = image.clone()
        for i in range(image.shape[0]):
            number_of_drops: int = int(params["number_of_drops_factor"][i])
            # We generate torch.Tensor with maximum number of drops, and then remove unnecessary drops.

            coordinates_of_drops: torch.Tensor = params["coordinates_factor"][i][:number_of_drops]
            height_of_drop: int = int(params["drop_height_factor"][i])
            width_of_drop: int = int(params["drop_width_factor"][i])

            # Generate how our drop will look like into the image
            size_of_line: int = max(height_of_drop, abs(width_of_drop))
            x = torch.linspace(start=0, end=height_of_drop, steps=size_of_line, dtype=torch.long).to(image.device)
            y = torch.linspace(start=0, end=width_of_drop, steps=size_of_line, dtype=torch.long).to(image.device)

            # A drop may start anywhere its far end stays inside the image. The far end is the line's last
            # offset, which is 0 for a single-pixel drop. The clamp keeps a draw that rounds up to 1.0 (half
            # precision) on the last admissible start.
            last_dy, last_dx = int(x[-1]), int(y[-1])
            rows, cols = image.shape[2] - last_dy, image.shape[3] - abs(last_dx)
            random_y_coords = (coordinates_of_drops[:, 0] * rows).long().clamp(max=rows - 1)
            random_x_coords = (coordinates_of_drops[:, 1] * cols).long().clamp(max=cols - 1) + max(-last_dx, 0)

            coords = torch.stack([random_y_coords, random_x_coords]).to(image.device)
            # Draw lines
            for k in range(x.shape[0]):
                modeified_img[i, :, coords[0] + x[k], coords[1] + y[k]] = 200 / 255
        return modeified_img
