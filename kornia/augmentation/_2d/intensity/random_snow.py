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

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.color import hls_to_rgb, rgb_to_hls
from kornia.core.check import KORNIA_CHECK


class RandomSnow(IntensityAugmentationBase2D):
    r"""Generates snow effect on given torch.Tensor image or a batch torch.Tensor images.

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        snow_coefficient: A tuple of floats (lower and upper bound) between 0 and 1 that control
        the amount of snow to add to the image, the larger value corresponds to the more snow.
        brightness: A tuple of floats (lower and upper bound) of ``1`` or greater that controls the
        brightness of the snow.
        same_on_batch: If True, apply the same transformation to each image in a batch. Default: False.
        p: Probability of applying the transformation. Default: 0.5.
        keepdim: Keep the output torch.Tensor with the same shape as input. Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Convention:
        - the input must have three channels: the effect is computed in HLS, and any other channel count
          raises on the forward pass.
        - ``snow_coefficient`` is checked against ``[0, 1]`` at construction, where ``brightness`` must be
          ``1`` or greater.
        - one ``snow_coefficient`` and one ``brightness`` are drawn per sample; ``same_on_batch=True``
          collapses both to a single value for the batch.
        - the output as a whole is not clamped. Only a snow-covered pixel -- one whose lightness is below the
          drawn ``snow_coefficient`` -- has its lightness scaled by ``brightness`` and clamped into ``[0, 1]``.
          A covered pixel comes back white once its scaled lightness reaches ``1``, and black when its
          lightness is zero or negative. A pixel the snow misses still goes through the HLS round trip
          unclamped, so one above ``1`` usually comes back above it. The two exceptions are the pixels where
          ``rgb_to_hls``'s saturation denominator vanishes, which collapse whether the snow covers them or not:
          lightness exactly ``1``, where ``2 - max - min`` is zero and ``(1.5, 0.5, 0.5)`` comes back white,
          and its mirror at lightness exactly ``0``, where ``max + min`` is zero and ``(2.0, -2.0, -2.0)``
          comes back black. The collapse is at the point, not around
          it: ``(1.4, 0.5, 0.5)`` and a lightness more than about ``1e-7`` off ``1`` come back close to their
          input. Within ``rgb_to_hls``'s ``eps`` of ``1e-8`` of the point, which only ``float64`` can
          represent, they do not: a lightness of ``1 + 1e-8`` comes back as ``(2, 0, 0)`` and ``1 + 5e-9`` as
          values of order ``1e8``. In half precision a value just above ``1`` can round onto the singular point.

    .. warning::
        An input whose values are all negative comes back as an all-zero image, and a pixel whose lightness is
        zero or negative comes back black in any image -- including an out-of-range pixel such as
        ``(2.0, -2.0, -2.0)``, which is neither all-negative nor has a negative lightness. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

    Examples:
        >>> inputs = torch.rand(2, 3, 4, 4)
        >>> snow = kornia.augmentation.RandomSnow(p=1.0, snow_coefficient=(0.1, 0.6), brightness=(1.0, 5.0))
        >>> output = snow(inputs)
        >>> output.shape
        torch.Size([2, 3, 4, 4])

    """

    def __init__(
        self,
        snow_coefficient: Tuple[float, float] = (0.5, 0.5),
        brightness: Tuple[float, float] = (2, 2),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        KORNIA_CHECK(all(0 <= el <= 1 for el in snow_coefficient), "Snow coefficient values must be between 0 and 1.")
        KORNIA_CHECK(all(1 <= el for el in brightness), "Brightness values must be 1 or greater.")

        self._param_generator = rg.PlainUniformGenerator(
            (snow_coefficient, "snow_coefficient", 0.5, (0.0, 1.0)), (brightness, "brightness", None, None)
        )

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        KORNIA_CHECK(input.shape[1] == 3, "Number of color channels should be 3.")
        KORNIA_CHECK(len(input.shape) in (3, 4), "Wrong input dimension.")

        if len(input.shape) == 3:
            input = input[None, :, :, :]
        input_HLS = rgb_to_hls(input)

        mask = torch.zeros_like(input_HLS)
        # Retrieve generated parameters
        snow_coefficient = params["snow_coefficient"].to(input)
        brightness = params["brightness"].to(input)
        snow_coefficient = snow_coefficient[:, None, None, None]
        brightness = brightness[:, None, None, None]

        mask[:, 1, :, :] = torch.where(input_HLS[:, 1, :, :] < snow_coefficient[:, 0, :, :], 1, 0)

        # Increase Light channel of the image by given brightness for areas based on snow coefficient.
        new_light = (input_HLS * mask * brightness).clamp(min=0.0, max=1.0)
        input_HLS = input_HLS * (1 - mask) + new_light

        output = hls_to_rgb(input_HLS)
        return output
