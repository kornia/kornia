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

from typing import Any, Dict, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _check_filter_min_size
from kornia.enhance import sharpness


class RandomSharpness(IntensityAugmentationBase2D):
    r"""Blend a random amount of a ``3 x 3`` smoothed copy with the image.

    A drawn factor above ``1`` sharpens and a factor below ``1`` blurs; the default
    ``sharpness=0.5`` draws from ``[0, 0.5]`` and therefore only ever blurs.

    .. image:: _static/img/RandomSharpness.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        p: probability of applying the transformation.
        sharpness: the blend factor between the blurred image and the input. If ``sharpness`` is a single
            non-negative number ``x``, the factor is sampled from ``[0, x]``; a tuple gives the range
            directly.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Convention:
        - the factor blends between the fully blurred image at ``0`` and the input at ``1``, and values
          above ``1`` sharpen. The one-pixel border is copied from the input at every factor, so it is never
          blurred or sharpened, although the final clamp into ``[0, 1]`` applies to it as to the rest.
        - a scalar argument is the upper bound of ``[0, x]`` -- the centred ``[-x, x]`` with its lower end
          floored at the non-negative bound -- so the default ``sharpness=0.5`` never reaches the identity
          and therefore never sharpens -- it blurs by a random amount.
        - the result is kept inside ``[0, 1]``.

    .. warning::
        An input whose values are all negative comes back as an all-zero image. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

    .. note::
        The smoothing kernel is a fixed ``3 x 3`` convolved with no padding, so both spatial sides
        must be at least ``3`` pixels; a smaller image raises a ``ValueError`` naming the class and
        the input shape.

    .. note::
        This function internally uses :func:`kornia.enhance.sharpness`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> sharpness = RandomSharpness(1., p=1.)
        >>> sharpness(input)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4810, 0.7367, 0.4177, 0.6323],
                  [0.3489, 0.4428, 0.1562, 0.2443, 0.2939],
                  [0.5185, 0.6462, 0.7050, 0.2288, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomSharpness(1., p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        sharpness: Union[torch.Tensor, float, Tuple[float, float]] = 0.5,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator((sharpness, "sharpness", 0.0, (0, float("inf"))))

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # the smoothing kernel is a fixed 3 x 3 convolved without padding of its own
        _check_filter_min_size("RandomSharpness", input, 3, border_type="valid")
        factor = params["sharpness"]
        return sharpness(input, factor)
