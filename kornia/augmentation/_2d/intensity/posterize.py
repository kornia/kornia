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
from kornia.enhance import posterize


class RandomPosterize(IntensityAugmentationBase2D):
    r"""Posterize given torch.Tensor image or a batch of torch.Tensor images randomly.

    .. image:: _static/img/RandomPosterize.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        p: probability of applying the transformation.
        bits: number of high bits to keep, in ``[0, 8]``, in which 0 gives a constant image and 8 gives the
            original. A non-integral argument is accepted; the drawn factor is rounded to an integer, half to
            even, so ``2.5`` gives ``2`` and ``3.5`` gives ``4``.
            If int x, bits will be generated from (x, 8) then convert to int.
            If tuple (x, y), bits will be generated from (x, y) then convert to int.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Convention:
        - ``bits=(k, k)`` with ``k < 8`` leaves at most ``2 ** k`` distinct values, so ``0`` gives a constant
          image; the reduction is a ``uint8`` round trip inside :func:`kornia.enhance.posterize`. A sample
          that draws ``8`` is returned unchanged, without the round trip. The drawn factor is integral: a
          non-integral ``bits`` is rounded half to even.
        - the round trip is a step function, so a posterized sample carries no gradient. The output still has
          ``requires_grad=True``, but the gradient with respect to the input is identically ``0`` below
          ``bits=8``; at ``bits=8``, which skips the round trip, it is the identity.
        - an ``int`` argument is the lower bound of the sampled range ``[x, 8]`` -- the opposite reading
          from :class:`RandomSharpness`, whose scalar argument is an upper bound.

    .. warning::
        Outside ``[0, 1]`` the output is the posterized ``uint8`` conversion of the raw float, which
        wraps or saturates depending on the platform and torch version, and bears no relation to the
        clamped input: an input above ``1`` can come back as a full-range posterized image instead of a
        clipped one, and an all-negative input as an all-zero one. A sample that draws ``bits=8`` skips the
        conversion and keeps its out-of-range values, and a scalar ``bits`` below ``8`` can draw ``8`` for any
        sample. Tracked in `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

    .. note::
        This function internally uses :func:`kornia.enhance.posterize`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> posterize = RandomPosterize(3., p=1.)
        >>> posterize(input)
        tensor([[[[0.4863, 0.7529, 0.0784, 0.1255, 0.2980],
                  [0.6275, 0.4863, 0.8941, 0.4549, 0.6275],
                  [0.3451, 0.3922, 0.0157, 0.1569, 0.2824],
                  [0.5176, 0.6902, 0.8000, 0.1569, 0.2667],
                  [0.6745, 0.9098, 0.3922, 0.8627, 0.4078]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomPosterize(3., p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        bits: Union[float, Tuple[float, float], torch.Tensor] = 3,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        # TODO: the generator should receive the device
        self._param_generator = rg.PosterizeGenerator(bits)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return posterize(input, params["bits_factor"].to(input.device))
