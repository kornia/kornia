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

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import solarize


class RandomSolarize(IntensityAugmentationBase2D):
    r"""Solarize given torch.Tensor image or a batch of torch.Tensor images randomly.

    .. image:: _static/img/RandomSolarize.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        p: probability of applying the transformation.
        thresholds:
            If float x, threshold will be generated from (0.5 - x, 0.5 + x).
            If tuple (x, y), threshold will be generated from (x, y).
        additions:
            If float x, addition will be generated from (-x, x).
            If tuple (x, y), addition will be generated from (x, y).
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Convention:
        - the addition comes first: the drawn ``additions`` value is added to the whole image and the sum
          is clamped into ``[0, 1]``, and only then is everything at or above the drawn ``thresholds``
          replaced by ``1 - value``.
        - the scalar forms are centred, not absolute: a scalar ``thresholds`` is a half-width around
          ``0.5`` and a scalar ``additions`` a half-width around ``0``, so the class defaults centre on
          :func:`kornia.enhance.solarize`'s own default threshold rather than equalling it.
        - an explicit ``additions`` range is checked against the closed ``[-0.5, 0.5]`` at construction, and
          :func:`kornia.enhance.solarize` accepts the same closed interval, so a range that reaches an endpoint,
          or draws one exactly, is applied rather than rejected. Because the two intervals are now the same,
          the function's own forward check cannot fire for this class; it still applies to a direct
          :func:`kornia.enhance.solarize` call, on the device its own ``.. note::`` describes.

    .. warning::
        An input entirely outside ``[0, 1]`` can collapse at either end. An all-negative input can come back
        as an all-zero image when the sampled addition does not raise it above zero; a positive sampled
        addition can recover values instead, and a drawn threshold of ``0`` inverts the clamped zeros into
        an all-ones image. At the upper end, ``clamp(x + a, 0, 1)``
        sends every ``x >= 1.5`` to exactly ``1.0`` for any admissible ``a``, and the inversion then returns
        ``1 - 1.0 == 0``, so an input drawn from ``[1.5, 3.0]`` is an all-zero image on every draw.
        Values between ``1`` and ``1.5`` can instead produce nonzero output after a negative addition:
        ``x=1.1``, ``additions=(-0.4, -0.4)`` and ``thresholds=(0.5, 0.5)`` give ``0.3``. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

    .. note::
        This function internally uses :func:`kornia.enhance.solarize`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> solarize = RandomSolarize(0.1, 0.1, p=1.)
        >>> solarize(input)
        tensor([[[[0.4132, 0.1412, 0.1790, 0.2226, 0.3980],
                  [0.2754, 0.4194, 0.0130, 0.4538, 0.2771],
                  [0.4394, 0.4923, 0.1129, 0.2594, 0.3844],
                  [0.3909, 0.2118, 0.1094, 0.2516, 0.3728],
                  [0.2278, 0.0000, 0.4876, 0.0353, 0.5100]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomSolarize(0.1, 0.1, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        thresholds: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.1,
        additions: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.1,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (thresholds, "thresholds", 0.5, (0.0, 1.0)), (additions, "additions", 0.0, (-0.5, 0.5))
        )

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        thresholds = params["thresholds"]
        additions: Optional[torch.Tensor]
        if "additions" in params:
            additions = params["additions"]
        else:
            additions = None
        return solarize(input, thresholds, additions)
