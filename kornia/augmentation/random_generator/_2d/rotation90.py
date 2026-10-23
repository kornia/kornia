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

import torch

from kornia.augmentation.random_generator._2d.plain_uniform import PlainUniformGenerator
from kornia.augmentation.random_generator.base import UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core.utils import _extract_device_dtype

__all__ = ["Rotation90Generator"]


class Rotation90Generator(PlainUniformGenerator):
    r"""Generate the number of quarter turns for a batch of images.

    The range is validated exactly as :class:`PlainUniformGenerator` validates it, with ``0.0`` as the center and
    ``(-3, 3)`` as the bounds. A range of whole numbers is closed and uniform: every whole number from the lower to
    the upper bound is drawn with the same probability. A range with a fractional bound keeps the continuous draw.

    Args:
        times: the range of whole numbers of 90 degree turns. A ``(low, high)`` pair, a ``(2,)`` shaped torch.Tensor,
            or a single number ``x`` for ``(-x, x)``.

    Returns:
        A dict of parameters to be passed for transformation.
            - times (torch.Tensor): the number of quarter turns, with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, times: Any) -> None:
        super().__init__((times, "times", 0.0, (-3, 3)))
        self._whole_range = bool(self.times.to(torch.float64).round().eq(self.times).all())

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        super().make_samplers(device, dtype)
        if self._whole_range:
            # The sampler covers ``[lo, hi + 1)`` and ``forward`` floors the draw, so every whole number from ``lo``
            # to ``hi`` owns a bucket of width one. Rounding a draw on ``[lo, hi]`` gave the two end points half of
            # that width, so ``times=(-1, 1)`` drew ``0`` about twice as often as ``-1`` or ``1``.
            sampler = self.sampler_dict["times"]
            self.sampler_dict["times"] = UniformDistribution(sampler.low, sampler.high + 1, validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([t for t, _, _, _ in self.samplers])
        sampler = self.sampler_dict["times"]
        drawn = _adapted_rsampling((batch_size,), sampler, same_on_batch)
        if self._whole_range:
            # The clamp removes the excluded end point: for the largest draw ``u = 1 - 2 ** -24``,
            # ``lo + u * (hi + 1 - lo)`` rounds up onto ``hi + 1`` for a range such as ``(1, 1)``, which ``floor``
            # would keep, so a rare draw would turn a quarter turn into a half turn.
            drawn = drawn.floor().clamp(max=sampler.high - 1)
        return {"times": drawn.to(device=_device, dtype=_dtype)}
