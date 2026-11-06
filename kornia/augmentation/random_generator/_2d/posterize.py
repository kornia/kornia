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

from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check
from kornia.augmentation.utils.random import _with_generator
from kornia.core.utils import _extract_device_dtype

__all__ = ["PosterizeGenerator"]


class PosterizeGenerator(RandomGeneratorBase):
    r"""Generate random posterize parameters for a batch of images.

    A range of whole numbers is closed and uniform: every whole number from the lower to the upper bound is
    drawn with the same probability. A range with a fractional bound keeps the continuous draw and rounds it.

    Args:
        bits: floats that ranged from [0, 8], in which 0 gives black image and 8 gives the original.
            If float x, bits will be generated from (x, 8).
            If tuple (x, y), bits will be generated from (x, y).

    Returns:
        A dict of parameters to be passed for transformation.
            - bits_factor (torch.Tensor): element-wise bit factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, bits: Union[float, Tuple[float, float], torch.Tensor]) -> None:
        super().__init__()
        self.bits_factor = bits

    def __repr__(self) -> str:
        return f"bits={self.bits_factor}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        r"""Create uniform distribution samplers for the bits parameter on the target device."""
        bits = torch.as_tensor(self.bits_factor, device=device, dtype=dtype)
        if len(bits.size()) == 0:
            bits = bits.repeat(2)
            bits[1] = 8
        elif not (len(bits.size()) == 1 and bits.size(0) == 2):
            raise ValueError(f"'bits' shall be either a scalar or a length 2 torch.Tensor. Got {bits}.")
        _joint_range_check(bits, "bits", (0, 8))
        self._whole_range = bool(bits.round().eq(bits).all())
        if self._whole_range:
            # The sampler covers ``[lo, hi + 1)`` and ``forward`` floors the draw, so every whole number from ``lo``
            # to ``hi`` owns a bucket of width one. Rounding a draw on ``[lo, hi]`` gave the two end points half of
            # that width, so ``bits=(0, 8)`` drew ``0`` and ``8`` about half as often as the interior values (#6334).
            self.bit_sampler = UniformDistribution(bits[0], bits[1] + 1, validate_args=False)
        else:
            self.bit_sampler = UniformDistribution(bits[0], bits[1], validate_args=False)

    @_with_generator
    def forward(
        self, batch_shape: Tuple[int, ...], same_on_batch: bool = False, *, generator: torch.Generator | None = None
    ) -> Dict[str, torch.Tensor]:
        r"""Sample random bits factors for the given batch shape."""
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _ = _extract_device_dtype([self.bits_factor if isinstance(self.bits_factor, torch.Tensor) else None])
        bits_factor = _adapted_rsampling((batch_size,), self.bit_sampler, same_on_batch, generator=self._generator)
        if getattr(self, "_whole_range", False):
            # The clamp removes the excluded end point: for the largest draw ``u = 1 - 2 ** -24``,
            # ``lo + u * (hi + 1 - lo)`` rounds up onto ``hi + 1`` for a range such as ``(1, 1)``,
            # which ``floor`` would keep.
            bits_factor = bits_factor.floor().clamp(max=self.bit_sampler.high - 1)
        else:
            bits_factor = bits_factor.round()
        return {"bits_factor": bits_factor.to(device=_device, dtype=torch.int32)}
