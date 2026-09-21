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
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core.utils import _extract_device_dtype

__all__ = ["MotionBlurGenerator"]


class MotionBlurGenerator(RandomGeneratorBase):
    r"""Get parameters for motion blur.

    Args:
        kernel_size: motion kernel size (odd and positive).
            If int, the kernel will have a fixed size.
            If Tuple[int, int], it will randomly generate one value from the range for the whole batch.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
            If float, it will generate the value from (-angle, angle).
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If float, it will generate the value from (-direction, direction).
            If Tuple[int, int], it will randomly generate the value from the range.

    Returns:
        A dict of parameters to be passed for transformation.
            - ksize_factor (torch.Tensor): one shared kernel size repeated to a shape of (B,).
            - angle_factor (torch.Tensor): element-wise angle factors with a shape of (B,).
            - direction_factor (torch.Tensor): element-wise direction factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        angle: Union[torch.Tensor, float, Tuple[float, float]],
        direction: Union[torch.Tensor, float, Tuple[float, float]],
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

    def __repr__(self) -> str:
        return f"kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        angle = _range_bound(self.angle, "angle", center=0.0, bounds=(-360, 360)).to(device=device, dtype=dtype)
        direction = _range_bound(self.direction, "direction", center=0.0, bounds=(-1, 1)).to(device=device, dtype=dtype)
        if isinstance(self.kernel_size, int):
            if not (self.kernel_size >= 3 and self.kernel_size % 2 == 1):
                raise AssertionError(f"`kernel_size` must be odd and greater than 3. Got {self.kernel_size}.")
            self.ksize_sampler = UniformDistribution(self.kernel_size // 2, self.kernel_size // 2, validate_args=False)
            self._ksize_half_max = self.kernel_size // 2
        elif isinstance(self.kernel_size, tuple):
            # kernel_size is fixed across the batch
            if len(self.kernel_size) != 2:
                raise AssertionError(f"`kernel_size` must be (2,) if it is a tuple. Got {self.kernel_size}.")
            # Draw the half-size h of the odd kernel 2h + 1 on [lo, hi + 1) and floor it, so every h in the
            # closed [lo, hi] is equally likely; truncating a draw on [lo, hi) never reached hi.
            # hi is the largest odd size not above the upper bound, and never below lo, which keeps an
            # even-only range such as (4, 4) drawing 5 as before.
            # A reversed pair is what the `max(half_lo, ...)` below would otherwise turn into a draw
            # above *both* bounds: `(20, 3)` drew a constant 21.  #4568 set the precedent for
            # RandomRain's closed integer ranges -- refuse it here rather than sample outside it.
            if self.kernel_size[0] > self.kernel_size[1]:
                raise ValueError(
                    f"`kernel_size`[0] should be smaller than or equal to `kernel_size`[1]. Got {self.kernel_size}."
                )
            half_lo = self.kernel_size[0] // 2
            half_hi = max(half_lo, (self.kernel_size[1] - 1) // 2)
            self.ksize_sampler = UniformDistribution(half_lo, half_hi + 1, validate_args=False)
            self._ksize_half_max = half_hi
        else:
            raise TypeError(f"Unsupported type: {type(self.kernel_size)}")

        self.angle_sampler = UniformDistribution(angle[0], angle[1], validate_args=False)
        self.direction_sampler = UniformDistribution(direction[0], direction[1], validate_args=False)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        # self.ksize_factor.expand((batch_size, -1))
        _device, _dtype = _extract_device_dtype([self.angle, self.direction])
        angle_factor = _adapted_rsampling((batch_size,), self.angle_sampler, same_on_batch)
        direction_factor = _adapted_rsampling((batch_size,), self.direction_sampler, same_on_batch)
        # A ranged kernel size is shared by the batch; angle and direction can still vary per sample.
        ksize_same_on_batch = same_on_batch or isinstance(self.kernel_size, tuple)
        ksize_half = _adapted_rsampling((batch_size,), self.ksize_sampler, ksize_same_on_batch).floor()
        # A float32 draw can round up onto the open upper end, hi + 1; keep it inside the closed range.
        ksize_factor = ksize_half.clamp_max(self._ksize_half_max).int() * 2 + 1

        return {
            "ksize_factor": ksize_factor.to(device=_device, dtype=torch.int32),
            "angle_factor": angle_factor.to(device=_device, dtype=_dtype),
            "direction_factor": direction_factor.to(device=_device, dtype=_dtype),
        }
