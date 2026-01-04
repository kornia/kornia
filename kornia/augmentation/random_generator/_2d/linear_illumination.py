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

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core.utils import _extract_device_dtype
from kornia.enhance.normalize import normalize_min_max


class LinearIlluminationGenerator(RandomGeneratorBase):
    r"""Generates random 2D Linear illumination patterns for image augmentation.

    Args:
        gain: Range for the gain factor applied to the generated illumination.
        sign: Range for the sign of the Linear distribution.

    Returns:
        A dictionary of parameters to be passed for transformation.
            - gradient: : Generated 2D Linear illumination pattern with shape (B, C, H, W).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(
        self,
        gain: tuple[float, float],
        sign: tuple[float, float],
    ) -> None:
        super().__init__()
        self.gain = gain
        self.sign = sign

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        repr = f"gain={self.gain}, sign={self.sign}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        r"""Create samplers for generating random gaussian illumination parameters."""
        gain = _range_bound(
            self.gain,
            "gain",
        ).to(device, dtype)
        self.gain_sampler = UniformDistribution(gain[0], gain[1], validate_args=False)

        sign = _range_bound(self.sign, "sign", bounds=(-1.0, 1.0), center=0.0).to(device, dtype)
        self.sign_sampler = UniformDistribution(sign[0], sign[1], validate_args=False)

        self.directions_sampler = UniformDistribution(0, 4, validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        r"""Generate random 2D Gaussian illumination patterns."""
        batch_size, channels, height, width = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.gain, self.sign])

        # Random gain and sign
        gain_factor = _adapted_rsampling((batch_size, 1, 1, 1), self.gain_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        sign = torch.where(
            _adapted_rsampling((batch_size, 1, 1, 1), self.sign_sampler, same_on_batch) >= 0.0,
            torch.tensor(1, device=_device, dtype=_dtype),
            torch.tensor(-1, device=_device, dtype=_dtype),
        )

        # Directions (0=lower,1=upper,2=left,3=right), shape [B,1,1,1]
        directions = _adapted_rsampling((batch_size, 1, 1, 1), self.directions_sampler, same_on_batch).to(
            device=_device, dtype=torch.int8
        )

        # Precompute 1D ramps
        ramp_h = torch.linspace(0, 1, height, device=_device, dtype=_dtype).view(1, 1, height, 1)
        ramp_h_rev = torch.linspace(1, 0, height, device=_device, dtype=_dtype).view(1, 1, height, 1)
        ramp_w = torch.linspace(0, 1, width, device=_device, dtype=_dtype).view(1, 1, 1, width)
        ramp_w_rev = torch.linspace(1, 0, width, device=_device, dtype=_dtype).view(1, 1, 1, width)

        # Broadcast masks for each direction
        d = directions.to(torch.int64)  # [B,1,1,1]
        m0 = (d == 0).to(_dtype)  # lower
        m1 = (d == 1).to(_dtype)  # upper
        m2 = (d == 2).to(_dtype)  # left
        m3 = (d == 3).to(_dtype)  # right

        # Build [B,1,H,W] gradient in one shot
        grad_b1 = m0 * ramp_h + m1 * ramp_h_rev + m2 * ramp_w + m3 * ramp_w_rev

        # Expand to [B,C,H,W]
        gradient = grad_b1.expand(batch_size, channels, height, width)

        # Apply sign and gain
        gradient = sign * gain_factor * gradient

        return {"gradient": gradient.to(device=_device, dtype=_dtype)}


class LinearCornerIlluminationGenerator(RandomGeneratorBase):
    r"""Generates random 2D Linear (from corner) illumination patterns for image augmentation.

    Args:
        gain: Range for the gain factor applied to the generated illumination.
        sign: Range for the sign of the linear distribution.

    Returns:
        A dictionary of parameters to be passed for transformation.
            - gradient: : Generated 2D Linear illumination pattern with shape (B, C, H, W).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(
        self,
        gain: tuple[float, float],
        sign: tuple[float, float],
    ) -> None:
        super().__init__()
        self.gain = gain
        self.sign = sign

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        repr = f"gain={self.gain}, sign={self.sign}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        r"""Create samplers for generating random gaussian illumination parameters."""
        gain = _range_bound(
            self.gain,
            "gain",
        ).to(device, dtype)
        self.gain_sampler = UniformDistribution(gain[0], gain[1], validate_args=False)

        sign = _range_bound(self.sign, "sign", bounds=(-1.0, 1.0), center=0.0).to(device, dtype)
        self.sign_sampler = UniformDistribution(sign[0], sign[1], validate_args=False)

        self.directions_sampler = UniformDistribution(0, 4, validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        r"""Generate random 2D Gaussian illumination patterns."""
        batch_size, channels, height, width = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.gain, self.sign])

        gain_factor = _adapted_rsampling((batch_size, 1, 1, 1), self.gain_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        sign = torch.where(
            _adapted_rsampling((batch_size, 1, 1, 1), self.sign_sampler, same_on_batch) >= 0.0,
            torch.tensor(1, device=_device, dtype=_dtype),
            torch.tensor(-1, device=_device, dtype=_dtype),
        )

        directions = _adapted_rsampling((batch_size, 1, 1, 1), self.directions_sampler, same_on_batch).to(
            device=_device,
            dtype=torch.long,  # int8 → long for gather
        )

        # Compute base gradients
        y_grad = torch.linspace(0, 1, height, device=_device, dtype=_dtype).unsqueeze(1).expand(height, width)
        x_grad = torch.linspace(0, 1, width, device=_device, dtype=_dtype).unsqueeze(0).expand(height, width)

        base = torch.stack(
            [
                x_grad + y_grad,  # 0: Bottom right
                -x_grad + y_grad,  # 1: Bottom left
                x_grad - y_grad,  # 2: Upper right
                1 - (x_grad + y_grad),  # 3: Upper left
            ],
            dim=0,
        )  # (4, H, W)

        # Expand to (4, C, H, W)
        base = base.unsqueeze(1).expand(-1, channels, -1, -1)

        # Index according to directions
        gradient = base[directions.view(-1), :, :, :]  # (B, C, H, W)

        gradient = sign * gain_factor * normalize_min_max(gradient)

        return {
            "gradient": gradient.to(device=_device, dtype=_dtype),
        }
