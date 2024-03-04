from __future__ import annotations

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor
from kornia.enhance.normalize import normalize_min_max
from kornia.utils import _extract_device_dtype


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

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        r"""Generate random 2D Gaussian illumination patterns."""
        batch_size, channels, height, width = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.gain, self.sign])

        gain_factor = _adapted_rsampling((batch_size, 1, 1, 1), self.gain_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        sign = torch.where(
            _adapted_rsampling((batch_size, 1, 1, 1), self.sign_sampler, same_on_batch) >= 0.0,
            torch.tensor(1),
            torch.tensor(-1),
        ).to(device=_device, dtype=_dtype)

        directions = _adapted_rsampling((batch_size, 1, 1, 1), self.directions_sampler, same_on_batch).to(
            device=_device,
            dtype=torch.int8,
        )

        gradient = torch.zeros(batch_shape)
        for _b in range(batch_size):
            if directions[_b] == 0:  # Lower
                gradient[_b] = torch.linspace(0, 1, height).unsqueeze(1).expand(channels, height, width)
            elif directions[_b] == 1:  # Upper
                gradient[_b] = torch.linspace(1, 0, height).unsqueeze(1).expand(channels, height, width)
            elif directions[_b] == 2:  # Left
                gradient[_b] = torch.linspace(0, 1, width).unsqueeze(0).expand(channels, height, width)
            elif directions[_b] == 3:  # Right
                gradient[_b] = torch.linspace(1, 0, width).unsqueeze(0).expand(channels, height, width)

        gradient = sign * gain_factor * gradient

        return {
            "gradient": gradient.to(device=_device, dtype=_dtype),
        }


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

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        r"""Generate random 2D Gaussian illumination patterns."""
        batch_size, channels, height, width = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.gain, self.sign])

        gain_factor = _adapted_rsampling((batch_size, 1, 1, 1), self.gain_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        sign = torch.where(
            _adapted_rsampling((batch_size, 1, 1, 1), self.sign_sampler, same_on_batch) >= 0.0,
            torch.tensor(1),
            torch.tensor(-1),
        ).to(device=_device, dtype=_dtype)

        directions = _adapted_rsampling((batch_size, 1, 1, 1), self.directions_sampler, same_on_batch).to(
            device=_device,
            dtype=torch.int8,
        )

        y_grad = torch.linspace(0, 1, height).unsqueeze(1).expand(channels, height, width)
        x_grad = torch.linspace(0, 1, width).unsqueeze(0).expand(channels, height, width)
        gradient = torch.zeros(batch_shape)
        for _b in range(batch_size):
            if directions[_b] == 0:  # Bottom right
                gradient[_b] = x_grad + y_grad
            elif directions[_b] == 1:  # Bottom left
                gradient[_b] = -x_grad + y_grad
            elif directions[_b] == 2:  # Upper right
                gradient[_b] = x_grad - y_grad
            elif directions[_b] == 3:  # Upper left
                gradient[_b] = 1 - (x_grad + y_grad)
        gradient = sign * gain_factor * normalize_min_max(gradient)

        return {
            "gradient": gradient.to(device=_device, dtype=_dtype),
        }
