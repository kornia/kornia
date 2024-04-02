from __future__ import annotations

import torch

from kornia.augmentation.random_generator.base import (
    RandomGeneratorBase,
    UniformDistribution,
)
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _common_param_check,
    _range_bound,
)
from kornia.core import Tensor
from kornia.enhance import normalize_min_max
from kornia.filters.kernels import gaussian


class GaussianIlluminationGenerator(RandomGeneratorBase):
    r"""Generates random 2D Gaussian illumination patterns for image augmentation.

    Args:
        gain: Range for the gain factor applied to the generated illumination.
        center: Range for the center coordinates of the Gaussian distribution.
        sigma: Range for the standard deviation of the Gaussian distribution.
        sign: Range for the sign of the Gaussian distribution.

    Returns:
        A dictionary of parameters to be passed for transformation.
            - gradient: : Generated 2D Gaussian illumination pattern with shape (B, C, H, W).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        gain: tuple[float, float],
        center: tuple[float, float],
        sigma: tuple[float, float],
        sign: tuple[float, float],
    ) -> None:
        super().__init__()
        self.gain = gain
        self.center = center
        self.sigma = sigma
        self.sign = sign

        self.gain_sampler: UniformDistribution
        self.center_sampler: UniformDistribution
        self.sigma_sampler: UniformDistribution
        self.sign_sampler: UniformDistribution

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        repr_buf = f"gain={self.gain}, center={self.center}, sigma={self.sigma}, sign={self.sign}"
        return repr_buf

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        r"""Create samplers for generating random gaussian illumination parameters."""

        gain = _range_bound(self.gain, "gain", device=device, dtype=dtype)
        self.gain_sampler = UniformDistribution(gain[0], gain[1], validate_args=False)

        center = _range_bound(self.center, "center", device=device, dtype=dtype)
        self.center_sampler = UniformDistribution(center[0], center[1], validate_args=False)

        sigma = _range_bound(self.sigma, "sigma", device=device, dtype=dtype)
        self.sigma_sampler = UniformDistribution(sigma[0], sigma[1], validate_args=False)

        sign = _range_bound(
            self.sign,
            "sign",
            bounds=(-1.0, 1.0),
            center=0.0,
            device=device,
            dtype=dtype,
        )
        self.sign_sampler = UniformDistribution(sign[0], sign[1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        r"""Generate random 2D Gaussian illumination patterns."""
        batch_size, channels, height, width = batch_shape

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = self.device, self.dtype

        # TODO: check whether we need generate all the parameters at once

        gain_factor = _adapted_rsampling((batch_size, 1, 1, 1), self.gain_sampler, same_on_batch)

        sigma_x = width * _adapted_rsampling((batch_size, 1), self.sigma_sampler, same_on_batch)

        center_x = torch.round(width * _adapted_rsampling((batch_size, 1), self.center_sampler, same_on_batch))

        sigma_y = height * _adapted_rsampling((batch_size, 1), self.sigma_sampler, same_on_batch)

        center_y = torch.round(height * _adapted_rsampling((batch_size, 1), self.center_sampler, same_on_batch))

        sign = torch.where(
            _adapted_rsampling((batch_size, 1, 1, 1), self.sign_sampler, same_on_batch) >= 0.0,
            torch.tensor(1.0, device=_device, dtype=_dtype),
            torch.tensor(-1.0, device=_device, dtype=_dtype),
        )

        # Generate random gaussian for create a 2D gaussian image.
        gauss_x = gaussian(width, sigma_x, mean=center_x, device=_device, dtype=_dtype).unsqueeze(1)

        gauss_y = gaussian(height, sigma_y, mean=center_y, device=_device, dtype=_dtype).unsqueeze(2)

        # gradient = (batch_size, channels, height, width)
        gradient = (gauss_y @ gauss_x).unsqueeze_(1).repeat(1, channels, 1, 1)

        # TODO: this will crash if the shape is not batched
        # Normalize between 0-1 to apply the gain factor.
        gradient = normalize_min_max(gradient, min_val=0.0, max_val=1.0)
        gradient = sign.mul_(gain_factor).mul(gradient)

        return {"gradient": gradient}
