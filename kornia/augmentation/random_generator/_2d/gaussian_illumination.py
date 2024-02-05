from __future__ import annotations

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor
from kornia.utils import _extract_device_dtype


def gaussian(window_size, sigma, center=None):
    if center is None:
        center = int((window_size - 1) / 2)

    x = torch.arange(window_size).float() - center
    gauss = torch.exp(-x.pow(2.0) / float(2 * sigma**2))

    return gauss / gauss.sum()


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

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        repr = f"gain={self.gain}, center={self.center}, sigma={self.sigma}, sign={self.sign}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        r"""Create samplers for generating random gaussian illumination parameters."""

        gain = _range_bound(
            self.gain,
            "gain",
        ).to(device, dtype)
        self.gain_sampler = UniformDistribution(gain[0], gain[1], validate_args=False)

        center = _range_bound(
            self.center,
            "center",
        ).to(device, dtype)
        self.center_sampler = UniformDistribution(center[0], center[1], validate_args=False)

        sigma = _range_bound(
            self.sigma,
            "sigma",
        ).to(device, dtype)
        self.sigma_sampler = UniformDistribution(sigma[0], sigma[1], validate_args=False)

        sign = _range_bound(self.sign, "sign", bounds=(-1.0, 1.0), center=0.0).to(device, dtype)
        self.sign_sampler = UniformDistribution(sign[0], sign[1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        r"""Generate random 2D Gaussian illumination patterns."""
        batch_size, channels, height, width = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.gain, self.center, self.sigma])

        gain_factor = _adapted_rsampling((batch_size,), self.gain_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        sigma_x = width * _adapted_rsampling((batch_size,), self.sigma_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        center_x = torch.round(width * _adapted_rsampling((batch_size,), self.center_sampler, same_on_batch)).to(
            device=_device, dtype=_dtype
        )
        sigma_y = height * _adapted_rsampling((batch_size,), self.sigma_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        center_y = torch.round(height * _adapted_rsampling((batch_size,), self.center_sampler, same_on_batch)).to(
            device=_device, dtype=_dtype
        )
        sign = torch.where(
            _adapted_rsampling((batch_size,), self.sign_sampler, same_on_batch) >= 0.0,
            torch.tensor(1),
            torch.tensor(-1),
        ).to(device=_device, dtype=_dtype)

        gradient = torch.zeros(batch_shape)
        for batch in range(batch_size):
            gauss_x = gaussian(width, sigma_x[batch], center_x[batch])
            gauss_y = gaussian(height, sigma_y[batch], center_y[batch])
            grad = torch.matmul(gauss_x.unsqueeze(-1), gauss_y.unsqueeze(-1).t()).unsqueeze(0).unsqueeze(0)
            # Normalize between 0-1 to apply the gain factor.
            grad = (grad - grad.min()) / (grad.max() - grad.min())
            grad = grad.repeat(1, channels, 1, 1)
            gradient[batch, :, :, :] = sign[batch] * gain_factor[batch] * grad

        return {
            "gradient": gradient.to(device=_device, dtype=_dtype),
        }
